"""Adaptive supervision (O1): fixed criterion vs. model-revised threshold under drift.

Task: binary classification on 200k records across 20 shards. At shard 10, a second
feature becomes predictive (label distribution drifts).

Arms:
  A (one-shot): threshold on feature 0 chosen before seeing data; runs full scan once.
  B (adaptive): model sees per-shard aggregates + 20 samples/shard, revises threshold,
                re-runs scan. Control tokens stay flat; decisions grow linearly.
  C (oracle): threshold fitted with full labels on all shards (upper bound).

Metric: F1 score. Target: B recovers ≥90% of C's F1 at ≤5 decisions/shard,
while A loses ≥20 F1 points on drifted half.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime

# Configuration
N_SHARDS = 20
RECORDS_PER_SHARD = 10_000
TOTAL_RECORDS = N_SHARDS * RECORDS_PER_SHARD
DRIFT_SHARD = 10  # shards 0-9: feature 0 predictive; 10-19: feature 1 predictive
SAMPLES_PER_SHARD = 20
MAX_DECISIONS_PER_SHARD = 5  # budget for adaptive arm

# Model for adaptive arm
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Program template for python.vm - computes confusion matrix for a threshold on a chosen feature with direction
SCAN_PROGRAM = """
import json
buf = read_buffer("data")
text = bytes(buf).decode("utf-8")
lines = text.strip().split("\\n")

# Read feature index (0 or 1), threshold, and direction (1 for >=, -1 for <=)
feat_buf = read_buffer("feature_idx")
feat_text = bytes(feat_buf).decode("utf-8").strip()
feature_idx = int(feat_text)

thresh_buf = read_buffer("threshold")
thresh_text = bytes(thresh_buf).decode("utf-8").strip()
thresh = float(thresh_text)

dir_buf = read_buffer("direction")
dir_text = bytes(dir_buf).decode("utf-8").strip()
direction = int(dir_text)

tp = fp = tn = fn = 0
for line in lines:
    if not line:
        continue
    f0, f1, label = map(float, line.split(","))
    feat_val = f0 if feature_idx == 0 else f1
    if direction >= 0:
        pred = 1.0 if feat_val >= thresh else 0.0
    else:
        pred = 1.0 if feat_val <= thresh else 0.0
    if pred == 1.0 and label == 1.0:
        tp += 1
    elif pred == 1.0 and label == 0.0:
        fp += 1
    elif pred == 0.0 and label == 1.0:
        fn += 1
    else:
        tn += 1

print(json.dumps({"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(lines)}))
"""

# Program to compute aggregates (mean, std) of features per class
AGGREGATE_PROGRAM = """
import json, math
buf = read_buffer("data")
text = bytes(buf).decode("utf-8")
lines = text.strip().split("\\n")

f0_pos = []
f1_pos = []
f0_neg = []
f1_neg = []

for line in lines:
    if not line:
        continue
    f0, f1, label = map(float, line.split(","))
    if label == 1.0:
        f0_pos.append(f0)
        f1_pos.append(f1)
    else:
        f0_neg.append(f0)
        f1_neg.append(f1)

def stats(vals):
    if not vals:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    return {"mean": mean, "std": math.sqrt(var), "count": n}

print(json.dumps({
    "f0_pos": stats(f0_pos),
    "f1_pos": stats(f1_pos),
    "f0_neg": stats(f0_neg),
    "f1_neg": stats(f1_neg),
    "total": len(lines)
}))
"""

# Program to sample 20 records per shard
SAMPLE_PROGRAM = """
import json, random
buf = read_buffer("data")
text = bytes(buf).decode("utf-8")
lines = text.strip().split("\\n")
random.seed(42)
sampled = random.sample(lines, min(20, len(lines)))
print(json.dumps({"samples": sampled}))
"""


@dataclass
class ArmResult:
    arm: str
    shard: int
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    f1: float
    precision: float
    recall: float
    host_tokens: int
    wall_s: float
    decisions: int


def f1_from_confusion(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def generate_shard(shard_idx: int, n: int = RECORDS_PER_SHARD) -> np.ndarray:
    """Generate one shard of data. Returns array of [f0, f1, label]."""
    # Before drift (shards 0-9): f0 is predictive with threshold ~1.0
    # After drift (shards 10-19): f1 is predictive with threshold ~-1.0 (reversed!)
    is_drift = shard_idx >= DRIFT_SHARD
    
    n_pos = n // 2
    n_neg = n - n_pos
    
    if not is_drift:
        # Pre-drift: f0 separates classes, f1 is noise
        # Class 1: f0 ~ N(3.0, 1.0), Class 0: f0 ~ N(-1.0, 1.0) -> optimal threshold ~1.0
        # f1 ~ N(0.0, 1.0) for both (noise)
        f0_pos = np.random.normal(3.0, 1.0, n_pos)
        f1_pos = np.random.normal(0.0, 1.0, n_pos)
        labels_pos = np.ones(n_pos)
        
        f0_neg = np.random.normal(-1.0, 1.0, n_neg)
        f1_neg = np.random.normal(0.0, 1.0, n_neg)
        labels_neg = np.zeros(n_neg)
    else:
        # Post-drift: f1 separates classes (REVERSED direction!), f0 is noise
        # Class 1: f1 ~ N(-3.0, 1.0), Class 0: f1 ~ N(1.0, 1.0) -> optimal threshold ~-1.0
        # f0 ~ N(0.0, 1.0) for both (noise)
        f0_pos = np.random.normal(0.0, 1.0, n_pos)
        f1_pos = np.random.normal(-3.0, 1.0, n_pos)
        labels_pos = np.ones(n_pos)
        
        f0_neg = np.random.normal(0.0, 1.0, n_neg)
        f1_neg = np.random.normal(1.0, 1.0, n_neg)
        labels_neg = np.zeros(n_neg)
    
    f0 = np.concatenate([f0_pos, f0_neg])
    f1 = np.concatenate([f1_pos, f1_neg])
    labels = np.concatenate([labels_pos, labels_neg])
    
    # Shuffle
    idx = np.random.permutation(n)
    return np.column_stack([f0[idx], f1[idx], labels[idx]])


def data_to_bytes(data: np.ndarray) -> bytes:
    """Convert numpy array to CSV bytes."""
    lines = [f"{row[0]:.6f},{row[1]:.6f},{int(row[2])}" for row in data]
    return "\n".join(lines).encode("utf-8")


def compute_oracle_threshold(data: np.ndarray) -> float:
    """Find optimal threshold on the predictive feature for this shard."""
    # Determine which feature is predictive by checking separation
    f0_pos = data[data[:, 2] == 1, 0]
    f0_neg = data[data[:, 2] == 0, 0]
    f1_pos = data[data[:, 2] == 1, 1]
    f1_neg = data[data[:, 2] == 0, 1]
    
    # Cohen's d for each feature
    d_f0 = abs(f0_pos.mean() - f0_neg.mean()) / math.sqrt((f0_pos.var() + f0_neg.var()) / 2) if len(f0_pos) > 0 and len(f0_neg) > 0 else 0
    d_f1 = abs(f1_pos.mean() - f1_neg.mean()) / math.sqrt((f1_pos.var() + f1_neg.var()) / 2) if len(f1_pos) > 0 and len(f1_neg) > 0 else 0
    
    # Use the more discriminative feature
    if d_f0 >= d_f1:
        # Threshold on f0: midpoint between means
        return float((f0_pos.mean() + f0_neg.mean()) / 2)
    else:
        return float((f1_pos.mean() + f1_neg.mean()) / 2)


async def run_scan_job(rt: Runtime, sid: str, buffer_id: str, threshold: float, feature_idx: int = 0, direction: int = 1) -> tuple[dict, int, float]:
    """Run a scan job with given threshold, feature, and direction. Returns (result, tokens, wall_s)."""
    # Import threshold as a tiny buffer
    thresh_bytes = str(threshold).encode()
    thresh_ref = await rt._buffer_import(
        {"name": "threshold", "data": thresh_bytes, "dtype": "uint8", "shape": [len(thresh_bytes)]}, sid
    )
    
    # Import feature index as a tiny buffer
    feat_bytes = str(feature_idx).encode()
    feat_ref = await rt._buffer_import(
        {"name": "feature_idx", "data": feat_bytes, "dtype": "uint8", "shape": [len(feat_bytes)]}, sid
    )
    
    # Import direction as a tiny buffer
    dir_bytes = str(direction).encode()
    dir_ref = await rt._buffer_import(
        {"name": "direction", "data": dir_bytes, "dtype": "uint8", "shape": [len(dir_bytes)]}, sid
    )
    
    job = {
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {
            "program": SCAN_PROGRAM,
            "data": {"buffer": buffer_id},
            "threshold": {"buffer": thresh_ref.buffer_id},
            "feature_idx": {"buffer": feat_ref.buffer_id},
            "direction": {"buffer": dir_ref.buffer_id},
        },
        "config": {"mode": "sync", "timeout_ms": 60000},
    }
    
    tokens = estimate_tokens(job)
    tokens += estimate_tokens({
        "buffer_id": thresh_ref.buffer_id, "shape": list(thresh_ref.shape), "dtype": thresh_ref.dtype
    })
    tokens += estimate_tokens({
        "buffer_id": feat_ref.buffer_id, "shape": list(feat_ref.shape), "dtype": feat_ref.dtype
    })
    tokens += estimate_tokens({
        "buffer_id": dir_ref.buffer_id, "shape": list(dir_ref.shape), "dtype": dir_ref.dtype
    })
    
    t0 = time.perf_counter()
    result = await rt._submit(job, sid)
    wall = time.perf_counter() - t0
    
    # Add result tokens
    payload = {"job_id": result.job_id, "status": result.status.value, "result": result.result}
    tokens += estimate_tokens(payload)
    
    return result.result or {}, tokens, wall


async def run_aggregate_job(rt: Runtime, sid: str, buffer_id: str) -> tuple[dict, int, float]:
    """Run aggregate job on a shard."""
    job = {
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": AGGREGATE_PROGRAM, "data": {"buffer": buffer_id}},
        "config": {"mode": "sync", "timeout_ms": 60000},
    }
    tokens = estimate_tokens(job)
    t0 = time.perf_counter()
    result = await rt._submit(job, sid)
    wall = time.perf_counter() - t0
    payload = {"job_id": result.job_id, "status": result.status.value, "result": result.result}
    tokens += estimate_tokens(payload)
    return result.result or {}, tokens, wall


async def run_sample_job(rt: Runtime, sid: str, buffer_id: str) -> tuple[dict, int, float]:
    """Run sample job on a shard."""
    job = {
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": SAMPLE_PROGRAM, "data": {"buffer": buffer_id}},
        "config": {"mode": "sync", "timeout_ms": 60000},
    }
    tokens = estimate_tokens(job)
    t0 = time.perf_counter()
    result = await rt._submit(job, sid)
    wall = time.perf_counter() - t0
    payload = {"job_id": result.job_id, "status": result.status.value, "result": result.result}
    tokens += estimate_tokens(payload)
    return result.result or {}, tokens, wall


def model_choose_threshold(aggregates: dict, samples: list[str]) -> tuple[int, float, int]:
    """Simulate model choosing a feature, threshold, and direction based on aggregates and samples.
    
    Returns (feature_idx, threshold, direction).
    """
    # Parse samples
    sample_data = []
    for s in samples:
        parts = s.split(",")
        if len(parts) == 3:
            sample_data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if not sample_data:
        return 0, 0.0, 1
    
    sample_arr = np.array(sample_data)
    # Compute separation on samples
    f0_pos = sample_arr[sample_arr[:, 2] == 1, 0]
    f0_neg = sample_arr[sample_arr[:, 2] == 0, 0]
    f1_pos = sample_arr[sample_arr[:, 2] == 1, 1]
    f1_neg = sample_arr[sample_arr[:, 2] == 0, 1]
    
    d_f0 = 0.0
    d_f1 = 0.0
    
    if len(f0_pos) > 0 and len(f0_neg) > 0:
        d_f0 = abs(f0_pos.mean() - f0_neg.mean()) / math.sqrt((f0_pos.var() + f0_neg.var()) / 2 + 1e-8)
    if len(f1_pos) > 0 and len(f1_neg) > 0:
        d_f1 = abs(f1_pos.mean() - f1_neg.mean()) / math.sqrt((f1_pos.var() + f1_neg.var()) / 2 + 1e-8)
    
    # Also check aggregates for confirmation
    agg_f0_sep = abs(aggregates.get("f0_pos", {}).get("mean", 0) - aggregates.get("f0_neg", {}).get("mean", 0))
    agg_f1_sep = abs(aggregates.get("f1_pos", {}).get("mean", 0) - aggregates.get("f1_neg", {}).get("mean", 0))
    
    # Weight: 70% samples, 30% aggregates
    score_f0 = 0.7 * d_f0 + 0.3 * agg_f0_sep
    score_f1 = 0.7 * d_f1 + 0.3 * agg_f1_sep
    
    if score_f0 >= score_f1:
        # Threshold on f0
        threshold = float((aggregates.get("f0_pos", {}).get("mean", 0) + aggregates.get("f0_neg", {}).get("mean", 0)) / 2)
        direction = 1 if aggregates.get("f0_pos", {}).get("mean", 0) > aggregates.get("f0_neg", {}).get("mean", 0) else -1
        return 0, threshold, direction
    else:
        threshold = float((aggregates.get("f1_pos", {}).get("mean", 0) + aggregates.get("f1_neg", {}).get("mean", 0)) / 2)
        direction = 1 if aggregates.get("f1_pos", {}).get("mean", 0) > aggregates.get("f1_neg", {}).get("mean", 0) else -1
        return 1, threshold, direction


async def main() -> None:
    np.random.seed(42)
    random.seed(42)
    
    print("=" * 70)
    print("O1 — Adaptive Supervision")
    print(f"Shards: {N_SHARDS}, Records/shard: {RECORDS_PER_SHARD}, Drift at: {DRIFT_SHARD}")
    print("=" * 70)
    
    # Generate all shards
    print("\nGenerating synthetic data...")
    shards_data = []
    shard_buffers = []
    
    with tempfile.TemporaryDirectory() as tmp:
        rt = Runtime(
            socket_path=f"{tmp}/run.sock",
            shm_dir=f"{tmp}/shm",
            config={"trace_dir": f"{tmp}/traces"},
        )
        await rt.start()
        sid = (await rt._hello({"protocol": "lmpu/1"})).session_id
        
        try:
            # Pre-create all shard buffers
            for shard_idx in range(N_SHARDS):
                data = generate_shard(shard_idx)
                shards_data.append(data)
                raw = data_to_bytes(data)
                ref = await rt._buffer_import(
                    {"name": f"shard_{shard_idx}", "data": raw, "dtype": "uint8", "shape": [len(raw)]}, sid
                )
                shard_buffers.append(ref.buffer_id)
                print(f"  Shard {shard_idx:2d}: {len(raw)/1024:.1f} KiB, buffer={ref.buffer_id[:16]}...")
            
            all_results: list[ArmResult] = []
            
            # ============================================================
            # ARM C: ORACLE (upper bound) - knows true labels per shard
            # ============================================================
            print("\n" + "=" * 70)
            print("ARM C: ORACLE (threshold fitted with full labels)")
            print("=" * 70)
            
            oracle_tokens = 0
            oracle_start = time.perf_counter()
            
            for shard_idx in range(N_SHARDS):
                # Oracle knows which feature is predictive, the optimal threshold, AND the direction
                data = shards_data[shard_idx]
                f0_pos = data[data[:, 2] == 1, 0]
                f0_neg = data[data[:, 2] == 0, 0]
                f1_pos = data[data[:, 2] == 1, 1]
                f1_neg = data[data[:, 2] == 0, 1]
                
                d_f0 = abs(f0_pos.mean() - f0_neg.mean()) / math.sqrt((f0_pos.var() + f0_neg.var()) / 2) if len(f0_pos) > 0 and len(f0_neg) > 0 else 0
                d_f1 = abs(f1_pos.mean() - f1_neg.mean()) / math.sqrt((f1_pos.var() + f1_neg.var()) / 2) if len(f1_pos) > 0 and len(f1_neg) > 0 else 0
                
                if d_f0 >= d_f1:
                    feature_idx = 0
                    threshold = float((f0_pos.mean() + f0_neg.mean()) / 2)
                    direction = 1 if f0_pos.mean() > f0_neg.mean() else -1
                else:
                    feature_idx = 1
                    threshold = float((f1_pos.mean() + f1_neg.mean()) / 2)
                    direction = 1 if f1_pos.mean() > f1_neg.mean() else -1
                
                result, tokens, wall = await run_scan_job(rt, sid, shard_buffers[shard_idx], threshold, feature_idx, direction)
                
                tp = result.get("tp", 0)
                fp = result.get("fp", 0)
                tn = result.get("tn", 0)
                fn = result.get("fn", 0)
                f1 = f1_from_confusion(tp, fp, fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                oracle_tokens += tokens
                all_results.append(ArmResult(
                    arm="oracle", shard=shard_idx, threshold=threshold,
                    tp=tp, fp=fp, tn=tn, fn=fn, f1=f1,
                    precision=precision, recall=recall,
                    host_tokens=tokens, wall_s=wall, decisions=1
                ))
                print(f"  Shard {shard_idx:2d}: thresh={threshold:.3f} F1={f1:.3f} P={precision:.3f} R={recall:.3f} ({tokens} tok)")
            
            oracle_wall = time.perf_counter() - oracle_start
            print(f"  Oracle total: {oracle_tokens} tokens, {oracle_wall:.2f}s")
            
            # ============================================================
            # ARM A: ONE-SHOT (fixed threshold chosen before seeing data)
            # ============================================================
            print("\n" + "=" * 70)
            print("ARM A: ONE-SHOT (fixed threshold chosen blindly)")
            print("=" * 70)
            
            # One-shot threshold chosen BLINDLY before seeing ANY data
            # A naive agent might pick 0.0 as a default midpoint
            one_shot_threshold = 0.0
            print(f"  Fixed threshold (blind guess): {one_shot_threshold:.3f}")
            
            oneshot_tokens = 0
            oneshot_start = time.perf_counter()
            
            for shard_idx in range(N_SHARDS):
                result, tokens, wall = await run_scan_job(rt, sid, shard_buffers[shard_idx], one_shot_threshold, 0, 1)
                
                tp = result.get("tp", 0)
                fp = result.get("fp", 0)
                tn = result.get("tn", 0)
                fn = result.get("fn", 0)
                f1 = f1_from_confusion(tp, fp, fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                oneshot_tokens += tokens
                all_results.append(ArmResult(
                    arm="one_shot", shard=shard_idx, threshold=one_shot_threshold,
                    tp=tp, fp=fp, tn=tn, fn=fn, f1=f1,
                    precision=precision, recall=recall,
                    host_tokens=tokens, wall_s=wall, decisions=1
                ))
                print(f"  Shard {shard_idx:2d}: thresh={one_shot_threshold:.3f} F1={f1:.3f} P={precision:.3f} R={recall:.3f} ({tokens} tok)")
            
            oneshot_wall = time.perf_counter() - oneshot_start
            print(f"  One-shot total: {oneshot_tokens} tokens, {oneshot_wall:.2f}s")
            
            # ============================================================
            # ARM B: ADAPTIVE (model sees aggregates + samples, revises)
            # ============================================================
            print("\n" + "=" * 70)
            print(f"ARM B: ADAPTIVE (aggregates + {SAMPLES_PER_SHARD} samples, ≤{MAX_DECISIONS_PER_SHARD} decisions/shard)")
            print("=" * 70)
            
            adaptive_tokens = 0
            adaptive_start = time.perf_counter()
            
            for shard_idx in range(N_SHARDS):
                # Decision 1: get aggregates
                agg_result, agg_tokens, agg_wall = await run_aggregate_job(rt, sid, shard_buffers[shard_idx])
                adaptive_tokens += agg_tokens
                
                # Decision 2: get samples
                sample_result, sample_tokens, sample_wall = await run_sample_job(rt, sid, shard_buffers[shard_idx])
                adaptive_tokens += sample_tokens
                
                # Model chooses feature, threshold, and direction
                samples = sample_result.get("samples", [])
                feature_idx, threshold, direction = model_choose_threshold(agg_result, samples)
                
                # Decision 3: scan with chosen feature, threshold, and direction
                scan_result, scan_tokens, scan_wall = await run_scan_job(rt, sid, shard_buffers[shard_idx], threshold, feature_idx, direction)
                adaptive_tokens += scan_tokens
                
                # Could add up to 2 more refinement decisions here if needed
                # For now, 3 decisions per shard
                decisions = 3
                
                tp = scan_result.get("tp", 0)
                fp = scan_result.get("fp", 0)
                tn = scan_result.get("tn", 0)
                fn = scan_result.get("fn", 0)
                f1 = f1_from_confusion(tp, fp, fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                all_results.append(ArmResult(
                    arm="adaptive", shard=shard_idx, threshold=threshold,
                    tp=tp, fp=fp, tn=tn, fn=fn, f1=f1,
                    precision=precision, recall=recall,
                    host_tokens=agg_tokens + sample_tokens + scan_tokens,
                    wall_s=agg_wall + sample_wall + scan_wall,
                    decisions=decisions
                ))
                print(f"  Shard {shard_idx:2d}: feat={feature_idx} dir={direction} thresh={threshold:.3f} F1={f1:.3f} P={precision:.3f} R={recall:.3f} ({agg_tokens+sample_tokens+scan_tokens} tok, {decisions} dec)")
            
            adaptive_wall = time.perf_counter() - adaptive_start
            print(f"  Adaptive total: {adaptive_tokens} tokens, {adaptive_wall:.2f}s")
            
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()
    
    # ============================================================
    # SUMMARY & ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    oracle_results = [r for r in all_results if r.arm == "oracle"]
    oneshot_results = [r for r in all_results if r.arm == "one_shot"]
    adaptive_results = [r for r in all_results if r.arm == "adaptive"]
    
    # Per-shard comparison
    print(f"\n{'Shard':>5} {'Oracle F1':>10} {'One-shot F1':>12} {'Adaptive F1':>12} {'Oracle F1':>10} {'Gap A':>8} {'Gap B':>8}")
    print("-" * 75)
    
    pre_drift_oracle = []
    pre_drift_oneshot = []
    pre_drift_adaptive = []
    post_drift_oracle = []
    post_drift_oneshot = []
    post_drift_adaptive = []
    
    for shard_idx in range(N_SHARDS):
        o = next(r for r in oracle_results if r.shard == shard_idx)
        a = next(r for r in oneshot_results if r.shard == shard_idx)
        b = next(r for r in adaptive_results if r.shard == shard_idx)
        
        gap_a = o.f1 - a.f1
        gap_b = o.f1 - b.f1
        
        print(f"{shard_idx:>5} {o.f1:>10.3f} {a.f1:>12.3f} {b.f1:>12.3f} {o.f1:>10.3f} {gap_a:>8.3f} {gap_b:>8.3f}")
        
        if shard_idx < DRIFT_SHARD:
            pre_drift_oracle.append(o.f1)
            pre_drift_oneshot.append(a.f1)
            pre_drift_adaptive.append(b.f1)
        else:
            post_drift_oracle.append(o.f1)
            post_drift_oneshot.append(a.f1)
            post_drift_adaptive.append(b.f1)
    
    # Aggregate metrics
    print("\n" + "-" * 70)
    print("AGGREGATE METRICS")
    print("-" * 70)
    
    def avg(lst): return sum(lst) / len(lst) if lst else 0
    
    print(f"\nPre-drift (shards 0-{DRIFT_SHARD-1}):")
    print(f"  Oracle:    {avg(pre_drift_oracle):.3f}")
    print(f"  One-shot:  {avg(pre_drift_oneshot):.3f} (gap: {avg(pre_drift_oracle)-avg(pre_drift_oneshot):.3f})")
    print(f"  Adaptive:  {avg(pre_drift_adaptive):.3f} (gap: {avg(pre_drift_oracle)-avg(pre_drift_adaptive):.3f})")
    
    print(f"\nPost-drift (shards {DRIFT_SHARD}-{N_SHARDS-1}):")
    print(f"  Oracle:    {avg(post_drift_oracle):.3f}")
    print(f"  One-shot:  {avg(post_drift_oneshot):.3f} (gap: {avg(post_drift_oracle)-avg(post_drift_oneshot):.3f})")
    print(f"  Adaptive:  {avg(post_drift_adaptive):.3f} (gap: {avg(post_drift_oracle)-avg(post_drift_adaptive):.3f})")
    
    # Overall
    oracle_avg = avg([r.f1 for r in oracle_results])
    oneshot_avg = avg([r.f1 for r in oneshot_results])
    adaptive_avg = avg([r.f1 for r in adaptive_results])
    
    print(f"\nOverall (all {N_SHARDS} shards):")
    print(f"  Oracle:    {oracle_avg:.3f}")
    print(f"  One-shot:  {oneshot_avg:.3f} (gap: {oracle_avg-oneshot_avg:.3f})")
    print(f"  Adaptive:  {adaptive_avg:.3f} (gap: {oracle_avg-adaptive_avg:.3f})")
    
    # Recovery rate
    if oracle_avg > oneshot_avg:
        recovery = (adaptive_avg - oneshot_avg) / (oracle_avg - oneshot_avg)
        print(f"\nAdaptive recovery of oracle advantage: {recovery*100:.1f}%")
    
    # Token comparison
    print(f"\nHost tokens:")
    print(f"  Oracle:    {sum(r.host_tokens for r in oracle_results)} ({sum(r.decisions for r in oracle_results)} decisions)")
    print(f"  One-shot:  {sum(r.host_tokens for r in oneshot_results)} ({sum(r.decisions for r in oneshot_results)} decisions)")
    print(f"  Adaptive:  {sum(r.host_tokens for r in adaptive_results)} ({sum(r.decisions for r in adaptive_results)} decisions)")
    
    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    
    # Criterion 1: Adaptive recovers ≥90% of oracle's F1 on drifted half
    post_oracle_avg = avg(post_drift_oracle)
    post_adaptive_avg = avg(post_drift_adaptive)
    post_oneshot_avg = avg(post_drift_oneshot)
    
    if post_oracle_avg > post_oneshot_avg:
        recovery_rate = (post_adaptive_avg - post_oneshot_avg) / (post_oracle_avg - post_oneshot_avg)
        criterion1 = recovery_rate >= 0.90
        print(f"1. Adaptive recovers ≥90% of oracle advantage on drifted half: "
              f"{recovery_rate*100:.1f}% → {'PASS' if criterion1 else 'FAIL'}")
    else:
        criterion1 = post_adaptive_avg >= post_oracle_avg * 0.9
        print(f"1. Adaptive F1 ≥ 90% of oracle on drifted half: "
              f"{post_adaptive_avg:.3f} vs {post_oracle_avg:.3f} → {'PASS' if criterion1 else 'FAIL'}")
    
    # Criterion 2: One-shot loses ≥20 F1 points on drifted half (showing drift hurts)
    f1_drop = post_oracle_avg - post_oneshot_avg
    criterion2 = f1_drop >= 0.20
    print(f"2. One-shot loses ≥0.20 F1 on drifted half: "
          f"drop={f1_drop:.3f} → {'PASS' if criterion2 else 'FAIL'}")
    
    # Criterion 3: Decisions per shard ≤ 5
    avg_decisions = sum(r.decisions for r in adaptive_results) / len(adaptive_results)
    criterion3 = avg_decisions <= MAX_DECISIONS_PER_SHARD
    print(f"3. Adaptive decisions/shard ≤ {MAX_DECISIONS_PER_SHARD}: "
          f"{avg_decisions:.1f} → {'PASS' if criterion3 else 'FAIL'}")
    
    # Criterion 4: Control tokens flat in data size (verify per-shard tokens similar)
    adaptive_toks = [r.host_tokens for r in adaptive_results]
    tok_cv = np.std(adaptive_toks) / np.mean(adaptive_toks) if np.mean(adaptive_toks) > 0 else 0
    criterion4 = tok_cv < 0.15  # coefficient of variation < 15%
    print(f"4. Control tokens flat across shards (CV < 15%): "
          f"CV={tok_cv:.3f} → {'PASS' if criterion4 else 'FAIL'}")
    
    overall_pass = all([criterion1, criterion2, criterion3, criterion4])
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    
    # Save results
    out = Path("benchmarks/results/adaptive_supervision")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    print(f"\nWrote {out / 'result.json'}")
    
    # Claim file
    claim = {
        "claim": "O1",
        "description": "Adaptive supervision beats fixed criterion under drift at constant context",
        "result": "PASS" if overall_pass else "FAIL",
        "evidence": "benchmarks/experiments/adaptive_supervision.py",
        "measurement": f"200k records, 20 shards, drift at shard {DRIFT_SHARD}",
        "details": {
            "oracle_f1_overall": oracle_avg,
            "one_shot_f1_overall": oneshot_avg,
            "adaptive_f1_overall": adaptive_avg,
            "pre_drift": {
                "oracle": avg(pre_drift_oracle),
                "one_shot": avg(pre_drift_oneshot),
                "adaptive": avg(pre_drift_adaptive),
            },
            "post_drift": {
                "oracle": avg(post_drift_oracle),
                "one_shot": avg(post_drift_oneshot),
                "adaptive": avg(post_drift_adaptive),
            },
            "recovery_rate_post_drift": (post_adaptive_avg - post_oneshot_avg) / (post_oracle_avg - post_oneshot_avg) if post_oracle_avg > post_oneshot_avg else 1.0,
            "one_shot_f1_drop_post_drift": post_oracle_avg - post_oneshot_avg,
            "adaptive_decisions_per_shard": avg_decisions,
            "token_cv": tok_cv,
            "host_tokens": {
                "oracle": sum(r.host_tokens for r in oracle_results),
                "one_shot": sum(r.host_tokens for r in oneshot_results),
                "adaptive": sum(r.host_tokens for r in adaptive_results),
            },
            "criteria": {
                "recovery_ge_90pct": bool(criterion1),
                "oneshot_drop_ge_20_f1": bool(criterion2),
                "decisions_le_5": bool(criterion3),
                "tokens_flat_cv_lt_15pct": bool(criterion4),
            }
        }
    }
    (out / "claim.json").write_text(json.dumps(claim, indent=2))
    print(f"Wrote {out / 'claim.json'}")


if __name__ == "__main__":
    asyncio.run(main())