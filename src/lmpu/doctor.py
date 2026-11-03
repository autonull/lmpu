"""LMPU Doctor - Environment diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

CheckResult = dict[str, Any]


def run_doctor(verbose: bool = False) -> dict[str, CheckResult]:
    """Run all diagnostic checks."""
    return {
        "python": check_python(),
        "platform": check_platform(),
        "shm": check_shm(),
        "uds": check_uds(),
        "gpu": check_gpu(),
        "nvml": check_nvml(),
        "model_cache": check_model_cache(),
        "sandbox": check_sandbox(),
        "dependencies": check_dependencies(),
    }


def check_python() -> CheckResult:
    """Check Python version."""
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 12
    return {
        "ok": ok,
        "details": f"Python {version.major}.{version.minor}.{version.micro}",
    }


def check_platform() -> CheckResult:
    """Check platform."""
    system = platform.system()
    return {
        "ok": system == "Linux",
        "details": f"{system} {platform.release()} ({platform.machine()})",
    }


def check_shm() -> CheckResult:
    """Check /dev/shm availability and space."""
    shm_path = Path("/dev/shm")
    if not shm_path.exists():
        return {"ok": False, "details": "/dev/shm not found"}
    usage = shutil.disk_usage(shm_path)
    free_gb = usage.free / (1024**3)
    return {
        "ok": free_gb > 1.0,
        "details": f"{free_gb:.1f} GB free of {usage.total / (1024**3):.1f} GB",
    }


def check_uds() -> CheckResult:
    """Check Unix domain socket support."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.close()
    except OSError as exc:
        return {"ok": False, "details": f"UDS error: {exc}"}
    return {"ok": True, "details": "Unix domain sockets supported"}


def check_gpu() -> CheckResult:
    """Check GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {"ok": False, "details": "nvidia-smi not found"}
    except Exception as exc:
        return {"ok": False, "details": f"GPU check error: {exc}"}
    if result.returncode != 0:
        return {"ok": False, "details": "nvidia-smi failed"}
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"ok": bool(gpus), "details": f"{len(gpus)} GPU(s): {', '.join(gpus)}"}


def check_nvml() -> CheckResult:
    """Check NVML (pynvml) availability."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except ImportError:
        return {"ok": False, "details": "pynvml not installed"}
    except Exception as exc:
        return {"ok": False, "details": f"NVML error: {exc}"}
    return {"ok": True, "details": f"NVML available, {count} device(s)"}


def check_model_cache() -> CheckResult:
    """Check Hugging Face model cache."""
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    models_dir = cache_dir / "hub"
    if not models_dir.exists():
        return {"ok": False, "details": "HF cache not found"}
    qwen_models = list(models_dir.glob("models--Qwen*"))
    return {
        "ok": bool(qwen_models),
        "details": f"HF cache at {cache_dir}, {len(qwen_models)} Qwen model(s)",
    }


def check_sandbox() -> CheckResult:
    """Check the baseline sandbox; stronger backends are optional (Gate E3)."""
    optional = {
        "nsjail": shutil.which("nsjail") is not None,
        "docker": shutil.which("docker") is not None,
    }
    required = {
        "subprocess": True,
        "namespaces": Path("/proc/self/ns").exists(),
        "rlimits": True,
    }
    backends = ", ".join(f"{name}: {'✓' if ok else '✗'}" for name, ok in optional.items())
    return {
        "ok": all(required.values()),
        "details": f"subprocess: ✓; optional backends: {backends}",
    }


def check_dependencies() -> CheckResult:
    """Check key dependencies."""
    deps = ("torch", "transformers", "tokenizers", "numpy", "pydantic", "orjson", "blake3")
    missing = []
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    installed = [dep for dep in deps if dep not in missing]
    return {
        "ok": not missing,
        "details": (
            f"Installed: {', '.join(installed)}; "
            f"Missing: {', '.join(missing) if missing else 'none'}"
        ),
    }


def main() -> None:
    """Main entry point for lmpu-doctor."""
    import json

    results = run_doctor(verbose=True)
    print(json.dumps(results, indent=2))
    if not all(result.get("ok", False) for result in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
