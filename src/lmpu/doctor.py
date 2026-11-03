"""LMPU Doctor - Environment diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict


def run_doctor(verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """Run all diagnostic checks."""
    results = {}
    
    results["python"] = check_python()
    results["platform"] = check_platform()
    results["shm"] = check_shm()
    results["uds"] = check_uds()
    results["gpu"] = check_gpu()
    results["nvml"] = check_nvml()
    results["model_cache"] = check_model_cache()
    results["sandbox"] = check_sandbox()
    results["dependencies"] = check_dependencies()
    
    return results


def check_python() -> Dict[str, Any]:
    """Check Python version."""
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 12
    return {
        "ok": ok,
        "details": f"Python {version.major}.{version.minor}.{version.micro}",
    }


def check_platform() -> Dict[str, Any]:
    """Check platform."""
    system = platform.system()
    ok = system == "Linux"
    return {
        "ok": ok,
        "details": f"{system} {platform.release()} ({platform.machine()})",
    }


def check_shm() -> Dict[str, Any]:
    """Check /dev/shm availability and space."""
    shm_path = "/dev/shm"
    if not os.path.exists(shm_path):
        return {"ok": False, "details": "/dev/shm not found"}
    
    stat = shutil.disk_usage(shm_path)
    free_gb = stat.free / (1024**3)
    ok = free_gb > 1.0  # At least 1 GB free
    return {
        "ok": ok,
        "details": f"{free_gb:.1f} GB free of {stat.total / (1024**3):.1f} GB",
    }


def check_uds() -> Dict[str, Any]:
    """Check Unix domain socket support."""
    try:
        import socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.close()
        return {"ok": True, "details": "Unix domain sockets supported"}
    except Exception as e:
        return {"ok": False, "details": f"UDS error: {e}"}


def check_gpu() -> Dict[str, Any]:
    """Check GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = [line.strip() for line in lines if line.strip()]
            return {
                "ok": len(gpus) > 0,
                "details": f"{len(gpus)} GPU(s): {', '.join(gpus)}",
            }
        else:
            return {"ok": False, "details": "nvidia-smi failed"}
    except FileNotFoundError:
        return {"ok": False, "details": "nvidia-smi not found"}
    except Exception as e:
        return {"ok": False, "details": f"GPU check error: {e}"}


def check_nvml() -> Dict[str, Any]:
    """Check NVML (pynvml) availability."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return {"ok": True, "details": f"NVML available, {count} device(s)"}
    except ImportError:
        return {"ok": False, "details": "pynvml not installed"}
    except Exception as e:
        return {"ok": False, "details": f"NVML error: {e}"}


def check_model_cache() -> Dict[str, Any]:
    """Check Hugging Face model cache."""
    from pathlib import Path
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    models_dir = cache_dir / "hub"
    
    if not models_dir.exists():
        return {"ok": False, "details": "HF cache not found"}
    
    # Check for Qwen models
    qwen_models = list(models_dir.glob("models--Qwen*"))
    return {
        "ok": len(qwen_models) > 0,
        "details": f"HF cache at {cache_dir}, {len(qwen_models)} Qwen model(s)",
    }


def check_sandbox() -> Dict[str, Any]:
    """Check sandbox capabilities."""
    checks = {}
    
    # Subprocess
    checks["subprocess"] = True
    
    # nsjail
    checks["nsjail"] = shutil.which("nsjail") is not None
    
    # Docker
    checks["docker"] = shutil.which("docker") is not None
    
    # Namespaces (requires root for some)
    checks["namespaces"] = os.path.exists("/proc/self/ns")
    
    # rlimits
    import resource
    checks["rlimits"] = True
    
    all_ok = all(checks.values())
    return {
        "ok": all_ok,
        "details": f"subprocess: ✓, nsjail: {'✓' if checks['nsjail'] else '✗'}, docker: {'✓' if checks['docker'] else '✗'}",
    }


def check_dependencies() -> Dict[str, Any]:
    """Check key dependencies."""
    deps = {
        "torch": False,
        "transformers": False,
        "tokenizers": False,
        "numpy": False,
        "pydantic": False,
        "orjson": False,
        "blake3": False,
    }
    
    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            pass
    
    all_ok = all(deps.values())
    installed = [k for k, v in deps.items() if v]
    missing = [k for k, v in deps.items() if not v]
    
    return {
        "ok": all_ok,
        "details": f"Installed: {', '.join(installed)}; Missing: {', '.join(missing) if missing else 'none'}",
    }


def main():
    """Main entry point for lmpu-doctor."""
    import json
    results = run_doctor(verbose=True)
    print(json.dumps(results, indent=2))
    
    if not all(r.get("ok", False) for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()