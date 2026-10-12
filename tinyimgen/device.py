"""Which accelerator to run on: CUDA (the RTX 2060 this was tuned on), Apple-silicon MPS, or CPU.

TINYIMGEN_DEVICE=cuda|mps|cpu overrides the choice. Shared by the image server and the 3D venv.
"""
import functools
import os
import platform
import subprocess

# MPS lacks a few ops (and all of float64); let PyTorch run those on the CPU instead of raising. Read when torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def _pick():
    want = os.environ.get("TINYIMGEN_DEVICE")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick()
CUDA, MPS = DEVICE == "cuda", DEVICE == "mps"


@functools.lru_cache(None)
def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
    except OSError:
        return ""


def ram_bytes():
    """Physical RAM (on Apple silicon this is also the GPU's memory)."""
    if platform.system() == "Darwin":
        return int(_sysctl("hw.memsize") or 0)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def name():
    if CUDA:
        return torch.cuda.get_device_name(0)
    if MPS:
        return _sysctl("machdep.cpu.brand_string") or "Apple silicon"
    return platform.processor() or "CPU"


def memory_gb():
    """Memory the device can use, in GB: the card's VRAM, or Metal's working-set limit on unified memory."""
    if CUDA:
        return torch.cuda.mem_get_info()[1] / 1e9
    if MPS:
        return torch.mps.recommended_max_memory() / 1e9
    return ram_bytes() / 1e9


def empty_cache():
    if CUDA:
        torch.cuda.empty_cache()
    elif MPS:
        torch.mps.empty_cache()


def synchronize():
    if CUDA:
        torch.cuda.synchronize()
    elif MPS:
        torch.mps.synchronize()


def reset_peak():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    """Peak memory of the last run in GB. MPS keeps no peak counter, so this is what Metal holds right after it."""
    if CUDA:
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if MPS:
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    return None


def generator(seed):
    # MPS generators work, but seeds then give different images than on CUDA anyway; a CPU generator is portable.
    return torch.Generator("cuda" if CUDA else "cpu").manual_seed(seed)


def is_oom(e):
    """MPS reports running out of memory as a plain RuntimeError."""
    return isinstance(e, torch.OutOfMemoryError) or (isinstance(e, RuntimeError) and "MPS backend out of memory" in str(e))
