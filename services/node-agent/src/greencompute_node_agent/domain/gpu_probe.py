"""Best-effort probe of ACTUAL GPU state via nvidia-smi.

Used to detect GPUs in use by processes GreenCompute doesn't know about
(co-tenants, or a crypto squatter like the golden-miner incidents) so the node
never ADVERTISES or ALLOCATES a physically-busy card. Without this the
allocator is "theft-blind" — it tracks only its own allocations and would
over-commit a customer onto a GPU another process is already maxing.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

# A card genuinely in use holds hundreds of MiB to tens of GiB of VRAM; an idle
# card reports ~1 MiB. 512 MiB cleanly separates "something is running here"
# from driver/context noise without flagging a truly-idle GPU.
_BUSY_VRAM_THRESHOLD_MIB = 512


def busy_gpu_devices(threshold_mib: int = _BUSY_VRAM_THRESHOLD_MIB) -> set[int]:
    """Device indices whose used VRAM exceeds `threshold_mib` — i.e. something
    is running on them.

    BEST-EFFORT: returns an EMPTY set on any failure (no nvidia-smi, timeout,
    parse error), so a missing/broken probe never blocks allocation — the node
    just falls back to its own bookkeeping (the pre-existing behavior).
    """
    if not shutil.which("nvidia-smi"):
        return set()
    try:
        out = subprocess.check_output(  # noqa: S603
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=5.0,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return set()
    return parse_busy_devices(out, threshold_mib)


def parse_busy_devices(nvidia_smi_csv: str, threshold_mib: int = _BUSY_VRAM_THRESHOLD_MIB) -> set[int]:
    """Pure parser for `index, memory.used` CSV rows — split out so the
    threshold logic is unit-testable without a GPU."""
    busy: set[int] = set()
    for line in nvidia_smi_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
            mem_used = float(parts[1])
        except ValueError:
            continue
        if mem_used > threshold_mib:
            busy.add(index)
    return busy
