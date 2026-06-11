from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_PATHS = [
    WORKSPACE_ROOT / "greencompute/protocol/src",
    ROOT / "services/node-agent/src",
]

for path in SRC_PATHS:
    sys.path.insert(0, str(path))
