"""nanosearch core. Every experiment reads the frozen corpus and indexes from CORE."""

import os

ROOT = os.environ.get("GRPOP_ROOT", ".")
CORE = os.path.join(ROOT, "core-v1")
DATA_DIR = os.path.join(CORE, "data")
INDEX_DIR = os.path.join(CORE, "indexes")
OUT_DIR = os.path.join(ROOT, "outputs")
