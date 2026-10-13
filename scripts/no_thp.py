"""no_thp.py — run a command with transparent huge pages disabled for it and its children.

sym8's container is pinned to one NUMA node; once page cache fills that node's RAM, every THP
fault goes through direct compaction and heavy jobs (eval, render) stall 3–20× at near-zero GPU
util. PR_SET_THP_DISABLE is inherited across fork/exec, so wrapping the outermost command covers
uv and the Python it spawns.

Run:
    python3 scripts/no_thp.py uv run python -m src.eval.runner --config <config.yaml>
"""
import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41

# disable THP for this process, then replace it with the wrapped command
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_THP_DISABLE) failed")
os.execvp(sys.argv[1], sys.argv[1:])
