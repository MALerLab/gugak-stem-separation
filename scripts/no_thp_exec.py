"""Exec a command with transparent huge pages disabled for it and all its children.

sym8's container is pinned to NUMA node 1. Once page cache fills that node, every THP
fault stalls on direct compaction, and DDP validation slows enough to trip NCCL's
10-minute watchdog (exp008, 2026-10-12). PR_SET_THP_DISABLE is inherited across
fork/exec, so the trainer, its spawned ranks and dataloader workers all run without THP.

Usage: python3 scripts/no_thp_exec.py <command> [args...]
"""
import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41


def disable_transparent_huge_pages() -> None:
    """Set PR_SET_THP_DISABLE on the current process, raising on failure."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_THP_DISABLE) failed: {os.strerror(errno)}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: no_thp_exec.py <command> [args...]")
    disable_transparent_huge_pages()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
