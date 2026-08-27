"""One-glance status of live training runs — one fixed-width line per run.

Usage: `uv run scripts/run_digest.py` (no flags). Runs and thresholds come from
`configs/run_digest.yaml`. Read-only: parses each run's MSST `train.log` directly and
never touches wandb or checkpoints, so it is safe to run while training writes.

Line format:
  <run> <epN> val <latest> Δ<vs prev> best <val>@<ep> log <age> <status>
Epoch numbers follow MSST's 0-based index (matches `model_*_ep_N_*.ckpt` names).
"""
from __future__ import annotations

import glob
import mmap
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "run_digest.yaml"
KST = ZoneInfo("Asia/Seoul")

# Each event kind gets its own literal-anchored search (Python's re fast-paths a literal
# prefix); the hits are then merged by byte offset so per-stem values attach to their epoch.
# One big alternation regex was ~8 s on a 260 MB log; this is ~0.5 s. The log is a
# \r-delimited tqdm stream, so nothing here anchors on line starts.
EVENT_PATTERNS: dict[str, re.Pattern[bytes]] = {
    "epoch": re.compile(rb"Train epoch: (\d+) Learning rate"),
    "stem": re.compile(rb"Instr (\S+) si_sdr: (-?[\d.]+)"),
    "avg_val": re.compile(rb"Metric avg si_sdr\s*: (-?[\d.]+)"),
    "nonfinite": re.compile(rb"Training loss: \S+ \(over \d+ finite steps; (\d+) non-finite skipped\)"),
    # tqdm prints "<00:00" only when a bar completes; the elapsed value sits just before it.
    "completed": re.compile(rb"<00:00"),
    "trailer": re.compile(
        rb"=== \S+ (finished clean|ended after resume|stopped by SIGINT|CRASHED TWICE|CRASHED)"
    ),
}
# "40000/40000 [6:44:12" right before "<00:00": the bar total identifies which bar completed.
BAR_BEFORE_COMPLETION = re.compile(rb"(\d+)/(\d+) \[(\d+:\d\d(?::\d\d)?)$")
NAN_LOSS_PATTERN = re.compile(rb"loss=(nan|inf|-inf)\b")


@dataclass
class RunHistory:
    """Everything the digest needs, extracted from one train.log."""

    epoch_avg_vals: list[float] = field(default_factory=list)
    epoch_stem_vals: list[dict[str, float]] = field(default_factory=list)
    epoch_wall_seconds: list[float] = field(default_factory=list)
    latest_nonfinite_steps: int = 0
    trailer: str | None = None
    tail_has_nan: bool = False


def parse_elapsed_seconds(elapsed: str) -> float:
    """Convert a tqdm elapsed string (M:SS or H:MM:SS) into seconds."""
    parts = [int(part) for part in elapsed.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def collect_events(data: mmap.mmap) -> list[tuple[int, str, re.Match[bytes]]]:
    """Run every event search and return the hits merged in file order."""
    events: list[tuple[int, str, re.Match[bytes]]] = []
    for kind, pattern in EVENT_PATTERNS.items():
        events.extend((match.start(), kind, match) for match in pattern.finditer(data))
    events.sort(key=lambda event: event[0])
    return events


def parse_train_log(log_path: Path) -> RunHistory:
    """Single ordered pass over the merged events of one log (mmap, ~0.5 s per 260 MB)."""
    history = RunHistory()
    current_stems: dict[str, float] = {}
    # tqdm rewrites its 100% line several times with a ticking elapsed; keep the max per bar
    # (keyed by the bar's total) so train and val bars each count exactly once per epoch.
    current_epoch_completions: dict[bytes, float] = {}
    with log_path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for offset, kind, match in collect_events(data):
            if kind == "epoch":
                # A new epoch header closes the previous epoch's wall-clock accumulation.
                if int(match.group(1)) > 0:
                    history.epoch_wall_seconds.append(sum(current_epoch_completions.values()))
                current_epoch_completions = {}
            elif kind == "stem":
                current_stems[match.group(1).decode("utf-8")] = float(match.group(2))
            elif kind == "avg_val":
                history.epoch_avg_vals.append(float(match.group(1)))
                history.epoch_stem_vals.append(current_stems)
                current_stems = {}
            elif kind == "nonfinite":
                history.latest_nonfinite_steps = int(match.group(1))
            elif kind == "completed":
                bar = BAR_BEFORE_COMPLETION.search(data[max(0, offset - 40):offset])
                if bar and bar.group(1) == bar.group(2):
                    bar_total = bar.group(2)
                    elapsed_seconds = parse_elapsed_seconds(bar.group(3).decode())
                    current_epoch_completions[bar_total] = max(
                        current_epoch_completions.get(bar_total, 0.0), elapsed_seconds
                    )
            elif kind == "trailer":
                history.trailer = match.group(1).decode()
        # NaN scan on the tail only: recent loss values, not the whole history.
        tail_bytes = int(CONFIG["nan_tail_bytes"])
        history.tail_has_nan = NAN_LOSS_PATTERN.search(data[max(0, len(data) - tail_bytes):]) is not None
    return history


def format_age(seconds: float) -> str:
    """Human age like '42m ago' / '6h ago' / '2d ago'; clock skew clamps to 0."""
    seconds = max(0.0, seconds)
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.0f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def head_drops(history: RunHistory, drop_db: float) -> list[str]:
    """Stems whose latest val SI-SDR sits more than `drop_db` below their own best."""
    if not history.epoch_stem_vals:
        return []
    latest = history.epoch_stem_vals[-1]
    dropped: list[str] = []
    for stem, latest_val in latest.items():
        best_val = max(epoch[stem] for epoch in history.epoch_stem_vals if stem in epoch)
        if best_val - latest_val > drop_db:
            dropped.append(stem)
    return dropped


def stall_window_seconds(history: RunHistory) -> float:
    """2× the run's own median epoch wall-time, or the fallback before that is known."""
    finished_epochs = [wall for wall in history.epoch_wall_seconds if wall > 0]
    if len(finished_epochs) < 2:
        return float(CONFIG["stalled_fallback_hours"]) * 3600
    return float(CONFIG["stalled_multiplier"]) * statistics.median(finished_epochs)


def decide_status(history: RunHistory, log_age_seconds: float) -> str:
    """Apply the status rules in priority order; NaN and HEAD flags stack after liveness."""
    flags: list[str] = []
    if history.trailer == "finished clean" or history.trailer == "ended after resume":
        liveness = "done"
    elif history.trailer == "stopped by SIGINT":
        liveness = "stopped"
    elif history.trailer in ("CRASHED", "CRASHED TWICE"):
        liveness = f"⚠ STALLED ({history.trailer.lower()})"
    elif log_age_seconds > stall_window_seconds(history):
        liveness = "⚠ STALLED"
    else:
        liveness = "ok"
    if history.latest_nonfinite_steps > 0 or history.tail_has_nan:
        flags.append("⚠ NaN")
    dropped = head_drops(history, float(CONFIG["head_drop_db"]))
    if dropped:
        flags.append("⚠ HEAD " + ",".join(dropped))
    return " ".join([liveness, *flags])


def resolve_log_path(pattern: str) -> Path | None:
    """Expand a literal path or glob relative to the repo root; newest match wins."""
    matches = sorted(glob.glob(str(REPO_ROOT / pattern)))
    return Path(matches[-1]) if matches else None


def digest_line(run_name: str, pattern: str, now: float) -> str:
    """Build the fixed-width status line for one run."""
    log_path = resolve_log_path(pattern)
    if log_path is None:
        return f"{run_name:<10} —     no log yet"
    history = parse_train_log(log_path)
    log_age = now - log_path.stat().st_mtime
    status = decide_status(history, log_age)
    if history.epoch_avg_vals:
        latest_epoch = len(history.epoch_avg_vals) - 1
        latest_val = history.epoch_avg_vals[-1]
        best_epoch = max(range(len(history.epoch_avg_vals)), key=history.epoch_avg_vals.__getitem__)
        epoch_field = f"ep{latest_epoch}"
        val_field = f"val {latest_val:+.2f}"
        delta_field = f"Δ{latest_val - history.epoch_avg_vals[-2]:+.2f}" if latest_epoch >= 1 else "Δ —"
        best_field = f"best {history.epoch_avg_vals[best_epoch]:+.2f}@{best_epoch}"
    else:
        epoch_field, val_field, delta_field, best_field = "ep—", "val —", "Δ —", "best —"
    return (
        f"{run_name:<10} {epoch_field:<5} {val_field:<10} {delta_field:<7} {best_field:<15} "
        f"log {format_age(log_age):<8} {status}"
    ).rstrip()


# Widths mirror digest_line(): run 10 · epoch 5 · val 10 · Δ 7 · best 15 · log-age 12 · status.
COLUMN_HEADER = f"{'run':<10} {'epoch':<5} {'val SI-SDR':<10} {'Δprev':<7} {'best@epoch':<15} {'last log':<12} status"


def main() -> None:
    """Print the timestamp header and one line per configured run."""
    now = time.time()
    print(datetime.fromtimestamp(now, KST).strftime("%Y-%m-%d %H:%M KST"))
    print(COLUMN_HEADER)
    for run_name, pattern in CONFIG["runs"].items():
        print(digest_line(run_name, pattern, now))


CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

if __name__ == "__main__":
    main()
