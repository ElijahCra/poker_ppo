#!/usr/bin/env python3
"""
Live-plot training metrics from a poker_ppo run directory.

The C++ trainer appends rows to:
    <run_dir>/metrics.csv   — per-update PPO stats + wall timings
    <run_dir>/elo.csv       — sparse, one row per Elo evaluation snapshot

This script re-reads both files every REFRESH_SEC seconds and re-renders a
3x3 grid of matplotlib panels. Close the figure window to exit.

Usage:
    python tools/plot_live.py runs/20260422_143012
    python tools/plot_live.py --latest                # auto-pick newest dir
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

REFRESH_SEC = 2.0


# The trainer writes runs/ relative to its cwd. Search the usual spots so
# `--latest` works whether you launched it from the repo root or a build dir.
SEARCH_BASES = (
    "runs",
    "cmake-build-debug/runs",
    "cmake-build-release/runs",
    "build/runs",
)


def latest_run_dir(bases=SEARCH_BASES) -> Optional[str]:
    candidates: list[tuple[float, str]] = []
    for base in bases:
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if os.path.isdir(p):
                candidates.append((os.path.getmtime(p), p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def safe_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV that the trainer is still appending to. Empty / partial
    writes return an empty frame instead of raising."""
    if not os.path.exists(path) or os.path.getsize(path) < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


PANELS = [
    ("policy_loss",        "Policy loss"),
    ("value_loss",         "Value loss"),
    ("entropy",            "Entropy"),
    ("approx_kl",          "Approx KL"),
    ("clip_fraction",      "Clip fraction"),
    ("explained_variance", "Explained variance"),
    ("learning_rate",      "Learning rate"),
    # Slot 7 (rollout/update timings) and slot 8 (Elo) are special-cased.
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="Path to runs/<timestamp>/ directory.")
    ap.add_argument("--latest", action="store_true",
                    help="Auto-pick the newest dir under ./runs/.")
    ap.add_argument("--refresh", type=float, default=REFRESH_SEC,
                    help=f"Seconds between redraws (default {REFRESH_SEC}).")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir and args.latest:
        run_dir = latest_run_dir()
        if not run_dir:
            print(
                f"--latest: no runs/ found under any of {list(SEARCH_BASES)}\n"
                f"          (cwd: {os.getcwd()})\n"
                f"          start a training run first, or pass an explicit path.",
                file=sys.stderr,
            )
            return 1
        print(f"--latest → {run_dir}")
    if not run_dir:
        print("usage: plot_live.py <run_dir> | --latest", file=sys.stderr)
        return 1
    if not os.path.isdir(run_dir):
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 1

    metrics_path = os.path.join(run_dir, "metrics.csv")
    elo_path     = os.path.join(run_dir, "elo.csv")

    print(f"Watching {run_dir} (refresh {args.refresh:.1f}s — close window or Ctrl-C to stop)")

    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    try:
        fig.canvas.manager.set_window_title(f"poker_ppo — {run_dir}")
    except Exception:
        pass

    while plt.fignum_exists(fig.number):
        m = safe_read_csv(metrics_path)
        e = safe_read_csv(elo_path)

        for ax in axes.flat:
            ax.clear()

        # Panels 0–6: scalar PPO metrics over global_step.
        if not m.empty and "global_step" in m.columns:
            x = m["global_step"]
            for (col, title), ax in zip(PANELS, axes.flat[:len(PANELS)]):
                if col in m.columns:
                    ax.plot(x, m[col], lw=1.0)
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("step", fontsize=8)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)

            # Panel 7: rollout_ms and update_ms together.
            ax = axes.flat[7]
            if "rollout_ms" in m.columns:
                ax.plot(x, m["rollout_ms"], lw=1.0, label="rollout")
            if "update_ms" in m.columns:
                ax.plot(x, m["update_ms"], lw=1.0, alpha=0.85, label="update")
            ax.set_title("Wall (ms / update)", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
        else:
            # Empty placeholder titles so the grid doesn't look broken on iter 0.
            for (_, title), ax in zip(PANELS, axes.flat[:len(PANELS)]):
                ax.set_title(title, fontsize=9)
                ax.grid(alpha=0.3)
            axes.flat[7].set_title("Wall (ms / update)", fontsize=9)
            axes.flat[7].grid(alpha=0.3)

        # Panel 8 (bottom-right): Elo + win-rates over time.
        ax = axes[2][2]
        ax.set_title("Elo + match win-rates", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if not e.empty and "global_step" in e.columns:
            ex = e["global_step"]
            if "latest_rating" in e.columns:
                ax.plot(ex, e["latest_rating"], "-o", lw=1.0, ms=3, label="elo")
            ax.set_ylabel("elo", fontsize=8)

            ax2 = ax.twinx()
            if "vs_initial_winrate" in e.columns:
                ax2.plot(ex, e["vs_initial_winrate"], "-s", color="tab:orange",
                         lw=0.8, ms=2, label="vs init")
            if "vs_uniform_winrate" in e.columns:
                ax2.plot(ex, e["vs_uniform_winrate"], "-^", color="tab:green",
                         lw=0.8, ms=2, label="vs unif")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("win rate", fontsize=8)
            ax2.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper left")
            ax2.legend(fontsize=7, loc="lower right")

        fig.suptitle(
            f"{run_dir} — {len(m)} updates · {len(e)} elo evals",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        try:
            fig.canvas.draw_idle()  # Request a redraw without forcing focus
            fig.canvas.flush_events()  # Process GUI events to keep the window responsive
        except KeyboardInterrupt:
            break
        start_wait = time.time()
        while time.time() - start_wait < 5:
            if not plt.fignum_exists(fig.number):
                break
            fig.canvas.flush_events()
            time.sleep(0.08)  # Small yield to prevent CPU pegging during wait

    return 0


if __name__ == "__main__":
    sys.exit(main())
