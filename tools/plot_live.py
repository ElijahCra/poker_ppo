#!/usr/bin/env python3
"""
Live-plot training metrics from a poker_ppo run directory.

The C++ trainer appends rows to:
    <run_dir>/metrics.csv   — per-update PPO stats + wall timings
    <run_dir>/league.csv    — long-format, one row per (snapshot, anchor) pair
                              with columns: update, global_step, anchor,
                              num_hands, bb_per_hand, win_rate

This script re-reads both files every REFRESH_SEC seconds and re-renders a
3x3 grid of matplotlib panels.  Panel 8 plots one bb/hand line per registered
anchor (uniform, random_init, always_call, always_raise, pair_all_in, ...).

Close the figure window to exit.

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
import numpy as np
import pandas as pd

REFRESH_SEC = 2.0

# Once the per-update CSV grows past this many rows, panels 0-7 saturate into a
# solid blob — drawing 6000 connected line segments in a 4-inch-wide panel
# leaves no whitespace between them.  Bin into TARGET_POINTS buckets and plot
# the per-bin mean as the line, with a faint min/max fill so single-update
# spikes are still visible.
TARGET_POINTS = 2000


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


def _bin_downsample(x, y, target: int = TARGET_POINTS):
    """Decimate (x, y) into ``target`` contiguous bins.

    Returns (x_centers, y_mean, y_min, y_max). When len(x) <= target the
    input is returned as-is with min/max set to None so the caller can skip
    drawing the envelope.

    Bins are uniform in *index* space (not x), which is fine for the trainer's
    metrics since global_step is monotone with near-constant per-update growth.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= target:
        return x, y, None, None
    edges = np.linspace(0, n, target + 1, dtype=int)
    xc  = np.empty(target)
    yav = np.empty(target)
    ymn = np.empty(target)
    ymx = np.empty(target)
    for i in range(target):
        s = edges[i]
        e = max(edges[i + 1], s + 1)  # guard against zero-width bins
        seg_x = x[s:e]
        seg_y = y[s:e]
        xc[i]  = seg_x.mean()
        yav[i] = seg_y.mean()
        ymn[i] = seg_y.min()
        ymx[i] = seg_y.max()
    return xc, yav, ymn, ymx


def _plot_metric(ax, x, y, color: str, label: str | None = None, alpha_line: float = 1.0):
    """Plot a (possibly large) scalar series with adaptive decimation.

    The mean line is drawn at lw=0.9; if decimation kicks in, a min/max fill
    is drawn underneath at low alpha so spikes remain visible without the
    middle of the panel saturating into a solid block.
    """
    xc, ym, ymn, ymx = _bin_downsample(x, y)
    if ymn is not None:
        ax.fill_between(xc, ymn, ymx, color=color, alpha=0.3, lw=0)
    ax.plot(xc, ym, lw=0.9, color=color, alpha=alpha_line, label=label)


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
    league_path  = os.path.join(run_dir, "league.csv")
    br_path      = os.path.join(run_dir, "br.csv")

    print(f"Watching {run_dir} (refresh {args.refresh:.1f}s — close window or Ctrl-C to stop)")

    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    try:
        fig.canvas.manager.set_window_title(f"poker_ppo — {run_dir}")
    except Exception:
        pass

    # Stable, distinguishable colour per anchor across re-renders.  Anchors are
    # discovered lazily: as the league.csv accumulates new anchor names, each
    # gets the next colour in the matplotlib default cycle.
    anchor_colors: dict[str, str] = {}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def color_for(anchor: str) -> str:
        if anchor not in anchor_colors:
            anchor_colors[anchor] = default_cycle[
                len(anchor_colors) % len(default_cycle)
            ]
        return anchor_colors[anchor]

    while plt.fignum_exists(fig.number):
        m = safe_read_csv(metrics_path)
        l = safe_read_csv(league_path)
        b = safe_read_csv(br_path)

        for ax in axes.flat:
            ax.clear()

        # Panels 0–6: scalar PPO metrics over global_step.
        if not m.empty and "global_step" in m.columns:
            x = m["global_step"]
            for (col, title), ax in zip(PANELS, axes.flat[:len(PANELS)]):
                if col in m.columns:
                    _plot_metric(ax, x, m[col], color="C0")
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("step", fontsize=8)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)

            # Panel 7: rollout_ms and update_ms together.
            ax = axes.flat[7]
            if "rollout_ms" in m.columns:
                _plot_metric(ax, x, m["rollout_ms"], color="C0", label="rollout")
            if "update_ms" in m.columns:
                _plot_metric(ax, x, m["update_ms"], color="C1", label="update",
                             alpha_line=0.85)
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

        # Panel 8 (bottom-right): bb/hand vs each league anchor over time.
        # league.csv is in long format — one row per (snapshot, anchor) pair —
        # so we filter the trainer's "final" row (global_step = -1) and plot
        # one line per anchor.
        ax = axes[2][2]
        ax.set_title("bb / hand vs anchors", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("bb / hand (A's perspective)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)  # break-even reference

        n_anchor_evals = 0
        if (not l.empty
                and {"global_step", "anchor", "bb_per_hand"}.issubset(l.columns)):
            mid_run = l[l["global_step"] >= 0]
            n_anchor_evals = len(mid_run)
            for anchor_name, group in mid_run.sort_values("global_step").groupby("anchor"):
                ax.plot(
                    group["global_step"], group["bb_per_hand"],
                    "-o", lw=1.0, ms=3,
                    color=color_for(anchor_name),
                    label=anchor_name,
                )
            if not mid_run.empty:
                ax.legend(fontsize=7, loc="upper left",
                          bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                          ncol=1, framealpha=0.85)

        # Overlay approximate-best-response curve on the same panel.
        # bb_per_hand is max-over-seeds — the tightest measured lower bound
        # on exploitability. When num_seeds > 1, also shade [min, max] over
        # seeds to visualise per-eval seed variance.
        if (not b.empty
                and {"global_step", "bb_per_hand"}.issubset(b.columns)):
            br_sorted = b.sort_values("global_step")
            multi_seed = (
                "num_seeds" in br_sorted.columns
                and "bb_per_hand_min" in br_sorted.columns
                and (br_sorted["num_seeds"] > 1).any()
            )
            if multi_seed:
                ax.fill_between(
                    br_sorted["global_step"],
                    br_sorted["bb_per_hand_min"],
                    br_sorted["bb_per_hand"],
                    color="black", alpha=0.12,
                    label="_nolegend_",
                )
            ax.plot(
                br_sorted["global_step"], br_sorted["bb_per_hand"],
                "--D", lw=1.4, ms=4,
                color="black",
                label="approx_BR (max)" if multi_seed else "approx_BR",
                alpha=0.85,
            )
            ax.legend(fontsize=7, loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                      ncol=1, framealpha=0.85)

        n_snapshots = (
            l["global_step"].nunique() if (not l.empty and "global_step" in l.columns) else 0
        )
        fig.suptitle(
            f"{run_dir} — {len(m)} updates · {n_snapshots} league snapshots "
            f"({n_anchor_evals} anchor evals)",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 0.92, 0.96))

        try:
            fig.canvas.draw_idle()  # Request a redraw without forcing focus
            fig.canvas.flush_events()  # Process GUI events to keep the window responsive
        except KeyboardInterrupt:
            break
        start_wait = time.time()
        while time.time() - start_wait < 0.1:
            if not plt.fignum_exists(fig.number):
                break
            fig.canvas.flush_events()
            time.sleep(0.08)  # Small yield to prevent CPU pegging during wait

    return 0


if __name__ == "__main__":
    sys.exit(main())
