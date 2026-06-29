#!/usr/bin/env python
"""
Simple plots of PERMISSIVE accuracy by framing x condition (A = genuine riddle,
B = riddle-riddle), read from scored.csv. Two panels:
  left  - grouped bars (A vs B) per framing, with 95% CIs and value labels
  right - paper-style A->B line per framing (shows the gap and how it shrinks)

Saves results_permissive.png. Open that file to view.

Error bars are 95% CIs from an ITEM-LEVEL CLUSTER BOOTSTRAP (2000 resamples of
the 30 riddle items with replacement). This respects the clustering (5 reps of
the same item are not independent) — a plain binomial CI on the pooled n would
be far too narrow. NB: it APPROXIMATES but does not equal the paper's intervals,
which come from mixed-effects logistic models with a random intercept for riddle
set (Fig 2/3); the paper's Fig 4 uses a bootstrap like this one. The definitive,
paper-matching CIs are the GLMM (R), later.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCORED = HERE / "scored.csv"
COL = "accuracy_permissive"
# Figure shows the core arc (matches the writeup's headline table). Other
# framings (nobrevity, unconstrained, seq_control) are in the appendix table.
FRAMING_ORDER = ["baseline", "nudge", "nudge_explicit"]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def load(mode):
    # (framing, version) -> {riddle_id: [0/1, ...]}, filtered to one thinking mode
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(SCORED.open(encoding="utf-8")):
        if r["thinking_mode"] == mode and r[COL] in ("0", "1"):
            data[(r["framing"], r["version"])][r["riddle_id"]].append(int(r[COL]))
    return data


def ci(item_map):
    """Point estimate (pooled proportion) + 95% item-cluster-bootstrap CI."""
    items = list(item_map.values())
    if not items:
        return float("nan"), 0.0, 0.0
    sums = np.array([sum(o) for o in items], dtype=float)
    counts = np.array([len(o) for o in items], dtype=float)
    p = sums.sum() / counts.sum()
    k = len(items)
    samp = RNG.integers(0, k, size=(N_BOOT, k))            # resample items
    boot = sums[samp].sum(1) / counts[samp].sum(1)         # pooled acc per resample
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return p, lo, hi


def main():
    ap = argparse.ArgumentParser(description="Plot permissive accuracy for one thinking mode.")
    ap.add_argument("--thinking", default="off", help="thinking mode to plot (off / adaptive)")
    mode = ap.parse_args().thinking
    out = HERE / f"results_permissive_{mode}.png"

    agg = load(mode)
    framings = [f for f in FRAMING_ORDER if any((f, v) in agg for v in ("A", "B"))]
    if not framings:
        raise SystemExit(f"No scored rows for thinking={mode!r} — run judge.py first.")

    # Compute each cell's (point, lo, hi) once so both panels share identical CIs.
    stats = {}
    for f in framings:
        for v in ("A", "B"):
            cell = agg.get((f, v))
            stats[(f, v)] = ci(cell) if cell else (float("nan"), float("nan"), float("nan"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cond_label = {"A": "A (genuine riddle)", "B": "B (riddle-riddle)"}
    x = list(range(len(framings)))
    w = 0.38

    # --- left: grouped bars ---
    for i, v in enumerate(("A", "B")):
        ps = [stats[(f, v)][0] for f in framings]
        lo = [0 if np.isnan(p) else p - stats[(f, v)][1] for f, p in zip(framings, ps)]
        hi = [0 if np.isnan(p) else stats[(f, v)][2] - p for f, p in zip(framings, ps)]
        ax1.bar([xi + (i - 0.5) * w for xi in x], ps, w,
                yerr=[lo, hi], capsize=4, label=cond_label[v])
        for xi, p in zip(x, ps):
            if not np.isnan(p):
                ax1.text(xi + (i - 0.5) * w, p + 0.02, f"{p:.0%}",
                         ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(framings, rotation=20, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("permissive accuracy")
    ax1.set_title(f"Accuracy by framing × condition (thinking {mode})")
    ax1.legend(loc="lower left", fontsize=9)

    # --- right: A->B line per framing, with error bars (paper style) ---
    for f in framings:
        ys = [stats[(f, v)][0] for v in ("A", "B")]
        los = [0 if np.isnan(stats[(f, v)][0]) else stats[(f, v)][0] - stats[(f, v)][1]
               for v in ("A", "B")]
        his = [0 if np.isnan(stats[(f, v)][0]) else stats[(f, v)][2] - stats[(f, v)][0]
               for v in ("A", "B")]
        ax2.errorbar([0, 1], ys, yerr=[los, his], marker="o", capsize=4, label=f)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["A (riddle)", "B (riddle-riddle)"])
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("permissive accuracy")
    ax2.set_title("A → B by framing")
    ax2.legend(fontsize=9)

    fig.text(0.5, 0.005,
             "Error bars: 95% CIs (item-level cluster bootstrap over the 30 items). "
             "Expected to be WIDER than the paper's, which plots ±1 SE "
             "(~half a 95% CI) from a partial-pooling model.",
             ha="center", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
