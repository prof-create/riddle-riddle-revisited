#!/usr/bin/env python
"""
Accuracy summary of scored.csv under BOTH coding schemes (strict, permissive),
broken down by framing x thinking_mode x condition (A = genuine riddle, B =
riddle-riddle). Also reports strict-vs-permissive divergence, which quantifies
how often the accepted inventive alternatives change the verdict (i.e. how
ambiguous "correct on B" actually is). Stdlib only.
"""
import csv
from collections import defaultdict
from pathlib import Path

SCORED = Path(__file__).resolve().parent / "scored.csv"


def table(rows, col, label):
    agg = defaultdict(lambda: [0, 0])  # (framing, thinking, version) -> [correct, total]
    for r in rows:
        if r[col] in ("0", "1"):
            k = (r["framing"], r["thinking_mode"], r["version"])
            agg[k][0] += int(r[col])
            agg[k][1] += 1
    print(f"\n=== {label} accuracy ===")
    print(f"{'framing':<14} {'thinking':<10} {'A (riddle)':<14} {'B (riddle-riddle)':<18} {'A-B gap':<8}")
    print("-" * 68)
    for fr in sorted({k[0] for k in agg}):
        for tm in sorted({k[1] for k in agg}):
            a, b = agg.get((fr, tm, "A")), agg.get((fr, tm, "B"))
            a_acc = a[0] / a[1] if a else None
            b_acc = b[0] / b[1] if b else None
            a_str = f"{a_acc:.0%} (n={a[1]})" if a else "-"
            b_str = f"{b_acc:.0%} (n={b[1]})" if b else "-"
            gap = f"{(a_acc - b_acc):+.0%}" if a and b else "-"
            print(f"{fr:<14} {tm:<10} {a_str:<14} {b_str:<18} {gap:<8}")


def divergence(rows):
    both = [r for r in rows
            if r["accuracy_strict"] in ("0", "1") and r["accuracy_permissive"] in ("0", "1")]
    diff = [r for r in both if r["accuracy_strict"] != r["accuracy_permissive"]]
    print(f"\n=== strict vs permissive divergence ===")
    if not both:
        print("no comparable rows")
        return
    print(f"{len(diff)}/{len(both)} scored responses differ ({len(diff)/len(both):.0%}).")
    items = defaultdict(lambda: [0, 0])  # riddle_id -> [diff, total]
    for r in both:
        items[r["riddle_id"]][1] += 1
    for r in diff:
        items[r["riddle_id"]][0] += 1
    flagged = sorted((rid, d, t) for rid, (d, t) in items.items() if d)
    if flagged:
        print("items where the two schemes ever disagree (the ambiguous ones):")
        for rid, d, t in flagged:
            print(f"  {rid}: {d}/{t} responses")


def main():
    rows = list(csv.DictReader(SCORED.open(encoding="utf-8")))
    table(rows, "accuracy_strict", "STRICT")
    table(rows, "accuracy_permissive", "PERMISSIVE")
    divergence(rows)


if __name__ == "__main__":
    main()
