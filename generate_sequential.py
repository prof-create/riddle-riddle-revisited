#!/usr/bin/env python
"""
Sequential / human-equivalent generation.

Presents several riddle items in ONE conversation (multi-turn), the way human
participants saw a mixed set, to test whether the model infers from CONTEXT
ALONE that some items are pseudo-riddles — without being told. Mirrors the
paper's human setup: a session of 6 items (3 condition-A + 3 condition-B, from
distinct pairs), randomized order.

DELIBERATELY NEUTRAL framing: the system prompt announces the sequence and the
answer format (same substance as `baseline`) but does NOT mention tricks /
straightforward problems — that would re-introduce the explicit cue and defeat
the fair human comparison. The question is whether context alone suffices.

Kept separate from generate.py / responses.csv so existing logic is untouched.
Writes responses_seq.csv (adds session_id, position columns). Judge it with:
    python judge.py --responses responses_seq.csv --scored scored_seq.csv
(the judge scores each response independently — no other changes needed).

Model: Opus 4.6, thinking adaptive (effort high). Resumable at session level.
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
STIMULI = HERE / "stimuli_answer_keys.csv"
OUT = HERE / "responses_seq.csv"

MODEL = "claude-opus-4-6"
EFFORT = "high"
MAX_TOKENS = 8000
SLEEP_S = 0.3

# Neutral preamble: sequence + answer format only. No hint about tricks.
SEQ_SYSTEM = (
    "You will be given a series of word problems, one at a time. For each, give "
    "one definitive answer and a one-sentence explanation for how you arrived at it."
)

FIELDS = [
    "model", "framing", "thinking_mode", "effort", "session_id", "position",
    "pair", "version", "riddle_id", "repetition", "riddle_text", "model_answer",
    "thinking_used", "input_tokens", "output_tokens", "error",
]


def load_key():
    kf = HERE / ".anthropic_key"
    if kf.exists():
        return kf.read_text(encoding="utf-8").strip()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    sys.exit("No API key: create .anthropic_key or set ANTHROPIC_API_KEY.")


def load_items():
    with STIMULI.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "A": {r["riddle_id"][:-1]: r for r in rows if r["version"] == "A"},
        "B": {r["riddle_id"][:-1]: r for r in rows if r["version"] == "B"},
    }


def build_session(by_ver, n_per_side, rng):
    """n_per_side A-items + n_per_side B-items, all from DISTINCT pairs, shuffled.

    Using distinct pairs avoids putting a matched A/B pair in the same session
    (which would near-give-away the trick); humans saw a varied set, not pairs.
    """
    pairs = sorted(by_ver["A"].keys())
    chosen = rng.sample(pairs, n_per_side * 2)
    a_pairs, b_pairs = chosen[:n_per_side], chosen[n_per_side:]
    items = [by_ver["A"][p] for p in a_pairs] + [by_ver["B"][p] for p in b_pairs]
    rng.shuffle(items)
    return items


def done_sessions():
    if not OUT.exists():
        return set()
    with OUT.open(encoding="utf-8") as f:
        return {r["session_id"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description="Sequential multi-turn generation.")
    ap.add_argument("--sessions", type=int, default=20, help="number of sessions")
    ap.add_argument("--per-side", type=int, default=3,
                    help="A-items and B-items per session (default 3 -> 6-item sessions)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=load_key())
    by_ver = load_items()
    rng = random.Random(args.seed)
    skip = done_sessions()

    new_file = not OUT.exists()
    f_out = OUT.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    for s in range(args.sessions):
        sid = f"s{args.seed}_{s:03d}"
        items = build_session(by_ver, args.per_side, rng)  # draw even if skipping, to keep rng stream stable
        if sid in skip:
            continue
        messages = []  # the growing conversation
        for pos, it in enumerate(items, 1):
            messages.append({"role": "user", "content": it["riddle_text"]})
            row = {
                "model": MODEL, "framing": "sequential", "thinking_mode": "adaptive",
                "effort": EFFORT, "session_id": sid, "position": pos,
                "pair": it["riddle_id"][:-1], "version": it["version"],
                "riddle_id": it["riddle_id"], "repetition": sid,
                "riddle_text": it["riddle_text"],
            }
            try:
                resp = client.messages.create(
                    model=MODEL, system=SEQ_SYSTEM, max_tokens=MAX_TOKENS,
                    thinking={"type": "adaptive"}, output_config={"effort": EFFORT},
                    messages=messages,
                )
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                tused = any(getattr(b, "type", None) == "thinking" and getattr(b, "thinking", "")
                            for b in resp.content)
                messages.append({"role": "assistant", "content": text})  # text-only history
                row.update(model_answer=text, thinking_used=int(tused),
                           input_tokens=resp.usage.input_tokens,
                           output_tokens=resp.usage.output_tokens, error="")
            except Exception as e:  # noqa: BLE001
                messages.append({"role": "assistant", "content": "(no answer)"})
                row.update(model_answer="", thinking_used="", input_tokens="",
                           output_tokens="", error=str(e)[:200])
            writer.writerow(row)
            f_out.flush()
            print(f"[session {s+1}/{args.sessions} {sid}] pos{pos} {it['riddle_id']}"
                  f"{'  ERROR' if row['error'] else ''}")
            time.sleep(SLEEP_S)
    f_out.close()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
