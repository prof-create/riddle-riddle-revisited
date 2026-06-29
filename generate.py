#!/usr/bin/env python
"""
Generate LLM responses to the riddle / riddle-riddle stimuli.

Reproduction experiment for "The Riddle Riddle" (Fascendini et al.), focused on
how *framing* and *thinking* affect accuracy. Model: Claude Opus 4.6.

Reads stimuli_answer_keys.csv (60 items = 30 matched pairs, conditions A/B).
For each (framing x thinking_mode x item x repetition) it calls the Anthropic
API and appends one row to responses.csv. Resumable: re-running skips rows
already present, so an interrupted run just continues.

Setup: see README.md. Needs an Anthropic API key in .anthropic_key (or the
ANTHROPIC_API_KEY env var) and `pip install anthropic`.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
STIMULI = HERE / "stimuli_answer_keys.csv"
OUT = HERE / "responses.csv"

MODEL = "claude-opus-4-6"
REPETITIONS = 5              # per (framing, thinking_mode, item)
THINKING_MODES = ["off", "adaptive"]
EFFORT = "high"             # only used when thinking is adaptive (low|medium|high|max)
MAX_TOKENS_OFF = 1024
MAX_TOKENS_ON = 8000        # headroom for adaptive thinking + the answer
SLEEP_S = 0.3              # gentle pacing between calls

# Temperature is intentionally not set: the API default is 1, which matches the
# paper's "default temperature settings (temperature = 1)".

# --- Framings: the main variable. Edit freely. -----------------------------
# {riddle} is replaced with the item text. `system` may be None.
FRAMINGS = {
    # Paper's exact instruction (Experiment 1, Section 4.1.2).
    "baseline": {
        "system": None,
        "template": (
            "Please provide one definitive answer to each word problem and a "
            "one-sentence explanation for how you arrived at it.\n\n{riddle}"
        ),
    },
    # Epistemic / anti-retrieval cue (content-free about the solution): tells the
    # model to think and not pattern-match to a remembered instance, WITHOUT
    # hinting literal-vs-lateral. The most conservative probe of the "LLMs only
    # do memorization" claim — if this lifts B, pure retrieval isn't the whole
    # story. Watch A too: if A drops (recognition was carrying A) while B rises,
    # that is itself diagnostic of memorization.
    "nudge": {
        "system": None,
        "template": (
            "Consider this problem carefully and do not assume you have "
            "encountered it before. Then give one definitive answer and a "
            "one-sentence explanation for how you arrived at it.\n\n{riddle}"
        ),
    },
    # Same cue, unconstrained (no one-sentence limit) — isolates the effect of
    # the paper's terse-answer constraint while holding the cue.
    "nudge_unconstrained": {
        "system": None,
        "template": (
            "Consider this problem carefully and do not assume you have "
            "encountered it before. Then answer it.\n\n{riddle}"
        ),
    },
    # No cue, unconstrained (bare riddle) — the cue-absent parallel of
    # nudge_unconstrained. Completes the cue x constraint 2x2: vs
    # nudge_unconstrained it isolates the cue; vs baseline it isolates the
    # paper's instruction wrapper.
    "baseline_unconstrained": {
        "system": None,
        "template": "{riddle}",
    },
    # Explicit symmetric cue: names both possibilities (lateral-thinking puzzle
    # vs straightforward problem that only looks like one) + anti-retrieval, in
    # the constrained format (matches `nudge`, the best cell so far). Tests
    # whether a more explicit but still non-cheating cue pushes B past the ~60%
    # the milder cues plateau at. Reveals neither which type an item is nor the
    # answer.
    "nudge_explicit": {
        "system": None,
        "template": (
            "This is either a lateral-thinking puzzle or a straightforward "
            "problem that only looks like a puzzle — work out which, and don't "
            "just assume you've seen this problem before. Then give one "
            "definitive answer and a one-sentence explanation for how you "
            "arrived at it.\n\n{riddle}"
        ),
    },
    # Single-shot control for the sequential run: the same neutral instruction in
    # a SYSTEM prompt (no trick hint), one item per call. sequential-vs-this
    # isolates the multi-turn-context effect; this-vs-baseline isolates the
    # system-prompt-vs-per-call-instruction placement difference.
    "seq_control": {
        "system": ("You will be given a word problem. Give one definitive answer "
                   "and a one-sentence explanation for how you arrived at it."),
        "template": "{riddle}",
    },
    # "drop ONLY the one-sentence brevity limit" versions of baseline / nudge:
    # keep the full answer-format instruction ("one definitive answer and an
    # explanation"), remove just "a one-sentence" -> "an". Isolates the brevity
    # constraint. (The *_unconstrained framings above removed the whole
    # instruction — a bare riddle / "answer it" — which is a different thing.)
    "baseline_nobrevity": {
        "system": None,
        "template": (
            "Please provide one definitive answer to each word problem and an "
            "explanation for how you arrived at it.\n\n{riddle}"
        ),
    },
    "nudge_nobrevity": {
        "system": None,
        "template": (
            "Consider this problem carefully and do not assume you have "
            "encountered it before. Then give one definitive answer and an "
            "explanation for how you arrived at it.\n\n{riddle}"
        ),
    },
}
# ---------------------------------------------------------------------------

FIELDS = [
    "model", "framing", "thinking_mode", "effort", "pair", "version",
    "riddle_id", "repetition", "riddle_text", "model_answer", "thinking_used",
    "input_tokens", "output_tokens", "error",
]


def load_key():
    keyfile = HERE / ".anthropic_key"
    if keyfile.exists():
        return keyfile.read_text(encoding="utf-8").strip()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    sys.exit("No API key: create .anthropic_key or set ANTHROPIC_API_KEY (see README).")


def load_items():
    with STIMULI.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_keys():
    if not OUT.exists():
        return set()
    with OUT.open(encoding="utf-8") as f:
        return {
            (r["framing"], r["thinking_mode"], r["riddle_id"], r["repetition"])
            for r in csv.DictReader(f)
        }


def call(client, riddle_text, framing, thinking_mode):
    spec = FRAMINGS[framing]
    user = spec["template"].format(riddle=riddle_text)
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": user}])
    if spec["system"]:
        kwargs["system"] = spec["system"]
    if thinking_mode == "off":
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["max_tokens"] = MAX_TOKENS_OFF
    else:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": EFFORT}
        kwargs["max_tokens"] = MAX_TOKENS_ON
    resp = client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    thinking_used = any(
        getattr(b, "type", None) == "thinking" and getattr(b, "thinking", "")
        for b in resp.content
    )
    return text, thinking_used, resp.usage


def main():
    ap = argparse.ArgumentParser(description="Generate riddle responses.")
    ap.add_argument("--reps", type=int, default=REPETITIONS,
                    help=f"repetitions per cell (default {REPETITIONS})")
    ap.add_argument("--framings", default=",".join(FRAMINGS),
                    help="comma-separated framing names to run (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N items, for smoke tests (0 = all 60)")
    ap.add_argument("--thinking", default=",".join(THINKING_MODES),
                    help="comma-separated thinking modes (off, adaptive); default: both")
    args = ap.parse_args()
    framings = [f.strip() for f in args.framings.split(",") if f.strip()]
    bad = [f for f in framings if f not in FRAMINGS]
    if bad:
        sys.exit(f"Unknown framing(s): {bad}. Choose from {list(FRAMINGS)}.")
    modes = [m.strip() for m in args.thinking.split(",") if m.strip()]
    bad_m = [m for m in modes if m not in THINKING_MODES]
    if bad_m:
        sys.exit(f"Unknown thinking mode(s): {bad_m}. Choose from {THINKING_MODES}.")

    client = anthropic.Anthropic(api_key=load_key())
    items = load_items()
    if args.limit:
        items = items[:args.limit]
    skip = done_keys()
    new_file = not OUT.exists()
    f_out = OUT.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    total = len(framings) * len(modes) * len(items) * args.reps
    n = 0
    for framing in framings:
        for tmode in modes:
            for it in items:
                for rep in range(1, args.reps + 1):
                    n += 1
                    key = (framing, tmode, it["riddle_id"], str(rep))
                    if key in skip:
                        continue
                    row = {
                        "model": MODEL, "framing": framing, "thinking_mode": tmode,
                        "effort": EFFORT if tmode != "off" else "",
                        "pair": it["riddle_id"][:-1], "version": it["version"],
                        "riddle_id": it["riddle_id"], "repetition": rep,
                        "riddle_text": it["riddle_text"],
                    }
                    try:
                        text, tused, usage = call(client, it["riddle_text"], framing, tmode)
                        row.update(
                            model_answer=text, thinking_used=int(tused),
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens, error="",
                        )
                    except Exception as e:  # noqa: BLE001 - log and continue
                        row.update(
                            model_answer="", thinking_used="",
                            input_tokens="", output_tokens="", error=str(e)[:200],
                        )
                    writer.writerow(row)
                    f_out.flush()
                    print(f"[{n}/{total}] {framing}/{tmode}/{it['riddle_id']} "
                          f"rep{rep}{'  ERROR' if row['error'] else ''}")
                    time.sleep(SLEEP_S)
    f_out.close()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
