#!/usr/bin/env python
"""
Score responses.csv with the paper's own LLM-as-judge code.

The judging functions below (prompts, get_all_answers, get_listed_strategies,
build_*_prompt, _extract_json, _call_judge, judge_strict, judge_permissive) are
COPIED VERBATIM from the upstream repo's LLM/scripts/llm_exp_judge_pipeline.py
(Fascendini et al., https://github.com/bellafascendini/riddle-riddle) — including
the "iswrong" typo in the strict prompt — so our accuracy coding is comparable to
the paper's. Only `main()` (the driver) and key-loading are ours, adapted to feed
our responses.csv (framing x thinking conditions) and to record BOTH schemes per
response. Settings match the paper: Sonnet 4.6, temperature=0, max_tokens=400.

Requires pandas (the upstream functions use pd.isna / pd.notna and a bool
is_open_ended column from pd.read_csv). Reads responses.csv +
stimuli_answer_keys.csv, writes scored.csv. Resumable.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd

HERE = Path(__file__).resolve().parent
RESP = HERE / "responses.csv"
KEY = HERE / "stimuli_answer_keys.csv"
OUT = HERE / "scored.csv"
JUDGE_MODEL = "claude-sonnet-4-6"

client = None  # set in main()


def load_key():
    kf = HERE / ".anthropic_key"
    if kf.exists():
        return kf.read_text(encoding="utf-8").strip()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    sys.exit("No API key: create .anthropic_key or set ANTHROPIC_API_KEY (see README).")


# ═════════════════════════════════════════════════════════════════════════════
# BELOW: verbatim from upstream llm_exp_judge_pipeline.py (do not edit — keep
# byte-comparable to the paper's judge).
# ═════════════════════════════════════════════════════════════════════════════

# Helpers for key-based strategy logic
def get_listed_strategies(key_row):
    s = {key_row['canonical_strategy']}
    if pd.notna(key_row['alternative_strategies']) and key_row['alternative_strategies']:
        s.update(key_row['alternative_strategies'].split('||'))
    return s


def get_all_answers(key_row):
    pairs = [(key_row['answer_canonical'], key_row['canonical_strategy'])]
    if pd.notna(key_row['answer_alternatives']) and key_row['answer_alternatives']:
        alts = key_row['answer_alternatives'].split('||')
        strats = key_row['alternative_strategies'].split('||')
        pairs.extend(zip(alts, strats))
    return pairs


JUDGE_SYSTEM_STRICT = """You are coding AI model responses to riddles in a psychology experiment.

STUDY DESIGN
Each riddle comes in two versions (A and B). You will be told which version.
  Version A — typically has a trick/non-obvious answer requiring inventive thinking.
  Version B — a modified version where the trick no longer applies.

YOUR TASK — STRICT ROUND
You are given ONE canonical correct answer per riddle. Code whether the model's
response matches that canonical answer (semantic equivalents OK; different
answers that happen to also work do NOT count). If the model gives more than one answer,
code "accuracy" = 0.

OPEN-ENDED RIDDLES: For a few riddles, the canonical answer is a CRITERION rather than
a specific item (e.g. "any solid container that holds water"). You will be told
when this is the case. For open-ended riddles, code "accuracy" = 1 if the model's
answer fits the criterion. If the model gives more than one answer, code "accuracy" = 0.

Also code the STRATEGY the model used:
  "inventive" — treats the riddle as a lateral thinking puzzle: wordplay, double meanings,
                naming tricks, mathematical tricks, paradoxes, creative reframes, metaphors, reinterpretations. Applies even when answer is wrong.
  "literal"   — takes the riddle at face value: straightforward reasoning from stated facts without any unusual interpretations.
                Applies even when answer iswrong.
  "ambiguous" — gives multiple answers or self-corrects mid-response, indicating uncertainty or a mix of strategies.

Respond ONLY with valid JSON:
{"accuracy": 0 or 1, "strategy": "inventive" or "literal" or "ambiguous", "rationale": "one sentence"}
"""

JUDGE_SYSTEM_PERMISSIVE = """You are coding AI model responses to riddles in a psychology experiment.

STUDY DESIGN
Each riddle comes in two versions (A and B). You will be told which version.
  Version A — typically has a trick/non-obvious answer requiring lateral thinking.
  Version B — a modified version where the trick no longer applies.

YOUR TASK — PERMISSIVE ROUND
You are given a list of valid answers (canonical first, then alternatives), each
tagged with its strategy type. Code whether the model's response matches ANY
listed answer (semantic equivalents OK), and if so, which one (by index: 0 =
canonical, 1 = first alternative, etc.). If the model gives more than one answer,
code "accuracy" = 0.

OPEN-ENDED RIDDLES: For a few riddles, the canonical answer is a CRITERION rather than
a specific item (e.g. "any solid container that holds water"). You will be told
when this is the case. For open-ended riddles, code "accuracy" = 1 if the model's
answer fits the criterion. If the model gives more than one answer, code "accuracy" = 0.

Also code the STRATEGY the model used:
  "inventive" — treats the riddle as a lateral thinking puzzle: wordplay, double meanings,
                naming tricks, mathematical tricks, paradoxes, creative reframes, metaphors, reinterpretations. Applies even when answer is wrong.
  "literal"   — takes the riddle at face value: straightforward reasoning from stated facts without any unusual interpretations.
                Applies even when answer is wrong.
  "ambiguous" — gives multiple answers or self-corrects mid-response, indicating uncertainty or a mix of strategies.

Respond ONLY with valid JSON:
{"accuracy": 0 or 1, "matched_answer_idx": integer or null, "strategy": "inventive" or "literal" or "ambiguous", "rationale": "one sentence"}

matched_answer_idx must be:
  - null if accuracy=0
  - 0 if matched the canonical answer
  - 1, 2, ... for alternatives in the order listed
"""


def build_strict_prompt(key_row, model_response):
    open_note = ("\n\nNOTE: This is an OPEN-ENDED riddle — the canonical describes a "
                 "criterion rather than a specific item. Accept any answer meeting the criterion."
                 if key_row['is_open_ended'] else "")
    return (f"Riddle (Version {key_row['version']}): {key_row['riddle_text']}\n\n"
            f"Canonical correct answer: {key_row['answer_canonical']}{open_note}\n\n"
            f"Model response:\n{model_response}")


def build_permissive_prompt(key_row, model_response):
    pairs = get_all_answers(key_row)
    ans_block = "\n".join(
        f"  [{i}] ({strat}) {ans}" for i, (ans, strat) in enumerate(pairs)
    )
    open_note = ("\n\nNOTE: This is an OPEN-ENDED riddle — the canonical describes a "
                 "criterion rather than a specific item. Accept any answer meeting the criterion."
                 if key_row['is_open_ended'] else "")
    return (f"Riddle (Version {key_row['version']}): {key_row['riddle_text']}\n\n"
            f"Valid answers:\n{ans_block}{open_note}\n\n"
            f"Model response:\n{model_response}")


def _extract_json(raw):
    """Parse JSON, tolerating code fences and preamble."""
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{': depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def _call_judge(system_prompt, user_prompt, max_attempts=4, temperature=0):
    """Low-level wrapper with retries. Returns (parsed_dict_or_None, error_or_None)."""
    raw = ''
    for attempt in range(max_attempts):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=400,
                temperature=temperature,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_prompt}]
            )
            raw = resp.content[0].text.strip()
            parsed = _extract_json(raw)
            if parsed is None:
                raise json.JSONDecodeError('no JSON found', raw, 0)
            return parsed, None
        except json.JSONDecodeError:
            if attempt == max_attempts - 1:
                return None, f'json_parse_error: {raw[:100]}'
        except anthropic.RateLimitError:
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            if attempt == max_attempts - 1:
                return None, f'api_error: {str(e)[:100]}'
            time.sleep(5)
    return None, 'max_attempts_exceeded'


def judge_strict(key_row, model_response):
    """Returns dict with accuracy, strategy, strategy_correctness, rationale.

    Closed-form riddles: strict comparison against the single canonical answer;
      strategy_correctness = 1 iff strategy matches canonical_strategy.
    Open-ended riddles: behave like permissive — no single answer is privileged,
      any listed valid answer (or anything meeting the criterion) counts as
      correct; strategy_correctness = 1 iff strategy is in the listed set.
    Strategy: always the judge's inferred label for the model's actual reasoning."""
    empty = {'accuracy': None, 'strategy': None, 'strategy_correctness': None, 'rationale': None}
    if pd.isna(model_response) or str(model_response).strip() == '':
        return {**empty, 'rationale': 'missing model response'}


    prompt = build_strict_prompt(key_row, model_response)
    parsed, err = _call_judge(JUDGE_SYSTEM_STRICT, prompt)

    if parsed is None:
        return {**empty, 'rationale': err}
    try:
        acc = int(parsed['accuracy'])
        strat = str(parsed['strategy'])
        sc = int(strat == key_row['canonical_strategy'])

        return {
            'accuracy':             acc,
            'strategy':             strat,
            'strategy_correctness': sc,
            'rationale':            str(parsed.get('rationale', '')),
        }
    except (KeyError, ValueError, TypeError) as e:
        return {**empty, 'rationale': f'field_parse_error: {e}'}


def judge_permissive(key_row, model_response):
    """Returns dict with accuracy, matched_answer_idx, strategy, strategy_correctness, rationale."""
    empty = {'accuracy': None, 'matched_answer_idx': None, 'strategy': None,
             'strategy_correctness': None, 'rationale': None}
    if pd.isna(model_response) or str(model_response).strip() == '':
        return {**empty, 'rationale': 'missing model response'}

    prompt = build_permissive_prompt(key_row, model_response)
    parsed, err = _call_judge(JUDGE_SYSTEM_PERMISSIVE, prompt)
    if parsed is None:
        return {**empty, 'rationale': err}
    try:
        acc = int(parsed['accuracy'])
        strat = str(parsed['strategy'])
        matched_idx = parsed.get('matched_answer_idx')
        if matched_idx is not None:
            matched_idx = int(matched_idx)

        # strategy_correctness: 1 iff judge's strategy is listed as valid for this riddle
        listed = get_listed_strategies(key_row)
        sc = int(strat in listed)

        return {
            'accuracy':             acc,
            'matched_answer_idx':   matched_idx,
            'strategy':             strat,
            'strategy_correctness': sc,
            'rationale':            str(parsed.get('rationale', '')),
        }
    except (KeyError, ValueError, TypeError) as e:
        return {**empty, 'rationale': f'field_parse_error: {e}'}


# ═════════════════════════════════════════════════════════════════════════════
# Driver (ours): feed responses.csv through both judges, write scored.csv.
# ═════════════════════════════════════════════════════════════════════════════
OUT_COLS = [
    "accuracy_strict", "strategy_strict", "strategy_correctness_strict", "rationale_strict",
    "accuracy_permissive", "matched_answer_idx", "strategy_permissive",
    "strategy_correctness_permissive", "rationale_permissive",
]


def main():
    ap = argparse.ArgumentParser(description="Score responses with strict + permissive judges.")
    ap.add_argument("--responses", default=str(RESP), help="input responses CSV")
    ap.add_argument("--scored", default=str(OUT), help="output scored CSV")
    args = ap.parse_args()
    resp_path, out_path = Path(args.responses), Path(args.scored)

    global client
    client = anthropic.Anthropic(api_key=load_key())

    key_df = pd.read_csv(KEY)
    key_by_id = {row["riddle_id"]: row for _, row in key_df.iterrows()}

    with resp_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{resp_path} is empty — run generation first.")

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {(r["framing"], r["thinking_mode"], r["riddle_id"], r["repetition"])
                    for r in csv.DictReader(f)}

    out_cols = list(rows[0].keys()) + OUT_COLS
    new_file = not out_path.exists()
    f_out = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=out_cols)
    if new_file:
        writer.writeheader()

    for i, r in enumerate(rows, 1):
        key = (r["framing"], r["thinking_mode"], r["riddle_id"], r["repetition"])
        if key in done:
            continue
        key_row = key_by_id[r["riddle_id"]]
        answer = r.get("model_answer") or None  # let the judges' missing-response guard handle it
        s = judge_strict(key_row, answer)
        p = judge_permissive(key_row, answer)
        out = dict(r)
        out.update(
            accuracy_strict=s["accuracy"], strategy_strict=s["strategy"],
            strategy_correctness_strict=s["strategy_correctness"], rationale_strict=s["rationale"],
            accuracy_permissive=p["accuracy"], matched_answer_idx=p["matched_answer_idx"],
            strategy_permissive=p["strategy"],
            strategy_correctness_permissive=p["strategy_correctness"],
            rationale_permissive=p["rationale"],
        )
        writer.writerow(out)
        f_out.flush()
        print(f"[{i}/{len(rows)}] {key} -> strict={out['accuracy_strict']} "
              f"permissive={out['accuracy_permissive']}")
        time.sleep(0.2)
    f_out.close()
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
