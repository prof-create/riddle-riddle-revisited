# Riddle Riddle — framing & thinking experiment

A small reproduction/extension of *The Riddle Riddle* (Fascendini et al.),
testing how **framing** and **thinking** affect **accuracy** on the paper's 30
matched riddle pairs. Model: **Claude Opus 4.6**.

> **Note:** this project (code and write-up) was primarily authored by Claude Code,
> with David P. Reichert directing. Details have not been exhaustively human-verified
> — see the Limitations in the write-up.

📄 **[Read the write-up → `WRITEUP.md`](WRITEUP.md)** — the findings, figures, and discussion. This README covers how to run the code.

The paper's repo ships the analysis but *not* the code that generated the LLM
responses, and disabled thinking without a reproducible log of the setting. This
re-generates responses ourselves with every setting recorded per row.

## Design

- **Model:** `claude-opus-4-6`
- **Thinking:** `off` (`thinking: disabled`) vs `adaptive` (effort = `high`)
- **Framings** (edit in `generate.py`):
  - `baseline` — the paper's exact instruction (one definitive answer + one-sentence explanation)
  - `nudge` — anti-retrieval cue: "consider carefully; do not assume you've encountered it before"
  - `nudge_unconstrained` — same cue, without the one-sentence limit
  - `baseline_unconstrained` — bare riddle (no cue, no constraint)
  - `nudge_explicit` — explicit symmetric cue ("either a lateral-thinking puzzle or a straightforward problem that only looks like one — work out which") + anti-retrieval
  - `seq_control` — single-shot control for the sequential run (the neutral instruction in a system prompt)
- **Sequential / human-equivalent** (`generate_sequential.py`): presents 6-item
  sessions (3 A + 3 B, distinct pairs, shuffled) in one multi-turn conversation
  with a neutral system prompt — tests whether the model infers the mix from
  context alone, the way human participants did. Writes `responses_seq.csv`.
- **Items:** 60 (30 pairs × condition A genuine riddle / B riddle-riddle)
- **Repetitions:** 5 per cell (`--reps`)
- **Metric:** accuracy, scored by an LLM judge (Sonnet 4.6) under BOTH the paper's
  strict (canonical only) and permissive (canonical + accepted alternatives)
  schemes. The judge code is copied verbatim from the paper's repo for comparability.

Scope each run with `--framings` / `--thinking` / `--reps` (and `--limit` for smoke tests). Full grid = 6 framings × 2 thinking × 60 items × 5 reps.

## Setup

```bash
git clone <repo-url> && cd riddle_experiment
python -m venv .venv
# activate — Windows: .venv\Scripts\activate   •   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Provide an Anthropic API key (the developer platform, separate from a Claude.ai
subscription and billed per token — create one at
<https://console.anthropic.com>) in **either**:

- a file named `.anthropic_key` in the repo root (gitignored), **or**
- the `ANTHROPIC_API_KEY` environment variable.

## Run

```powershell
python generate.py    # -> responses.csv   (resumable; re-run to continue)
python judge.py       # -> scored.csv      (strict + permissive accuracy)
python analyze.py     # accuracy by framing x thinking x A/B (both schemes) + strict/permissive divergence
python plot.py --thinking off        # -> results_permissive_off.png
python plot.py --thinking adaptive   # -> results_permissive_adaptive.png

# Sequential / human-equivalent run + its judging (separate files):
python generate_sequential.py --sessions 50           # -> responses_seq.csv
python judge.py --responses responses_seq.csv --scored scored_seq.csv
```

`generate.py`, `generate_sequential.py`, and `judge.py` are all resumable: if
interrupted, just run again and they skip work already written. Prefix with
`PYTHONUNBUFFERED=1` for live progress logs.

## Files

- `stimuli_answer_keys.csv` — the 60 items + accepted answers (from the paper's repo)
- `generate.py` — single-shot generation across framing × thinking → `responses.csv`
- `generate_sequential.py` — multi-turn human-equivalent sessions → `responses_seq.csv`
- `judge.py` — strict + permissive accuracy (verbatim paper judge); `--responses/--scored` to point at other files
- `analyze.py` — accuracy tables (both schemes) + strict/permissive divergence
- `plot.py` — permissive accuracy figure per thinking mode (`--thinking off|adaptive`)

## License & attribution

- This repo's code and write-up: MIT (see `LICENSE`).
- `judge.py` reuses the paper's judge code **verbatim** (MIT), and
  `stimuli_answer_keys.csv` is the paper's stimulus set (**CC-BY 4.0**). Full
  notices and attribution in `THIRD_PARTY_LICENSES.md`. Original work:
  Fascendini, McGregor, Gupta & Griffiths, *"The Riddle Riddle"*
  (arXiv:2606.27103), <https://github.com/bellafascendini/riddle-riddle>.
