# Third-party licenses & attribution

This project reuses material from the original *Riddle Riddle* repository:
Fascendini, McGregor, Gupta & Griffiths, *"The Riddle Riddle: Testing Flexible
Reasoning in Large Language Models and Humans"* (arXiv:2606.27103),
<https://github.com/bellafascendini/riddle-riddle>.

## Code — MIT

`judge.py` copies the LLM-as-judge functions and prompt strings **verbatim** from
the upstream repository's `LLM/scripts/llm_exp_judge_pipeline.py` (for
comparability with the paper). The upstream MIT license and copyright notice are
reproduced in full below, as the license requires:

```
MIT License

Copyright (c) 2026 the Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Stimuli / data — CC-BY 4.0

`stimuli_answer_keys.csv` is the upstream stimulus set (the 30 matched riddle
pairs with answer keys), licensed **CC-BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>).

- **Attribution:** Fascendini, McGregor, Gupta & Griffiths, *"The Riddle Riddle"*
  (arXiv:2606.27103); from <https://github.com/bellafascendini/riddle-riddle>.
- **Changes:** used unchanged — copied from the upstream `stimuli/` directory; no
  riddle text or answer keys were altered.

Everything else in this repository (the rest of the code, and `WRITEUP.md`) is
covered by this repo's own `LICENSE`.
