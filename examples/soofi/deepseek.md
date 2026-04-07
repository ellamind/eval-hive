# DeepSeek-V3 Evaluation Issues

## Problem

DeepSeek-V3 Base scores are anomalously low on multiple-choice (`_mc`) tasks and BBH CoT tasks.
All other task types (`_rc`, `_bpb`, generation tasks like GSM8K) produce expected scores.

## Root Cause: BPE Tokenizer Boundary Merge

The DeepSeek-V3 tokenizer has an unusually aggressive BPE vocabulary that merges
`: ` + letter (colon + space + letter) into a single token:

```
tok("Answer:")       → [7805, 28]       (2 tokens)
tok("Answer: A")     → [7805, 48389]    (2 tokens — ": A" merged into ":A")
tok("Answer: B")     → [7805, 59716]    (2 tokens — ": B" merged into ":B")
```

In `lm_eval/api/model.py`, `_encode_pair()` computes the continuation as:

```python
whole_enc = self.tok_encode(context + continuation)   # [7805, 48389]
context_enc = self.tok_encode(context)                # [7805, 28]
continuation_enc = whole_enc[len(context_enc):]       # [] ← EMPTY
```

Since all choices get an empty continuation → logprob = 0 → random selection.
This is why `_mc` scores perfectly match 1/N_choices (e.g., ~0.25 for 4-choice, ~0.10 for 10-choice).

Other models (Kimi K2, Nemotron, etc.) do NOT merge `: A` into a single token,
so the continuation retains at least one token and gets a real logprob.

## Impact

- **All `_mc` tasks**: Broken (random chance). Affects MMLU, ARC, AGIEval, GPQA, etc.
- **`_rc` tasks**: Working correctly. Use these instead.
- **`_bpb` tasks**: Working correctly.
- **Generation tasks** (GSM8K, math, etc.): Server and model are fine. Scores are plausible.
- **BBH CoT tasks**: All 27 tasks score 0.0 — see separate section below.
- **`deu_mmlu_pro_*_cot` (strict-match)**: Near 0.0 — same root cause as BBH.

## Affected Models

Any model using the DeepSeek tokenizer: DeepSeek-V3, DeepSeek-V3.1, DeepSeek-V3.2-Exp, etc.

## Known lm-eval Issues and PRs

- **[EleutherAI/lm-evaluation-harness#1297](https://github.com/EleutherAI/lm-evaluation-harness/issues/1297)** — `AssertionError: len(continuation_enc) > 0` from BPE boundary merge; reported with Persimmon tokenizer; still open.
- **[EleutherAI/lm-evaluation-harness#1053](https://github.com/EleutherAI/lm-evaluation-harness/issues/1053)** — Same assertion failure; reported with Shisa 7B / extended-vocab Mistral where `": C"` is a single token; still open.
- **[EleutherAI/lm-evaluation-harness#1322](https://github.com/EleutherAI/lm-evaluation-harness/pull/1322)** — Draft PR "Deal with `_encode_pair()` / Llama token 29871 / SPIECE_UNDERLINE better"; never merged, blocked on upstream HF transformers PR.

No fix has been merged in any lm-eval release through v0.4.11.

## Proposed Fix

Patch `_encode_pair()` in `lm_eval/api/model.py` (around line 395 in v0.4.12.dev0):

```python
if self.backend == "causal":
    whole_enc = self.tok_encode(context + continuation)
    context_enc = self.tok_encode(context)

    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]

    # Guard against BPE merges across the context/continuation boundary.
    # Some tokenizers (e.g. DeepSeek-V3) merge the last context token with
    # the first continuation token, producing fewer tokens than encoding
    # each part separately. Fall back to separate encoding so the
    # continuation is never empty.  See lm-eval #1297 / #1053.
    if len(continuation_enc) == 0:
        continuation_enc = self.tok_encode(
            continuation, add_special_tokens=False
        )
```

The file to edit is:
```
.pixi/envs/default/lib/python3.13/site-packages/lm_eval/api/model.py
```

### Caveats

This fallback handles the complete-merge case (continuation becomes empty).
It does NOT handle partial merges where only the first continuation token gets absorbed —
but that case still leaves a non-empty continuation with meaningful logprobs,
so it is far less impactful in practice.

## BBH CoT and Other CoT Tasks: Answer Format Mismatch

**Not a stop sequence issue.** The model generates full CoT reasoning (requests take
30-120 seconds each — real generation is happening). The problem is the answer
extraction regex.

BBH CoT uses the `get-answer` filter with regex:
```
(?<=the answer is )(.*)(?=.)
```
This requires the literal phrase `"the answer is"` (lowercase) in the model output.

Similarly, `deu_mmlu_pro_*_cot` uses `strict-match`:
```
(?:Die Antwort ist|[Tt]he answer is)\s*\(?([A-J])\)?
```

**Evidence:** Tasks that use `flexible-extract` (e.g., `multi_choice_regex` looking for
`(A)`, `(B)`, etc. anywhere) work fine for DSV3:
- `gpqa_diamond_cot`: 0.28 (flexible-extract)
- `deu_gsm8k_platinum_cot`: 0.75 (flexible-extract) vs 0.003 (strict-match)

Kimi K2 scores 0.66 avg on `deu_mmlu_pro_*_cot` strict-match. DSV3 scores 0.003.
Both use the same task templates and fewshot examples. The difference is that DSV3 base
does not reliably reproduce the `"the answer is X"` phrasing from the fewshot examples —
it expresses the final answer in other ways.

### Potential fixes for BBH CoT

1. Switch to `multi_choice_regex` (like GPQA's `flexible-extract`) instead of the
   `"the answer is"` regex — this extracts the last `(A)`/`(B)` etc. from the output.
2. Use a more permissive regex that also matches "The answer is", "the answer is",
   "answer is", etc.
3. Add `--log_samples` to future runs to inspect actual model outputs and tune the filter.
