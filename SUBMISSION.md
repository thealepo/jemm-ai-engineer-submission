# Alex Sanchez — AI Engineer assessment submission

This repository preserves the supplied Solstice Support Agent baseline and
adds two independently inspectable implementation stages:

1. a simplified, uncontaminated, stage-aware evaluator plus the direct
   retrieval fix that retains rank 1; and
2. a deterministic BM25 reranking experiment that removes the 12 sequential
   model calls from each knowledge-base search.

The presentation is the primary deliverable. This repository provides the
working code, tests, and reproducible measurements behind the implemented
portions of the proposal. Conversation-state redesign remains proposed work;
it is not represented here as implemented.

## Fastest review path

Python 3.9 or newer is sufficient. The project uses only the standard library
and requires no API key or network access.

```bash
cd solstice-agent

# Fast implementation tests
python3 -m unittest discover -s tests -v

# Uncontaminated, stage-aware evaluation
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_eval

# Preserved legacy evaluator, for comparison only
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_legacy_eval
```

The clean evaluator intentionally reports separate routing, retrieval, tool,
answer, single-turn, and multi-turn results. Its scores are deterministic
assertion coverage, not a claim of general production accuracy.

## Reproduce the implementation stages

Three tags make the comparison explicit:

| Tag | Contents |
|---|---|
| `original-baseline` | Unmodified repository supplied for the challenge |
| `evaluator-rank1` | Simplified clean evaluator, tests, and rank-1 retention |
| `bm25-final` | Final submission with deterministic BM25 reranking |

Run the original contaminated evaluation:

```bash
git switch --detach original-baseline
cd solstice-agent
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_eval
```

Run the clean evaluator after retaining rank 1:

```bash
git switch --detach evaluator-rank1
cd solstice-agent
python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_eval
```

Run the final BM25 experiment:

```bash
git switch main
cd solstice-agent
python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_eval
```

Return to `main` after inspecting a tag:

```bash
git switch main
```

## Verified comparison

These measurements were collected under the same self-contained mock
workload. Wall time varies by host; attempts, prompt volume, and estimated
cost are the more stable comparison metrics.

| Clean-evaluation result | Retain rank 1 + LLM reranker | BM25 experiment |
|---|---:|---:|
| Deterministic answer assertions | 7/14 | 6/14 |
| Single-turn scenarios | 4/9 | 4/9 |
| Multi-turn scenarios | 1/3 | 0/3 |
| Model attempts | 119 | 23 |
| Prompt tokens | 31,770 | 8,268 |
| Estimated cost | $0.1054 | $0.0342 |

BM25 reduced measured estimated cost by 67.6%, model attempts by 80.7%, and
prompt tokens by 74.0%. It did **not** preserve answer quality in this small
suite, so it should be read as an efficiency experiment and design signal—not
as a production-ready replacement.

## What changed

### Trustworthy evaluation

- References never enter agent inputs.
- Scenarios use fresh agents while turns within a multi-turn scenario share
  state.
- Assertions check required and forbidden phrases, exact values, polarity,
  routing, selected evidence, and record-tool evidence.
- Execution errors are diagnosed without aborting the remaining cases.
- The contaminated evaluator remains available as
  `eval/run_legacy_eval.py`, clearly labeled as legacy.

### Rank-1 correctness fix

The baseline reranker sorted candidates and then returned ranks 2–5. The
implementation retains ranks 1–4 instead. This is deliberately isolated from
broader retrieval redesign so the effect remains attributable.

### BM25 cost experiment

The final stage preserves embedding recall and replaces 12 sequential LLM
relevance calls with deterministic BM25 scoring over the candidate set, using
document frequencies from the full chunk corpus. Tests cover discriminative
term ranking, stable tie ordering, corpus-level document frequency, and the
absence of reranker model calls.

## Known limitations and remaining work

- Routing and tool-argument extraction still fail important cases.
- Empty evidence can still produce an unsupported answer.
- Conversation history reaches final generation but not routing, argument
  resolution, or retrieval; stateful follow-ups remain unreliable.
- Summarization and prefix truncation can discard durable facts and the newest
  evidence.
- The BM25 result requires further relevance tuning or a measured hybrid
  design before production use.
- Authentication and authorization requirements for ticket and invoice data
  are unspecified and therefore not invented in this submission.

These limitations are kept visible because the objective is a defensible
diagnosis and prioritized engineering plan, not an artificial perfect score.
