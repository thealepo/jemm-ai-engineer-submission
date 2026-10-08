# My AI Engineer Assessment Submission

Alex Eduardo Sanchez

## Presentation

[View or download the final presentation (PDF)](presentation/Jemm_Tec_Competition.pdf)

## Fastest review path

```bash
cd solstice-agent

# Fast implementation tests
python3 -m unittest discover -s tests -v

# New evaluation
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_eval

# Preserved legacy evaluator, for comparison only
PYTHONDONTWRITEBYTECODE=1 python3 -m eval.run_legacy_eval
```

## Reproduce the implementation stages

| Tag | Contents |
|---|---|
| `original-baseline` | Unmodified repository supplied for the challenge |
| `evaluator-rank1` | Simplified clean evaluator, tests, and rank-1 retention |
| `bm25-final` | Final submission with deterministic BM25 reranking |

Run the original evaluation:

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
