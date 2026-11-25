# Distributed training: technical analysis

**Decision: we do not use DDP, FSDP or ZeRO.** A single CPU is enough, with a wide margin.

## The evidence

| Fact | Value | Source |
|---|---|---|
| Training time | 0.38 s (5 fits, min 0.36, max 0.40) | measured, `benchmarks/pipeline-steps-results.csv` |
| Training data | 2,790 rows × 62 features, 1.8 MB | measured |
| Memory added by training | 17 MB | measured |
| Model size | 9.76 MB | measured |
| Cross-validation (5 folds) | a few seconds | observed in the pipeline |

## Why each technique does not apply

| Technique | What it solves | Why not here |
|---|---|---|
| **DDP** (Distributed Data Parallel) | Splits each batch across GPUs and averages gradients, to shorten long neural-network training | No gradients, no GPU: a random forest builds trees independently. scikit-learn already uses all CPU cores (`n_jobs=-1`), and the fit lasts 0.38 s |
| **FSDP** (Fully Sharded Data Parallel) | Shards parameters, gradients and optimizer state so a model too large for one GPU can train | The model is 9.76 MB. It fits in memory about a thousand times over |
| **ZeRO** | Same idea in stages (optimizer state, then gradients, then parameters) | Same reason: there is no optimizer state to shard |

Distribution also has a cost. In the course lab on DDP, running DDP with a single process was
already slower than plain training, because of the synchronisation machinery. On a 0.38 s job
the start-up time of the workers alone would exceed the job.

## What we would parallelise instead

Feature extraction: about 20 s for 3,490 rides in one process. Each ride is independent, so
with 100 times more rides the right tool is **data parallelism on CPU** (a process pool, or
Spark/Dask on a cluster), not distributed training.

## When the decision would change

- The model becomes a neural network on the raw time series (1D CNN or transformer on
  5,000 samples × 6 channels per ride) **and** one epoch takes long enough to matter, hours
  rather than seconds: then DDP on several GPUs.
- That network no longer fits in one GPU's memory: then FSDP or ZeRO.

With the current data (268 MB) even a small neural network would train on one GPU. And the
random forest already reaches 0.99 ROC AUC, so there is no accuracy problem to solve with a
bigger model.
