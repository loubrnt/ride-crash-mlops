# AI Infrastructure contribution

Advanced Cloud AI Infrastructure (MSA-DAT09-05) · Group 3 · extension of our MLOps project.

This folder answers one question: **what does this model need to run, how fast is it, and how
would we serve it at scale for a reasonable cost?**

Rule followed throughout: **measure, do not claim.** Every number below is tagged
**[measured]** (produced by a script in this folder, raw results in the CSV files) or
**[estimate]** (computed from stated assumptions, not tested).

| Section | Folder |
|---|---|
| Current and proposed architecture | [`architecture/`](architecture/) |
| Baseline benchmark | [`benchmarks/`](benchmarks/) |
| Optimisation | [`optimization/`](optimization/) |
| Distributed training decision | [`distributed-training/`](distributed-training/technical-analysis.md) |
| Scaling, cost, resilience | [`cost-resilience/`](cost-resilience/infrastructure-plan.md) |

## Key findings

1. **The workload is small.** Training takes 0.38 s on a laptop CPU. No GPU, no distributed training.
2. **The baseline has a hidden cost.** The random forest spends 27 ms per prediction, mostly
   starting a thread pool for a single row.
3. **ONNX Runtime removes it:** 0.01 ms per prediction instead of 27 ms, with identical
   predictions (0 decision changed on 700 test rides) and a model file 7 times smaller.
4. **One optimisation is deployed and measured end to end:** `INFERENCE_THREADS=1` in the API
   image. Mean latency −7 %, p95 −9 % for sequential requests. The gain is smaller than
   in-process, and we say so.
5. **One API container serves about 30 requests per second.** For 100,000 rides a day that is
   one small serverless instance, a few dollars a month.

## 1. Resource sizing

| Item | Value | Source |
|---|---|---|
| Model | Random forest, 300 trees, max depth 8, median imputation (scikit-learn 1.9.1 pipeline) | `params.yaml` |
| Raw dataset | 268 MB parquet, 16.9 million rows, 3,490 rides | **[measured]** file size |
| Training input | 2.3 MB CSV: 3,490 rides × 62 features (2,790 used for training) | **[measured]** |
| Feature extraction, whole dataset | about 20 s, single process (load + 3,490 rides) | **[measured]** wall clock of the `features` stage |
| Training duration | **0.38 s** mean over 5 fits (min 0.36, max 0.40) | **[measured]** `pipeline-steps-results.csv` |
| Memory added by training | 17 MB | **[measured]** |
| Model file | 9.76 MB (`model.skops`) | **[measured]** |
| Hardware | Apple M3 Pro, 11 cores, 19.3 GB RAM, no GPU used | `benchmarks/environment.json` |
| API container memory | 243 MiB at rest after the benchmark | **[measured]** `docker stats` |
| API image | 501 MB compressed, 2.3 GB on disk (MLflow and scikit-learn inside) | **[measured]** |

**Is one machine enough?** Yes. Training fits in 17 MB and 0.4 s. The heaviest step is feature
extraction (20 s), which is independent per ride and would scale by processing rides in
parallel, not with a GPU.

## 2. Baseline benchmark

Fixed model version (registry version 2, the one in production), the 700 held-out test rides,
50 warm-up calls, then 1,000 timed predictions. CPU only, so no GPU synchronisation is needed.

### Model alone, in process

| Metric | Value | Conditions |
|---|---|---|
| Mean latency | **27.3 ms** | batch size 1, 1,000 repetitions |
| p95 latency | 28.9 ms | batch size 1 |
| Throughput | 36.6 predictions/s | batch size 1, sequential |
| Throughput | 19,326 rides/s | batch size 700 |
| Process memory | 264 MB (81 MB added by loading the model) | fresh Python process |
| Model size | 9.76 MB | serialized file |
| Quality | ROC AUC 0.9905 | 700 held-out rides |

### One request, step by step

Mean over the 4 real example rides (5,400 samples, 278 KB of JSON each):

| Step | Time | Share |
|---|---|---|
| JSON parsing and validation | 1.1 ms | 3 % |
| Conversion to a DataFrame | 2.9 ms | 9 % |
| Feature extraction | 1.5 ms | 5 % |
| **Model prediction** | **26.9 ms** | **83 %** |

The model is the bottleneck, so that is where we optimise.

### API over HTTP, Docker container

`POST /predict` on the running container, real rides, single uvicorn worker.

| Concurrency | Mean | p95 | Throughput |
|---|---|---|---|
| 1 | 40.0 ms | 44.4 ms | 25.0 requests/s |
| 4 | 135 ms | 208 ms | 29.5 requests/s |
| 16 | 504 ms | 615 ms | 31.6 requests/s |

Throughput stops growing at about 30 requests/s: one worker handles one prediction at a time,
so extra concurrent requests only wait in line.

## 3. Optimisation

**Hypothesis, written before testing:** a random forest with `n_jobs=-1` starts a thread pool
at every call. For one row this overhead should exceed the work itself. Removing it, or
replacing the runtime with ONNX Runtime, should cut latency without changing predictions.

Fair comparison: same model version, same 700 test rides, same machine, each runtime in a
fresh process.

### Model alone

| Metric | Baseline | 1 thread | ONNX Runtime | Interpretation |
|---|---|---|---|---|
| Mean latency, batch 1 | 27.3 ms | 11.8 ms | **0.010 ms** | ONNX is about 2,700 times faster |
| p95 latency, batch 1 | 28.9 ms | 12.6 ms | 0.012 ms | |
| Throughput, batch 700 | 19,326 rides/s | 34,798 | **400,610** | 20 times more in batch |
| Process memory | 264 MB | 307 MB | **231 MB** | lighter runtime |
| Model file | 9.76 MB | 9.76 MB | **1.42 MB** | 7 times smaller |
| ROC AUC | 0.9905 | 0.9905 | 0.9905 | unchanged |
| Decisions changed | — | 0 / 700 | **0 / 700** | max probability gap 0.000001 |

Both hypotheses hold. Even with one thread, scikit-learn spends 12 ms in Python overhead per
call for 300 trees; ONNX Runtime evaluates the same trees in compiled code.

### Deployed and measured end to end: `INFERENCE_THREADS=1`

This one is a configuration change, so we shipped it in the API image and measured the
container again.

| Concurrency | Metric | Baseline | Optimised | Change |
|---|---|---|---|---|
| 1 | mean / p95 | 40.0 / 44.4 ms | 37.0 / 40.3 ms | −7 % / −9 % |
| 4 | mean / p95 | 135 / 208 ms | 120 / 155 ms | −11 % / −25 % |
| 4 | throughput | 29.5 req/s | 33.2 req/s | +13 % |
| 16 | mean / p95 | 504 / 615 ms | 524 / 665 ms | no improvement |

**Honest reading.** In process the model step drops by 15 ms; in the container the request
only drops by 3 ms. The gain did not fully carry over and we have not found out why (thread
scheduling inside the Linux container is our guess, not verified). At 16 concurrent requests
there is no gain: the single worker is saturated either way.

### Not done

ONNX Runtime is measured on the model only. Serving it in the API would mean adding the ONNX
export to the pipeline and to the evaluation gate. From the step-by-step table, a request
would then cost about 5.5 ms instead of 32 ms in process **[estimate]**: that is the next step.

## 4. Distributed training decision

**Not justified.** Details in [`distributed-training/technical-analysis.md`](distributed-training/technical-analysis.md).

- DDP, FSDP and ZeRO distribute gradient-based training of neural networks across GPUs. A
  random forest has no gradients and no GPU code path in scikit-learn.
- Training takes 0.38 s and 17 MB. There is nothing to speed up or to fit in memory.
- The step worth parallelising is feature extraction (20 s, independent per ride): that is
  data parallelism across CPU cores, not distributed training.

## 5. Scaling, cost and resilience

Details and assumptions in [`cost-resilience/infrastructure-plan.md`](cost-resilience/infrastructure-plan.md).

- **Serving:** the existing stateless API image on a serverless container platform (Cloud Run),
  autoscaled on concurrent requests.
- **Sizing:** 100,000 rides a day is 1.2 requests/s on average. One instance measured at
  about 30 requests/s covers a 10× peak.
- **Cost:** a few dollars a month on Cloud Run, against about 15 for an always-on small VM and
  about 250 for a GPU that the model cannot use **[estimate]**.
- **Resilience:** images tagged by commit, health check, gradual rollout, rollback by
  redeploying the previous image or moving the `production` alias back in MLflow.

Nothing in this section is deployed: it is a design, not a tested system.

## Reproduce

```bash
uv sync --all-groups
make pipeline                     # or: split + train + evaluate + export
docker compose up -d
uv run python ai-infrastructure/benchmarks/benchmark.py --api http://localhost:8000
uv run python ai-infrastructure/benchmarks/request_breakdown.py
```
