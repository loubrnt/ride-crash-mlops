# Scaling, cost and resilience plan

**Status: design only.** Nothing here is deployed or tested. Capacity figures come from our
benchmark; prices are list prices recalled from the GCP pricing pages and must be checked in
the pricing calculator before any decision.

## Workload assumptions

| Assumption | Value |
|---|---|
| Rides scored per day | 100,000 (one request at the end of each ride) |
| Average load | 1.2 requests/s |
| Peak load | 10 × average = 12 requests/s |
| Request | 278 KB of JSON, about 5,400 samples |
| Region | europe-west1 |

## Capacity

Measured on our laptop: one container, one worker, about **30 requests/s** at 40 ms mean
latency. A cloud vCPU is slower than an M3 Pro core; we assume half, **15 requests/s per
instance** (assumption, to re-measure on the target machine).

- Average load: 1 instance at 8 % utilisation.
- Peak load: 1 instance is still enough; 2 give headroom.
- Scaling rule: add an instance above 8 concurrent requests per instance; maximum 5 instances.
- Scale the number of instances, not their size: the API is stateless and one request uses one core.

## Serving options and monthly cost

| Option | Monthly cost (estimate) | Comment |
|---|---|---|
| **Cloud Run**, 1 vCPU, 512 MiB, scale to zero | **about 2 to 8 USD** | 3 M requests × 0.08 s = 240,000 vCPU-seconds, partly covered by the free tier |
| Cloud Run with 1 instance always warm | about 50 USD | removes cold starts (our image is 500 MB compressed) |
| One small always-on VM (e2-small) | about 15 USD | cheapest fixed option, no autoscaling, manual recovery |
| GKE cluster | 75 USD management fee + nodes | justified only with several services |
| One T4 GPU always on | about 250 USD | **no benefit:** scikit-learn does not use it |

**Choice: Cloud Run.** The load is low and irregular, the service is stateless, and we pay per
request. Training costs nothing: 0.38 s, inside the free GitHub Actions run.

## Monitoring

| Signal | Why | Alert |
|---|---|---|
| p95 latency, error rate, requests/s | Service health | p95 > 200 ms or errors > 1 % for 5 minutes |
| Instance count, CPU, memory | Capacity | at maximum instances for 10 minutes |
| Share of rides predicted as crash | Model health without labels | moves by more than 50 % week over week |
| Feature distributions per city | Drift (8 cities in the dataset) | PSI above 0.25 on a key feature |

One caution on the third signal: our dataset contains 20 % crashes because of how it was built
(4 baseline rides per crash). Real traffic has far fewer, so the decision threshold must be
re-calibrated on production data before alerts mean anything.

## Resilience and rollback

| Failure | Response |
|---|---|
| Container crashes | Health check on `/health`; the platform restarts it. The API keeps no state |
| Bad release (code) | Images are tagged with the commit SHA: route traffic back to the previous revision |
| Bad model | Move the `production` alias back to the previous version in MLflow, re-export, redeploy |
| Model fails the gate in CD | Already handled: nothing is built or published |
| Region outage | Redeploy the same image in a second region; no data to migrate |

Releases would go out gradually: 10 % of traffic to the new revision, check latency and error
rate, then 100 %.

## Cost and risk reduction, next steps

1. Serve the model with ONNX Runtime: measured 2,700 times faster on the model and a 7 times
   smaller file. It also removes MLflow and scikit-learn from the image, so cold starts shrink.
2. Send the 62 features, or a compressed payload, instead of 278 KB of raw samples when the
   client can compute them.
3. Re-run the benchmark on the target cloud machine before fixing the instance count.
