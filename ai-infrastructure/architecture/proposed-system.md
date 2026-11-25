# Proposed system (design, not deployed)

```mermaid
flowchart LR
    client[Ride ends:<br/>fleet backend] --> lb[HTTPS endpoint]
    lb --> run[Cloud Run service<br/>API image, 1 vCPU, 512 MiB<br/>min 1, max 5 instances]
    run --> logs[(Prediction log)]
    run --> mon[Monitoring<br/>latency p95, error rate,<br/>share of crashes predicted]
    mon --> alert[Alerts]
    logs --> drift[Drift check<br/>per city]
    drift -. triggers .-> cd[CD workflow:<br/>train, gate, build, e2e]
    cd --> reg[(Image registry<br/>tag = commit SHA)]
    reg -. gradual rollout 10 % then 100 % .-> run
    mlflow[(MLflow server<br/>shared)] --- cd
```

What changes compared to today:

1. The API runs on a managed, autoscaled platform instead of a laptop.
2. Predictions are logged, which makes monitoring and drift detection possible.
3. MLflow moves from a local SQLite file to a shared server.
4. Releases go through a gradual rollout with an instant rollback path.
