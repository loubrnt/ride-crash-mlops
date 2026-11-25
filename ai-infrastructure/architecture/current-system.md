# Current system (what exists and runs today)

```mermaid
flowchart LR
    subgraph Laptop or CI runner
        raw[(SRV3490.parquet<br/>268 MB)] --> feat[features<br/>20 s]
        feat --> split[split]
        split --> train[train<br/>0.38 s, CPU]
        train --> mlflow[(MLflow<br/>SQLite)]
        mlflow --> gate{gate<br/>AUC >= 0.70}
        gate --> export[export ./model<br/>9.8 MB]
    end
    export --> img[API image<br/>Docker Hub]
    img --> api[1 container<br/>FastAPI, 1 worker<br/>~30 req/s]
    api --> app[Streamlit web app]
    gha[GitHub Actions<br/>CI + CD] -. builds and tests .-> img
```

| Component | Status |
|---|---|
| Preprocessing, training, evaluation gate | Implemented, run by DVC and by the CD workflow |
| Model versioning | MLflow registry on SQLite, `production` alias |
| Prediction API, web app | Implemented, two public Docker images |
| CI/CD | GitHub Actions: lint, tests, train, gate, build, end-to-end test |
| Where it runs | Local machine and GitHub runners. **No cloud deployment.** |
| Monitoring, autoscaling | **Not implemented** |
