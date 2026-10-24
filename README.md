# General Forecast

Predictors on tabular and time series foundation models. One manifest, `gf.yaml`, runs the same predictor on your machine and hosted.

- Website: https://generalforecast.com
- Docs: https://docs.generalforecast.com/

```sh
pip install generalforecast
```

## A predictor from a CSV

In a directory with a CSV, `gf init` asks for the target, the model, the id column, the inputs and which rows are queries, then writes `gf.yaml`.

```
$ cd churn && gf init
Target       churned
Model        general-forecast/tabicl-v2-classifier
Id           customer_id
Inputs       spend, plan, signups
Queries      212 rows with no target

 Initialized predictor churn in ./gf.yaml
```

`gf check` resolves the schema against the context and pins the model in `gf.lock`.

```
$ gf check
    Resolved churn  classification, 3 inputs, 1,840 context rows, 212 query rows
      Locked gf.lock in 0.21s
```

`gf eval` scores the held-out rows.

```
$ gf eval
    Resolved churn  classification, 1,840 context rows, 212 query rows
   Predicted 212 rows in 0.19s

             accuracy   0.934
             log loss   0.187
             auroc      0.961
```

`gf serve` runs the predictor on this machine. The first run pulls the model's weights into `~/.cache/huggingface`.

```
$ gf serve
     Pulling general-forecast/tabicl-v2-classifier  (1.2 GB) to ~/.cache/huggingface
     Loading general-forecast/tabicl-v2-classifier (cached)
    Compiled 12 programs in 6.8s
     Serving churn at http://127.0.0.1:8000
```

```sh
curl -s http://127.0.0.1:8000/v1/projects/local/predictors/churn/predict \
  -H 'content-type: application/json' \
  -d '{
    "api_version": "gf/v1",
    "project": "local",
    "request_id": "r-1",
    "resource": {"kind": "predictor", "name": "churn"},
    "payload": {"queries": [{"customer_id": 9001, "spend": 42.0, "plan": "pro", "signups": 3}]}
  }'
```

The same predictor from Python:

```python
import generalforecast as gf
import pandas as pd

df = pd.read_csv("new_customers.csv")
scored = gf.Predictor("./churn").predict(df)  # columns: customer_id, label, p_0, p_1
```

`gf push` deploys it. `gf predict` and `gf.Predictor("churn")` then score rows against the hosted revision.

```
$ gf push
   Uploading churn revision 7  snapshot, 1,840 rows, 3.1 kB
    Deployed churn revision 7 to production in 41s
```

## Estimators

`Classifier`, `Regressor` and `Forecaster` have the sklearn shape and take a catalog model. Weights pull on first use; `download=False` or `GF_OFFLINE=1` refuses the download and names the `gf model pull` command instead.

```python
import generalforecast as gf

clf = gf.Classifier(model="general-forecast/tabicl-v2-classifier")
clf.fit(X_context, y_context)
clf.predict_proba(X_queries)

fc = gf.Forecaster(model="general-forecast/chronos-2", horizon=7)
fc.fit(history)  # columns: date, sku, units
fc.forecast()  # quantiles 0.1, 0.5, 0.9 per sku per step
```

`gf model list` prints the catalog with each model's limits on context rows, queries and inputs.

## Contributing

We welcome and value any contributions and collaborations. For more
specific questions, please contact hello@generalforecast.com.

## License

Licensed under the [Apache License 2.0](LICENSE).

If you use General Forecast, please cite our work:
```bibtex
@misc{generalforecast2026,
    title={General Forecast: A unified high-throughput inference engine for structured data foundation models},
    author={Tejas Prabhune and Ayushi Batwara},
    year={2026},
    publisher={GitHub},
    howpublished={\url{https://github.com/generalforecast/gf}}
}
```
