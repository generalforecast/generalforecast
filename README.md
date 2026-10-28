# General Forecast

Predictors on tabular and time series foundation models. One manifest, `gf.yaml`, runs the same predictor on your machine and hosted.

- Website: https://generalforecast.com
- Docs: https://docs.generalforecast.com/

```sh
uv pip install generalforecast
```

Predict or forecast with an `sklearn`-compatible interface.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import generalforecast as gf

X, y = load_breast_cancer(return_X_y=True)
X_context, X_queries, y_context, y_queries = train_test_split(X, y, random_state=0)

clf = gf.Classifier(model="general-forecast/tabicl-v2-classifier")
clf.fit(X_context, y_context)
probabilities = clf.predict_proba(X_queries)

days = pd.date_range("2025-01-01", periods=60, freq="D")
history = pd.DataFrame(
    {
        "date": np.tile(days, 2),
        "sku": np.repeat(["a", "b"], 60),
        "units": np.r_[np.arange(60.0) % 7, 10 + np.arange(60.0) % 5],
    }
)
fc = gf.Forecaster(target="units", time="date", series="sku", horizon=7)
fc.fit(history)
quantiles = fc.forecast()  # 0.1, 0.5 and 0.9 for each sku and step
```

For deployments on large, live datasets, please take a look at the [docs](https://docs.generalforecast.com/).

## Contributing

We welcome and value any contributions and collaborations; see
[CONTRIBUTING.md](CONTRIBUTING.md). For more specific questions, please
contact hello@generalforecast.com.

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
