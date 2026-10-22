# General Forecast

General Forecast is a easy-to-use unified inference
and deployment engine for structured data foundation models (SDFMs).
It provides low-latency, high-throughput in-context
learning predictions for live tabular and time-series data
through high-level APIs and a Python SDK.

- Website: https://generalforecast.com
- Docs: https://docs.generalforecast.com/

## Quickstart

```python
uv venv --python 3.12
source .venv/bin/activate

uv pip install "generalforecast[sklearn]"
```

Predict locally in Python with the `scikit-learn` API:
```python
import generalforecast as gf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = gf.Classifier(
    "general-forecast/tabicl-v2-classifier",
    allow_downloads=True,
).fit(X_train, y_train)

print("accuracy:", classifier.score(X_test, y_test))
print("probabilities:\n", classifier.predict_proba(X_test[:3]).round(3))
```

Serve models locally with a prediction package:
```python
# Create a classifier as shown above

# Creates a declarative package directory with gf.yaml, context.arrow, gf.lock
package = classifier.to_package("./breast-cancer-pkg", name="breast-cancer")
print("locked digest:", package.lock.package_digest)
```
```bash
gf serve ./breast-cancer-pkg --port 8000
```

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
