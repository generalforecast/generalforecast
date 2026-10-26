# General Forecast

Predictors on tabular and time series foundation models. One manifest, `gf.yaml`, runs the same predictor on your machine and hosted.

- Website: https://generalforecast.com
- Docs: https://docs.generalforecast.com/

```sh
uv pip install generalforecast
```

Predict or forecast with an `sklearn`-compatible interface.

```python
import generalforecast as gf

clf = gf.Classifier(model="general-forecast/tabicl-v2-classifier")
clf.fit(X_context, y_context)
clf.predict_proba(X_queries)

fc = gf.Forecaster(model="general-forecast/chronos-2", horizon=7)
fc.fit(history)  # columns: date, sku, units
fc.forecast()  # quantiles 0.1, 0.5, 0.9 per sku per step
```

For deployments on large, live datasets, please take a look at the [docs](https://docs.generalforecast.com/).

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
