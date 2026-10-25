# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One capability admission for compiled, registered and executed predictors.

`compile_plan`, revision registration and the predictor runtime all call
`admit`, so a lock forged to be self-consistent still cannot carry a quantile,
output, horizon or bound its pinned model and the serving policy do not allow.
"""

from typing import Any

from generalforecast.predictors.validation import ensure


def _on_grid(level: float, step: float) -> bool:
    return abs(level / step - round(level / step)) < 1e-9


def admit(spec: Any, pin: Any, task: str, outputs: tuple[str, ...]) -> None:
    """Refuses intent the pinned model or applied serving policy cannot honor.

    Args:
        spec: The authored execution plan.
        pin: The model pin it compiled against.
        task: The compiled task.
        outputs: The compiled outputs.

    Raises:
        PredictorError: `UnsupportedTask`, `UnsupportedCapability` or
            `CapacityExceeded`, naming the first refused intent.
    """
    from generalforecast.models.catalog import V1_POLICY, ModelRef

    ensure(task in pin.tasks, "model does not support declared task", "UnsupportedTask")
    ensure(
        set(outputs) <= set(pin.outputs),
        "model does not support requested outputs",
        "UnsupportedCapability",
    )
    levels = spec.quantiles
    ensure(
        not levels or "quantiles" in outputs, "quantiles require the quantiles output"
    )
    ensure(
        task != "classification" or not levels,
        "classification does not support quantiles",
    )
    ensure(
        not levels or not pin.quantiles or set(levels) <= set(pin.quantiles),
        "model does not provide requested quantile levels",
        "UnsupportedCapability",
    )
    ensure(
        pin.quantile_step is None
        or all(_on_grid(q, pin.quantile_step) for q in levels),
        "model does not provide requested quantile precision",
        "UnsupportedCapability",
    )
    ensure(
        not levels or pin.quantiles or pin.quantile_step is not None,
        "model does not provide requested quantile levels",
        "UnsupportedCapability",
    )
    if spec.context.max_rows is not None:
        ensure(
            spec.context.max_rows <= pin.max_context_rows,
            "context selection exceeds model capacity",
            "CapacityExceeded",
        )
    if task == "forecast":
        ensure(
            spec.horizon is None
            or (pin.max_horizon is not None and spec.horizon <= pin.max_horizon),
            "forecast horizon exceeds model capability",
            "CapacityExceeded",
        )
    else:
        ensure(
            not any(
                (
                    spec.time,
                    spec.series,
                    spec.horizon,
                    spec.frequency,
                    spec.past_inputs,
                    spec.future_inputs,
                )
            ),
            "time and series roles require forecast task",
        )
    policy = V1_POLICY
    refused = policy.admits(ModelRef(pin.repository, pin.codes, pin.estimators))
    ensure(refused is None, f"serving policy allows {refused}", "CapacityExceeded")
    ensure(
        pin.max_context_rows <= policy.context_rows
        and pin.max_features <= max(policy.features, policy.covariates)
        and (pin.max_horizon is None or pin.max_horizon <= policy.horizon),
        "locked bounds exceed the serving policy",
        "CapacityExceeded",
    )
