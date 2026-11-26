"""Summarize untouched holdout batches and rank pending destructive checks."""

from collections import defaultdict

import numpy as np

from .pls import PLSPath
from .records import predictors, strengths
from .selection import calibration_interval, interval_half_width
from .settings import FEATURES, TARGET

COVERAGE_TOLERANCE = 1e-10


def _margin_value(half_width) -> float:
    """Accept a scalar or the small result objects used by selection helpers."""
    if half_width is None:
        return 0.0
    if isinstance(half_width, (int, float)):
        return float(half_width)
    if isinstance(half_width, np.ndarray):
        values = np.abs(np.asarray(half_width, dtype=float)).ravel()
        return float(values.max()) if values.size else 0.0
    if isinstance(half_width, (list, tuple)):
        if not half_width:
            return 0.0
        return max(abs(float(value)) for value in half_width)
    for attribute in ("half_width", "margin_mpa", "half_width_mpa", "value"):
        if hasattr(half_width, attribute):
            return float(getattr(half_width, attribute))
    if isinstance(half_width, dict):
        for key in ("half_width", "half_width_mpa", "margin_mpa", "interval_half_width_mpa"):
            if key in half_width:
                return float(half_width[key])
    raise TypeError("区间半宽必须是数值或包含 half_width 的结果对象")


def trace_prediction(records: list[dict], path: PLSPath, count: int,
                     half_width: float | None = None) -> list[dict]:
    x = predictors(records)
    predictions = path.predict(x, count)
    scores = path.scores(x, count)
    outside = (x < path.low) | (x > path.high)
    margin = None if half_width is None else _margin_value(half_width)
    result = []
    for record, estimate, latent, flags in zip(records, predictions, scores, outside):
        estimate = float(estimate)
        row = {
            "id": record["id"], "batch_id": record["batch_id"],
            "predicted_mpa": estimate,
            "outside_features": ",".join(field for field, flag in zip(FEATURES, flags) if flag),
            **{f"score_{number}": round(float(score), 8) for number, score in enumerate(latent, 1)},
        }
        if margin is not None:
            lower = estimate - margin
            upper = estimate + margin
            row.update({
                "interval_half_width_mpa": margin,
                "half_width_mpa": margin,
                "interval_lower_mpa": lower,
                "interval_upper_mpa": upper,
                # Common explicit aliases for handoff tables and callers.
                "lower_bound_mpa": lower,
                "upper_bound_mpa": upper,
                "lower_mpa": lower,
                "upper_mpa": upper,
            })
        result.append(row)
    return result


def holdout_diagnostics(records: list[dict], path: PLSPath, count: int,
                        half_width: float = 0.0) -> tuple[list[dict], list[dict]]:
    margin = _margin_value(half_width)
    trace = trace_prediction(records, path, count, margin)
    groups = defaultdict(list)
    for index, (row, record) in enumerate(zip(trace, records)):
        actual = record[TARGET]
        error = row["predicted_mpa"] - actual
        covered = abs(error) <= margin + COVERAGE_TOLERANCE
        row.update(
            actual_mpa=actual,
            error_mpa=error,
            covered=covered,
            coverage=covered,
            within_interval=covered,
            interval_covered=covered,
            is_covered=covered,
            is_within_interval=covered,
        )
        groups[row["batch_id"]].append(index)
    predictions = np.array([row["predicted_mpa"] for row in trace])
    actual = strengths(records)
    residual = predictions - actual
    baseline_residual = path.target_mean - actual
    results = []
    for scope, batch, index in [
        ("overall", "", np.arange(len(trace))),
        *(("batch", batch, selected) for batch, selected in sorted(groups.items())),
    ]:
        errors = residual[index]
        covered_strips = sum(bool(trace[i]["covered"]) for i in index)
        strips = len(index)
        coverage_rate = covered_strips / strips if strips else None
        results.append({
            "scope": scope, "batch_id": batch, "strips": strips,
            "rmse_mpa": round(float(np.sqrt(np.mean(errors ** 2))), 8),
            "mae_mpa": round(float(np.mean(np.abs(errors))), 8),
            "bias_mpa": round(float(np.mean(errors)), 8),
            "baseline_rmse_mpa": round(float(np.sqrt(np.mean(baseline_residual[index] ** 2))), 8),
            "outside_strips": sum(bool(trace[i]["outside_features"]) for i in index),
            "interval_half_width_mpa": round(margin, 8),
            "half_width_mpa": round(margin, 8),
            "covered_strips": covered_strips,
            "coverage_count": covered_strips,
            "coverage": round(coverage_rate, 8) if coverage_rate is not None else None,
            "coverage_rate": round(coverage_rate, 8) if coverage_rate is not None else None,
            "coverage_ratio": round(coverage_rate, 8) if coverage_rate is not None else None,
            "coverage_pct": round(100 * coverage_rate, 8) if coverage_rate is not None else None,
            "interval_coverage_pct": round(100 * coverage_rate, 8) if coverage_rate is not None else None,
        })
    return results, trace


def pending_queue(trace: list[dict], threshold: float, half_width: float | None = None) -> tuple[list[dict], int]:
    # Prefer a band already attached by trace_prediction; an explicit argument
    # supports direct calls with lightweight trace fixtures.
    if half_width is not None:
        margin = _margin_value(half_width)
    elif trace and ("interval_half_width_mpa" in trace[0] or "half_width_mpa" in trace[0]):
        margin = _margin_value(trace[0].get("interval_half_width_mpa", trace[0].get("half_width_mpa")))
    else:
        margin = 0.0

    for row in trace:
        estimate = float(row["predicted_mpa"])
        lower = estimate - margin
        row.update({
            "interval_half_width_mpa": margin,
            "half_width_mpa": margin,
            "interval_lower_mpa": lower,
            "interval_upper_mpa": estimate + margin,
            "lower_bound_mpa": lower,
            "upper_bound_mpa": estimate + margin,
            "lower_mpa": lower,
            "upper_mpa": estimate + margin,
        })

        if row["outside_features"]:
            row.update(action="核对读数并优先实测", priority=1)
        elif lower < threshold:
            if estimate < threshold:
                action = "低于筛查线，优先实测"
            else:
                action = "区间下限低于筛查线，优先实测"
            row.update(action=action, priority=2)
        else:
            row.update(action="按班次常规实测", priority=3)
    trace.sort(key=lambda row: (row["priority"], row["predicted_mpa"], row["id"]))
    for index, row in enumerate(trace, 1):
        row["position"] = index
    return trace, sum(row["priority"] < 3 for row in trace)


# Compatibility facades: interval construction lives with component selection,
# but tests and callers often discover it through the evaluation module.
prediction_interval = calibration_interval
build_interval = calibration_interval
cross_validated_interval = calibration_interval
out_of_fold_interval = calibration_interval
maximum_out_of_fold_absolute_error = interval_half_width
calibration_interval_width = interval_half_width
prediction_interval_width = interval_half_width
out_of_fold_half_width = interval_half_width
