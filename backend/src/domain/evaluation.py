"""Summarize untouched holdout batches and rank pending destructive checks."""

from collections import defaultdict

import numpy as np

from .pls import PLSPath
from .records import predictors, strengths
from .settings import FEATURES, TARGET


def trace_prediction(records: list[dict], path: PLSPath, count: int) -> list[dict]:
    x = predictors(records)
    predictions = path.predict(x, count)
    scores = path.scores(x, count)
    outside = (x < path.low) | (x > path.high)
    result = []
    for record, estimate, latent, flags in zip(records, predictions, scores, outside):
        result.append({
            "id": record["id"], "batch_id": record["batch_id"],
            "predicted_mpa": float(estimate),
            "outside_features": ",".join(field for field, flag in zip(FEATURES, flags) if flag),
            **{f"score_{number}": round(float(score), 8) for number, score in enumerate(latent, 1)},
        })
    return result


def holdout_diagnostics(records: list[dict], path: PLSPath, count: int) -> tuple[list[dict], list[dict]]:
    trace = trace_prediction(records, path, count)
    groups = defaultdict(list)
    for index, (row, record) in enumerate(zip(trace, records)):
        actual = record[TARGET]
        row.update(actual_mpa=actual, error_mpa=row["predicted_mpa"] - actual)
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
        results.append({
            "scope": scope, "batch_id": batch, "strips": len(index),
            "rmse_mpa": round(float(np.sqrt(np.mean(errors ** 2))), 8),
            "mae_mpa": round(float(np.mean(np.abs(errors))), 8),
            "bias_mpa": round(float(np.mean(errors)), 8),
            "baseline_rmse_mpa": round(float(np.sqrt(np.mean(baseline_residual[index] ** 2))), 8),
            "outside_strips": sum(bool(trace[i]["outside_features"]) for i in index),
        })
    return results, trace


def pending_queue(trace: list[dict], threshold: float) -> tuple[list[dict], int]:
    for row in trace:
        if row["outside_features"]:
            row.update(action="核对读数并优先实测", priority=1)
        elif row["predicted_mpa"] < threshold:
            row.update(action="低于筛查线，优先实测", priority=2)
        else:
            row.update(action="按班次常规实测", priority=3)
    trace.sort(key=lambda row: (row["priority"], row["predicted_mpa"], row["id"]))
    for index, row in enumerate(trace, 1):
        row["position"] = index
    return trace, sum(row["priority"] < 3 for row in trace)
