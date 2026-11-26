"""Select component counts and build the batch-cross-validated error band."""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .pls import fit_path
from .records import predictors, strengths


@dataclass(frozen=True)
class FoldError:
    batch: str
    count: int
    samples: int
    sse: float | None


@dataclass(frozen=True)
class Candidate:
    count: int
    folds: int
    samples: int
    rmse: float | None


@dataclass(frozen=True)
class IntervalFold:
    """The component count reselected inside one outer calibration fold."""

    batch: str
    selected_components: int
    samples: int
    max_absolute_error: float


@dataclass(frozen=True)
class IntervalBand:
    half_width: float
    folds: list[IntervalFold]
    absolute_errors: list[float]

    @property
    def margin_mpa(self) -> float:
        return self.half_width


def select_components(train: list[dict], maximum: int) -> tuple[Candidate, list[Candidate], list[FoldError]]:
    x, y = predictors(train), strengths(train)
    groups = np.array([row["batch_id"] for row in train])
    batches = np.unique(groups)
    fold_errors = {count: [] for count in range(1, maximum + 1)}
    for batch in batches:
        validation = groups == batch
        path = fit_path(x[~validation], y[~validation], maximum)
        samples = int(validation.sum())
        for count, errors in fold_errors.items():
            sse = None
            if count <= path.components:
                residual = path.predict(x[validation], count) - y[validation]
                sse = float(residual @ residual)
            errors.append(FoldError(str(batch), count, samples, sse))

    candidates = []
    for count, errors in fold_errors.items():
        usable = [error for error in errors if error.sse is not None]
        samples = sum(error.samples for error in usable)
        rmse = sqrt(sum(error.sse for error in usable) / samples) if len(usable) == len(batches) else None
        candidates.append(Candidate(count, len(usable), samples, rmse))
    eligible = [candidate for candidate in candidates if candidate.rmse is not None]
    if not eligible:
        raise ValueError("至少一个训练折缺少可提取的 PLS 分量，请核对恒定读数、抗折标注及批次分布。")
    best_rmse = min(candidate.rmse for candidate in eligible)
    selected = next(candidate for candidate in eligible if candidate.rmse - best_rmse <= 1e-10)
    return selected, candidates, [error for errors in fold_errors.values() for error in errors]


def calibration_interval(train: list[dict], maximum: int) -> IntervalBand:
    """Build a max-residual band using nested batch-wise component selection.

    Each calibration batch is the outer validation batch.  On the remaining
    batches, component count is selected again with leave-one-batch-out CV,
    then that inner model is refitted and used to predict the outer batch.
    The band half-width is the largest absolute outer-fold strip error.
    """
    x, y = predictors(train), strengths(train)
    groups = np.array([row["batch_id"] for row in train])
    batches = np.unique(groups)
    absolute_errors: list[float] = []
    fold_results: list[IntervalFold] = []

    for batch in batches:
        validation = groups == batch
        rest = [row for row, is_validation in zip(train, validation) if not is_validation]
        selected, _, _ = select_components(rest, maximum)
        path = fit_path(x[~validation], y[~validation], selected.count)
        if path.components != selected.count:
            raise ValueError("区间折内数据无法重拟合所选分量，请核对各批次的测量与实测分布。")
        residual = path.predict(x[validation], selected.count) - y[validation]
        errors = np.abs(np.asarray(residual, dtype=float))
        batch_errors = [float(value) for value in errors]
        absolute_errors.extend(batch_errors)
        fold_results.append(IntervalFold(
            batch=str(batch), selected_components=selected.count,
            samples=int(validation.sum()), max_absolute_error=max(batch_errors),
        ))

    if not absolute_errors:
        raise ValueError("至少一个训练折缺少可用于确定区间半宽的折外误差。")
    return IntervalBand(max(absolute_errors), fold_results, absolute_errors)


def interval_half_width(train: list[dict], maximum: int) -> float:
    """Convenience scalar facade for callers that only need the half-width."""
    return calibration_interval(train, maximum).half_width


# Descriptive aliases used by nearby domain/service callers and tests.
calibration_margin = calibration_interval
cross_validation_interval = calibration_interval
maximum_out_of_fold_error = interval_half_width
calibration_interval_width = interval_half_width
prediction_interval_width = interval_half_width
out_of_fold_half_width = interval_half_width
