"""Select the component count by leave-one-production-batch-out validation."""

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
