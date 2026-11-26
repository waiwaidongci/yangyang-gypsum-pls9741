"""Fit a centered, scaled, single-response partial least squares regression path."""

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class PLSPath:
    mean: np.ndarray
    scale: np.ndarray
    low: np.ndarray
    high: np.ndarray
    target_mean: float
    target_scale: float
    weights: np.ndarray
    loadings: np.ndarray
    target_loadings: np.ndarray
    score_sd: np.ndarray

    @property
    def components(self) -> int:
        return len(self.target_loadings)

    def rotation(self, count: int) -> np.ndarray:
        weights = self.weights[:, :count]
        gram = self.loadings[:, :count].T @ weights
        return np.linalg.solve(gram.T, weights.T).T

    def scores(self, x: np.ndarray, count: int) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.rotation(count)

    def coefficients(self, count: int) -> np.ndarray:
        return self.rotation(count) @ self.target_loadings[:count] * self.target_scale / self.scale

    def predict(self, x: np.ndarray, count: int) -> np.ndarray:
        return self.target_mean + (x - self.mean) @ self.coefficients(count)


def fit_path(x: np.ndarray, y: np.ndarray, maximum: int) -> PLSPath:
    mean = x.mean(axis=0)
    deviations = x.std(axis=0, ddof=1)
    scale = np.where(deviations > EPS, deviations, 1.0)
    target_mean = float(y.mean())
    target_sd = float(y.std(ddof=1))
    target_scale = target_sd if target_sd > EPS else 1.0
    residual_x = (x - mean) / scale
    residual_y = (y - target_mean) / target_scale
    residual_x[:, deviations <= EPS] = 0
    if target_sd <= EPS:
        residual_y.fill(0)
    weights, loadings, target_loadings, score_sd = [], [], [], []
    limit = min(maximum, len(x) - 1, int(np.linalg.matrix_rank(residual_x, tol=EPS)))
    for _ in range(limit):
        direction = residual_x.T @ residual_y
        magnitude = np.linalg.norm(direction)
        if magnitude <= EPS:
            break
        weight = direction / magnitude
        if weight[np.argmax(np.abs(weight))] < 0:
            weight = -weight
        score = residual_x @ weight
        energy = float(score @ score)
        if energy <= EPS ** 2:
            break
        loading = residual_x.T @ score / energy
        target_loading = float(residual_y @ score / energy)
        residual_x -= np.outer(score, loading)
        residual_y -= score * target_loading
        weights.append(weight)
        loadings.append(loading)
        target_loadings.append(target_loading)
        score_sd.append((energy / (len(x) - 1)) ** 0.5)
    return PLSPath(
        mean, scale, x.min(axis=0), x.max(axis=0), target_mean, target_scale,
        np.array(weights).reshape(-1, x.shape[1]).T,
        np.array(loadings).reshape(-1, x.shape[1]).T,
        np.array(target_loadings), np.array(score_sd),
    )
