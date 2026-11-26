"""Build scalar, typed CSV/JSON tables for the calibration handoff."""

from math import sqrt

from .pls import PLSPath
from .selection import Candidate, FoldError
from .settings import FEATURES


def table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def selection_tables(selected: Candidate, candidates: list[Candidate], folds: list[FoldError]) -> list[dict]:
    candidate_rows = [{
        "components": candidate.count, "usable_folds": candidate.folds, "validation_strips": candidate.samples,
        "cv_rmse_mpa": round(candidate.rmse, 8) if candidate.rmse is not None else None,
        "selected": candidate == selected,
    } for candidate in candidates]
    fold_rows = [{
        "components": fold.count, "batch_id": fold.batch, "validation_strips": fold.samples,
        "rmse_mpa": round(sqrt(fold.sse / fold.samples), 8) if fold.sse is not None else None,
    } for fold in folds]
    return [
        table("component_selection", "训练批折外分量选型", {
            "components": "潜变量数", "usable_folds": "可拟合折数", "validation_strips": "有效留批试条数",
            "cv_rmse_mpa": "折外汇总 RMSE（MPa）", "selected": "选中",
        }, candidate_rows),
        table("selection_folds", "训练批逐折误差", {
            "components": "潜变量数", "batch_id": "折内留出的制板批", "validation_strips": "折内留出试条数",
            "rmse_mpa": "该折 RMSE（MPa）",
        }, fold_rows),
    ]


def calibration_tables(path: PLSPath, count: int) -> list[dict]:
    rotations = path.rotation(count)
    coefficients = path.coefficients(count)
    strengths = []
    loadings = []
    covariance_total = 0.0
    for index in range(count):
        # Covariance of the unit-variance score with the original standardized target.
        covariance = float(path.score_sd[index] * path.target_loadings[index])
        covariance_total += covariance ** 2
        strengths.append({
            "component": index + 1, "score_sd": round(float(path.score_sd[index]), 8),
            "covariance": round(covariance, 8), "explained_covariance_pct": round(100 * covariance ** 2, 8),
            "cumulative_covariance_pct": round(100 * covariance_total, 8),
        })
        for feature_index, feature in enumerate(FEATURES):
            loadings.append({
                "component": index + 1, "feature": feature,
                "weight": round(float(path.weights[feature_index, index]), 8),
                "loading": round(float(path.loadings[feature_index, index]), 8),
                "rotation": round(float(rotations[feature_index, index]), 8),
            })
    calibration = [{
        "feature": feature, "train_mean": float(path.mean[index]), "train_scale": float(path.scale[index]),
        "train_min": float(path.low[index]), "train_max": float(path.high[index]),
        "coefficient_mpa": float(coefficients[index]),
    } for index, feature in enumerate(FEATURES)]
    return [
        table("component_covariance", "潜变量与抗折的解释协方差", {
            "component": "潜变量序号", "score_sd": "训练得分样本标准差",
            "covariance": "单位方差得分与标准化抗折的协方差",
            "explained_covariance_pct": "平方协方差贡献（%）", "cumulative_covariance_pct": "累计贡献（%）",
        }, strengths),
        table("component_loadings", "各测量通道权重及载荷", {
            "component": "潜变量序号", "feature": "测量字段", "weight": "权重", "loading": "读数载荷",
            "rotation": "标准化原始读数到得分的旋转权重",
        }, loadings),
        table("frozen_calibration", "冻结训练尺度及原始量纲系数", {
            "feature": "测量字段", "train_mean": "训练均值", "train_scale": "训练尺度",
            "train_min": "训练最小值", "train_max": "训练最大值", "coefficient_mpa": "原始量纲回归系数",
        }, calibration),
    ]


def handoff_tables(summary: list[dict], holdout: list[dict], pending: list[dict], count: int) -> list[dict]:
    trace_labels = {"id": "试条编号", "batch_id": "制板批号", "predicted_mpa": "估计抗折（MPa）",
                    "outside_features": "超出训练范围的字段"}
    scores = {f"score_{index}": f"潜变量 {index} 得分" for index in range(1, count + 1)}
    return [
        table("pending_estimates", "待测试条及实测优先顺序", {
            "position": "交接顺序", **trace_labels, "priority": "实测优先级（1最高）", "action": "实测安排", **scores,
        }, pending),
        table("holdout_errors", "独立批抗折误差", {
            "scope": "汇总层级", "batch_id": "留出制板批", "strips": "试条数", "rmse_mpa": "RMSE（MPa）",
            "mae_mpa": "MAE（MPa）", "bias_mpa": "平均估计偏差（MPa）",
            "baseline_rmse_mpa": "训练均值基线 RMSE（MPa）", "outside_strips": "读数越界条数",
        }, summary),
        table("holdout_predictions", "独立批逐条抗折对照", {
            **trace_labels, "actual_mpa": "实测抗折（MPa）", "error_mpa": "估计减实测（MPa）", **scores,
        }, holdout),
    ]
