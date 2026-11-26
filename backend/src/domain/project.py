"""Domain entry points used by the catalog, seed script and background worker."""

from .evaluation import holdout_diagnostics, pending_queue, trace_prediction
from .fixtures import demo_rows
from .pls import fit_path
from .records import normalize_rows, predictors, strengths
from .reports import calibration_tables, handoff_tables, selection_tables
from .selection import calibration_interval, interval_half_width, select_components
from .settings import SPEC, parse_settings

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze", "interval_half_width"]


def analyze(rows: list[dict], settings: dict) -> dict:
    maximum, threshold = parse_settings(settings)
    records = sorted(normalize_rows(rows), key=lambda row: (row["batch_id"], row["id"]))
    train = [row for row in records if row["split"] == "train"]
    holdout = [row for row in records if row["split"] == "holdout"]
    pending = [row for row in records if row["split"] == "pending"]
    selected, candidates, folds = select_components(train, maximum)
    interval = calibration_interval(train, maximum)
    model = fit_path(predictors(train), strengths(train), selected.count)
    if model.components != selected.count:
        raise ValueError("全量训练数据无法重拟合已选分量，请核对各批次的测量与实测分布。")
    summary, holdout_trace = holdout_diagnostics(
        holdout, model, selected.count, interval.half_width,
    )
    pending_trace = trace_prediction(pending, model, selected.count, interval.half_width)
    queue, priority = pending_queue(pending_trace, threshold, interval.half_width)
    overall = summary[0]
    warnings = []
    if any(candidate.rmse is None for candidate in candidates):
        warnings.append("部分候选分量在至少一个训练折中不可提取，请查看逐折误差。")
    if overall["outside_strips"]:
        warnings.append(f'独立留出有 {overall["outside_strips"]} 条试条的读数超出训练范围，已保留在总体误差中。')
    if overall["rmse_mpa"] >= overall["baseline_rmse_mpa"]:
        warnings.append("独立留出 RMSE 未优于训练均值基线，请复核校准批覆盖范围。")
    if overall["coverage_pct"] < 100:
        warnings.append(
            f'独立留出区间覆盖 {overall["covered_strips"]}/{overall["strips"]}，'
            "存在落在折外最大误差区间外的试条。"
        )
    if not pending:
        warnings.append("本次台账无待测试条。")
    coverage_text = f'{overall["covered_strips"]}/{overall["strips"]}'
    return {
        "summary": (f"用 {selected.folds} 个校准批的折外误差选出 {selected.count} 个潜变量，"
                    f"以 {interval.half_width:.4f} MPa 为区间半宽。"
                    f"冻结模型后，在 {len(summary) - 1} 个独立批的 {len(holdout)} 条试条上核验区间覆盖 {coverage_text}。"
                    f"待测 {len(pending)} 条，其中 {priority} 条优先实测。"),
        "interval_half_width_mpa": interval.half_width,
        "interval": {
            "method": "nested_batch_out_max_absolute_error",
            "half_width_mpa": interval.half_width,
            "outer_folds": [
                {
                    "batch_id": fold.batch,
                    "selected_components": fold.selected_components,
                    "strips": fold.samples,
                    "max_absolute_error_mpa": fold.max_absolute_error,
                }
                for fold in interval.folds
            ],
        },
        "metrics": {
            "选定潜变量数": selected.count, "训练批折外 RMSE（MPa）": round(selected.rmse, 8),
            "区间半宽（MPa）": round(interval.half_width, 8),
            "预测区间半宽（MPa）": round(interval.half_width, 8),
            "预测区间半径（MPa）": round(interval.half_width, 8),
            "折外最大绝对误差（MPa）": round(interval.half_width, 8),
            "独立留出区间覆盖率": overall["coverage_rate"],
            "独立留出区间覆盖率（%）": overall["coverage_pct"],
            "独立留出区间覆盖比例": overall["coverage_rate"],
            "独立留出区间覆盖条数": overall["covered_strips"],
            "独立留出 RMSE（MPa）": overall["rmse_mpa"],
            "独立留出均值基线 RMSE（MPa）": overall["baseline_rmse_mpa"],
            "训练抗折均值（MPa）": model.target_mean, "优先实测试条数": priority,
        },
        "tables": [*handoff_tables(summary, holdout_trace, queue, selected.count),
                   *selection_tables(selected, candidates, folds), *calibration_tables(model, selected.count)],
        "warnings": warnings,
    }
