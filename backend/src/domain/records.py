"""Validate the physical measurements and prevent leakage across batch cohorts."""

from collections import Counter
import re

import numpy as np

from .settings import COLUMNS, FEATURES, FIELDS, TARGET

CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SPLITS = {"train", "holdout", "pending"}
LIMITS = {
    **{field: (-0.2, 5.0) for field in FEATURES if field.startswith("nir_")},
    "density_kg_m3": (200.0, 2500.0),
    "thickness_mm": (1.0, 60.0),
    TARGET: (0.0, 30.0),
}


def parse_code(value, field: str, line: int) -> str:
    if isinstance(value, str) and CODE.fullmatch(value.strip()):
        return value.strip()
    raise ValueError(f"第 {line} 条{FIELDS[field]}须为 1 至 64 位编号，以英文字母或数字开头。")


def parse_measurement(value, field: str, line: int) -> float:
    if isinstance(value, str) or type(value) in (int, float):
        try:
            measured = float(value)
        except (ValueError, OverflowError):
            measured = float("nan")
        low, high = LIMITS[field]
        if low <= measured <= high:
            return measured
    low, high = LIMITS[field]
    raise ValueError(f"第 {line} 条{FIELDS[field]}须为 {low:g} 至 {high:g} 的有限数值。")


def normalize_rows(rows: list[dict]) -> list[dict]:
    ids = set()
    batch_splits = {}
    counts = Counter()
    records = []
    for line, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict) or raw.keys() != set(COLUMNS):
            raise ValueError(f"第 {line} 条的列须与模板一致：{', '.join(COLUMNS)}。")
        row_id = parse_code(raw["id"], "id", line)
        batch_id = parse_code(raw["batch_id"], "batch_id", line)
        if row_id in ids:
            raise ValueError(f"第 {line} 条的试条编号 {row_id} 重复。")
        ids.add(row_id)
        split = raw["split"].strip() if isinstance(raw["split"], str) else ""
        if split not in SPLITS:
            raise ValueError(f"第 {line} 条的样本用途须为 train、holdout 或 pending。")
        if batch_splits.setdefault(batch_id, split) != split:
            raise ValueError(f"制板批 {batch_id} 跨样本用途，请将同批试条整批划分。")
        measurements = {field: parse_measurement(raw[field], field, line) for field in FEATURES}
        target = raw[TARGET]
        if split == "pending":
            if target is not None and not (isinstance(target, str) and not target.strip()):
                raise ValueError(f"第 {line} 条待测试条的实测抗折强度须留空。")
            target = None
        else:
            target = parse_measurement(target, TARGET, line)
        records.append({"id": row_id, "batch_id": batch_id, "split": split, **measurements, TARGET: target})
        counts[split] += 1
    if counts["train"] < 6 or sum(split == "train" for split in batch_splits.values()) < 3:
        raise ValueError("校准至少需要 6 条有抗折实测的试条，覆盖至少 3 个制板批。")
    if counts["holdout"] < 2:
        raise ValueError("独立留出至少需要 2 条有抗折实测的试条。")
    return records


def predictors(rows: list[dict]) -> np.ndarray:
    return np.array([[row[field] for field in FEATURES] for row in rows], dtype=float).reshape(-1, len(FEATURES))


def strengths(rows: list[dict]) -> np.ndarray:
    return np.array([row[TARGET] for row in rows], dtype=float)
