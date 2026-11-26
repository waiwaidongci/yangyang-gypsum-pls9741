"""Columns, measurement units and run options shared by the application and domain."""

FEATURES = (
    "nir_1200", "nir_1450", "nir_1550", "nir_1720", "nir_1940", "nir_2100",
    "density_kg_m3", "thickness_mm",
)
TARGET = "flexural_mpa"
COLUMNS = ("id", "batch_id", "split", *FEATURES, TARGET)
FIELDS = {
    "id": "试条编号",
    "batch_id": "制板批号",
    "split": "样本用途",
    "nir_1200": "1200 nm 吸光度",
    "nir_1450": "1450 nm 吸光度",
    "nir_1550": "1550 nm 吸光度",
    "nir_1720": "1720 nm 吸光度",
    "nir_1940": "1940 nm 吸光度",
    "nir_2100": "2100 nm 吸光度",
    "density_kg_m3": "体积密度（kg/m³）",
    "thickness_mm": "实测试条厚度（mm）",
    TARGET: "实测抗折强度（MPa）",
}
SETTINGS = [
    {"key": "max_components", "label": "候选潜变量上限", "default": 6,
     "min": 1, "max": len(FEATURES), "step": 1},
    {"key": "screening_mpa", "label": "优先实测筛查线（MPa）", "default": 4.2,
     "min": 0.1, "max": 30, "step": 0.1},
]
SPEC = {
    "columns": list(COLUMNS),
    "fields": FIELDS,
    "settings": SETTINGS,
    "modules": {
        "datasets": "试条测量台账",
        "runs": "潜变量校准",
        "results": "批次留出结果",
        "reviews": "工坊实测交接",
    },
    "limitation": "抗折估计用于安排破坏性实测。新批超出校准范围时优先实测，放行仍依据实测。",
    "dataset_name": "20261125-再生石膏板试条.csv",
    "recommendation": "先复核独立批区间覆盖与误差，再领取待测试条清单，优先实测越界或区间下限低于筛查线的试条。",
}


def parse_settings(settings: dict) -> tuple[int, float]:
    if settings.keys() - {option["key"] for option in SETTINGS}:
        raise ValueError("运行参数仅接受 max_components 和 screening_mpa。")
    parsed = []
    for option in SETTINGS:
        value = settings.get(option["key"], option["default"])
        low, high = option["min"], option["max"]
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f'{option["label"]}须为 {low} 至 {high} 的数值。')
        parsed.append(value)
    maximum, threshold = parsed
    if maximum != int(maximum):
        raise ValueError("候选潜变量上限须为整数。")
    return int(maximum), float(threshold)
