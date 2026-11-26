"""Deterministic synthetic shop measurements sharing latent material properties."""

from random import Random

from .settings import COLUMNS

SEED = 26112592

# Absorbance offsets and responses to moisture, compaction and binder variation.
CHANNELS = (
    (0.29, 0.045, -0.017, 0.030),
    (0.52, 0.088, -0.029, 0.014),
    (0.43, 0.067, -0.025, 0.022),
    (0.36, 0.041, -0.013, 0.029),
    (0.73, 0.110, -0.043, 0.019),
    (0.47, 0.060, -0.012, 0.047),
)


def demo_rows() -> list[dict]:
    rng = Random(SEED)
    rows = []
    for split, prefix, batches, strips in (
        ("train", "C", 8, 8), ("holdout", "H", 3, 6), ("pending", "P", 2, 5),
    ):
        for batch in range(1, batches + 1):
            batch_id = f"GP-{prefix}{batch:02d}"
            moisture_shift = rng.uniform(-0.75, 0.75)
            press_shift = rng.uniform(-0.65, 0.65)
            binder_shift = rng.uniform(-0.55, 0.55)
            kiln_shift = rng.uniform(-0.12, 0.12)
            for strip in range(1, strips + 1):
                moisture = moisture_shift + rng.gauss(0, 0.44)
                press = press_shift + rng.gauss(0, 0.36)
                binder = binder_shift + rng.gauss(0, 0.36)
                if split == "pending" and batch == 2 and strip == 5:
                    moisture = 2.3
                spectrum = [round(offset + wet * moisture + dense * press + glue * binder + rng.gauss(0, 0.006), 5)
                            for offset, wet, dense, glue in CHANNELS]
                density = round(820 + 50 * press - 13 * moisture + rng.gauss(0, 9), 2)
                thickness = round(12.5 + 0.18 * moisture - 0.12 * press + rng.gauss(0, 0.12), 3)
                measured = round(4.45 - 0.72 * moisture + 0.56 * press + 0.75 * binder
                                 + kiln_shift + rng.gauss(0, 0.13), 3)
                values = (f"{batch_id}-{strip:02d}", batch_id, split, *spectrum, density, thickness,
                          None if split == "pending" else measured)
                rows.append(dict(zip(COLUMNS, values)))
    return rows
