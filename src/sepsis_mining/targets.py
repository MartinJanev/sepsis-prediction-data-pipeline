"""Target columns (Y_K future window, discrete-time hazard) derived from `label_sepsis`."""

import polars as pl

_GROUP_EDGES = [(6, "0-6"), (12, "7-12"), (24, "13-24"), (48, "25-48")]


def add_targets[F: (pl.DataFrame, pl.LazyFrame)](frame: F, k: int = 6) -> F:
    """Add row_idx, T_i and the label_* targets. Independent of input row order."""
    pid = "patient_id"
    row = pl.col("iculos") - pl.col("iculos").min().over(pid)
    onset = pl.col("row_idx").filter(pl.col("label_sepsis") == 1).min().over(pid)
    last = pl.col("row_idx").max().over(pid)
    t, r = pl.col("label_onset_row_idx"), pl.col("row_idx")

    group = pl.when(False).then(None)
    for edge, name in _GROUP_EDGES:
        group = group.when(pl.col("iculos") <= edge).then(pl.lit(name))
    group = group.otherwise(pl.lit(">48"))

    return (
        frame.with_columns(row.alias("row_idx"))
        .with_columns(onset.alias("label_onset_row_idx"))
        .with_columns(
            ((r >= t - k) & (r < t)).fill_null(False).alias(f"label_y{k}"),
            pl.when(t.is_null()).then(last - r >= k).otherwise(r < t).alias(f"label_y{k}_in_train"),
            (t.is_null() | (r <= t)).alias("label_hazard_at_risk"),
            (r == t).fill_null(False).alias("label_hazard_event"),
            group.alias("feat_icu_time_group"),
        )
    )
