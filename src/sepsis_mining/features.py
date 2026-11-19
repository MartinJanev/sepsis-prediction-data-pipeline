"""Feature sets (S1/S2, D9) and train-only preprocessing for linear models (D7/D8/D11)."""

import re

import polars as pl

# Column patterns come from int_patient_hour_imputed.sql / int_patient_hour_features.sql,
# prefixed with `feat_` by ml_features_patient_hour.sql.
_S1_NAMES = {
    "feat_age",
    "feat_gender",
    "feat_hosp_adm_time",
    "feat_iculos",
    "feat_delta_sofa_partial_24h",
    "feat_shock_index",
    "feat_pulse_pressure",
    "feat_map_or_derived",
    "feat_bun_creatinine_ratio",
    "feat_spo2_fio2_ratio",
    "feat_lactate_max_so_far",
}
_S2_NAMES = {
    "feat_n_vitals_measured",
    "feat_n_labs_measured",
    "feat_lab_draw_hours_24h",
    "feat_hours_since_any_lab",
    "feat_map_is_derived",
    "feat_n_abnormal",
    "feat_n_abnormal_cumulative",
}
_S1_RE = re.compile(
    r"feat_\w+_last"
    r"|feat_\w+_(mean|min|max|std)_(6h|24h)"
    r"|feat_\w+_(slope_24h|delta_1h|delta_6h)"
    r"|feat_(sirs|qsofa|sofa)_(?!components_available$)\w+"
)
_S2_RE = re.compile(
    r"feat_\w+_(measured|hours_since|never_measured|count_6h|count_24h)"
    r"|feat_(sirs|qsofa|sofa)_components_available"
)
_BINARY_NAMES = {"feat_gender", "feat_map_is_derived"}


def _is_binary_name(c: str) -> bool:
    # feat_n_*_measured are counts, not flags
    return c in _BINARY_NAMES or (
        c.endswith(("_measured", "_never_measured")) and not c.startswith("feat_n_")
    )


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    """S1 / S2 (= S1 + measurement-process features) from mart column names."""
    present = set(columns)
    s1, s2 = [], []
    for c in columns:
        if not c.startswith("feat_"):
            continue
        if c == "feat_icu_time_group" or f"{c}_last" in present:  # excluded: raw observed
            continue
        if c in _S1_NAMES or _S1_RE.fullmatch(c) and c not in _S2_NAMES:
            s1.append(c)
        elif c in _S2_NAMES or _S2_RE.fullmatch(c):
            s2.append(c)
        else:
            raise ValueError(f"unclassified feature column: {c}")
    derived = [
        c.replace("_hours_since", "_never_measured")
        for c in s2
        if c.endswith("_hours_since") and c.replace("_hours_since", "_never_measured") not in s2
    ]
    return {"S1": s1, "S2": s1 + s2 + derived}


def preprocess(frame, stats: pl.DataFrame, scheme: str, feature_set: str):
    """Clip to [p01, p99], fill, z-score using only train stats; works on DataFrame/LazyFrame.

    The output keeps patient_id, iculos, row_idx, feat_icu_time_group, label_* and meta_* next to
    the preprocessed features, so it feeds fit_hazard / prediction_frame / alarm_timing directly
    (alarm_timing: `pre.with_columns(binary_alarm=...)`.) Model input X is the S1/S2 list from
    `feature_sets(mart_columns)`; S2 additionally has the derived `*_never_measured` columns
    (already in the output), so take X as the output's feat_* columns minus feat_icu_time_group.
    """
    schema = frame.collect_schema()
    cols = feature_sets(schema.names())[feature_set]
    st = {r["feature"]: r for r in stats.filter(pl.col("split_scheme") == scheme).to_dicts()}
    keep = [
        c
        for c in schema.names()
        if c in ("patient_id", "iculos", "row_idx", "feat_icu_time_group")
        or c.startswith(("label_", "meta_"))
    ]

    lf = frame.lazy()
    # never_measured is computed before hours_since is filled
    lf = lf.with_columns(
        [
            pl.col(c.replace("_never_measured", "_hours_since")).is_null().alias(c)
            for c in cols
            if c not in schema.names() and c.endswith("_never_measured")
        ]
    )
    schema = lf.collect_schema()

    exprs = []
    for c in cols:
        col = pl.col(c)
        if schema[c] == pl.Boolean or _is_binary_name(c):
            # null -> 0: absent flag / unmeasured component counts as "not set"
            exprs.append(col.cast(pl.Int8).fill_null(0).alias(c))
            continue
        if c not in st:
            raise ValueError(f"no reference stats for {c!r} in scheme {scheme!r}")
        s = st[c]
        if not s["std"]:  # null/0 train std: uninformative in this training set -> constant 0.0
            exprs.append(pl.lit(0.0).alias(c))
            continue
        is_hours_since = c.endswith("_hours_since") or c == "feat_hours_since_any_lab"
        fill = -1 if is_hours_since else s["median"]  # D7 cold start: never measured yet -> -1
        exprs.append(
            (
                (col.cast(pl.Float64).clip(s["p01"], s["p99"]).fill_null(fill) - s["mean"])
                / s["std"]
            ).alias(c)
        )
    out = lf.select(*keep, *exprs)
    return out if isinstance(frame, pl.LazyFrame) else out.collect()
