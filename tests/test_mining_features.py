"""Feature sets and train-only preprocessing (task 3)."""

import polars as pl
import pytest

from sepsis_mining.data import scan_mart
from sepsis_mining.features import feature_sets, preprocess

# Real naming patterns: dbt/sepsis_dw/models/silver/int_patient_hour_{imputed,features}.sql,
# prefixed `feat_` by models/marts/ml_features_patient_hour.sql.
S1_ONLY = [
    "feat_age",
    "feat_gender",
    "feat_hosp_adm_time",
    "feat_iculos",
    "feat_hr_last",
    "feat_hr_mean_6h",
    "feat_hr_min_24h",
    "feat_hr_max_6h",
    "feat_hr_std_24h",
    "feat_hr_slope_24h",
    "feat_hr_delta_1h",
    "feat_hr_delta_6h",
    "feat_sirs_temp",
    "feat_sirs_count",
    "feat_qsofa_partial",
    "feat_sofa_renal",
    "feat_sofa_partial",
    "feat_delta_sofa_partial_24h",
    "feat_shock_index",
    "feat_pulse_pressure",
    "feat_map_or_derived",
    "feat_bun_creatinine_ratio",
    "feat_spo2_fio2_ratio",
    "feat_lactate_max_so_far",
]
S2_EXTRA = [
    "feat_hr_measured",
    "feat_hr_hours_since",
    "feat_hr_never_measured",
    "feat_hr_count_6h",
    "feat_hr_count_24h",
    "feat_n_vitals_measured",
    "feat_n_labs_measured",
    "feat_lab_draw_hours_24h",
    "feat_hours_since_any_lab",
    "feat_map_is_derived",
    "feat_sirs_components_available",
    "feat_qsofa_components_available",
    "feat_sofa_components_available",
    "feat_n_abnormal",
    "feat_n_abnormal_cumulative",
]
EXCLUDED = ["feat_hr", "feat_icu_time_group"]
# feat_hr_never_measured is derived by preprocess, not a mart column
MART_COLS = (
    ["patient_id", "iculos", "meta_hospital", "label_sepsis"]
    + S1_ONLY
    + [c for c in S2_EXTRA if c != "feat_hr_never_measured"]
    + EXCLUDED
)

STATS = pl.DataFrame(
    {
        "split_scheme": ["random", "random", "other"],
        "feature": ["feat_hr_last", "feat_hr_hours_since", "feat_hr_last"],
        "mean": [80.0, 2.0, 0.0],
        "std": [10.0, 2.0, 1.0],
        "median": [78.0, 1.0, 0.0],
        "p01": [50.0, 0.0, 0.0],
        "p99": [150.0, 10.0, 1.0],
    }
)


def _frame():
    return pl.DataFrame(
        {
            "patient_id": ["a", "a", "a"],
            "iculos": [1, 2, 3],
            "label_sepsis": [0, 0, 1],
            "feat_hr_last": [200.0, None, 60.0],
            "feat_hr_hours_since": [None, 3, 20],
            "feat_hr_measured": [True, False, True],
        }
    )


def test_feature_sets_membership():
    s = feature_sets(MART_COLS)
    assert sorted(s["S1"]) == sorted(S1_ONLY)
    assert sorted(s["S2"]) == sorted(S1_ONLY + S2_EXTRA)
    assert set(s["S1"]) < set(s["S2"])
    assert not any(t in c for c in s["S1"] for t in ("_measured", "_hours_since", "_count_"))


def test_unknown_feature_raises_naming_it():
    with pytest.raises(ValueError, match="feat_mystery"):
        feature_sets([*MART_COLS, "feat_mystery"])


def test_s1_values_match_oracle():
    out = preprocess(_frame(), STATS, "random", "S1")
    assert out.columns == ["patient_id", "iculos", "label_sepsis", "feat_hr_last"]
    assert out["feat_hr_last"].to_list() == pytest.approx([7.0, -0.2, -2.0])


def test_s2_hours_since_sentinel_never_measured_and_binary():
    out = preprocess(_frame(), STATS, "random", "S2")
    assert out["feat_hr_hours_since"].to_list() == pytest.approx([-1.5, 0.5, 4.0])
    assert out["feat_hr_never_measured"].to_list() == [1, 0, 0]
    assert out["feat_hr_measured"].to_list() == [1, 0, 1]


def test_hours_since_any_lab_null_gets_minus_one_sentinel():
    stats = pl.concat(
        [
            STATS,
            pl.DataFrame(
                {
                    "split_scheme": ["random"],
                    "feature": ["feat_hours_since_any_lab"],
                    "mean": [3.0],
                    "std": [2.0],
                    "median": [2.0],
                    "p01": [0.0],
                    "p99": [12.0],
                }
            ),
        ]
    )
    df = _frame().with_columns(feat_hours_since_any_lab=pl.Series([None, 1, 30]))
    out = preprocess(df, stats, "random", "S2")
    # null → -1 (not the median 2), clipped before the fill: [-1, 1, 12] z-scored with mean 3, std 2
    assert out["feat_hours_since_any_lab"].to_list() == pytest.approx([-2.0, -1.0, 4.5])


def test_numeric_binary_unchanged_without_stats():
    df = _frame().with_columns(feat_hr_measured=pl.Series([1, 0, 1]))
    out = preprocess(df, STATS, "random", "S2")
    assert out["feat_hr_measured"].to_list() == [1, 0, 1]


def test_boolean_with_null_and_no_stats_row_becomes_zero_one():
    df = _frame().with_columns(feat_sirs_temp=pl.Series([True, None, False]))
    out = preprocess(df, STATS, "random", "S1")
    assert out["feat_sirs_temp"].to_list() == [1, 0, 0]


def test_rows_are_independent():
    df = _frame()
    whole = preprocess(df, STATS, "random", "S2")
    for i in range(df.height):
        assert preprocess(df[i], STATS, "random", "S2").equals(whole[i])


def test_lazy_in_lazy_out():
    out = preprocess(_frame().lazy(), STATS, "random", "S1")
    assert isinstance(out, pl.LazyFrame)
    assert out.collect()["feat_hr_last"].to_list() == pytest.approx([7.0, -0.2, -2.0])


def test_missing_stats_raises_and_wrong_scheme_ignored():
    with pytest.raises(ValueError, match="feat_hr_last"):
        preprocess(_frame(), STATS, "nonexistent", "S1")
    only_other = STATS.filter(pl.col("feature") == "feat_hr_last")
    with pytest.raises(ValueError, match="feat_hr_hours_since"):
        preprocess(_frame(), only_other, "random", "S2")


@pytest.mark.export
def test_real_mart_columns_all_classified(export_dir):
    feature_sets(scan_mart(export_dir).collect_schema().names())


def test_null_or_zero_train_std_gives_constant_zero():
    stats = STATS.with_columns(
        pl.when(pl.col("feature") == "feat_hr_last").then(None).otherwise(pl.col(c)).alias(c)
        for c in ("mean", "std", "median")
    ).with_columns(
        std=pl.when(pl.col("feature") == "feat_hr_hours_since").then(0.0).otherwise(pl.col("std"))
    )
    out = preprocess(_frame(), stats, "random", "S2")
    for c in ("feat_hr_last", "feat_hr_hours_since"):
        assert out[c].to_list() == [0.0, 0.0, 0.0]


def test_preprocess_keeps_keys_row_idx_group_and_meta():
    from sepsis_mining.targets import add_targets

    df = add_targets(_frame()).with_columns(meta_hospital=pl.lit("A"))
    out = preprocess(df, STATS, "random", "S1")
    for c in ("patient_id", "iculos", "row_idx", "feat_icu_time_group", "meta_hospital"):
        assert c in out.columns
    assert "label_hazard_event" in out.columns


def test_preprocess_chain_into_hazard_and_prediction_frame():
    import numpy as np

    from sepsis_mining.evaluate import prediction_frame
    from sepsis_mining.hazard import fit_hazard
    from sepsis_mining.simulate import simulate_cohort
    from sepsis_mining.targets import add_targets

    sim = simulate_cohort(300, 1, {"feat_x1": 0.5}, alpha=-3.0)
    df = add_targets(sim.rename({"feat_x1": "feat_hr_last", "feat_x2": "feat_hr_mean_6h"}))
    feats = ["feat_hr_last", "feat_hr_mean_6h", "feat_age"]
    stats = pl.DataFrame(
        {
            "split_scheme": "random",
            "feature": feats,
            "mean": [0.0, 0.0, 60.0],
            "std": [1.0, 1.0, 15.0],
            "median": [0.0, 0.0, 60.0],
            "p01": [-3.0, -3.0, 20.0],
            "p99": [3.0, 3.0, 100.0],
        }
    )
    pre = preprocess(df, stats, "random", "S1")
    fit_hazard(pre, feats)
    p = prediction_frame(
        pre, np.full(pre.height, 0.1), 0.5, model_name="m", feature_set="S1",
        target="hazard", split="val", split_scheme="random", run_id="r",
    )  # fmt: skip
    assert p.height == pre.height


@pytest.mark.export
@pytest.mark.parametrize("feature_set", ["S1", "S2"])
def test_real_export_preprocess_every_scheme_has_no_nulls(export_dir, feature_set):
    stats = pl.read_parquet(next(export_dir.glob("**/train_reference_stats.parquet")))
    mart = scan_mart(export_dir)
    sample = mart.head(2000).collect()
    for scheme in stats["split_scheme"].unique():
        out = preprocess(sample, stats, scheme, feature_set)
        x = out.select(pl.col("^feat_.*$").exclude("feat_icu_time_group"))
        assert x.null_count().sum_horizontal().item() == 0, scheme
        assert not any(x.select(pl.col(pl.Float64).is_nan().any()).row(0)), scheme
