import polars as pl
import pytest

from sepsis_mining.data import scan_mart
from sepsis_mining.targets import add_targets


def _rows(df, col, pid):
    return df.filter(pl.col("patient_id") == pid).sort("row_idx")[col].to_list()


def test_row_idx_and_onset(mini_mart):
    out = add_targets(mini_mart)
    assert _rows(out, "row_idx", "p_sep") == list(range(12))
    onset = out.group_by("patient_id").agg(pl.col("label_onset_row_idx").first())
    assert dict(onset.iter_rows()) == {"p_non": None, "p_sep": 8, "p_left": 0, "p_early": 3}


def test_y6_training_rows_and_positives(mini_mart):
    out = add_targets(mini_mart)
    train = out.filter(pl.col("label_y6_in_train"))
    assert train.height == 15
    assert train["label_y6"].sum() == 9
    pos = out.filter(pl.col("label_y6"))
    assert sorted(pos.filter(pl.col("patient_id") == "p_sep")["row_idx"]) == [2, 3, 4, 5, 6, 7]
    assert sorted(pos.filter(pl.col("patient_id") == "p_early")["row_idx"]) == [0, 1, 2]
    assert train.filter(pl.col("patient_id") == "p_left").height == 0
    assert out["label_y6"].null_count() == 0
    assert out["label_y6_in_train"].null_count() == 0


def test_k_is_parameter(mini_mart):
    out = add_targets(mini_mart, k=2)
    assert "label_y2" in out.columns and "label_y2_in_train" in out.columns
    assert out["label_y2"].sum() == 4  # p_sep 6,7 and p_early 1,2


def test_hazard(mini_mart):
    out = add_targets(mini_mart)
    assert out["label_hazard_at_risk"].sum() == 24
    ev = out.filter(pl.col("label_hazard_event"))
    assert ev.height == 3
    assert dict(zip(ev["patient_id"], ev["row_idx"], strict=True)) == {
        "p_sep": 8,
        "p_left": 0,
        "p_early": 3,
    }
    assert out["label_hazard_event"].null_count() == 0
    assert out["label_hazard_at_risk"].null_count() == 0


@pytest.mark.parametrize(
    ("iculos", "group"),
    [(1, "0-6"), (6, "0-6"), (7, "7-12"), (12, "7-12"), (13, "13-24"), (24, "13-24"),
     (25, "25-48"), (48, "25-48"), (49, ">48")],
)  # fmt: skip
def test_icu_time_group(iculos, group):
    df = pl.DataFrame({"patient_id": ["a"], "iculos": [iculos], "label_sepsis": [0]})
    assert add_targets(df)["feat_icu_time_group"].to_list() == [group]


def test_order_independent(mini_mart):
    shuffled = add_targets(mini_mart)
    ordered = add_targets(mini_mart.sort("patient_id", "iculos"))
    keys = ["patient_id", "iculos"]
    assert shuffled.sort(keys).equals(ordered.sort(keys))


def test_lazy_in_lazy_out(mini_mart):
    out = add_targets(mini_mart.lazy())
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().height == mini_mart.height


@pytest.mark.export
def test_real_export_counts(export_dir):
    out = add_targets(scan_mart(export_dir))
    s = out.select(
        pl.col("label_y6_in_train").sum().alias("train"),
        (pl.col("label_y6") & pl.col("label_y6_in_train")).sum().alias("pos"),
        pl.col("label_hazard_at_risk").sum().alias("hazard"),
        pl.col("label_hazard_event").sum().alias("events"),
        pl.col("patient_id").filter(pl.col("label_onset_row_idx") == 0).n_unique().alias("left"),
    ).collect()
    assert s.row(0) == (1_299_870, 14_079, 1_527_226, 2_932, 426)
