import subprocess
import sys

import polars as pl
import pytest

from sepsis_mining.hazard import fit_hazard
from sepsis_mining.simulate import recovery_table, simulate_cohort
from sepsis_mining.targets import add_targets

BETA = {"feat_x1": 0.8, "feat_x2": 0.5, "feat_age": 0.03}
FEATS = list(BETA)


def _fit_and_check(link, seed=0, n=3000):
    frame = add_targets(simulate_cohort(n, seed, BETA, link=link))
    table = recovery_table(fit_hazard(frame, FEATS, link=link), -5.0, BETA)
    return table


def test_same_seed_gives_identical_frame():
    a = simulate_cohort(200, 3, BETA)
    assert a.equals(simulate_cohort(200, 3, BETA))
    assert not a.equals(simulate_cohort(200, 4, BETA))


def test_simulated_frame_has_contiguous_labels_and_one_event_per_septic_patient():
    frame = add_targets(simulate_cohort(500, 1, BETA))
    assert {"patient_id", "iculos", "label_sepsis", *FEATS} <= set(frame.columns)
    per = frame.group_by("patient_id").agg(
        pl.col("label_sepsis").sort_by("iculos").alias("s"),
        pl.col("label_onset_row_idx").first().alias("t"),
    )
    septic = per.filter(pl.col("t").is_not_null())
    assert septic.height > 0
    for s, t in zip(septic["s"], septic["t"], strict=True):
        assert s.to_list() == [0] * t + [1] * (len(s) - t)
    assert frame["label_hazard_event"].sum() == septic.height


@pytest.mark.parametrize("link", ["logit", "cloglog"])
def test_fit_recovers_known_coefficients(link):
    t = _fit_and_check(link)
    betas = t.filter(pl.col("term").str.starts_with("feat_"))
    assert betas.height == 3
    assert betas["covered"].all(), t
    if link == "logit":
        assert t.filter(pl.col("term").str.starts_with("alpha["))["covered"].all(), t


def test_fit_hazard_table_shape():
    frame = add_targets(simulate_cohort(1000, 0, BETA))
    out = fit_hazard(frame, FEATS)
    assert out.columns == ["term", "estimate", "std_error", "odds_ratio", "ci_low", "ci_high"]
    assert out["term"].to_list()[:5] == [
        f"alpha[{g}]" for g in ("0-6", "7-12", "13-24", "25-48", ">48")
    ]
    assert out["odds_ratio"].null_count() == 0
    cl = fit_hazard(frame, FEATS, link="cloglog")
    assert cl["odds_ratio"].null_count() == cl.height


def test_fit_hazard_names_missing_group():
    frame = add_targets(simulate_cohort(300, 0, BETA)).filter(
        pl.col("feat_icu_time_group") != ">48"
    )
    with pytest.raises(ValueError, match=">48"):
        fit_hazard(frame, FEATS)


def test_broken_risk_set_is_detected():
    frame = add_targets(simulate_cohort(3000, 0, BETA)).with_columns(
        pl.lit(True).alias("label_hazard_at_risk"),
        (pl.col("label_sepsis") == 1).alias("label_hazard_event"),
    )
    t = recovery_table(fit_hazard(frame, FEATS), -5.0, BETA)
    assert not t.filter(pl.col("term").str.starts_with("feat_"))["covered"].all(), t


def test_main_writes_outputs(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "sepsis_mining.simulate", "--out", str(tmp_path), "--n", "300"],
        check=True,
    )
    assert (tmp_path / "recovery.csv").exists()
    assert (tmp_path / "recovery.png").exists()


def test_fit_hazard_accepts_lazyframe():
    frame = add_targets(simulate_cohort(500, 0, BETA))
    eager = fit_hazard(frame, FEATS)
    assert fit_hazard(frame.lazy(), FEATS).equals(eager)
