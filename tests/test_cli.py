from typer.testing import CliRunner

from sepsis_pipeline.cli import app

runner = CliRunner()


def test_help_lists_all_stages():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "doctor",
        "extract",
        "load-bronze",
        "profile",
        "transform",
        "export",
        "timescale",
        "run-all",
        "dbt",
    ):
        assert command in result.output


def test_run_all_offers_sample_and_timescale_options():
    result = runner.invoke(app, ["run-all", "--help"])

    assert result.exit_code == 0
    assert "--sample" in result.output
    assert "--with-timescale" in result.output


def test_profile_dbt_build_skips_tests_that_need_later_stages():
    """On a fresh database, profile runs before transform: a test that also reads a gold/mart
    model must not be pulled in just because one of its parents is selected (dbt's eager mode)."""
    from sepsis_pipeline.cli import PROFILE_DBT_BUILD
    from sepsis_pipeline.transform.dbt_runner import run_dbt

    _, *selection = PROFILE_DBT_BUILD
    result = run_dbt(
        ["ls", *selection, "--resource-type", "test", "--output", "name"], capture=True
    )

    assert result.returncode == 0, result.stdout[-2000:]
    tests = set(result.stdout.split())
    assert "assert_split_proportions" in tests  # the split's own tests still run
    assert "assert_reference_stats_train_only" not in tests  # needs mart.split_scheme_membership
    assert "assert_gold_reconciles_with_silver" not in tests  # needs gold.*
