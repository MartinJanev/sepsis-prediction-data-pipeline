import pytest

from sepsis_pipeline.config import get_settings
from sepsis_pipeline.db import connect


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings are cached; clear the cache so tests that change env vars see their changes."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db_conn():
    """Autocommit connection to the warehouse; skips the test if the database is not running."""
    try:
        conn = connect(autocommit=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"warehouse database not reachable: {exc}")
    with conn:
        yield conn


@pytest.fixture
def export_dir():
    """Newest real export; skips the test when none exists."""
    from pathlib import Path

    from sepsis_mining.data import latest_export

    try:
        return latest_export(Path(__file__).resolve().parents[1] / "exports")
    except FileNotFoundError:
        pytest.skip("no export under exports/v*")


@pytest.fixture
def mini_mart():
    """Four patients (shuffled rows): columns patient_id, iculos, label_sepsis."""
    import random

    import polars as pl

    # (patient, first iculos, n rows, first row_idx with label 1 or None)
    spec = [("p_non", 1, 10, None), ("p_sep", 3, 12, 8), ("p_left", 1, 5, 0), ("p_early", 1, 6, 3)]
    rows = [
        (pid, start + i, int(onset is not None and i >= onset))
        for pid, start, n, onset in spec
        for i in range(n)
    ]
    random.Random(0).shuffle(rows)
    return pl.DataFrame(rows, schema=["patient_id", "iculos", "label_sepsis"], orient="row")
