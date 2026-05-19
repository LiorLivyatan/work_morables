import csv
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.results import (
    get_run_path,
    append_query_row,
    count_completed_rows,
    load_completed_rows,
    append_summary_row,
    RUN_FIELDNAMES,
    UNIFIED_FIELDNAMES,
)


def test_get_run_path_format():
    base = Path("/tmp/results")
    path = get_run_path(base, "GPT-4o", "minimal", "2026-05-14")
    assert path == Path("/tmp/results/runs/2026-05-14_GPT-4o_minimal.csv")


def test_append_and_count(tmp_path):
    path = tmp_path / "run.csv"
    row = {f: "x" for f in RUN_FIELDNAMES}
    append_query_row(path, row)
    append_query_row(path, row)
    assert count_completed_rows(path) == 2


def test_count_nonexistent_file(tmp_path):
    assert count_completed_rows(tmp_path / "missing.csv") == 0


def test_load_completed_rows(tmp_path):
    path = tmp_path / "run.csv"
    row = {f: "val" for f in RUN_FIELDNAMES}
    append_query_row(path, row)
    rows = load_completed_rows(path)
    assert len(rows) == 1
    assert rows[0]["moral_id"] == "val"


def test_load_completed_rows_casts_numerics(tmp_path):
    """CSV round-trip must preserve numeric types — needed for aggregate_metrics after resume."""
    path = tmp_path / "run.csv"
    row = {f: "" for f in RUN_FIELDNAMES}
    row.update({
        "moral_id": "moral_0001",
        "relevant_fable": "fable_0001",
        "ranked_ids": "fable_0001|fable_0002",
        "reciprocal_rank": "1.0",
        "r_at_1": "1.0",
        "r_at_5": "1.0",
        "r_at_10": "1.0",
        "ndcg_at_10": "1.0",
        "rank": "1",
        "latency_s": "0.5",
    })
    append_query_row(path, row)
    rows = load_completed_rows(path)
    assert isinstance(rows[0]["reciprocal_rank"], float)
    assert isinstance(rows[0]["r_at_1"], float)
    assert isinstance(rows[0]["rank"], int)
    assert isinstance(rows[0]["latency_s"], float)


def test_load_completed_rows_null_rank(tmp_path):
    path = tmp_path / "run.csv"
    row = {f: "" for f in RUN_FIELDNAMES}
    row["rank"] = ""
    append_query_row(path, row)
    rows = load_completed_rows(path)
    assert rows[0]["rank"] is None


def test_append_summary_row(tmp_path):
    unified = tmp_path / "unified.csv"
    summary = {f: "x" for f in UNIFIED_FIELDNAMES}
    append_summary_row(unified, summary)
    append_summary_row(unified, summary)
    with open(unified) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
