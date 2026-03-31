"""Tests for runner.py utility functions.

Run from repo root:
    pytest experiments/08_autoresearch/tests/test_runner.py -v
"""
import sys
from pathlib import Path

# runner.py lives one level up from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

import runner  # noqa: E402  (not yet implemented — tests will fail)


# ── parse_metric ────────────────────────────────────────────────────────────

def test_parse_mrr_basic():
    stdout = "mrr: 0.2100\nr@1: 0.1400\n"
    assert runner.parse_metric(stdout, "mrr") == 0.2100


def test_parse_mrr_missing_returns_none():
    assert runner.parse_metric("nothing here", "mrr") is None


def test_parse_metric_r1_and_r5():
    stdout = "mrr: 0.2100\nr@1: 0.1400\nr@5: 0.3640\nr@10: 0.4200\n"
    assert runner.parse_metric(stdout, "r@1") == 0.1400
    assert runner.parse_metric(stdout, "r@5") == 0.3640
    assert runner.parse_metric(stdout, "r@10") == 0.4200


def test_parse_metric_with_extra_noise():
    stdout = "encoding 709 texts...\nmrr: 0.2456\nsome other output\n"
    assert runner.parse_metric(stdout, "mrr") == 0.2456


# ── get_best_mrr ─────────────────────────────────────────────────────────────

def test_get_best_mrr_no_file_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    assert runner.get_best_mrr() == 0.0


def test_get_best_mrr_includes_baseline_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "baseline")
    assert runner.get_best_mrr() == 0.2100  # baseline counts toward best


def test_get_best_mrr_excludes_discard_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "baseline")
    runner.append_result("def5678", 0.1900, 0.12, 0.32, 0.38, "discard", "worse attempt")
    assert runner.get_best_mrr() == 0.2100  # discard row ignored, baseline still wins


def test_get_best_mrr_returns_max_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "baseline")
    runner.append_result("def5678", 0.2300, 0.16, 0.38, 0.44, "keep", "better model")
    runner.append_result("ghi9012", 0.2250, 0.15, 0.37, 0.43, "discard", "not better")
    assert runner.get_best_mrr() == 0.2300


# ── append_result ────────────────────────────────────────────────────────────

def test_append_result_creates_file_with_header(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.1400, 0.3640, 0.4200, "baseline", "Linq baseline")
    lines = (tmp_path / "results.tsv").read_text().strip().split("\n")
    assert lines[0].startswith("commit")   # header is first line
    assert "abc1234" in lines[1]           # data is in the data row
    assert "0.2100" in lines[1]            # MRR value is in the data row
    assert "baseline" in lines[1]          # status is in the data row
    assert "Linq baseline" in lines[1]     # description is in the data row


def test_append_result_no_duplicate_header(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "first")
    runner.append_result("def5678", 0.2300, 0.16, 0.38, 0.44, "keep", "second")
    lines = (tmp_path / "results.tsv").read_text().strip().split("\n")
    header_lines = [l for l in lines if l.startswith("commit")]
    assert len(header_lines) == 1     # exactly one header


def test_append_result_tab_separated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "test")
    lines = (tmp_path / "results.tsv").read_text().strip().split("\n")
    assert len(lines[1].split("\t")) == 7  # 7 tab-separated columns
