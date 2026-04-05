"""Tests for lib/pipeline/run_utils.py"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.run_utils import (
    load_env,
    make_run_dir,
    find_latest_run_dir,
    write_manifest,
    read_manifest,
)


def test_make_run_dir_creates_directory(tmp_path):
    run_dir = make_run_dir(tmp_path, tag="sample10")
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert "sample10" in run_dir.name


def test_make_run_dir_timestamp_format(tmp_path):
    run_dir = make_run_dir(tmp_path)
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir.name)


def test_make_run_dir_no_tag(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert run_dir.exists()


def test_find_latest_run_dir(tmp_path):
    (tmp_path / "2026-04-01_run").mkdir()
    (tmp_path / "2026-04-02_run").mkdir()
    (tmp_path / "2026-04-03_run").mkdir()
    latest = find_latest_run_dir(tmp_path)
    assert latest.name == "2026-04-03_run"


def test_find_latest_run_dir_missing_base(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_run_dir(tmp_path / "nonexistent")


def test_write_and_read_manifest_roundtrip(tmp_path):
    cfg = {"n_fables": 10, "model": "gemini-3-flash"}
    write_manifest(tmp_path, "generate_corpus_summaries", cfg)
    manifest = read_manifest(tmp_path)
    assert "generate_corpus_summaries" in manifest["steps_completed"]
    assert manifest["config_snapshot"]["n_fables"] == 10


def test_write_manifest_appends_steps(tmp_path):
    write_manifest(tmp_path, "step_one", {})
    write_manifest(tmp_path, "step_two", {})
    manifest = read_manifest(tmp_path)
    assert manifest["steps_completed"] == ["step_one", "step_two"]


def test_write_manifest_no_duplicate_steps(tmp_path):
    write_manifest(tmp_path, "step_one", {})
    write_manifest(tmp_path, "step_one", {})
    manifest = read_manifest(tmp_path)
    assert manifest["steps_completed"].count("step_one") == 1


def test_read_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path)


def test_load_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY_XYZ=hello_world\n# comment\nANOTHER=value # inline comment\n")
    os.environ.pop("TEST_KEY_XYZ", None)
    os.environ.pop("ANOTHER", None)
    load_env(tmp_path)
    assert os.environ["TEST_KEY_XYZ"] == "hello_world"
    assert os.environ["ANOTHER"] == "value"


def test_load_env_does_not_override_existing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=new_value\n")
    os.environ["EXISTING_KEY"] = "original"
    load_env(tmp_path)
    assert os.environ["EXISTING_KEY"] == "original"
    del os.environ["EXISTING_KEY"]


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "nonexistent")  # should not raise
