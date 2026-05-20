# tests/experiments/test_build_tf1_corpus.py
import json
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = Path(__file__).parent.parent.parent / "experiments" / "11_tf1_diagnostic" / "build_tf1_corpus.py"
build = _load("build_tf1_corpus", _PATH)
group_by_moral = build.group_by_moral
first_seen_order = build.first_seen_order
assign_moral_ids = build.assign_moral_ids
sample_n_per_moral = build.sample_n_per_moral


def _row(idx, moral, fable="...", chunk=0, prompt_hash="h"):
    return {"idx": idx, "chunk": chunk, "prompt_hash": prompt_hash, "moral": moral, "fable": fable}


def test_group_by_moral_lowercases_and_strips():
    rows = [_row(0, " Greed Leads To Downfall "), _row(1, "greed leads to downfall")]
    groups = group_by_moral(rows)
    assert list(groups.keys()) == ["greed leads to downfall"]
    assert len(groups["greed leads to downfall"]) == 2


def test_first_seen_order_preserves_insertion():
    rows = [_row(0, "B"), _row(1, "A"), _row(2, "B"), _row(3, "C")]
    assert first_seen_order(rows) == ["b", "a", "c"]


def test_assign_moral_ids_zero_padded():
    ids = assign_moral_ids(["a", "b", "c"])
    assert ids == {"a": "moral_tf1_000", "b": "moral_tf1_001", "c": "moral_tf1_002"}


def test_sample_n_per_moral_is_deterministic_with_same_seed():
    grouped = {"m": [_row(i, "m") for i in range(20)]}
    out_a = sample_n_per_moral(grouped, n=5, seed=42)
    out_b = sample_n_per_moral(grouped, n=5, seed=42)
    assert [r["idx"] for r in out_a["m"]] == [r["idx"] for r in out_b["m"]]


def test_sample_n_per_moral_differs_with_different_seed():
    grouped = {"m": [_row(i, "m") for i in range(100)]}
    out_a = sample_n_per_moral(grouped, n=10, seed=1)
    out_b = sample_n_per_moral(grouped, n=10, seed=2)
    assert [r["idx"] for r in out_a["m"]] != [r["idx"] for r in out_b["m"]]


def test_sample_n_per_moral_errors_when_not_enough_rows():
    grouped = {"m": [_row(i, "m") for i in range(3)]}
    with pytest.raises(ValueError, match="only 3"):
        sample_n_per_moral(grouped, n=10, seed=42)


build_morals_corpus = build.build_morals_corpus
build_fables_corpus = build.build_fables_corpus
build_qrels_moral_to_fable = build.build_qrels_moral_to_fable
build_qrels_fable_to_moral = build.build_qrels_fable_to_moral


def test_build_morals_corpus_shape():
    unique = ["greed leads to downfall", "honesty wins"]
    ids = {"greed leads to downfall": "moral_tf1_000", "honesty wins": "moral_tf1_001"}
    out = build_morals_corpus(unique, ids)
    assert out == [
        {"doc_id": "moral_tf1_000", "text": "greed leads to downfall"},
        {"doc_id": "moral_tf1_001", "text": "honesty wins"},
    ]


def test_build_fables_corpus_assigns_globally_unique_ids():
    sampled = {
        "a": [_row(10, "a", fable="F0", chunk=1, prompt_hash="h0"),
              _row(11, "a", fable="F1", chunk=1, prompt_hash="h1")],
        "b": [_row(20, "b", fable="G0", chunk=2, prompt_hash="h2"),
              _row(21, "b", fable="G1", chunk=2, prompt_hash="h3")],
    }
    unique = ["a", "b"]
    ids = {"a": "moral_tf1_000", "b": "moral_tf1_001"}
    out = build_fables_corpus(sampled, unique, ids, n=2)
    assert [f["doc_id"] for f in out] == [
        "fable_tf1_00000", "fable_tf1_00001",
        "fable_tf1_00002", "fable_tf1_00003",
    ]
    assert out[0]["moral_id"] == "moral_tf1_000"
    assert out[2]["moral_id"] == "moral_tf1_001"
    assert out[0]["source_idx"] == 10
    assert out[0]["prompt_hash"] == "h0"
    assert out[0]["text"] == "F0"
    assert out[2]["text"] == "G0"
    assert out[0]["source_chunk"] == 1
    assert out[2]["source_chunk"] == 2


def test_build_fables_corpus_rejects_mismatched_row_count():
    sampled = {"a": [_row(0, "a"), _row(1, "a")]}  # 2 rows
    with pytest.raises(AssertionError, match="Expected 3 rows"):
        build_fables_corpus(sampled, ["a"], {"a": "moral_tf1_000"}, n=3)


def test_build_qrels_moral_to_fable_pair_per_row():
    fables = [
        {"doc_id": "fable_tf1_00000", "moral_id": "moral_tf1_000"},
        {"doc_id": "fable_tf1_00001", "moral_id": "moral_tf1_000"},
    ]
    qrels = build_qrels_moral_to_fable(fables)
    assert qrels == [
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00001", "relevance": 1},
    ]


def test_build_qrels_fable_to_moral_is_inverse():
    fables = [
        {"doc_id": "fable_tf1_00000", "moral_id": "moral_tf1_000"},
        {"doc_id": "fable_tf1_00001", "moral_id": "moral_tf1_001"},
    ]
    qrels = build_qrels_fable_to_moral(fables)
    assert qrels == [
        {"query_id": "fable_tf1_00000", "doc_id": "moral_tf1_000", "relevance": 1},
        {"query_id": "fable_tf1_00001", "doc_id": "moral_tf1_001", "relevance": 1},
    ]


run_build = build.run_build


def test_run_build_writes_all_four_files(tmp_path):
    samples = [
        {"idx": 0, "chunk": 0, "prompt_hash": "h0", "moral": "A", "fable": "fa0"},
        {"idx": 1, "chunk": 0, "prompt_hash": "h1", "moral": "A", "fable": "fa1"},
        {"idx": 2, "chunk": 0, "prompt_hash": "h2", "moral": "A", "fable": "fa2"},
        {"idx": 3, "chunk": 0, "prompt_hash": "h3", "moral": "B", "fable": "fb0"},
        {"idx": 4, "chunk": 0, "prompt_hash": "h4", "moral": "B", "fable": "fb1"},
        {"idx": 5, "chunk": 0, "prompt_hash": "h5", "moral": "B", "fable": "fb2"},
    ]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("\n".join(json.dumps(r) for r in samples))

    out_dir = tmp_path / "out"
    run_build(samples_path=samples_path, n=2, seed=42, out_dir=out_dir, expected_unique_morals=2)

    processed = out_dir / "processed"
    morals = json.loads((processed / "morals_corpus.json").read_text())
    fables = json.loads((processed / "fables_corpus.json").read_text())
    qmf = json.loads((processed / "qrels_moral_to_fable.json").read_text())
    qfm = json.loads((processed / "qrels_fable_to_moral.json").read_text())

    assert len(morals) == 2
    assert len(fables) == 4
    assert len(qmf) == 4 and len(qfm) == 4
    assert (out_dir / "README.md").exists()

    fable_ids = {f["doc_id"] for f in fables}
    assert {q["doc_id"] for q in qmf} == fable_ids
    assert {q["query_id"] for q in qfm} == fable_ids


def test_run_build_fails_on_wrong_unique_moral_count(tmp_path):
    samples = [{"idx": i, "chunk": 0, "prompt_hash": str(i), "moral": "Only", "fable": "f"}
               for i in range(5)]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in samples))
    with pytest.raises(ValueError, match="expected 99"):
        run_build(samples_path=p, n=2, seed=42, out_dir=tmp_path / "o", expected_unique_morals=99)


def test_run_build_deduplicates_on_duplicate_prompt_hash(tmp_path):
    # 5 rows all sharing the same prompt_hash (as produced by multi-chunk sampling).
    # After dedup only 1 unique row remains, which is fewer than n=2, so the build
    # should fail with a "not enough rows" error — proving duplicates are removed.
    samples = [{"idx": i, "chunk": 0, "prompt_hash": "DUP", "moral": "M", "fable": "f"}
               for i in range(5)]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in samples))
    with pytest.raises(ValueError, match="only 1 cached rows"):
        run_build(samples_path=p, n=2, seed=42, out_dir=tmp_path / "o", expected_unique_morals=1)


has_explicit_moral = build.has_explicit_moral


def test_has_explicit_moral_detects_phrase_the_moral_of():
    assert has_explicit_moral("...and the moral of the story is to never lie.", "honesty wins")


def test_has_explicit_moral_detects_phrase_this_fable_teaches():
    assert has_explicit_moral("They learned much. This fable teaches us that kindness wins.", "be kind")


def test_has_explicit_moral_detects_phrase_lesson_here_is():
    assert has_explicit_moral("In the end, the lesson here is patience.", "patience pays")


def test_has_explicit_moral_detects_explicit_moral_label():
    assert has_explicit_moral("A short story.\nMoral: always tell the truth.\n", "honesty wins")


def test_has_explicit_moral_detects_high_overlap_sentence():
    fable = "After the storm they learned that timely help earns lasting loyalty."
    moral = "timely help earns lasting loyalty"
    assert has_explicit_moral(fable, moral)


def test_has_explicit_moral_false_when_clean():
    fable = ("In a quiet glade lived a young deer who often wandered far from home. "
             "One day a wise owl told him a strange riddle, and after much thought "
             "the deer understood that he had been chasing illusions all along.")
    moral = "patience pays"
    assert not has_explicit_moral(fable, moral)


def test_has_explicit_moral_safe_for_short_morals():
    assert not has_explicit_moral("A quiet story with nothing notable.", "patience")


select_low_iou_clean = build.select_low_iou_clean


def _row_with_iou(idx, moral, fable, iou):
    return {
        "idx": idx, "chunk": 0, "prompt_hash": f"h{idx}",
        "moral": moral, "fable": fable,
        "iou_no_stop": iou,
    }


def test_select_low_iou_clean_takes_n_lowest_iou_per_moral():
    grouped = {
        "moral_a": [
            _row_with_iou(0, "moral_a", "clean fable 1", iou=0.05),
            _row_with_iou(1, "moral_a", "clean fable 2", iou=0.02),
            _row_with_iou(2, "moral_a", "clean fable 3", iou=0.03),
            _row_with_iou(3, "moral_a", "clean fable 4", iou=0.04),
        ],
    }
    out = select_low_iou_clean(grouped, n=2)
    assert len(out["moral_a"]) == 2
    assert [r["iou_no_stop"] for r in out["moral_a"]] == [0.02, 0.03]


def test_select_low_iou_clean_filters_leaky_fables_before_sorting():
    grouped = {
        "honesty wins": [
            _row_with_iou(0, "honesty wins", "The moral of the story is honesty.", iou=0.01),
            _row_with_iou(1, "honesty wins", "Clean narrative about birds.", iou=0.02),
            _row_with_iou(2, "honesty wins", "Another clean fable.", iou=0.05),
        ],
    }
    out = select_low_iou_clean(grouped, n=2)
    fables = [r["fable"] for r in out["honesty wins"]]
    assert "The moral of the story is honesty." not in fables
    assert fables == ["Clean narrative about birds.", "Another clean fable."]


def test_select_low_iou_clean_raises_when_not_enough_clean():
    grouped = {
        "m": [
            _row_with_iou(0, "m", "Moral: be kind.", iou=0.01),  # leaky
            _row_with_iou(1, "m", "Clean.", iou=0.02),
        ],
    }
    with pytest.raises(ValueError, match="only 1 fable remain"):
        select_low_iou_clean(grouped, n=2)


def test_run_build_with_selection_low_iou_clean_uses_clean_fables(tmp_path):
    samples = [
        {"idx": 0, "chunk": 0, "prompt_hash": "h0", "moral": "be kind",
         "fable": "A clean story about a fox.", "iou_no_stop": 0.02},
        {"idx": 1, "chunk": 0, "prompt_hash": "h1", "moral": "be kind",
         "fable": "Another clean story.", "iou_no_stop": 0.03},
        {"idx": 2, "chunk": 0, "prompt_hash": "h2", "moral": "be kind",
         "fable": "The moral of the story is to be kind.", "iou_no_stop": 0.01},
    ]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("\n".join(json.dumps(r) for r in samples))

    out_dir = tmp_path / "out"
    build.run_build(
        samples_path=samples_path, n=2, seed=42,
        out_dir=out_dir, expected_unique_morals=1,
        selection="low_iou_clean",
    )
    fables = json.loads((out_dir / "processed" / "fables_corpus.json").read_text())
    fable_texts = [f["text"] for f in fables]
    assert "The moral of the story is to be kind." not in fable_texts
    assert len(fables) == 2


def test_select_low_iou_clean_is_deterministic():
    grouped = {
        "m1": [
            _row_with_iou(0, "m1", "clean a", 0.05),
            _row_with_iou(1, "m1", "clean b", 0.02),
            _row_with_iou(2, "m1", "clean c", 0.03),
        ],
        "m2": [
            _row_with_iou(3, "m2", "clean d", 0.10),
            _row_with_iou(4, "m2", "clean e", 0.04),
        ],
    }
    a = select_low_iou_clean(grouped, n=2)
    b = select_low_iou_clean(grouped, n=2)
    assert [(r["idx"], r["iou_no_stop"]) for r in a["m1"]] == \
           [(r["idx"], r["iou_no_stop"]) for r in b["m1"]]
    assert [(r["idx"], r["iou_no_stop"]) for r in a["m2"]] == \
           [(r["idx"], r["iou_no_stop"]) for r in b["m2"]]


def test_run_build_rejects_unknown_selection(tmp_path):
    samples = [
        {"idx": 0, "chunk": 0, "prompt_hash": "h0", "moral": "be kind",
         "fable": "Clean.", "iou_no_stop": 0.02},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(samples[0]))
    with pytest.raises(ValueError, match="unknown selection"):
        build.run_build(
            samples_path=p, n=1, seed=42, out_dir=tmp_path / "o",
            expected_unique_morals=1, selection="bogus",
        )
