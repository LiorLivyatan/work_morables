import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = (
    Path(__file__).parent.parent.parent
    / "finetuning" / "ft_13_tf1_transfer_clustered" / "train.py"
)
train = _load("ft_13_train", _PATH)


def _pair(idx, moral_id="moral_tf1_000", fable_id_suffix=0):
    return {
        "moral": f"moral text {moral_id}",
        "story": f"fable text {fable_id_suffix}",
        "moral_id": moral_id,
        "fable_id": f"fable_tf1_{fable_id_suffix:05d}",
    }


def test_subsample_morals_keeps_all_when_unrestricted():
    pairs = [_pair(i, "moral_tf1_000", i) for i in range(10)]
    out = train._subsample_morals(pairs, {"n_morals": None, "n_fables_per_moral": 10}, seed=42)
    assert len(out) == 10


def test_subsample_morals_caps_fables_per_moral():
    pairs = [_pair(i, "moral_tf1_000", i) for i in range(20)]
    out = train._subsample_morals(pairs, {"n_morals": None, "n_fables_per_moral": 5}, seed=42)
    assert len(out) == 5
    # First 5 by file order (insertion preserved)
    assert [p["fable_id"] for p in out] == [f"fable_tf1_{i:05d}" for i in range(5)]


def test_subsample_morals_samples_morals_then_takes_all_fables():
    pairs = []
    for moral_idx in range(4):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    out = train._subsample_morals(pairs, {"n_morals": 2, "n_fables_per_moral": 10}, seed=42)
    morals = {p["moral_id"] for p in out}
    assert len(morals) == 2
    assert len(out) == 20  # 2 morals x 10 fables
    # Pin which morals are selected so algorithmic regressions are caught.
    assert morals == {"moral_tf1_000", "moral_tf1_003"}


def test_subsample_morals_is_seed_stable():
    pairs = []
    for moral_idx in range(10):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    a = train._subsample_morals(pairs, {"n_morals": 3, "n_fables_per_moral": 5}, seed=42)
    b = train._subsample_morals(pairs, {"n_morals": 3, "n_fables_per_moral": 5}, seed=42)
    assert [(p["moral_id"], p["fable_id"]) for p in a] == [(p["moral_id"], p["fable_id"]) for p in b]


def test_split_tf1_groups_separates_by_moral_id():
    pairs = []
    for moral_idx in range(10):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    train_rows, val_rows = train.split_tf1_groups(pairs, seed=42, validation_ratio=0.20)
    train_morals = {p["moral_id"] for p in train_rows}
    val_morals = {p["moral_id"] for p in val_rows}
    assert train_morals.isdisjoint(val_morals)
    assert len(val_morals) == 2  # 20% of 10 morals
    assert len(train_morals) == 8
    assert len(train_rows) == 80
    assert len(val_rows) == 20


def test_split_tf1_groups_seed_stable():
    pairs = [_pair(i, f"moral_tf1_{i//10:03d}", i) for i in range(100)]
    a = train.split_tf1_groups(pairs, seed=42, validation_ratio=0.10)
    b = train.split_tf1_groups(pairs, seed=42, validation_ratio=0.10)
    assert [p["fable_id"] for p in a[0]] == [p["fable_id"] for p in b[0]]
    assert [p["fable_id"] for p in a[1]] == [p["fable_id"] for p in b[1]]


def test_make_tf1_dataset_assigns_integer_labels_per_moral():
    pairs = [
        _pair(0, "moral_tf1_000", 0),
        _pair(1, "moral_tf1_000", 1),
        _pair(2, "moral_tf1_001", 2),
    ]
    ds = train.make_tf1_dataset(pairs, instruction="Q: ")
    assert ds["anchor"] == ["Q: moral text moral_tf1_000", "Q: moral text moral_tf1_000", "Q: moral text moral_tf1_001"]
    assert ds["positive"] == ["fable text 0", "fable text 1", "fable text 2"]
    # Same moral_id -> same int label
    assert ds["label"][0] == ds["label"][1]
    # Different moral_id -> different int label
    assert ds["label"][0] != ds["label"][2]


def test_make_tf1_dataset_empty_instruction_passes_moral_verbatim():
    pairs = [_pair(0, "moral_tf1_000", 0)]
    ds = train.make_tf1_dataset(pairs, instruction="")
    assert ds["anchor"] == ["moral text moral_tf1_000"]


def test_split_tf1_groups_rejects_single_moral():
    import pytest as _pytest
    pairs = [_pair(0, "moral_tf1_000", 0)]
    with _pytest.raises(ValueError, match="at least 2 distinct morals"):
        train.split_tf1_groups(pairs, seed=42, validation_ratio=0.10)


def test_load_tf1_synthetic_exact_from_tmp_dir(tmp_path):
    import json as _json
    processed = tmp_path / "processed"
    processed.mkdir()
    morals = [
        {"doc_id": "moral_tf1_000", "text": "be kind"},
        {"doc_id": "moral_tf1_001", "text": "be brave"},
    ]
    fables = [
        {"doc_id": "fable_tf1_00000", "text": "Kind story 0",
         "moral_id": "moral_tf1_000", "source_idx": 0, "source_chunk": 0, "prompt_hash": "h0"},
        {"doc_id": "fable_tf1_00001", "text": "Kind story 1",
         "moral_id": "moral_tf1_000", "source_idx": 1, "source_chunk": 0, "prompt_hash": "h1"},
        {"doc_id": "fable_tf1_00002", "text": "Brave story 0",
         "moral_id": "moral_tf1_001", "source_idx": 2, "source_chunk": 0, "prompt_hash": "h2"},
    ]
    qrels = [
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00001", "relevance": 1},
        {"query_id": "moral_tf1_001", "doc_id": "fable_tf1_00002", "relevance": 1},
    ]
    (processed / "morals_corpus.json").write_text(_json.dumps(morals))
    (processed / "fables_corpus.json").write_text(_json.dumps(fables))
    (processed / "qrels_moral_to_fable.json").write_text(_json.dumps(qrels))

    pairs, stats = train.load_tf1_synthetic_exact(
        size_cfg={"n_morals": None, "n_fables_per_moral": 10},
        seed=42,
        source_dir=tmp_path,
    )
    assert len(pairs) == 3
    # Pair schema
    sample = pairs[0]
    assert set(sample.keys()) >= {"moral", "story", "moral_id", "fable_id"}
    # Stats keys and content
    assert stats["raw_total"] == 3
    assert stats["selected_rows"] == 3
    assert stats["selected_morals"] == 2
    # source_dir.name returns the last path segment; tmp_path typically ends with "test_..."
    # but for portability we just assert it's a non-empty string equal to tmp_path.name
    assert stats["selection_strategy"] == tmp_path.name
    assert stats["size_config"]["n_fables_per_moral"] == 10


def test_load_tf1_synthetic_exact_size_subsamples(tmp_path):
    import json as _json
    processed = tmp_path / "processed"
    processed.mkdir()
    morals = [{"doc_id": f"moral_tf1_{i:03d}", "text": f"moral {i}"} for i in range(5)]
    fables = []
    qrels = []
    for moral_idx in range(5):
        for fable_idx in range(4):
            fid = f"fable_tf1_{moral_idx * 4 + fable_idx:05d}"
            fables.append({
                "doc_id": fid, "text": f"story {moral_idx}/{fable_idx}",
                "moral_id": f"moral_tf1_{moral_idx:03d}",
                "source_idx": moral_idx * 4 + fable_idx,
                "source_chunk": 0, "prompt_hash": f"h{moral_idx}{fable_idx}",
            })
            qrels.append({
                "query_id": f"moral_tf1_{moral_idx:03d}",
                "doc_id": fid, "relevance": 1,
            })
    (processed / "morals_corpus.json").write_text(_json.dumps(morals))
    (processed / "fables_corpus.json").write_text(_json.dumps(fables))
    (processed / "qrels_moral_to_fable.json").write_text(_json.dumps(qrels))

    pairs, stats = train.load_tf1_synthetic_exact(
        size_cfg={"n_morals": 2, "n_fables_per_moral": 3},
        seed=42, source_dir=tmp_path,
    )
    assert len(pairs) == 6  # 2 morals * 3 fables
    assert stats["selected_morals"] == 2
    assert stats["selected_rows"] == 6
