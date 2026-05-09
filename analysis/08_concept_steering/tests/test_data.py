import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import load_corpus, build_tag_index


def test_build_tag_index_groups_doc_ids_by_tag_value(tmp_path):
    elements = [
        {"doc_id": "f0", "characters": ["fox", "crow"], "moral_category": "deception",
         "character_roles": {"fox": "trickster", "crow": "victim"}},
        {"doc_id": "f1", "characters": ["fox"],          "moral_category": "greed",
         "character_roles": {"fox": "protagonist"}},
        {"doc_id": "f2", "characters": ["wolf"],         "moral_category": "deception",
         "character_roles": {"wolf": "antagonist"}},
    ]
    p = tmp_path / "fable_elements.json"
    p.write_text(json.dumps(elements))

    idx = build_tag_index(p, fields=["characters", "moral_category", "character_roles"])
    assert idx["characters"]["fox"]              == {"f0", "f1"}
    assert idx["characters"]["wolf"]             == {"f2"}
    assert idx["moral_category"]["deception"]    == {"f0", "f2"}
    # character_roles is a dict per fable — explode the DICT VALUES (roles)
    assert idx["character_roles"]["trickster"]   == {"f0"}
    assert idx["character_roles"]["victim"]      == {"f0"}
    assert idx["character_roles"]["protagonist"] == {"f1"}
    assert idx["character_roles"]["antagonist"]  == {"f2"}


def test_load_corpus_returns_aligned_lists(tmp_path):
    morals_data = [{"moral_id": "m0", "moral": "be kind"},
                    {"moral_id": "m1", "moral": "be brave"}]
    fables_data = [{"doc_id": "f0", "fable": "once upon ..."},
                    {"doc_id": "f1", "fable": "long ago ..."}]
    qrels = {"m0": "f0", "m1": "f1"}

    mp = tmp_path / "morals.json"; mp.write_text(json.dumps(morals_data))
    fp = tmp_path / "fables.json"; fp.write_text(json.dumps(fables_data))
    qp = tmp_path / "qrels.json"; qp.write_text(json.dumps(qrels))

    corpus = load_corpus(morals_path=mp, fables_path=fp, qrels_path=qp)
    assert corpus.moral_texts   == ["be kind", "be brave"]
    assert corpus.fable_texts   == ["once upon ...", "long ago ..."]
    assert corpus.gt_fable_idx  == [0, 1]
    assert corpus.fable_doc_ids == ["f0", "f1"]


def test_real_data_smoke():
    """Sanity: real corpus loads and shapes are right."""
    ROOT = Path(__file__).resolve().parents[3]
    corpus = load_corpus(
        morals_path=ROOT / "data/processed/morals_corpus.json",
        fables_path=ROOT / "data/processed/fables_corpus.json",
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    assert len(corpus.moral_texts)  == 709
    assert len(corpus.fable_texts)  == 709
    assert len(corpus.gt_fable_idx) == 709
    assert all(0 <= i < 709 for i in corpus.gt_fable_idx)


def test_real_metadata_smoke():
    ROOT = Path(__file__).resolve().parents[3]
    idx = build_tag_index(
        ROOT / "data/enriched/fable_elements.json",
        fields=["characters", "character_roles", "moral_category", "setting", "fable_type", "themes"],
    )
    assert "fox"        in idx["characters"]
    assert "trickster"  in idx["character_roles"]   # dict-values exploded
    assert "deception"  in idx["moral_category"]
    assert "deception"  in idx["themes"]            # also appears as a theme
    assert len(idx["characters"]["fox"]) >= 60      # spec verified n=67
    assert len(idx["character_roles"]["trickster"]) >= 50  # spec verified n=64
