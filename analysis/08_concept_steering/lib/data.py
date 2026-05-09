"""Load morals/fables/metadata into aligned arrays. No model code here."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable


@dataclass
class Corpus:
    moral_ids:     list[str]
    moral_texts:   list[str]
    fable_doc_ids: list[str]
    fable_texts:   list[str]
    gt_fable_idx:  list[int]   # for each moral, index into fable_texts of its ground-truth fable


def _pick(d: dict, *candidates: str) -> str:
    """Return the first present field from `candidates`."""
    for c in candidates:
        if c in d:
            return d[c]
    raise KeyError(f"none of {candidates} found in keys={list(d.keys())}")


def _moral_id(m: dict) -> str:
    return _pick(m, "moral_id", "doc_id", "id", "query_id")


def _moral_text(m: dict) -> str:
    return _pick(m, "moral", "text", "moral_text")


def _fable_id(f: dict) -> str:
    return _pick(f, "doc_id", "fable_id", "id")


def _fable_text(f: dict) -> str:
    return _pick(f, "fable", "text", "fable_text")


def _qrels_to_map(qrels) -> dict[str, str]:
    """Normalize qrels into {moral_id: fable_id}.

    Accepts:
      - dict {moral_id: fable_id}
      - list of {query_id|moral_id, doc_id|fable_id, [relevance]}
    """
    if isinstance(qrels, dict):
        return dict(qrels)
    out: dict[str, str] = {}
    for r in qrels:
        if r.get("relevance", 1) == 0:
            continue
        mid = _pick(r, "query_id", "moral_id", "id")
        fid = _pick(r, "doc_id", "fable_id")
        out[mid] = fid
    return out


def load_corpus(*, morals_path: Path, fables_path: Path, qrels_path: Path) -> Corpus:
    morals = json.loads(Path(morals_path).read_text())
    fables = json.loads(Path(fables_path).read_text())
    qrels  = _qrels_to_map(json.loads(Path(qrels_path).read_text()))

    moral_ids   = [_moral_id(m)   for m in morals]
    moral_texts = [_moral_text(m) for m in morals]
    fable_ids   = [_fable_id(f)   for f in fables]
    fable_texts = [_fable_text(f) for f in fables]

    fid_to_idx = {fid: i for i, fid in enumerate(fable_ids)}
    gt_idx = [fid_to_idx[qrels[mid]] for mid in moral_ids]

    return Corpus(moral_ids, moral_texts, fable_ids, fable_texts, gt_idx)


def build_tag_index(metadata_path: Path, *, fields: Iterable[str]) -> dict[str, dict[str, set[str]]]:
    """
    Return: {field_name: {tag_value: set_of_doc_ids}}

    Field shapes seen in `data/enriched/fable_elements.json`:
      - list of strings:  characters=[fox, crow], themes=[deception, greed]
      - scalar string:    setting=forest, moral_category=greed, fable_type=animal_only
      - dict[str, str]:   character_roles={androcles: protagonist, lion: helper}
                          -> we explode the DICT's VALUES (the controlled-vocabulary
                          roles), not the keys (free-form character mentions).
    """
    elements = json.loads(Path(metadata_path).read_text())
    index: dict[str, dict[str, set[str]]] = {f: {} for f in fields}
    for el in elements:
        doc_id = el["doc_id"]
        for field in fields:
            value = el.get(field)
            if value is None:
                continue
            if isinstance(value, dict):
                # explode dict VALUES; deduplicate so two characters with the
                # same role don't double-count one fable
                for v in set(value.values()):
                    index[field].setdefault(v, set()).add(doc_id)
            elif isinstance(value, list):
                for v in value:
                    index[field].setdefault(v, set()).add(doc_id)
            else:
                index[field].setdefault(value, set()).add(doc_id)
    return index
