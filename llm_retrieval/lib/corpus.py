def build_corpus_block(fables: list[dict]) -> str:
    if not fables:
        return ""
    return "\n\n".join(f"[{f['doc_id']}] {f['text']}" for f in fables)
