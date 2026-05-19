"""
Quick token measurement: run 1 moral through Gemini-3-Flash and print actual
input/output token counts from the API response metrics.
"""
import asyncio
import json
import random
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from llm_retrieval.lib.corpus import build_corpus_block
from llm_retrieval.lib.prompt import render_prompt
from llm_retrieval.lib.providers import make_agno_model

FABLES_PATH = ROOT / "data/processed/fables_corpus.json"
MORALS_PATH = ROOT / "data/processed/morals_corpus.json"


class RankedFables(BaseModel):
    ids: list[str]


async def main():
    fables = json.loads(FABLES_PATH.read_text())
    random.seed(42)
    random.shuffle(fables)
    corpus_block = build_corpus_block(fables)

    morals = json.loads(MORALS_PATH.read_text())
    moral = next(m for m in morals if m.get("fable_id"))

    print(f"Moral: {moral['doc_id']} — {moral['text'][:80]}...")
    print(f"Ground truth fable: {moral['fable_id']}")
    print()
    print(f"Corpus chars:  {len(corpus_block):,}")
    print(f"Char/4 estimate: {len(corpus_block) // 4:,} tokens")
    print()

    model_cfg = {"provider": "google", "id": "gemini-2.5-flash", "concurrency": 5}
    variant_cfg = {
        "label": "minimal",
        "system": "You are a retrieval system. Return only the requested JSON.",
        "user_template": (
            "Moral: {moral}\n\nCorpus:\n{corpus}\n\n"
            "Return a JSON array of the {top_k} fable IDs most relevant to this moral, "
            "ranked by relevance. IDs are in the format fable_XXXX."
        ),
    }

    from agno.agent import Agent
    model = make_agno_model(model_cfg)
    agent = Agent(
        model=model,
        instructions=variant_cfg["system"],
        output_schema=RankedFables,
        stream=False,
    )

    _, user_msg = render_prompt(moral["text"], corpus_block, variant_cfg, top_k=10)

    print(f"User message chars: {len(user_msg):,}")
    print(f"User message char/4 estimate: {len(user_msg) // 4:,} tokens")
    print()
    print("Running agent...")

    response = await agent.arun(user_msg)

    print()
    print("=" * 50)
    print("API METRICS (actual token counts)")
    print("=" * 50)
    m = response.metrics
    print(f"  input_tokens:  {m.input_tokens}")
    print(f"  output_tokens: {m.output_tokens}")
    print(f"  total_tokens:  {m.total_tokens}")
    if hasattr(m, "duration"):
        print(f"  duration:      {m.duration:.2f}s")
    if hasattr(m, "cost") and m.cost:
        print(f"  cost:          ${m.cost:.6f}")
    print()

    content = response.content
    if isinstance(content, RankedFables):
        ranked = content.ids[:10]
    elif isinstance(content, dict):
        ranked = content.get("ids", [])[:10]
    elif content:
        ranked = json.loads(str(content)).get("ids", [])[:10]
    else:
        print("No content returned (API error above)")
        return

    print(f"Top-10 ranked: {ranked}")
    hit = moral["fable_id"] in ranked
    rank = ranked.index(moral["fable_id"]) + 1 if hit else None
    print(f"Ground truth {moral['fable_id']}: {'rank ' + str(rank) if hit else 'NOT IN TOP-10'}")


if __name__ == "__main__":
    asyncio.run(main())
