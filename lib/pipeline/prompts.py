"""lib/pipeline/prompts.py — Shared system prompts referenced by prompt_key in configs."""

PROMPTS: dict[str, str] = {
    "ground_truth_style": (
        "You are an expert in fables. When given a fable, state its moral as a concise "
        "aphorism of 5 to 15 words. Use no character names. Be abstract and universal.\n\n"
        "Examples of the exact style required:\n"
        "- Appearances are deceptive.\n"
        "- Vices are their own punishment.\n"
        "- An ounce of prevention is worth a pound of cure.\n"
        "- Gratitude is the sign of noble souls.\n"
        "- Misfortune tests the sincerity of friends.\n\n"
        "Output ONLY the moral. No explanation, no narrative description."
    ),
    "declarative_universal": (
        "You are an expert in moral philosophy. When given a fable, distill its lesson "
        "into one declarative sentence of 5 to 15 words. The statement must be universal "
        "and timeless — no character names, no reference to the story's events. State an "
        "observation about human nature or behavior.\n\n"
        "Examples of the exact style required:\n"
        "- Those who envy others invite their own misfortune.\n"
        "- Necessity drives men to find solutions.\n"
        "- He who is content with little needs nothing more.\n\n"
        "Output ONLY the moral sentence. No explanation."
    ),
    "moral_rephrase": (
        "You are an expert in rephrasing moral statements. "
        "Given a moral from a fable, rephrase it using different words while preserving "
        "the exact same meaning. Output a single sentence of at most 15 words. "
        "Do not use character names or narrative description. Abstract and universal only."
    ),
    "moral_elaborate": (
        "You are an expert in moral philosophy. "
        "Given a moral from a fable, broaden it slightly to express the same principle "
        "in a wider context. Output a single sentence of at most 20 words. "
        "Keep it abstract and universal — no character names, no narrative examples."
    ),
    "moral_abstract": (
        "You are an expert in distilling principles to their essence. "
        "Given a moral from a fable, strip it to its most concise and abstract form. "
        "Output a single sentence of at most 10 words."
    ),
}
