import os


def make_agno_model(model_cfg: dict):
    provider = model_cfg["provider"]
    model_id = model_cfg["id"]

    if provider == "openai":
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=model_id)

    if provider == "anthropic":
        from agno.models.anthropic import Claude
        return Claude(id=model_id)

    if provider == "google":
        from agno.models.google import Gemini
        return Gemini(id=model_id)

    if provider == "openrouter":
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=model_id,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    if provider == "together":
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=model_id,
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )

    raise ValueError(f"Unknown provider: {provider!r}")
