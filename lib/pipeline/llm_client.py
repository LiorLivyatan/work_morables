"""lib/pipeline/llm_client.py — Gemini API client with retry/backoff."""
import os
import time
from typing import Optional


def create_client(api_key: Optional[str] = None):
    """Create and return a Gemini genai.Client from env or explicit key."""
    from google import genai
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=key)


def call(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
) -> dict:
    """
    Call Gemini with exponential backoff on rate-limit errors.

    Returns:
        dict with keys: text, input_tokens, output_tokens, thinking_tokens, total_tokens
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config={"system_instruction": system_prompt},
            )
            text = response.text.strip()
            usage = response.usage_metadata
            return {
                "text": text,
                "input_tokens": usage.prompt_token_count if usage else 0,
                "output_tokens": usage.candidates_token_count if usage else 0,
                "thinking_tokens": (usage.thoughts_token_count or 0) if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0,
            }
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {"text": f"[ERROR: {e}]", "input_tokens": 0,
                            "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
                time.sleep(2)

    return {"text": "[ERROR: max retries exceeded]", "input_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
