import string


def render_prompt(
    moral: str,
    corpus_block: str,
    variant: dict,
    top_k: int,
) -> tuple[str, str]:
    system = variant["system"]
    template = variant["user_template"]

    # Extract all field names from the template
    formatter = string.Formatter()
    field_names = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}

    # Check that all required placeholders are present
    required = {"moral", "corpus", "top_k"}
    missing = required - field_names
    if missing:
        raise KeyError(missing.pop())

    user = template.format(
        moral=moral,
        corpus=corpus_block,
        top_k=top_k,
    )
    return system, user
