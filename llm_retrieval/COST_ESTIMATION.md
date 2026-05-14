# LLM Retrieval — Cost Estimation

## Token Measurement

Actual token counts measured via Gemini 2.5 Flash API metrics (`response.metrics`), 1 query with the full 709-fable corpus:

| Measure | char/4 estimate | **Actual API** |
|---------|-----------------|----------------|
| Corpus chars | 444,799 | — |
| Input tokens | 111,244 | **104,782** |
| Output tokens | ~150 (guessed) | **122** |

The char/4 rule overestimates by ~6% — fable prose has longer average words than typical English, so tokenizes more efficiently.

**Full run token budget (709 morals, 1 variant, 1 model):**

| | Tokens |
|-|--------|
| Input | 709 × 104,782 = **74.3M** |
| Output | 709 × 122 = **86.5k** |

Output is <0.12% of input — cost is almost entirely input tokens.

**All 3 variants (1 model):** 3 × above = **222.9M input / 259.5k output**

---

## Pricing & Cost Table

Prices sourced May 2026. Costs rounded to nearest dollar.

### Anthropic (direct API)

| Model | Input $/M | Output $/M | Per variant | All 3 variants | Batch (50% off) |
|-------|-----------|------------|-------------|----------------|-----------------|
| Claude Haiku 4.5 | $1.00 | $5.00 | **$75** | **$225** | ~$113 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | **$224** | **$672** | ~$336 |
| Claude Opus 4.7 | $5.00 | $25.00 | **$374** | **$1,121** | ~$561 |

### Google Gemini (direct API)

Our input (~105k tokens) falls under the ≤200k pricing tier.

| Model (config alias) | Input $/M | Output $/M | Per variant | All 3 variants | Batch (50% off) |
|----------------------|-----------|------------|-------------|----------------|-----------------|
| Gemini 3.1 Flash-Lite *(not in config yet)* | $0.25 | $1.50 | **$19** | **$56** | ~$28 |
| Gemini 2.5 Flash *(Gemini-3-Flash)* | $0.30 | $2.50 | **$23** | **$68** | ~$34 |
| Gemini 2.5 Pro *(Gemini-2.5-Pro)* | $1.25 | $10.00 | **$94** | **$281** | ~$141 |
| Gemini 3.1 Pro Preview *(Gemini-3.1-Pro)* | $2.00 | $12.00 | **$150** | **$449** | ~$225 |

> **Note:** Config IDs `gemini-3-flash` and `gemini-3.1-pro` need to be verified against current Google API model IDs before running. `gemini-3.1-flash-lite` is not yet in `config.yaml` — add if desired.

### OpenAI (direct API)

| Model | Input $/M | Output $/M | Per variant | All 3 variants |
|-------|-----------|------------|-------------|----------------|
| GPT-4o-mini | $0.15 | $0.60 | **$11** | **$34** |
| GPT-4o | $2.50 | $10.00 | **$187** | **$560** |
| GPT-5.5 | ⚠️ TBD | ⚠️ TBD | — | — |

> **⚠️ GPT-5.5:** Not a known OpenAI model ID as of May 2026. Update `config.yaml` to a valid ID before running (e.g. `gpt-4.1` at $2/$8 per M → ~$149/variant).

### OpenRouter — OpenAI OSS

| Model | Input $/M | Output $/M | Per variant | All 3 variants |
|-------|-----------|------------|-------------|----------------|
| GPT-OSS-20B | $0.03 | $0.14 | **$2** | **$7** |
| GPT-OSS-120B | $0.04 | $0.18 | **$3** | **$9** |

### OpenRouter — Meta

| Model | Input $/M | Output $/M | Per variant | All 3 variants |
|-------|-----------|------------|-------------|----------------|
| Llama-4-Scout-17B | $0.08 | $0.30 | **$6** | **$18** |

### OpenRouter — Qwen

Config IDs (`qwen3.5-flash`, `qwen3.6-plus`, `qwen3.6-27b`) are not available on OpenRouter. Available alternatives:

| Model | Input $/M | Output $/M | Per variant | All 3 variants |
|-------|-----------|------------|-------------|----------------|
| qwen/qwen3-8b | $0.05 | $0.40 | **$4** | **$11** |
| qwen/qwen3-30b-a3b | $0.09 | $0.45 | **$7** | **$20** |
| qwen/qwen3-235b-a22b | $0.46 | $1.82 | **$34** | **$103** |

> Update `config.yaml` Qwen IDs to one of the above before running.

---

## Full Experiment Summary

All 3 variants, sorted cheapest first (valid IDs only):

| Rank | Model | All 3 variants | Notes |
|------|-------|----------------|-------|
| 1 | GPT-OSS-20B | **~$7** | Free-tier rate limits apply |
| 2 | GPT-OSS-120B | **~$9** | Free-tier rate limits apply |
| 3 | Llama-4-Scout-17B | **~$18** | Via OpenRouter |
| 4 | GPT-4o-mini | **~$34** | Direct OpenAI |
| 5 | Gemini 3.1 Flash-Lite | **~$56** | (~$28 with batch) — add to config |
| 6 | Gemini 2.5 Flash | **~$68** | (~$34 with batch API) |
| 7 | qwen3-30b-a3b | **~$20** | ID needs update in config |
| 8 | Claude Haiku 4.5 | **~$225** | (~$113 with batch API) |
| 9 | Gemini 2.5 Pro | **~$281** | (~$141 with batch API) |
| 10 | Gemini 3.1 Pro Preview | **~$449** | (~$225 with batch API) |
| 11 | GPT-4o | **~$560** | Direct OpenAI |
| 12 | Claude Sonnet 4.6 | **~$672** | (~$336 with batch API) |
| 13 | Claude Opus 4.7 | **~$1,121** | (~$561 with batch API) |
| — | GPT-5.5 | **Unknown** | Invalid model ID — fix before running |
| — | Qwen3.5/3.6 variants | **Unknown** | IDs not on OpenRouter — fix before running |

**Estimated total (all valid models, all 3 variants): ~$3,530** *(includes Gemini 3.1 Flash-Lite if added to config)*

---

## Recommendations

1. **Fix 4 model IDs before any run:** `gpt-5.5`, `qwen/qwen3.5-flash`, `qwen/qwen3.6-plus`, `qwen/qwen3.6-27b`
2. **Run `minimal` variant only first** — ⅓ the cost, gives full comparison baseline before committing to 3 variants
3. **Use Batch API** for Anthropic and Google when budget is a concern — 50% off with ~24h turnaround
4. **Skip Opus + Sonnet for 3 variants** unless early results justify it — those two alone are ~$1,800
