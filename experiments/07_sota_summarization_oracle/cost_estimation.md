# Cost Estimation: Experiment 07 — SOTA Summarization Oracle

## Run Parameters
- **Fables:** 709
- **Prompt variants:** 3 (direct_moral, narrative_distillation, conceptual_abstract)
- **Total API calls:** 709 x 3 = **2,127 calls**

## Per-Call Token Estimates (from 3-fable sample runs)

Based on sample runs with the `direct_moral` variant. The other two variants have
slightly longer system prompts (~10-20 extra tokens), so we add a 10% buffer to
input estimates.

| Model | Input/call | Thinking/call | Output/call |
|-------|-----------|---------------|-------------|
| gemini-3.1-pro-preview | ~278 | ~711 | ~8 |
| gemini-3-flash-preview | ~278 | ~366 | ~13 |
| gemini-3.1-flash-lite-preview | ~278 | 0 | ~15 |

## Full Run Token Estimates (2,127 calls)

| Model | Input | Thinking | Output | Total |
|-------|-------|----------|--------|-------|
| **gemini-3.1-pro-preview** | ~650K | ~1,512K | ~18K | ~2,180K |
| **gemini-3-flash-preview** | ~650K | ~778K | ~28K | ~1,456K |
| **gemini-3.1-flash-lite-preview** | ~650K | 0 | ~32K | ~682K |

## Pricing (per 1M tokens, paid tier)

Source: https://ai.google.dev/gemini-api/docs/pricing (retrieved 2026-03-29)

| Model | Input $/M | Output $/M | Notes |
|-------|----------|-----------|-------|
| gemini-3.1-pro-preview | $2.00 | $12.00 | Thinking billed at output rate |
| gemini-3-flash-preview | $0.50 | $3.00 | Thinking billed at output rate |
| gemini-3.1-flash-lite-preview | $0.25 | $1.50 | No thinking |

Note: Flash and Flash Lite have **free tiers** with rate limits.

## Cost Estimates (paid tier)

### gemini-3.1-pro-preview
| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| Input | 0.650M | $2.00/M | $1.30 |
| Output + Thinking | (0.018 + 1.512) = 1.530M | $12.00/M | $18.36 |
| **Total** | | | **$19.66** |

### gemini-3-flash-preview
| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| Input | 0.650M | $0.50/M | $0.33 |
| Output + Thinking | (0.028 + 0.778) = 0.806M | $3.00/M | $2.42 |
| **Total** | | | **$2.75** |

### gemini-3.1-flash-lite-preview
| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| Input | 0.650M | $0.25/M | $0.16 |
| Output | 0.032M | $1.50/M | $0.05 |
| **Total** | | | **$0.21** |

## Summary

| Model | Thinking? | Est. Cost | Quality (sample) |
|-------|-----------|-----------|-----------------|
| gemini-3.1-pro-preview | Yes | **$19.66** | Most concise (avg 6.3 words) |
| gemini-3-flash-preview | Yes | **$2.75** | Matched exact GT moral on 1/3 samples (avg 10.7 words) |
| gemini-3.1-flash-lite-preview | No | **$0.21** | Longest outputs (avg 13.0 words) |

## Recommendation

Run **all three models** for the full corpus. Total cost: ~$22.62.
This gives us a clean comparison of how model capability and thinking affect
summary quality, and whether that translates to retrieval performance differences.

If budget is a constraint, start with **gemini-3-flash-preview** ($2.75) — best
quality-to-cost ratio based on the sample run, and it produced the closest match
to ground truth morals.
