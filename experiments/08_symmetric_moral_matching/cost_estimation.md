# Cost Estimation: Experiment 08 — Symmetric Moral Matching

## Run Parameters

Two generation steps per fable:

| Step | What | Calls per fable | Total calls (709) |
|------|------|-----------------|-------------------|
| Corpus summaries | 2 variants × 709 fables | 2 | 1,418 |
| Query expansions | 3 variants × 709 morals | 3 | 2,127 |
| **Total** | | | **3,545** |

Model used: `gemini-3-flash-preview` (thinking enabled)

---

## Per-Fable Token Actuals (from 50-fable sample run)

### Corpus Summaries (2 variants per fable)

| Metric | Total (50 fables) | Per fable | Per call |
|--------|-------------------|-----------|----------|
| Input tokens | 26,936 | 538.7 | 269.4 |
| Output tokens | 1,180 | 23.6 | 11.8 |
| Thinking tokens | 188,212 | 3,764.2 | 1,882.1 |

### Query Expansions (3 variants per moral)

| Metric | Total (50 morals) | Per moral | Per call |
|--------|-------------------|-----------|----------|
| Input tokens | 9,721 | 194.4 | 64.8 |
| Output tokens | 1,848 | 37.0 | 12.3 |
| Thinking tokens | 198,480 | 3,969.6 | 1,323.2 |

---

## Full-Run Token Estimates (709 fables)

| Step | Input | Output | Thinking | Total |
|------|-------|--------|----------|-------|
| Corpus summaries | 381,938 | 16,732 | 2,668,818 | 3,067,488 |
| Query expansions | 137,829 | 26,233 | 2,814,446 | 2,978,508 |
| **Combined** | **519,767** | **42,965** | **5,483,264** | **6,045,996** |
| *(in millions)* | *0.520M* | *0.043M* | *5.483M* | *6.046M* |

---

## Pricing (gemini-3-flash-preview, paid tier)

Source: https://ai.google.dev/gemini-api/docs/pricing

| Token type | Rate |
|------------|------|
| Input | $0.50 / 1M tokens |
| Output | $3.00 / 1M tokens |
| Thinking | $3.00 / 1M tokens (billed at output rate) |

---

## Cost Estimate — Full 709 Fables

### Corpus Summaries

| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| Input | 0.382M | $0.50/M | $0.19 |
| Output + Thinking | (0.017 + 2.669)M = 2.686M | $3.00/M | $8.06 |
| **Subtotal** | | | **$8.25** |

### Query Expansions

| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| Input | 0.138M | $0.50/M | $0.07 |
| Output + Thinking | (0.026 + 2.814)M = 2.840M | $3.00/M | $8.52 |
| **Subtotal** | | | **$8.59** |

### Combined

| Step | Cost |
|------|------|
| Corpus summaries (1,418 calls) | $8.25 |
| Query expansions (2,127 calls) | $8.59 |
| **Total** | **$16.84** |

---

## Notes

- **Thinking tokens dominate cost** — 5.48M thinking vs 0.52M input and 0.04M output. Thinking is ~91% of total token spend.
- The 50-fable sample run consumed ~426K total tokens across both steps. Scaling linearly to 709: ~6.05M tokens (14.2× factor).
- Exp 07 full run cost ~$2.75 (2,127 calls, 1 step). Exp 08 is ~6× more expensive due to the second generation step (query expansion) and heavier thinking per call.
- **Free tier feasibility:** Flash free tier allows 1,500 requests/day with rate limits. At 3,545 total calls, a free-tier run would take ~3 days with careful batching.

## Recommendation

Run the full 709 at ~**$16.84** on the paid tier. The 50-fable pilot showed B_expand achieves 70% R@1 (+8% over exp07 baseline) — the full run is needed to confirm this generalises before committing to fine-tuning.

If budget is tight, the query expansion step ($8.59) can be run first on exp07's existing summaries to test whether QE alone recovers most of the gain without regenerating corpus summaries.
