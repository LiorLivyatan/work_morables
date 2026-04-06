# Generic Experiment Pipeline

A configurable, reusable pipeline for moral-matching retrieval experiments. Each experiment needs only a `config.yaml` and a thin `run_pipeline.py`.

## Architecture

```
lib/pipeline/
├── __init__.py            # load_config() + run_experiment() orchestrator
├── llm_client.py          # Gemini API wrapper with retry/backoff
├── corpus_generator.py    # Step 1: fable × variant → corpus_summaries.json
├── query_expander.py      # Step 2: moral × variant → query_expansions.json
├── retrieval_eval.py      # Step 3: embed + fuse + score → retrieval_results.json
├── run_utils.py           # Run dir creation, .env loading, manifest I/O
├── prompts.py             # Shared prompt library (reusable across experiments)
└── default_config.yaml    # Base defaults inherited by every experiment
```

### Pipeline Flow

```
config.yaml
    │
    ▼
load_config()          ← merges defaults + resolves prompts
    │
    ▼
run_experiment()
    ├── Step 1: generate_corpus_summaries   → corpus_summaries.json
    ├── Step 2: generate_query_expansions   → query_expansions.json
    └── Step 3: run_retrieval_eval          → retrieval_results.json
```

Each step is **idempotent** — it skips if its output file already exists (pass `--force` to re-run).

---

## Creating a New Experiment

### 1. Create the experiment directory

```
experiments/09_my_experiment/
├── config.yaml        # your experiment config
└── run_pipeline.py    # thin entry point (copy from exp08)
```

### 2. Write `run_pipeline.py`

This is always the same boilerplate:

```python
"""run_pipeline.py — entry point for exp09."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.pipeline import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=None,
                    help="Existing run dir to continue (default: create new)")
parser.add_argument("--force", action="store_true",
                    help="Re-run steps even if output already exists")
args = parser.parse_args()

run_experiment(
    config_path=Path(__file__).parent / "config.yaml",
    run_dir=args.run_dir,
    force=args.force,
)
```

### 3. Write `config.yaml`

Only specify what differs from the defaults. See the full config reference below.

**Minimal example** — just one corpus variant, no query expansion:

```yaml
n_fables: 10

corpus_variants:
  - name: concise_moral
    prompt: "State the fable's moral in one sentence of at most 10 words."

retrieval_configs:
  - name: base
    corpus_variant: concise_moral
    use_expansion: false
```

**Full example** — multiple variants, expansions, fusion, baseline:

```yaml
n_fables: 50

corpus_variants:
  - name: aphorism
    prompt_key: ground_truth_style          # reuse from lib/pipeline/prompts.py
  - name: declarative
    prompt_file: prompts/my_custom.txt      # load from file
  - name: dramatic
    prompt: |                               # inline multi-line
      Rewrite the fable's lesson with emotional intensity.
      At most 15 words. No character names.

query_expansion_variants:
  - name: rephrase
    prompt: "Rephrase this moral using different words. At most 15 words."
  - name: abstract
    prompt: "Strip this moral to its most concise form. At most 10 words."

retrieval_configs:
  - name: A
    corpus_variant: aphorism
    use_expansion: false

  - name: A_expand
    corpus_variant: aphorism
    use_expansion: true
    expansion_variants: [rephrase, abstract]

  - name: B
    corpus_variant: declarative
    use_expansion: false

  - name: fused
    fusion: rrf
    source_configs: [A, A_expand, B]
    k: 60

baseline:
  path: experiments/07_.../golden_summaries.json
  variant: conceptual_abstract
```

### 4. Run

```bash
# Create new run
.venv/bin/python experiments/09_my_experiment/run_pipeline.py

# Continue from existing run dir (e.g., skip generation, only run retrieval)
.venv/bin/python experiments/09_my_experiment/run_pipeline.py --run-dir path/to/run

# Force re-run all steps
.venv/bin/python experiments/09_my_experiment/run_pipeline.py --force
```

---

## Config Reference

All fields below are set in `default_config.yaml`. Your experiment config overrides only what differs.

### Models

| Key | Default | Description |
|-----|---------|-------------|
| `corpus_generation_model` | `gemini-3-flash-preview` | Model for corpus summary generation |
| `query_expansion_model` | `gemini-3-flash-preview` | Model for query expansion generation |
| `embed_model` | `Linq-AI-Research/Linq-Embed-Mistral` | Sentence-transformer model for embeddings |
| `embed_query_instruction` | `"Given a text, retrieve..."` | E5-instruct query prefix |

### Scale & Steps

| Key | Default | Description |
|-----|---------|-------------|
| `n_fables` | `null` (= all 709) | Number of fables to use |
| `steps.generate_corpus_summaries` | `true` | Enable/disable step 1 |
| `steps.generate_query_expansions` | `true` | Enable/disable step 2 |
| `steps.run_retrieval_eval` | `true` | Enable/disable step 3 |
| `api_delay_seconds` | `0.5` | Sleep between LLM API calls |
| `cache_dir` | `null` (= `<run_dir>/embedding_cache/`) | Embedding cache location |

### Corpus Variants

Each entry in `corpus_variants` defines a way to summarize fables:

```yaml
corpus_variants:
  - name: my_style                          # unique name (used as key everywhere)
    prompt: "Inline prompt text"            # option 1: inline
    prompt_file: prompts/custom.txt         # option 2: file (relative to experiment dir)
    prompt_key: ground_truth_style          # option 3: key from lib/pipeline/prompts.py
    user_prompt_template: "Fable: {text}"   # optional (this is the default)
```

> **Priority:** `prompt` > `prompt_file` > `prompt_key`. Must have exactly one. Error if none.

### Query Expansion Variants

Same structure as corpus variants, but applied to moral texts:

```yaml
query_expansion_variants:
  - name: rephrase
    prompt: "Rephrase this moral..."
    user_prompt_template: "Moral: {text}"   # optional (this is the default)
```

### Retrieval Configs

Each entry defines one retrieval evaluation run:

```yaml
retrieval_configs:
  # Simple: embed corpus variant, compute cosine similarity
  - name: baseline
    corpus_variant: my_style
    use_expansion: false

  # With query expansion: max-score fusion of original + expanded queries
  - name: expanded
    corpus_variant: my_style
    use_expansion: true
    expansion_variants: [rephrase, abstract]   # which expansions to fuse

  # Fusion across multiple configs (RRF)
  - name: fused_all
    fusion: rrf
    source_configs: [baseline, expanded]       # names of other configs to fuse
    k: 60                                      # RRF k parameter
```

### Baseline (optional)

Compare all retrieval configs against a previous experiment's results:

```yaml
baseline:
  path: experiments/07_.../golden_summaries.json   # relative to project root
  variant: conceptual_abstract                      # which summary variant to use
```

---

## Shared Prompts (`prompts.py`)

Available prompt keys for `prompt_key` references:

| Key | Description |
|-----|-------------|
| `ground_truth_style` | Concise aphorism, 5-15 words |
| `declarative_universal` | Declarative sentence about human nature, 5-15 words |
| `moral_rephrase` | Same meaning, different words, ≤15 words |
| `moral_elaborate` | Broader principle, ≤20 words |
| `moral_abstract` | Most concise form, ≤10 words |

To add a new shared prompt, add an entry to `PROMPTS` in `lib/pipeline/prompts.py`.

---

## Output Files

Each pipeline run creates a timestamped directory:

```
experiments/<exp>/results/pipeline_runs/
  2026-04-05_10-00-00_sample10/
    run_manifest.json              # steps completed + config snapshot
    corpus_summaries.json          # fable summaries per variant
    token_usage.json               # LLM token costs for corpus generation
    query_expansions.json          # moral paraphrases per variant
    query_expansion_token_usage.json
    retrieval_results.json         # metrics per retrieval config
    embedding_cache/               # .npy files (reused across runs)
```

### Key Output: `retrieval_results.json`

```json
{
  "baseline": { "MRR": 0.45, "Recall@1": 0.40, "Recall@5": 0.70, ... },
  "A":        { "MRR": 0.52, "Recall@1": 0.48, ... },
  "A_expand": { "MRR": 0.58, "Recall@1": 0.55, ... },
  "RRF_all":  { "MRR": 0.61, "Recall@1": 0.58, ... }
}
```

---

## Tips

- **Reuse generation, re-run retrieval only:** Set `steps.generate_corpus_summaries: false` and `steps.generate_query_expansions: false`, then pass `--run-dir` pointing to an existing run.
- **Experiment with more fables:** Change `n_fables` (or set to `null` for all 709).
- **Different LLM model:** Override `corpus_generation_model` or `query_expansion_model`.
- **Custom user prompt format:** Set `user_prompt_template` on any variant (e.g., `"Story to analyze: {text}"`).
