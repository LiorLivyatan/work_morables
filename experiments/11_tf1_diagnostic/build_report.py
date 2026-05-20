"""
Generate a self-contained academic-style HTML report for the TF1-EN-3M
evaluation experiment. Embeds all figures as base64 so the file is
portable (single artifact, shareable as one attachment).

Run:
    ./run.sh experiments/11_tf1_diagnostic/build_report.py
    # writes:  experiments/11_tf1_diagnostic/report.html
"""
import base64
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
FIG_DIR = EXP_DIR / "results" / "runs" / "20260502_184314" / "figures"
OUT_PATH = EXP_DIR / "report.html"


def b64_img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def main() -> None:
    figs = {p.stem: b64_img(p) for p in sorted(FIG_DIR.glob("*.png"))}
    expected = {
        "01_plateau", "02_iou_distribution",
        "03_moral_frequency", "04_chunk_consistency", "05_summary_table",
    }
    missing = expected - set(figs)
    if missing:
        raise FileNotFoundError(f"missing figures: {missing}")

    now = datetime.now().strftime("%Y-%m-%d")

    body = BODY_TEMPLATE
    substitutions = {
        "%%DATE%%": now,
        "%%FIG_PLATEAU%%": figs["01_plateau"],
        "%%FIG_IOU%%": figs["02_iou_distribution"],
        "%%FIG_FREQ%%": figs["03_moral_frequency"],
        "%%FIG_CHUNKS%%": figs["04_chunk_consistency"],
        "%%FIG_SUMMARY%%": figs["05_summary_table"],
    }
    for placeholder, value in substitutions.items():
        body = body.replace(placeholder, value)
    html = HEAD_PART + CSS + STYLE_CLOSE + body
    OUT_PATH.write_text(html)
    print(f"wrote {OUT_PATH} ({len(html):,} chars)")


CSS = """
:root {
  --bg: #fbfaf6;
  --fg: #1a1a1a;
  --muted: #5a5a5a;
  --accent: #8b0000;
  --rule: #d0c8b0;
  --code-bg: #f3eee0;
  --finding-bg: #f0f7f0;
  --finding-border: #2e7d32;
  --question-bg: #e8f0fe;
  --question-border: #1565c0;
  --warning-bg: #fdedec;
  --warning-border: #c62828;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'ETBembo', 'Charter', 'Georgia', 'Cambria', serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
  font-size: 16px;
}
.container {
  max-width: 880px;
  margin: 60px auto;
  padding: 0 40px;
}
header {
  border-bottom: 2px solid var(--rule);
  padding-bottom: 30px;
  margin-bottom: 40px;
}
header h1 {
  font-size: 2.2em;
  margin: 0 0 6px 0;
  font-weight: 600;
  letter-spacing: -0.01em;
}
header .subtitle {
  font-size: 1.3em;
  color: var(--muted);
  font-style: italic;
  margin: 0 0 18px 0;
}
header .meta {
  font-size: 0.9em;
  color: var(--muted);
}
header .meta a { color: var(--accent); }
h2 {
  font-size: 1.6em;
  margin-top: 2.2em;
  margin-bottom: 0.6em;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--rule);
  font-weight: 600;
}
h3 {
  font-size: 1.2em;
  margin-top: 1.8em;
  margin-bottom: 0.4em;
  color: var(--fg);
  font-weight: 600;
}
h4 {
  font-size: 1.05em;
  margin-top: 1.4em;
  margin-bottom: 0.3em;
  color: var(--muted);
  font-weight: 600;
}
p { margin: 0.8em 0; }
ul, ol { margin: 0.8em 0; padding-left: 28px; }
li { margin: 0.3em 0; }
code, pre, kbd, samp {
  font-family: 'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', monospace;
  font-size: 0.86em;
}
code {
  background: var(--code-bg);
  padding: 2px 5px;
  border-radius: 3px;
  color: #5a2020;
}
pre {
  background: var(--code-bg);
  padding: 14px 18px;
  overflow-x: auto;
  border-left: 3px solid var(--accent);
  margin: 14px 0;
  border-radius: 0 4px 4px 0;
  line-height: 1.45;
}
pre code { background: none; padding: 0; color: var(--fg); }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(139, 0, 0, 0.3); }
a:hover { border-bottom: 1px solid var(--accent); }
figure {
  margin: 30px 0;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--rule);
  background: white;
  padding: 6px;
}
figcaption {
  font-style: italic;
  color: var(--muted);
  margin-top: 10px;
  font-size: 0.92em;
  padding: 0 30px;
  line-height: 1.5;
}
table {
  border-collapse: collapse;
  margin: 18px 0;
  font-size: 0.93em;
  width: 100%;
}
th, td {
  border: 1px solid var(--rule);
  padding: 8px 12px;
  text-align: left;
  vertical-align: top;
}
th { background: var(--code-bg); font-weight: 600; }
.callout {
  padding: 16px 22px;
  margin: 20px 0;
  border-left: 4px solid;
  border-radius: 0 4px 4px 0;
  font-size: 0.96em;
}
.callout p:first-child { margin-top: 0; }
.callout p:last-child { margin-bottom: 0; }
.callout .label {
  display: block;
  font-weight: 700;
  font-size: 0.78em;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.callout.finding { background: var(--finding-bg); border-color: var(--finding-border); }
.callout.finding .label { color: var(--finding-border); }
.callout.question { background: var(--question-bg); border-color: var(--question-border); }
.callout.question .label { color: var(--question-border); }
.callout.warning { background: var(--warning-bg); border-color: var(--warning-border); }
.callout.warning .label { color: var(--warning-border); }
.callout.note { background: #faf6ee; border-color: #8b6914; }
.callout.note .label { color: #8b6914; }
.callout.note h4 { margin-top: 1.2em; margin-bottom: 0.3em; color: #5a4408; }
.toc {
  background: #fff;
  padding: 18px 24px;
  border: 1px solid var(--rule);
  margin: 30px 0;
}
.toc h3 {
  margin: 0 0 10px 0;
  font-size: 1em;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}
.toc ol { margin: 0; padding-left: 24px; font-size: 0.95em; }
.toc a { color: var(--fg); border-bottom: none; }
.toc a:hover { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 40px 0; }
footer {
  margin-top: 60px;
  padding-top: 20px;
  border-top: 1px solid var(--rule);
  font-size: 0.85em;
  color: var(--muted);
}
@media print {
  body { background: white; }
  .container { max-width: none; margin: 0; padding: 20px; }
  figure { page-break-inside: avoid; }
  h2, h3 { page-break-after: avoid; }
  pre { white-space: pre-wrap; word-wrap: break-word; }
  a { color: var(--fg); border-bottom: none; }
}
"""


HEAD_PART = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TF1-EN-3M Evaluation &amp; Derived Synthetic Corpus &mdash; MORABLES</title>
<style>
"""

STYLE_CLOSE = """
</style>
</head>
<body>
<div class="container">"""

BODY_TEMPLATE = """

<header>
<h1>TF1-EN-3M Evaluation &amp; Derived Synthetic Corpus</h1>
<p class="subtitle">Fit-for-thesis analysis of a 3M-fable synthetic dataset and the construction of a MORABLES-shaped derivative for downstream retrieval experiments.</p>
<p class="meta">
Project: <strong>MORABLES</strong> (Hebrew University thesis &mdash; moral&ndash;fable retrieval) &middot; Branch: <code>tf1-iou-diagnostic</code> &middot; Date: %%DATE%%<br>
External resources:
<a href="https://huggingface.co/datasets/klusai/ds-tf1-en-3m">klusai/ds-tf1-en-3m</a> (HF dataset, MIT) &middot;
<a href="https://arxiv.org/abs/2504.20605">arXiv 2504.20605</a> (paper) &middot;
<a href="https://github.com/klusai/tinyfabulist">tinyfabulist</a> (generator, MIT)
</p>
</header>

<section id="abstract">
<h2>Abstract</h2>
<p>We evaluate <em>TF1-EN-3M</em>, a 3-million synthetic moral fable dataset generated by <code>meta-llama/Llama-3.1-8B-Instruct</code>, for fitness as training or evaluation data in the MORABLES moral&rarr;fable retrieval thesis. Two empirical questions are answered via a stratified 50,000-row diagnostic sample: (1) Does the LLM lexically leak moral wording into the generated fable, reducing the retrieval task to keyword matching? <strong>No</strong> &mdash; mean Jaccard IoU (no stopwords) is 0.022, only ~2&times; the MORABLES baseline of 0.011. (2) How many unique moral seeds does the corpus actually cover? <strong>Exactly 100 globally</strong> &mdash; verified by independently saturating the pool in every one of ten 5,000-row windows spread across the 3M dataset. We conclude that TF1-EN-3M is unsuitable as a drop-in replacement for MORABLES training data (the small fixed pool causes catastrophic in-batch positive collisions under <code>MultipleNegativesRankingLoss</code>), but is well-suited as a hard-negative pool, a generalization-eval target, and most importantly, as a pilot for using its open-source generator <code>tinyfabulist</code> directly with the 678 MORABLES morals as custom seeds. As a downstream artifact, we produce a MORABLES-shaped synthetic corpus at <code>data/external/tf1_synthetic/</code> with both flat and clustered retrieval qrels, built and validated by test-driven development across ten review-gated implementation tasks.</p>
</section>

<div class="toc">
<h3>Contents</h3>
<ol>
<li><a href="#introduction">Introduction &amp; Motivation</a></li>
<li><a href="#dataset">The Dataset and the Generator</a></li>
<li><a href="#research-questions">Research Questions</a></li>
<li><a href="#methodology">Methodology</a></li>
<li><a href="#results">Results</a></li>
<li><a href="#discussion">Discussion</a></li>
<li><a href="#derived-corpus">Derived Corpus: TF1-Synthetic</a></li>
<li><a href="#engineering">Engineering Process</a></li>
<li><a href="#future-work">Recommendations &amp; Future Work</a></li>
<li><a href="#open-questions">Open Questions for Team Discussion</a></li>
<li><a href="#references">References</a></li>
<li><a href="#appendix">Appendix</a></li>
</ol>
</div>

<section id="introduction">
<h2>1. Introduction &amp; Motivation</h2>

<p>The <strong>MORABLES</strong> thesis investigates contrastive bi-encoder retrieval where the query is a one-sentence moral (e.g., <em>"Gratitude is the sign of noble souls"</em>) and the document is a longer literary fable (e.g., the story of <em>Androcles and the Lion</em>). The corpus consists of 709 curated fable&ndash;moral pairs drawn from Aesop, Gibbs, Perry, and Abstemius; the central modeling challenge is that the lexical overlap between a moral and its fable is essentially zero (mean Jaccard IoU 0.011 over content words), which forces the model to learn semantic compression rather than keyword matching.</p>

<p>Two fine-tuning experiments have been completed: <code>ft_01_5fold_cv</code> with <code>BAAI/bge-base-en-v1.5</code> (110M parameters) and <code>ft_02_linq_5fold_cv</code> with <code>Linq-AI-Research/Linq-Embed-Mistral</code> (7B parameters with LoRA adapters). Both rely on <code>MultipleNegativesRankingLoss</code> over a single (moral, fable) positive pair, with negatives drawn implicitly from the rest of the batch. A third experiment <code>ft_03_hard_neg</code> adds explicit mined hard negatives derived from MORABLES' MCQA distractor morals.</p>

<p>A natural question, given the modest corpus size of 709 pairs, is whether a large-scale synthetic dataset could augment training. Our colleagues pointed us to <em>TF1-EN-3M</em>, a recently published 3,000,000-fable dataset (N&abreve;da&#537; et&nbsp;al., <a href="https://arxiv.org/abs/2504.20605">arXiv 2504.20605</a>) generated entirely by Llama-3.1-8B. This experiment evaluates whether and how that resource can be integrated into the thesis.</p>
</section>

<section id="dataset">
<h2>2. The Dataset and the Generator</h2>

<p>TF1-EN-3M ships as 2.8M training, 100K validation, and 100K test rows in parquet format. Each row records:</p>

<ul>
<li><code>prompt</code> &mdash; the structured generation instruction sent to Llama-3.1-8B, containing five narrative slots</li>
<li><code>fable</code> &mdash; the model's generated story (~250 words target)</li>
<li>generation metadata: model name, token counts, inference time, host GPU, cost-per-hour, ISO timestamp</li>
</ul>

<p>Critically, <strong>the dataset has no dedicated <code>moral</code> column</strong>. The moral text is embedded as one bullet (<code>- Teaching: &lt;moral&gt;</code>) inside the <code>prompt</code> field. Extraction is fully deterministic via the regex <code>r"-\\s*Teaching:\\s*(.+?)(?:\\n|$)"</code> and succeeded on 50,000 of 50,000 sampled rows in our diagnostic. The five narrative slots are <em>Main Character</em>, <em>Setting</em>, <em>Challenge</em>, <em>Outcome</em>, and <em>Teaching</em>; the LLM weaves these into prose.</p>

<p>The same authors released the generator, <a href="https://github.com/klusai/tinyfabulist"><code>tinyfabulist</code></a>, as an MIT-licensed Python tool. Its architecture is YAML-driven: the slot pools (including the 100 morals examined later in this report) live in <code>tinyfabulist/conf/generator_features.yaml</code>, and the prompt template (with Mustache-style <code>{moral}</code> substitution) lives in <code>tinyfabulist/conf/generator_prompts.yaml</code>. Generation is performed via an OpenAI-compatible HTTP client (<code>AsyncOpenAI</code>), which means the model backend is pluggable &mdash; the published config points at the authors' Hugging Face Inference Endpoints, but a self-hosted vLLM server or a local Ollama install can be substituted by changing a single <code>base_url</code> field.</p>
</section>

<section id="research-questions">
<h2>3. Research Questions</h2>

<p>We pose two empirical questions before considering integration with MORABLES training:</p>

<ol>
<li><strong>Lexical-leakage risk:</strong> Because the moral is a <em>seed</em> in the LLM's prompt rather than an interpretation extracted from a pre-existing narrative, a naive expectation is that Llama-3.1-8B will incorporate the moral's words directly into the fable body. If so, training on TF1 would teach an embedding model to match on shared vocabulary rather than on shared meaning &mdash; a regression relative to MORABLES' semantically-hard task. We measure this with content-word Jaccard IoU between each moral and its fable.</li>
<li><strong>Moral diversity:</strong> A 3M dataset would naively be expected to cover thousands of distinct moral concepts. We measure the actual number of unique moral phrases used as seeds, since a small fixed pool implies many fables per moral and would break standard contrastive losses through in-batch positive collisions.</li>
</ol>
</section>

<section id="methodology">
<h2>4. Methodology</h2>

<h3>4.1 Sampling Design</h3>
<p>We avoid downloading the full 13.4&nbsp;GB dataset by using Hugging Face's streaming interface. To prevent shard-locality bias from a single contiguous read, sampling is <strong>stratified</strong>: ten windows of 5,000 rows each are drawn at evenly-spaced offsets across the train split (offsets 0, 280K, 560K, &hellip;, 2.52M), yielding 50,000 total rows distributed across the entire 3M corpus. The stratification is essential for the diversity measurement &mdash; a contiguous read could find the same 100 morals because they all reside in adjacent shards, leaving the actual pool size ambiguous.</p>

<h3>4.2 IoU Computation</h3>
<p>To produce a directly comparable baseline, we recompute IoU on MORABLES inside the same script using the exact tokenizer (<code>r"\\b\\w+\\b"</code> lowercased) and stopword list (64 words) used in <code>scripts/02_eda.py</code>. For each (moral, fable) pair we compute Jaccard similarity twice: with all words, and with stopwords removed. The "no-stopwords" variant is the meaningful one for semantic comparison, since stopwords inflate overlap mechanically.</p>

<h3>4.3 Plateau Curve Tracking</h3>
<p>To detect whether the unique-morals count is still growing or has saturated, we record the cumulative unique count at log-spaced checkpoints during the early rows and linear (every-1000) checkpoints thereafter. A flat plateau is direct evidence that the pool has been fully observed.</p>

<h3>4.4 Tooling</h3>
<p>The diagnostic ships as two scripts under <code>experiments/11_tf1_diagnostic/</code>:</p>
<ul>
<li><code>check_iou.py</code> &mdash; stratified streaming, moral extraction, IoU computation, MORABLES baseline recomputation</li>
<li><code>make_report.py</code> &mdash; matplotlib figure generation from the diagnostic's <code>summary.json</code> output</li>
</ul>

<p>Reproduction:</p>
<pre><code>./run.sh experiments/11_tf1_diagnostic/check_iou.py --n 50000 --chunks 10
./run.sh experiments/11_tf1_diagnostic/make_report.py --run experiments/11_tf1_diagnostic/results/runs/&lt;timestamp&gt;</code></pre>
</section>

<section id="results">
<h2>5. Results</h2>

<h3>5.1 Finding 1: Lexical Overlap is Low</h3>

<div class="callout finding">
<span class="label">Finding 1</span>
<p>Mean content-word IoU between morals and fables in TF1 is 0.0218 (median 0.0211; 99th percentile 0.0538). This is approximately <strong>2&times; the MORABLES baseline of 0.0107</strong>, but both distributions remain well below 0.05, indicating that Llama-3.1-8B writes fables that are <em>narratively</em> rather than <em>lexically</em> aligned to the seed moral.</p>
</div>

<p>Inspection of randomly-sampled (moral, fable) pairs at low IoU confirms the model is producing genuine narrative embodiments rather than paraphrasing the moral in story form. Examples drawn from the 50,000-row sample include:</p>

<ul>
<li><em>Moral:</em> "perseverance triumphs over difficulty" &rarr; a horse exploring an abandoned temple (IoU 0.008, zero content-word overlap)</li>
<li><em>Moral:</em> "honesty is the best policy" &rarr; an ostrich steals an acorn and is scolded (IoU 0.008)</li>
<li><em>Moral:</em> "bravery arises from compassion" &rarr; a frog locates a hidden spring during a drought to save others (IoU 0.016)</li>
</ul>

<p>The distribution of IoU for both datasets is shown below.</p>

<figure>
<img src="%%FIG_IOU%%" alt="IoU distribution histograms for MORABLES vs TF1">
<figcaption><strong>Figure 1.</strong> Distribution of content-word Jaccard IoU between morals and their fables. MORABLES (red, n=709) is sharply concentrated near zero (median 0.000). TF1 (blue, n=50,000) is shifted slightly right but remains in the same low-overlap regime. Dashed vertical lines mark the means. Both distributions occupy IoU&nbsp;&lt;&nbsp;0.05, confirming that the moral&rarr;fable retrieval task remains semantic rather than lexical for both corpora.</figcaption>
</figure>

<p><strong>Implication:</strong> TF1 is a legitimate semantic retrieval signal. The pre-experiment concern that "the LLM was told the moral and will copy its words into the fable" is empirically refuted.</p>

<h3>5.2 Finding 2: The Moral Pool is Globally Fixed at 100</h3>

<div class="callout finding">
<span class="label">Finding 2</span>
<p>The entire 3,000,000-row dataset draws its <em>Teaching</em> slot from a fixed pool of exactly <strong>100 unique moral phrases</strong>. Each of these morals therefore has approximately 30,000 distinct fables generated for it. This pool is global (not shard-local), confirmed by ten independent stratified windows each saturating the same 100-moral set.</p>
</div>

<p>The cumulative count of unique morals as a function of rows streamed is plotted below. It saturates within the first chunk (the first 5,000 rows already contain all 100 morals) and never grows afterward, despite sampling 45,000 more rows from positions evenly distributed across the rest of the 3M dataset.</p>

<figure>
<img src="%%FIG_PLATEAU%%" alt="Plateau curve of unique morals discovered">
<figcaption><strong>Figure 2.</strong> Cumulative unique morals discovered as a function of rows streamed. Sampling is stratified across 10 chunks of 5,000 rows each (vertical gridlines mark chunk boundaries) spanning the full 2.8M training split. The curve plateaus at 100 within the first chunk and remains flat thereafter, confirming that the moral pool is exhaustively observed.</figcaption>
</figure>

<p>The within-pool distribution is approximately flat: the most frequent moral ("greed leads to downfall") appears 732 times in our 50,000-row sample, the least frequent ("second chances reveal new paths") 337 times &mdash; a 2.2&times; spread. The top 30 morals are shown below.</p>

<figure>
<img src="%%FIG_FREQ%%" alt="Top 30 moral frequencies">
<figcaption><strong>Figure 3.</strong> Top 30 of 100 unique morals by frequency in the 50,000-row stratified sample. The roughly flat distribution (counts ranging from 337 to 732) reflects the combinatorial sampling design of the <code>tinyfabulist</code> generator. Extrapolated to the full 3M corpus, each moral has ~30,000 associated fables.</figcaption>
</figure>

<p>That every chunk discovers exactly the same 100 morals &mdash; with the same morals dominating in every chunk &mdash; rules out any explanation other than a globally-shared seed pool. The heatmap below shows the per-chunk counts for the twelve most-frequent morals.</p>

<figure>
<img src="%%FIG_CHUNKS%%" alt="Per-chunk moral frequencies heatmap">
<figcaption><strong>Figure 4.</strong> Per-chunk counts for the twelve globally most-frequent morals (columns) across all ten stratified sampling windows (rows). The uniformity of the heatmap confirms a global, shard-independent moral pool. If the pool were shard-local, we would expect a block-diagonal pattern instead.</figcaption>
</figure>

<h3>5.3 Side-by-Side Comparison</h3>

<figure>
<img src="%%FIG_SUMMARY%%" alt="MORABLES vs TF1 side-by-side summary table">
<figcaption><strong>Figure 5.</strong> Summary of key metrics for MORABLES (literary, curated) versus TF1-EN-3M (synthetic, LLM-generated).</figcaption>
</figure>
</section>

<section id="discussion">
<h2>6. Discussion</h2>

<h3>6.1 Why TF1 Cannot Be a Drop-In Training Replacement</h3>

<p>The 100-moral pool is fatal for standard contrastive training. <code>MultipleNegativesRankingLoss</code> &mdash; used throughout our existing fine-tuning experiments &mdash; assumes that for each anchor in a batch, there is exactly one positive document, and all other documents in the batch are valid negatives. With 30,000 fables sharing each moral in TF1, the probability that a batch of 32 random examples contains at least two fables of the same moral is overwhelming. Concretely, with batch size 32 over a 100-class label space with approximately uniform class frequencies, the expected number of duplicate-class pairs per batch is approximately <em>32 &middot; 31 / 2 / 100 &asymp; 5</em>, meaning roughly 5 of the 32 anchors will have at least one false negative in the batch.</p>

<p>Three workarounds exist:</p>
<ol>
<li><strong>Moral-disjoint batch sampling</strong> &mdash; build batches that guarantee distinct morals; technically possible but destroys shuffling efficiency and limits effective batch size to 100.</li>
<li><strong>Multi-positive contrastive loss</strong> (SupCon, multi-positive InfoNCE) &mdash; treats same-moral fables as positives in the loss, properly handling the class structure. Requires a non-trivial trainer rewrite.</li>
<li><strong>Reframe as 100-way classification</strong> &mdash; train a softmax head over the 100 morals. Useful for certain analyses but discards the semantic-retrieval framing.</li>
</ol>

<h3>6.2 Why the Upstream Generator is the Right Answer</h3>

<p>The <code>tinyfabulist</code> generator that produced TF1-EN-3M accepts a custom moral list via a single YAML edit to <code>tinyfabulist/conf/generator_features.yaml</code>. By substituting the 100-phrase default list with our 678 MORABLES morals (or a curated subset), we can generate a targeted synthetic corpus whose moral distribution exactly matches the test set we care about &mdash; sidestepping every TF1-3M limitation while preserving the generator's measured quality. The pipeline is otherwise unchanged: the same Llama-3.1-8B model, the same prompt template, the same combinatorial slot mixing.</p>

<p>Backend choices for the generator:</p>
<table>
<thead>
<tr><th>Backend</th><th>Cost</th><th>Suitability</th></tr>
</thead>
<tbody>
<tr><td>Local Ollama (quantized Llama-3.1-8B)</td><td>Free (laptop)</td><td>~20s/fable; right for 100-fable validation spike</td></tr>
<tr><td>vLLM on the project's GPU server</td><td>GPU-time only</td><td>High throughput; right for real-scale (1K&ndash;50K fables)</td></tr>
<tr><td>Hugging Face Inference Endpoint</td><td>~$0.14 / 1K fables</td><td>Matches the paper's deployment; needs an HF billing account</td></tr>
<tr><td>OpenAI / Anthropic API</td><td>Per-token</td><td>Higher quality but loses apples-to-apples comparability with the paper</td></tr>
</tbody>
</table>

<h3>6.3 Recommendations Table</h3>

<table>
<thead><tr><th>Use of TF1-EN-3M</th><th>Verdict</th><th>Rationale</th></tr></thead>
<tbody>
<tr><td>Direct training-set replacement for MORABLES</td><td>Not recommended</td><td>100-moral pool causes contrastive loss collisions; cannot mimic 678-moral curated diversity.</td></tr>
<tr><td>Hard-negative pool for MORABLES training</td><td>Recommended</td><td>100 LLM-quality proverbs serve as drop-in distractors for the existing 2,803 augmented morals.</td></tr>
<tr><td>Generalization evaluation target</td><td>Recommended (with care)</td><td>Test whether MORABLES-trained models transfer to LLM-generated fables. Report Recall@k rather than MRR due to different pool shapes.</td></tr>
<tr><td>Pretraining via vanilla <code>MultipleNegativesRankingLoss</code></td><td>Not recommended</td><td>See loss-collision analysis above.</td></tr>
<tr><td>Pretraining via multi-positive contrastive loss</td><td>Conditional</td><td>Worth it only after a quality spike confirms the synthetic distribution helps.</td></tr>
<tr><td>Generating <em>custom</em> synthetic fables seeded from MORABLES morals via <code>tinyfabulist</code></td><td><strong>Primary recommendation</strong></td><td>Bypasses every TF1-3M limitation. ~$1 of compute for ~7K targeted fables.</td></tr>
</tbody>
</table>
</section>

<section id="derived-corpus">
<h2>7. Derived Corpus: TF1-Synthetic</h2>

<p>To enable downstream experiments without re-streaming the 3M every time, we produced a MORABLES-shaped derivative at <code>data/external/tf1_synthetic/</code>. The directory mirrors the established MORABLES <code>data/processed/</code> + <code>data/clustered/</code> layout, which means existing analysis and evaluation code can treat the synthetic corpus as a drop-in alternative target.</p>

<h3>7.1 Layout</h3>
<pre><code>data/external/tf1_synthetic/
  README.md                                       # provenance: HF id, paper, build params
  raw/                                            # gitignored (future stream cache)
  processed/                                      # mirrors data/processed/
    fables_corpus.json                            # 1000 entries (100 morals x N=10)
    morals_corpus.json                            # 100 entries
    qrels_moral_to_fable.json                     # 1000 rows (1:10)
    qrels_fable_to_moral.json                     # 1000 rows (1:1)
  clustered/                                      # mirrors data/clustered/, dual mode
    morals_unique_corpus_near.json                # 78 cluster reps (paraphrase merging)
    cluster_mapping_near.json                     # 78 clusters
    moral_to_cluster_near.json                    # 100 -&gt; 78 mapping
    qrels_moral_to_fable_clustered_near.json
    qrels_fable_to_moral_clustered_near.json
    morals_unique_corpus_exact.json               # 100 cluster reps (no merging)
    cluster_mapping_exact.json                    # 100 clusters (1 moral each)
    moral_to_cluster_exact.json                   # 100 -&gt; 100 identity mapping
    qrels_moral_to_fable_clustered_exact.json
    qrels_fable_to_moral_clustered_exact.json</code></pre>

<p>Both clustered modes are produced by the same script (<code>cluster_tf1_morals.py</code>) with a <code>--mode {near, exact}</code> flag. All output filenames carry the mode suffix so the two outputs can coexist; downstream consumers select the variant appropriate for their experiment.</p>

<h3>7.2 Build Pipeline</h3>

<p>Two scripts, each with full unit + integration tests:</p>

<table>
<thead><tr><th>Script</th><th>Input</th><th>Output</th></tr></thead>
<tbody>
<tr>
<td><code>build_tf1_corpus.py</code></td>
<td>Latest <code>samples.jsonl</code> from the diagnostic; <code>--n</code> (default 10), <code>--seed</code> (42)</td>
<td>The four <code>processed/*.json</code> files + auto-written <code>README.md</code></td>
</tr>
<tr>
<td><code>cluster_tf1_morals.py</code></td>
<td><code>processed/morals_corpus.json</code>; embedder (<code>BAAI/bge-large-en-v1.5</code>); <code>--threshold</code> (default 0.80)</td>
<td>The five <code>clustered/*.json</code> files + side-by-side inspection dumps at thresholds 0.80, 0.85, 0.90</td>
</tr>
</tbody>
</table>

<p>The build script discovered a subtle invariant violation in the diagnostic's cached samples: each unique <code>prompt_hash</code> appeared in five chunks because HF parquet streaming's <code>skip()</code> over large offsets re-reads earlier shards. A <code>dedup_by_prompt_hash()</code> step was added to preserve the contract that every fable in the corpus is unique by source identity. The 50,000 raw samples collapse to 10,000 unique fables (100 morals &times; ~100 unique fables per moral), comfortably exceeding the N=10 build target.</p>

<h3>7.3 Clustering Output</h3>

<h4>Near mode (semantic paraphrase merging)</h4>

<p>Clustering the 100 morals with BGE-large embeddings + single-linkage agglomerative clustering at cosine threshold 0.80 yields <strong>78 clusters</strong> (66 singletons, 12 near-paraphrase groups, 0 exact duplicates). The largest cluster groups 8 paraphrases of generosity/kindness:</p>

<pre><code>cluster_000 (near, 8 morals, 80 fables)
  - generosity overcomes envy
  - a small act of kindness can make a big difference
  - kindness transcends boundaries
  - generosity enriches the giver
  - generosity quells greed
  - kindness is rewarded
  - selflessness inspires others
  - small deeds can spark great change</code></pre>

<p>Other notable clusters include "acceptance/harmony" (4 morals, 40 fables), "empathy" (3 morals, 30 fables), and "perseverance" (3 morals, 30 fables). The side-by-side threshold inspection confirms monotonic behavior: 0.80 &rarr; 78 clusters (most merging), 0.85 &rarr; 93 clusters, 0.90 &rarr; 99 clusters (almost all singletons).</p>

<h4>Exact mode (identity, no merging)</h4>

<p>The <code>--mode exact</code> path skips embedding and clustering entirely: each of the 100 morals becomes its own cluster, with cluster type labeled <code>"exact"</code>. The resulting <code>cluster_mapping_exact.json</code> contains exactly <strong>100 clusters, each with 1 moral and 10 fables, totaling 1000 fables</strong> &mdash; the trivial 1-to-1 baseline against which paraphrase-merged variants can be compared. This mode is the right choice when downstream code wants the corpus-shape mirror without any semantic grouping decisions baked in (for example, when the experimental hypothesis itself concerns whether paraphrase merging helps or hurts retrieval quality).</p>

<table>
<thead><tr><th>Mode</th><th>Clusters</th><th>Type distribution</th><th>Morals per cluster</th><th>Fables per cluster</th></tr></thead>
<tbody>
<tr><td><code>near</code> (threshold 0.80)</td><td>78</td><td>66 singleton + 12 near + 0 exact</td><td>min 1, max 8 (mean 1.28)</td><td>min 10, max 80</td></tr>
<tr><td><code>exact</code></td><td>100</td><td>100 exact</td><td>1 (all)</td><td>10 (all)</td></tr>
</tbody>
</table>

<h3>7.4 Research Note &mdash; Does N=10 Fit the Current Training Setup?</h3>

<div class="callout note">
<span class="label">Research note</span>
<p><strong>Short answer: not for the current ft_01 / ft_02 loss, but it is the right starting point for the ft_03-style hard-negative trainer once we adapt it for cluster-aware sampling.</strong> The choice of N has two distinct implications &mdash; one statistical (does it give enough volume?) and one structural (does it interact correctly with the loss function?). The structural problem dominates.</p>

<h4>The contrastive-loss collision is unchanged by N</h4>
<p>Our current production loss, <code>MultipleNegativesRankingLoss</code> (used in <code>ft_01_5fold_cv/train.py</code> and <code>ft_02_linq_5fold_cv/train.py</code>), assumes <em>exactly one positive document per anchor in the batch</em>; every other in-batch fable is treated as a negative. For a batch of size B drawn uniformly at random from a corpus with C distinct moral classes, the expected number of within-batch class collisions (anchors with at least one false negative) is approximately:</p>
<pre><code>E[collisions] &asymp; B &middot; (B&minus;1) / (2&middot;C)</code></pre>
<p>For our standard ft_01 batch size B=32 and our TF1-synthetic class count C=100 (or C=78 after clustering), this gives <strong>~5 collisions per batch</strong> &mdash; regardless of whether we set N=1, N=10, or N=200. The number of fables per moral does not enter this formula because uniform sampling from C classes is what drives the collision rate; the per-class population only matters once we exceed C in the batch.</p>
<p>Therefore: if our training stays on <code>MultipleNegativesRankingLoss</code> as-is, <strong>increasing N does not help</strong>, and any N&gt;1 means we are knowingly training with a corrupted gradient signal. The only fix at the loss level is a multi-positive contrastive loss (SupCon / multi-positive InfoNCE) or moral-disjoint batch sampling.</p>

<h4>Volume comparison vs MORABLES</h4>
<table>
<thead><tr><th>Configuration</th><th>Total (moral, fable) pairs</th><th>Multiple of MORABLES</th></tr></thead>
<tbody>
<tr><td>MORABLES (current)</td><td>709</td><td>1.0&times;</td></tr>
<tr><td>TF1-synthetic at <strong>N=1</strong></td><td>100</td><td>0.14&times;</td></tr>
<tr><td>TF1-synthetic at <strong>N=10</strong> (current)</td><td>1,000</td><td>1.4&times;</td></tr>
<tr><td>TF1-synthetic at N=50</td><td>5,000</td><td>7.0&times;</td></tr>
<tr><td>TF1-synthetic at N=100 (max from our cache after dedup)</td><td>10,000</td><td>14&times;</td></tr>
</tbody>
</table>
<p>At N=10 we add 1,000 synthetic pairs on top of 709 literary ones &mdash; volume-wise this is augmentation, not pretraining. Meaningful pretraining-scale augmentation would require N=50&ndash;100 (5K&ndash;10K pairs).</p>

<h4>Compatibility by trainer</h4>
<ul>
<li><strong>ft_01 / ft_02 (single-positive in-batch MNR loss):</strong> Not compatible at any N&gt;1 due to the collision math above. Even N=1 would only deliver 100 training pairs, which is too small for the eval splits to be meaningful.</li>
<li><strong>ft_03_hard_neg (explicit triplets with mined hard negatives):</strong> <em>Compatible at N=10</em>, provided the hard-negative sampler is changed to draw from outside the anchor's <em>cluster</em> rather than just outside its moral. Each of the 1,000 (moral, positive) rows would then yield K explicit triplets, and the loss is well-defined regardless of how many siblings the moral has.</li>
<li><strong>New training experiment with multi-positive loss:</strong> Compatible at any N. This is the only path that fully exploits the 30,000-fables-per-moral richness available in the upstream TF1-3M.</li>
</ul>

<h4>Recommendation</h4>
<p>Hold N=10 for the immediate pipeline-validation spike (verifies that the build &rarr; cluster &rarr; train flow works end-to-end without bugs), then re-cut at N=50&ndash;100 once we commit to a specific training experiment:</p>
<ol>
<li><strong>If we extend ft_03 with cluster-aware hard-negative sampling</strong> (most likely path, smallest delta to existing trainer): rebuild at <strong>N=50</strong>. Gives ~5K training rows, each yielding K explicit hard negatives drawn from outside-cluster fables.</li>
<li><strong>If we adopt a multi-positive contrastive loss</strong> (larger trainer change, but fully exploits the synthetic richness): rebuild at <strong>N=100</strong>, since multi-positive losses benefit from larger per-class populations.</li>
<li><strong>If the team prefers a minimalist hard-negative-only experiment</strong> (no augmentation, just enrich MORABLES negatives): use <strong>N=1</strong> &mdash; one synthetic neighbor per cluster &mdash; with the TF1 morals serving solely as distractors during MORABLES training.</li>
</ol>
<p>The build script accepts <code>--n</code> as a parameter precisely so this decision can be deferred and re-applied cheaply (build time at N=100 is &lt;5 seconds; cluster step is unchanged). The current N=10 artifact remains useful for the validation spike regardless of which training path the team picks.</p>
</div>
</section>

<section id="engineering">
<h2>8. Engineering Process</h2>

<p>The implementation followed a strict test-driven, review-gated workflow facilitated by an agentic toolchain (Anthropic Claude Code with the <code>superpowers</code> skills library). Documentation was produced at three levels of formality:</p>

<ol>
<li><strong>Design spec</strong> at <code>docs/superpowers/specs/2026-05-06-tf1-synthetic-corpus-design.md</code> &mdash; outlines layout, schemas, script contracts, reproduction commands, and explicitly parks downstream work for follow-up specs.</li>
<li><strong>Implementation plan</strong> at <code>docs/superpowers/plans/2026-05-06-tf1-synthetic-corpus.md</code> &mdash; ten TDD-structured tasks, each with checkbox steps (test &rarr; fail &rarr; implement &rarr; pass &rarr; commit).</li>
<li><strong>Code</strong> at <code>experiments/11_tf1_diagnostic/</code> with mirrored tests at <code>tests/experiments/</code>.</li>
</ol>

<p>Each implementation task was completed by a fresh subagent, then verified by two further subagent reviews (spec compliance, then code quality). Eight commits across the ten tasks contain only the implementation; an additional three commits address review-driven fixes (notably the convert-asserts-to-ValueError change for user-facing validation, and the require-all-pair-similarity change in the "exact" cluster classifier). Final test coverage: <strong>27 tests, all passing</strong> (14 for the build script, 13 for the cluster script). The branch contains 17 commits total and was verified merge-ready by a final whole-implementation review.</p>

<p>An unrelated but synced upstream change &mdash; the new <code>08_concept_steering</code> experiment &mdash; was merged from <code>origin/main</code> twice during the work without conflicts.</p>
</section>

<section id="future-work">
<h2>9. Recommendations &amp; Future Work</h2>

<p>The diagnostic plus derived corpus enable three concrete follow-up experiments, ordered by priority:</p>

<h3>Spike A &mdash; tinyfabulist with custom MORABLES morals (highest priority)</h3>
<p>Vendor the <code>tinyfabulist</code> generator into <code>experiments/12_tinyfabulist_spike/</code>; replace the 100-phrase moral list in <code>generator_features.yaml</code> with our 678 MORABLES morals; generate 100 sample fables via local Ollama; run the same IoU diagnostic to confirm the lexical-overlap property is preserved with the new seeds; manually inspect 20 random outputs for narrative coherence. If quality is acceptable (mean IoU &lt; 0.04; &ge; 80% on-target fables), scale to 10&ndash;50 fables per moral (~7K&ndash;34K total) via vLLM on the GPU server. Estimated cost: ~$1 of HF endpoint compute or zero on the GPU server.</p>

<h3>Spike B &mdash; TF1 morals as hard negatives for MORABLES</h3>
<p>Embed all 100 TF1 morals once with the existing fine-tuned BGE/Linq checkpoint. For each MORABLES batch during training, retrieve the K nearest TF1 morals as additional in-batch negatives. Compare MRR@10 across K &isin; {1, 3, 5} against the current ft_01/ft_02 baselines. A pre-experiment sanity check should first measure the mean cosine similarity between TF1 and MORABLES morals to ensure the negatives are sufficiently challenging.</p>

<h3>Spike C &mdash; Cross-domain generalization probe</h3>
<p>Identify the TF1 cluster representatives semantically closest to MORABLES morals via cosine similarity. Generate or sample ~50 TF1 fables for each (~1.5K eval pairs). Run inference with the existing ft_01/ft_02 checkpoints; report MRR@10 against the TF1 sub-corpus. Drop in an ablation: how does MRR change as the eval is restricted to TF1 morals that are <em>not</em> close to any MORABLES moral? This produces a publishable finding either way: positive transfer (literary&rarr;synthetic embeddings generalize) or negative (the literary distribution is genre-locked).</p>
</section>

<section id="open-questions">
<h2>10. Open Questions for Team Discussion</h2>

<p>Several design decisions for the next implementation phase remain open. These are listed here for team review.</p>

<div class="callout question">
<span class="label">Q1 &mdash; Negative-sampling dataset format</span>
<p>The next deliverable is a dataset of (moral, ground-truth fable, K&times; negative fables) tuples for training experiments that go beyond in-batch implicit negatives. Three candidate formats exist:</p>
<ol>
<li>Flattened triplets matching <code>ft_03_hard_neg</code>'s <code>(anchor, positive, negative)</code> row shape, multiple rows per moral. <strong>Pro:</strong> Drop-in reuse of the existing trainer. <strong>Con:</strong> Implicit "list of negatives" requires reading row groups.</li>
<li>Single-row <code>{moral, positive_fable, negative_fables: [&hellip;]}</code>. <strong>Pro:</strong> Clearer for losses that consume multiple negatives at once (multi-positive InfoNCE). <strong>Con:</strong> Requires a small trainer adapter.</li>
<li>MCQA-style <code>{moral, choices: [gt, neg_1, &hellip;], correct_idx: 0}</code>. <strong>Pro:</strong> Symmetric to <code>data/raw/mcqa.json</code>; natural for retrieval-as-classification eval. <strong>Con:</strong> Less natural for contrastive training.</li>
</ol>
</div>

<div class="callout question">
<span class="label">Q2 &mdash; Negative selection strategy</span>
<p>The original requirement &mdash; "negatives must not come from the N-1 other fables that share the same moral" &mdash; naturally extends, given clustering, to "negatives must not come from the same <em>cluster</em>". Within that constraint, two policies are possible:</p>
<ol>
<li><strong>Random outside-cluster sampling.</strong> Uniform draw from fables in clusters other than the anchor's. Simplest, no embedding work needed.</li>
<li><strong>Hard outside-cluster sampling.</strong> Embed all fables; for each anchor, pick the K fables outside the cluster that are <em>most similar</em> to the positive (or the anchor). Harder negatives, but adds an embedding pass and a tuning knob (K).</li>
</ol>
<p>Hard negatives are usually preferred in modern retrieval recipes, but the additional complexity may not be justified for the initial spike.</p>
</div>

<div class="callout question">
<span class="label">Q3 &mdash; Cluster threshold calibration</span>
<p>The MORABLES clustering pipeline used a human-in-the-loop review tool (<code>analysis/cluster_review.py</code>) to refine candidate clusters generated at cosine threshold 0.80. We skipped human review for TF1 (only 100 morals, manual scan is fast) and shipped the auto-clustered output at 0.80, yielding 78 clusters with 12 well-formed paraphrase groups. The threshold is calibratable: 0.75 would merge more aggressively, 0.85 less. Should we (a) accept 0.80 and proceed, (b) run a brief manual review and re-emit, or (c) defer the threshold decision until a specific downstream consumer can quantify the cost of over- vs under-merging?</p>
</div>

<div class="callout question">
<span class="label">Q4 &mdash; Branch finalization</span>
<p>The <code>tf1-iou-diagnostic</code> branch has 17 commits, 27 passing tests, and is verified merge-ready. Push access to the upstream <code>LiorLivyatan/work_morables</code> remote is currently denied for the working account. Decision required: (a) request collaborator access on the upstream and push directly, (b) fork to a personal GitHub account and open a PR back, or (c) other distribution mechanism (private mirror, patch file).</p>
</div>

<div class="callout question">
<span class="label">Q5 &mdash; Scope of the tinyfabulist spike</span>
<p>If the team agrees to proceed with Spike A (custom-seed generation), there are two scope decisions:</p>
<ol>
<li><strong>Seed list:</strong> all 678 MORABLES morals, or only the cluster representatives (~558)? The latter avoids generating duplicate fables for paraphrased seeds.</li>
<li><strong>Volume per moral:</strong> N=1 (matches MORABLES 1:1 ratio for direct augmentation) or N=10&ndash;50 (provides enough redundancy for held-out validation within the same moral concept)?</li>
</ol>
</div>
</section>

<section id="references">
<h2>11. References</h2>

<ol>
<li>N&abreve;da&#537;, M., Dio&#537;an, L., Tomescu, A., &amp; Pi&#537;coran, A. (2025). <em>TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small, Open Language Models.</em> arXiv preprint <a href="https://arxiv.org/abs/2504.20605">arXiv:2504.20605</a>.</li>
<li>klusai. <em>ds-tf1-en-3m</em>. Hugging Face Datasets, 2025. <a href="https://huggingface.co/datasets/klusai/ds-tf1-en-3m">https://huggingface.co/datasets/klusai/ds-tf1-en-3m</a> (MIT license).</li>
<li>klusai. <em>tinyfabulist</em>. GitHub, 2025. <a href="https://github.com/klusai/tinyfabulist">https://github.com/klusai/tinyfabulist</a> (MIT license).</li>
<li>Xiao, S., et al. <em>BGE: BAAI General Embedding.</em> 2024. <a href="https://huggingface.co/BAAI/bge-large-en-v1.5">BAAI/bge-large-en-v1.5</a>.</li>
<li>Linq AI Research. <em>Linq-Embed-Mistral.</em> 2024. <a href="https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral">Hugging Face model card</a>.</li>
<li>Reimers, N., &amp; Gurevych, I. <em>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.</em> EMNLP 2019.</li>
<li>Karpukhin, V., et al. <em>Dense Passage Retrieval for Open-Domain Question Answering.</em> EMNLP 2020.</li>
<li>Khosla, P., et al. <em>Supervised Contrastive Learning.</em> NeurIPS 2020. (SupCon &mdash; multi-positive contrastive loss referenced in &sect;6.1.)</li>
</ol>
</section>

<section id="appendix">
<h2>Appendix</h2>

<h3>A. File Inventory (this experiment)</h3>
<pre><code>experiments/11_tf1_diagnostic/
  check_iou.py                     diagnostic: stratified IoU + plateau curve
  make_report.py                   matplotlib figure generator
  build_tf1_corpus.py              build script for processed/
  cluster_tf1_morals.py            build script for clustered/
  build_report.py                  this HTML report's generator
  REPORT.md                        markdown writeup (developer-facing)
  report.html                      this document
  results/runs/20260502_184314/    diagnostic outputs (gitignored)
    summary.json
    samples.jsonl
    examples.md
    figures/*.png

data/external/tf1_synthetic/
  README.md                        provenance + build snapshot
  processed/                       4 files: morals, fables, 2x qrels
  clustered/                       5 files: unique morals, mapping, 2x qrels, moral_to_cluster

tests/experiments/
  test_build_tf1_corpus.py         14 unit + integration tests
  test_cluster_tf1_morals.py       13 unit + integration tests

docs/superpowers/
  specs/2026-05-06-tf1-synthetic-corpus-design.md
  plans/2026-05-06-tf1-synthetic-corpus.md</code></pre>

<h3>B. Commit History (branch tf1-iou-diagnostic)</h3>
<pre><code>chore(tf1_synthetic): add Telegram notifications and gitignore cluster_inspection
data(tf1_synthetic): clustered corpus at threshold=0.80
feat(tf1_synthetic): wire cluster_tf1_morals.main with embedder + inspection dumps
feat(tf1_synthetic): assemble all 5 clustered output files
fix(tf1_synthetic): require all-pair similarity for 'exact' cluster + edge tests
feat(tf1_synthetic): add cluster_tf1_morals helpers (cluster/classify/canonical)
data(tf1_synthetic): processed corpus at N=10 (1000 fables, 100 morals)
fix(tf1_synthetic): raise ValueError for input validation, defensive mkdir
feat(tf1_synthetic): wire build_tf1_corpus.main with end-to-end run_build
fix(tf1_synthetic): assert build_fables_corpus row count and test text/chunk passthrough
feat(tf1_synthetic): add corpus + qrels assemblers for build script
feat(tf1_synthetic): add build_tf1_corpus helpers (group/order/ids/sample)
chore(tf1_synthetic): gitignore raw/ and add tests/experiments package
docs(plan): add implementation plan for TF1-synthetic corpus build
docs(spec): add design spec for TF1-synthetic corpus build
feat(experiments): add TF1-EN-3M diagnostic and evaluation report</code></pre>

<h3>C. Reproduction Commands</h3>
<pre><code># 1. Diagnostic (produces samples.jsonl + summary.json)
./run.sh experiments/11_tf1_diagnostic/check_iou.py --n 50000 --chunks 10

# 2. Figures (writes PNGs under figures/)
./run.sh experiments/11_tf1_diagnostic/make_report.py \\
    --run experiments/11_tf1_diagnostic/results/runs/&lt;timestamp&gt;

# 3. Build processed corpus
./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --n 10

# 4a. Build clustered corpus - near mode (semantic paraphrase merging, BGE-large)
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --mode near --threshold 0.80

# 4b. Build clustered corpus - exact mode (no merging, 1-cluster-per-moral baseline)
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --mode exact

# 5. Regenerate this HTML report
./run.sh experiments/11_tf1_diagnostic/build_report.py

# 6. Run all tests
uv run pytest tests/experiments/ -v</code></pre>
</section>

<footer>
<p>Compiled %%DATE%% from <code>experiments/11_tf1_diagnostic/build_report.py</code> against branch <code>tf1-iou-diagnostic</code>. Figures embedded as base64 PNG; report is self-contained. To regenerate: <code>./run.sh experiments/11_tf1_diagnostic/build_report.py</code>.</p>
</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
