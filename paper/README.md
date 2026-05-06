# ParabeLink — Paper Writing Guide

## What This Paper Is

**ParabeLink** is a retrieval benchmark investigating whether embedding models can perform abstract moral reasoning: given a short moral lesson, can a system retrieve the fable that teaches it from a corpus of 709 candidates (MORABLES dataset, Marcuzzo et al., EMNLP 2025)?

The core finding is a hard gap between current models and an oracle ceiling, and a systematic exploration of what closes it — summarization, fine-tuning with contrastive loss (STORAL augmentation), symmetric representation, and query expansion.

---

## Submission Target

| Property | Value |
|---|---|
| Venue | ACL Rolling Review → ARR May 2026 (targeting ACL 2026) |
| Paper type | Long paper |
| Page limit | **8 pages** main body + unlimited references |
| Format | Standard `*ACL` two-column LaTeX |
| Review mode | `\usepackage[review]{acl}` (adds line numbers, anonymizes) |
| Citation style | natbib — `\citet{}`, `\citep{}`, `\citealp{}`, `\citeposs{}` |
| Required sections | Abstract, Introduction, [core sections], **Limitations**, References |
| Optional sections | Ethics Statement, Acknowledgments, Appendices |

---

## Planned Paper Structure

```
Abstract                      (~150–180 words)
1  Introduction               (~1.5 columns)
2  Related Work               (~1.5 columns, 3 subsections)
3  The ParabeLink Benchmark   (~2 columns)
   3.1 Problem Formulation
   3.2 Dataset
4  Experiments                (~1.5 columns)
   4.1 Evaluation Protocol
   4.2 Models & Baselines
5  Results and Discussion     (~2 columns)
   5.1 Main Results
   5.2 Analysis
6  Conclusions                (~0.5 column)
Limitations
Ethics Statement
References
Appendix (hyperparams, full results tables, prompts)
```

---

## Style Guide (distilled from ID10M-JAM and IdioLink)

Both reference papers are from the same lab. Match their conventions exactly.

### Title
Format: `Name: Descriptive Subtitle`
Examples: *"ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context"*, *"IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions"*

### Abstract
- Opens on the broad challenge (1 sentence), then the specific gap (1 sentence)
- Introduces the dataset/benchmark name in **bold**
- States contributions and key finding concisely
- No citations in abstract

### Introduction
Paragraph flow:
1. Broad domain framing with dense citations
2. Narrow to the specific problem / gap
3. Why it's hard / what's missing
4. "In this work, we introduce **ParabeLink**, a benchmark..." — introduce with bold name
5. "Building on [MORABLES dataset]..."
6. Numbered contribution list: "Our contributions are threefold: 1. We introduce... 2. We provide... 3. We show..."

### Related Work
- Numbered subsections (e.g., 2.1, 2.2, 2.3)
- Dense inline citations with `\citet` / `\citep`
- Each subsection ends with a sentence positioning this work relative to prior art

Likely subsections:
- 2.1 Abstract Moral Reasoning in NLP
- 2.2 Semantic Retrieval and Embedding Models
- 2.3 Synthetic Data for Retrieval

### Main Benchmark Section
- Opens with a **Problem Formulation** subsection (formal notation, task definition)
- Uses bold inline subheadings within paragraphs: **Dataset definition.** **Data generation.** **Human validation.**
- Figure 1 = illustrative example of the query–document mismatch

### Experiments / Results
- Bold bolded finding headers: **First, [finding].** followed by 1–2 paragraphs of evidence
- Numbered findings: First, Second, Third, Fourth
- Results tables: caption **below**, best result in **bold**, ↓/↑ for metric direction, ± for std
- End Results with: **Summary of key observations.** as a bulleted list

### Conclusions
- Opens: "We introduce **ParabeLink**, ..."
- Recaps each contribution in 1 sentence
- Closes with broader implications / future directions

### Limitations
- Unnumbered section (no section number)
- Honest about scope: dataset size, English-only, benchmark vs. real-world deployment, etc.

### Ethics Statement
- Unnumbered section
- Brief: dataset is publicly available, no PII, research purpose only

---

## Language and Wording Conventions

| Convention | Rule |
|---|---|
| Tense | Present for describing this paper's work; past for prior work |
| Voice | Active preferred: "We evaluate", "We introduce", "We show" |
| Benchmark name | **ParabeLink** in bold on first mention per section |
| "in order to" | Replace with "to" |
| Em-dash | Use `---` for asides (no spaces) |
| Numerals | Spell out one–nine; use digits for 10+ |
| Percent | Use % symbol with no space: "70%" |
| Citations in-text | `\citet{X}` → "Marcuzzo et al. (2025)" |
| Citations parenthetical | `\citep{X}` → "(Marcuzzo et al., 2025)" |
| Avoid | "in this paper we show" (redundant) — just show it |
| Model names | Linq-Embed-Mistral, BGE-base, STORAL — exact capitalization |

### Signature patterns from the example papers
```
"X pose[s] a [persistent/fundamental/longstanding] challenge for..."
"We introduce X, a [type] designed to..."
"Unlike [prior work], which [did Y], our [approach]..."
"Building on [X], we [contribution]."
"Our results reveal that..."
"[Finding]. This suggests that..."
"Notably, [surprising observation]."
"Taken together, these findings indicate that..."
```

---

## Writing Workflow

We write **section by section**, in this order:

1. Abstract (after all sections drafted — written last but placed first)
2. Introduction
3. Related Work
4. The ParabeLink Benchmark (§3)
5. Experiments (§4)
6. Results and Discussion (§5)
7. Conclusions
8. Limitations + Ethics Statement
9. Abstract (final revision)
10. Appendices

Each session: pick one section, draft it fully, then revise together.

---

## Key Numbers to Know

| Metric | Value |
|---|---|
| Corpus size | 709 fable–moral pairs |
| Avg fable length | ~133 words |
| Avg moral length | ~12 words |
| Lexical overlap (IoU) | 0.011 (near zero) |
| Random baseline R@1 | 0.14% |
| Oracle ceiling R@1 | 82.7% |
| Best off-the-shelf MRR | 0.210 (Linq-Embed-Mistral) |
| Best summarization R@1 | 26.5% (Gemini, full 709) |
| Best pilot result R@1 | 70.0% (symmetric + QE, 50-fable pilot) |

---

## Files in This Directory

| File | Purpose |
|---|---|
| `ParabeLink___Lior_ARR_May_2026.pdf` | ACL LaTeX formatting instructions / submission template |
| `Dec_25_Kai_ID10M_JAM (2).pdf` | Style reference — same lab, adversarial benchmark paper |
| `Kai_IdioLink_Jan_25 (1).pdf` | Style reference — same lab, retrieval benchmark paper |
| `README.md` | This file |
