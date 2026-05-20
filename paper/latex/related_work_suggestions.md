# Related Work Suggestions

This note collects proposed edits for the Related Work section and a complete
rephrased version that keeps the same thesis and wording style as the
Introduction.

## Main Positioning

The Related Work should keep serving this claim:

**MoraLink is not mainly a moral-judgment benchmark. It is a retrieval
benchmark for semantic-level alignment between abstract principles and concrete
narratives.**

The section should therefore avoid becoming a broad survey of moral reasoning.
Each paragraph should explain how prior work relates to one of three ideas:

- moral or narrative understanding;
- semantic retrieval beyond surface similarity;
- representation interventions for retrieval under abstraction mismatch.

## What To Keep

- Keep ETHICS and Delphi, but only as background for moral reasoning in NLP.
  They test moral judgment over explicit situations, while MoraLink tests
  retrieval between abstract morals and implicit narrative realizations.

- Keep MORABLES and STORAL. These are the closest narrative/moral datasets.
  STORAL studies story-moral understanding and generation. MORABLES is the
  direct source dataset, but its task is multiple-choice moral selection, not
  open retrieval.

- Keep dense retrieval, BEIR, MTEB, SBERT, Contriever, E5, BGE, and
  Linq-Embed-Mistral. This connects MoraLink to embedding benchmarks and
  makes the failure of strong retrievers meaningful.

- Keep HyDE, Query2doc, InPars, and synthetic embedding data, but frame them as
  LLM-assisted retrieval alignment. The important contrast is that MoraLink
  uses document-side abstraction: fables are transformed into moral-level
  representations.

- Keep contrastive fine-tuning and hard negatives, but focus on the distinction
  that MoraLink's hard negatives are conceptually close but morally wrong,
  rather than merely lexically or topically similar.

## What To Cut Or Shrink

- Shrink the multimodal moral storytelling paragraph about Tales. It is relevant
  to moral narrative generation, but not central to retrieval. One sentence is
  enough, or it can be omitted.

- Do not expand figurative language too much. It is useful as motivation, but
  fables and parables are not exactly idioms or metaphors. Mention figurative
  language briefly only if it helps support the broader abstraction argument.

- Remove `\citep{moralink}` unless MoraLink refers to a prior released
  version. In this paper, we can just say "our work" or "MoraLink" without
  citing ourselves.

- Consider adding BRIGHT, because it shows recent interest in retrieval
  benchmarks that go beyond surface matching and require reasoning. It is not
  exactly the same problem, but it gives useful context for reasoning-intensive
  retrieval.

## Suggested Structure

Use three subsections:

```latex
\subsection{Moral and Narrative Understanding}
\subsection{Semantic Retrieval Beyond Surface Similarity}
\subsection{LLM-Augmented Retrieval and Task-Specific Adaptation}
```

This is cleaner than separating contrastive fine-tuning from retrieval
augmentation, because in MoraLink both are interventions for the same
abstraction-gap problem.

## Citation And Bib Notes

The current `references.bib` already uses several keys that differ from the
earlier draft:

- Use `hendrycks2021aligning`, not `hendrycks2021ethics`.
- Use `karpukhin2020dense`, not `karpukhin2020dpr`.
- Use `izacard2022unsupervised`, not `izacard2022contriever`.
- Use `wang2022text`, not `wang2022e5`.
- Use `xiao2023cpack`, not `xiao2023bge`.
- Use `kim2024linq`, not `linqembed`.
- Use `marcuzzo-etal-2025-morables`, not `marcuzzo2025morables`.

New citations worth adding if we use them:

- STORAL
- Moral Stories
- FLUTE, if we keep figurative language
- HyDE
- Query2doc
- BRIGHT
- possibly AnaloBench, if we want an analogy-specific reference
- Easy as PIE?
- Not Just a Piece of Cake
- ID10M-JAM
- IdioLink

## Figurative Language And Idiom Citations

It makes sense to cite ID10M-JAM and IdioLink, but they should be used in a
controlled way. The goal is not to turn the Related Work into an idiom survey.
Instead, they should support the broader argument that current models still
struggle when intended meaning diverges from surface form.

Suggested role for each citation:

- **Easy as PIE?**: LLMs can identify idiomatic expressions in context using
  prompting, showing progress on figurative-language identification.
- **Not Just a Piece of Cake**: cross-lingual fine-tuning improves idiom
  identification and studies how idiomatic meaning is represented across model
  layers.
- **ID10M-JAM**: even when models perform well on idiom identification, they can
  be misled by adversarial but human-resolvable context.
- **IdioLink**: the closest structural comparison, because it formulates
  figurative meaning as a retrieval task over conceptually equivalent literal,
  idiomatic, and paraphrased realizations.

This citation plan should work if:

- ID10M-JAM has been camera-ready submitted and can be cited as forthcoming or
  accepted, depending on its final venue metadata.
- IdioLink is uploaded to arXiv before submission, so it can be cited with an
  arXiv entry rather than as unpublished internal work.

Suggested paragraph:

```latex
A related line of work studies figurative meaning, especially idioms, where
surface form can diverge sharply from intended meaning. Recent work shows that
LLMs can identify idiomatic expressions in context using prompting
\citep{hashiloni-etal-2025-easy}, and that cross-lingual fine-tuning can improve
idiom identification while revealing how idiomatic meaning is represented across
model layers \citep{hefetz-etal-2025-just}. ID10M-JAM further stress-tests this
ability by introducing adversarial contexts that remain clear to humans but
mislead models when distinguishing literal from figurative uses
\citep{hashiloni-etal-2025-id10mjam}. Most closely related to our setting,
IdioLink formulates idiom understanding as a retrieval task, asking whether
models can retrieve conceptually equivalent literal or paraphrased meanings for
idiomatic expressions despite surface divergence
\citep{hashiloni-etal-2025-idiolink}. MoraLink shares this focus on meaning
beyond lexical form, but shifts the challenge from idiomatic expressions to
semantic-level alignment between abstract moral principles and full narrative
realizations.
```

Suggested BibTeX entries for the two ACL Anthology papers:

```bibtex
@inproceedings{hashiloni-etal-2025-easy,
    title = "Easy as {PIE}? Identifying Multi-Word Expressions with {LLM}s",
    author = "Hashiloni, Kai Golan and Hefetz, Ofri and Bar, Kfir",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1213/",
    doi = "10.18653/v1/2025.emnlp-main.1213",
    pages = "23771--23790",
}

@inproceedings{hefetz-etal-2025-just,
    title = "Not Just a Piece of Cake: Cross-Lingual Fine-Tuning for Idiom Identification",
    author = "Hefetz, Ofri and Hashiloni, Kai Golan and Mannor, Alon and Bar, Kfir",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.136/",
    doi = "10.18653/v1/2025.ijcnlp-long.136",
    pages = "2521--2537",
}
```

Suggested provisional BibTeX entries:

```bibtex
@inproceedings{hashiloni-etal-2025-id10mjam,
  title = "{ID10M-JAM}: Stress-Testing Idiom Identification Under Challenging Context",
  author = "Hashiloni, Kai Golan and Livyatan, Lior and Hefetz, Ofri and Mannor, Alon and Cohen, Bar and Bar, Kfir",
  booktitle = "{TODO: add final venue proceedings}",
  year = "2025",
  note = "Camera-ready submitted"
}

@article{hashiloni-etal-2026-idiolink,
  title = "{IdioLink}: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions",
  author = "Hashiloni, Kai Golan and others",
  journal = "arXiv preprint",
  year = "2026",
  note = "arXiv ID forthcoming"
}
```

Once IdioLink is on arXiv, replace the provisional entry with the exact arXiv
metadata. If its arXiv version appears in 2025 rather than 2026, update the key
and year accordingly.

## Complete Rephrased Suggestion

```latex
\section{Related Work}
\label{sec:related-work}

\subsection{Moral and Narrative Understanding}

Moral reasoning has been studied in NLP through benchmarks that require models
to judge explicit situations, rules, or social norms. ETHICS and Delphi, for
example, evaluate whether models can approximate human moral judgments over
short descriptions of everyday scenarios
\citep{hendrycks2021aligning,jiang2021delphi}. These settings demonstrate the
difficulty of moral reasoning, but they largely operate over explicit
descriptions rather than narratives whose moral content must be inferred
indirectly.

A closer line of work studies moral meaning in stories. Moral Stories and
STORAL connect narratives with norms, intentions, consequences, or explicit
morals, supporting tasks such as moral selection and story generation
\citep{emelin-etal-2021-moral,guan-etal-2022-corpus}. MORABLES further focuses
on fables and abstract moral reasoning, asking models to select the correct
moral for a given fable from a small candidate set
\citep{marcuzzo-etal-2025-morables}. MoraLink builds on this setting but
reverses and expands the task: given a short abstract moral, the system must
retrieve the corresponding fable from a large candidate corpus. This turns moral
understanding into an open retrieval problem under extreme lexical and
abstraction divergence.

\subsection{Semantic Retrieval Beyond Surface Similarity}

Dense retrieval has become a central approach for semantic search, with
sentence embedding and contrastive retrieval models enabling retrieval beyond
exact keyword matching \citep{reimers2019sentence,izacard2022unsupervised}.
Large-scale benchmarks such as BEIR and MTEB have driven progress by evaluating
retrieval and embedding quality across diverse domains and tasks
\citep{thakur2021beir,muennighoff-etal-2023-mteb}. Modern embedding models such
as E5, BGE, and Linq-Embed-Mistral further improve performance through weak
supervision, instruction tuning, and large-scale contrastive training
\citep{wang2022text,xiao2023cpack,kim2024linq}.

However, most retrieval benchmarks still emphasize topical relevance,
paraphrase, or domain transfer. Recent reasoning-intensive benchmarks such as
BRIGHT show that strong retrievers struggle when relevance requires deeper
inference rather than surface semantic matching. MoraLink isolates a different
but related failure mode: the query and document differ not only in wording, but
also in semantic level. The query states a compressed abstract principle, while
the document expresses that principle indirectly through characters, events, and
consequences.

\subsection{LLM-Augmented Retrieval and Task-Specific Adaptation}

Several methods use LLMs to improve retrieval by rewriting, expanding, or
synthesizing text. HyDE generates a hypothetical document from the query and
retrieves real documents in the resulting embedding space, while Query2doc
expands queries with LLM-generated pseudo-documents
\citep{gao-etal-2023-precise,wang-etal-2023-query2doc}. InPars and related work
generate synthetic queries or training pairs for dense retrieval, reducing the
need for manual relevance labels \citep{bonifacio2022inpars}. Other approaches
use synthetic data at larger scale to improve general-purpose embedding models
\citep{wang2024improving}.

MoraLink follows the same broad intuition that retrieval can be improved by
changing the representation space, but makes the intervention on the document
side. Instead of expanding moral queries into narrative-like pseudo-documents,
we abstract fables into moral-style summaries that better align with the query
level. We also study task-specific contrastive fine-tuning with morally related
hard negatives, targeting conceptual distinctions that are not captured by
lexical or topical similarity alone.
```

## Notes Before Pasting Into `main.tex`

- Add missing bibliography entries before compiling.
- Decide whether to keep BRIGHT. If kept, add its BibTeX entry.
- Decide whether to mention figurative language explicitly. If kept, use one
  concise sentence and cite FLUTE or another suitable figurative-language
  benchmark.
- Check whether STORAL and Moral Stories citation keys match the final BibTeX
  entries.
- If the final experiments include a broader set of embedding models than E5,
  BGE, and Linq-Embed-Mistral, the retrieval paragraph can mention them as
  "modern instruction-tuned embedding models" rather than listing every model.
