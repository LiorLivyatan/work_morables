# Fable Enrichment — Generation Notes

## What this is

`fable_elements.json` contains structured semantic metadata for all 709 fables in the MORABLES corpus, extracted by Claude sub-agents from the raw fable texts in `data/processed/fables_corpus.json`.

## Fields

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Copied from source (e.g. `"fable_0000"`) |
| `title` | string | Copied from source |
| `themes` | list[str] | 2–5 thematic keywords |
| `characters` | list[str] | Character types present |
| `character_roles` | dict[str, str] | Main character → narrative role |
| `narrative_elements` | list[str] | 2–4 key story actions/events |
| `setting` | string | Physical location of the story |
| `moral_category` | string | Single top-level moral theme |
| `fable_type` | string | Nature of the cast |

### Controlled vocabularies

**`character_roles`** values: `deceiver`, `victim`, `helper`, `trickster`, `bystander`, `judge`, `seeker`, `antagonist`, `protagonist`

**`setting`** values: `forest`, `farm`, `sea`, `city`, `divine/olympus`, `field`, `home`, `river`, `unknown`

**`moral_category`** values: `trust`, `greed`, `pride`, `hubris`, `wisdom`, `patience`, `friendship`, `justice`, `nature`, `fate`, `deception`, `courage`, `contentment`, `vanity`, `ingratitude`, `ambition`

**`fable_type`** values: `animal_only`, `human_only`, `animal_human`, `divine`, `mixed`

## Generation method

8 Claude sub-agents were dispatched in parallel, each processing ~89 fables and writing to a separate chunk file (`elements_chunk_0.json` through `elements_chunk_7.json`). The chunks were then merged into `fable_elements.json`.

### Prompt template (identical for all agents, only index range and output path varied)

```
You are doing structured data extraction for a research project. Your job is to
read Aesop's fables and extract structured semantic elements from each one.

Read the file `data/processed/fables_corpus.json` (it's a JSON array of 709
fables). Process ONLY indices {start} through {end} (inclusive).

For each fable, extract the following fields:
- `doc_id`: copy from the source (string)
- `title`: copy from the source (string)
- `themes`: list of 2-5 thematic keywords (e.g. ["deception", "greed", "pride", "betrayal", "wisdom"])
- `characters`: list of character types present (e.g. ["fox", "crow", "farmer", "god"])
- `character_roles`: dict mapping each main character to their narrative role.
  Roles should be one of: "deceiver", "victim", "helper", "trickster",
  "bystander", "judge", "seeker", "antagonist", "protagonist"
- `narrative_elements`: list of key story actions/events (2-4 items,
  e.g. ["trick", "flattery", "food_lost", "warning_ignored"])
- `setting`: one of "forest", "farm", "sea", "city", "divine/olympus", "field",
  "home", "river", "unknown"
- `moral_category`: one of "trust", "greed", "pride", "hubris", "wisdom",
  "patience", "friendship", "justice", "nature", "fate", "deception", "courage",
  "contentment", "vanity", "ingratitude", "ambition"
- `fable_type`: one of "animal_only", "human_only", "animal_human", "divine", "mixed"

Output a JSON array (not an object, just a flat array) of these records. Write
the output to `data/enriched/elements_chunk_{n}.json`

Important:
- Process ALL fables in the assigned range, no skipping
- Be consistent with theme vocabulary across fables
- Use lowercase for all list items and dict values
- Write valid JSON only — no comments, no trailing commas
```

### Chunk assignment

| Chunk | Indices | Output file |
|---|---|---|
| 0 | 0–88 | `elements_chunk_0.json` |
| 1 | 89–177 | `elements_chunk_1.json` |
| 2 | 178–266 | `elements_chunk_2.json` |
| 3 | 267–355 | `elements_chunk_3.json` |
| 4 | 356–444 | `elements_chunk_4.json` |
| 5 | 445–533 | `elements_chunk_5.json` |
| 6 | 534–622 | `elements_chunk_6.json` |
| 7 | 623–708 | `elements_chunk_7.json` |
