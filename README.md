# Term Accuracy Evaluation Pipeline

A pipeline for evaluating **term accuracy in machine translation** test sets. Given a target term, optional distractor terms and a target sentence, the pipeline applies four matching strategies optimized for the German language (1. surface-level lookup, 2. lemmatisation, 3. compound splitting 4. inflectional fuzzy matching) to determine whether the expected term was correctly rendered by the MT system. For a full description of the methodology, refer to:

Paolo Di Natale, Elena Chiocchetti, Marlies Alber, and Egon Waldemar Stemle. 2026. Beyond simple term injection: Reasoning models for legal translation in a non-dominant language variety. In Proceedings of the 26th Annual Conference of the European Association for Machine Translation, Tilburg, the Netherlands. European Association for Machine Translation.

---

## Requirements

### Python dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### spaCy models

Download the spaCy model(s) for your target language:

```bash
# German (default)
python -m spacy download de_dep_news_trf

# Italian
python -m spacy download it_core_news_sm
```

---

## Pipeline Overview

The pipeline consists of two scripts that must be run in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `preproc.py` | Lemmatises MT outputs and reference terms; saves preprocessed data |
| 2 | `find_terms.py` | Runs term matching across all models; computes and saves accuracy rates |

---

## Usage

### Step 1 — Preprocessing

This pipeline has been designed for use in two test sets and their subsections. Refer to the papers, or contact me at pdinatale@eurac.edu, for further details. 

However, the mechanism is the same. if you want to use it on your own data, you may want to choose ```--testset bistro``` on any mode.

```bash
python preproc.py --testset <testset> --mode <mode> --lang <lang>
```

**Arguments:**

| Argument | Required | Values | Description |
|----------|----------|--------|-------------|
| `--testset` | ✅ | `legistyr`, `bistro` | Test set specification |
| `--mode` | ✅ | See table below | Terminology subset |
| `--lang` | ❌ | `de` (default), `it` | Target language |

**Valid modes per testset:**

| Testset | Modes |
|---------|-------|
| `legistyr` | `hom`, `abbr`, `simple_terms` |
| `bistro` | `var`, `rel`, `hom` |

**Examples:**

```bash
python preproc.py --testset legistyr --mode hom --lang de
python preproc.py --testset bistro --mode var --lang de
```

**Output:** Preprocessed and raw CSV/TSV files saved under `data/preprocessed_texts/<testset>/`.


---

### Step 2 — Term Matching

```bash
python find_terms.py --testset <testset> --mode <mode> --lang <lang>
```

Arguments are identical to `preproc.py`.

**Examples:**

```bash
python find_terms.py --testset legistyr --mode hom --lang de
python find_terms.py --testset bistro --mode var --lang de
```

**Output:**
- Per-sentence term match, represented by a list → `data/results/<testset>/<mode>/`
- Accuracy rate summary: readable percentages of accuracy rate per model → `data/results_analysis/<testset>/<mode>/term_accuracy_rates.txt`

---

## Input File Requirements

Both testsets require a **reference terminology file** and one or more **MT output files**, all placed under `data/<testset>/<mode>/`.

---

### Reference terminology file

This is the main testset file containing the source sentences and reference terms. It must be placed at:
```
data/<testset>/<mode>/<TESTSET>__<mode>
```

For example: `data/legistyr/hom/LegISTyr__hom.csv` or `data/bistro/var/BISTRO__var.tsv`.

#### LegISTyr — semicolon-delimited `.csv`

| Column name | Description |
|-------------|-------------|
| `TARGET HYPOTHESIS (DE SOUTH TYROL)` | Primary target term (South Tyrolean legal system) |
| `OTHER TERMS SOUTH TYROL (CSV)` | Alternative accepted terms (South Tyrolean) |
| `TERMS FROM OTHER LEGAL SYTEMS (CSV)` | Terms from other legal systems (used as distractors) |
| `OPTIONS` | Homonym candidates — **required in `hom` mode only** |

Any additional columns present in the file are preserved but not used for term matching.

#### BISTRO — tab-delimited `.tsv`

| Column name | Description |
|-------------|-------------|
| `tgt_term_*` | One or more target term columns (e.g. `tgt_term_1`, `tgt_term_2`). All columns whose name starts with `tgt_term_` are detected automatically and treated as separate matching term categories. |

Terms in bistro testset should be separated by commas with no spaces.

If you have **reference translations**, place them under a `context` column.

Any additional columns present in the file are preserved but not used for term matching.

---

### MT files — `.txt`

Each MT system to evaluate must be provided as a plain `.txt` file in the same folder as the reference file:

- **One translated sentence per line**, in the same order as the rows in the reference file.
- **The filename (without extension) is used as the model identifier** in all results and reports. Name files descriptively, e.g. `gpt4o.txt`, `deepl.txt`, `opus-mt.txt`.
- Multiple `.txt` files can be placed in the same folder to evaluate several systems at once.

**Example folder layout for `bistro/hom`:**

```
data/legistyr/hom/
├── Bistro__hom.tsv     # Reference terminology file
├── gpt4o.txt             # MT output — model "gpt4o"
├── deepl.txt             # MT output — model "deepl"
└── opus-mt.txt           # MT output — model "opus-mt"
```


## Limitations

A `config.ini` file is automatically created/updated by `preproc.py` to store the list of detected model names. This is read by `find_terms.py` and does not need to be edited manually. I am aware this implementation is a bottleneck to running preprecessing on different test sets simultaneously, as the model list is overwritten every time the preprocessing pipeline starts. I will fix in the future.


## Extra features
## Generic Tools
 
Two standalone utilities are provided for use outside the main evaluation pipeline. They are not tied to any specific testset and can be applied to arbitrary corpora and term lists.
 
---
 
### `generic_preproc.py` — Lemmatize a corpus
 
Lemmatizes a corpus from a `.txt`, `.csv`, or `.tsv` file and writes the result as a single flat `.txt` file, suitable as input for `generic_term_finder.py`.
 
```bash
python generic_preproc.py -i <input_file> -o <output_file> [-c <column_name>]
```
 
**Arguments:**
 
| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input-file` | `-i` | ✅ | — | Path to a `.txt`, `.csv`, or `.tsv` input file |
| `--output-file` | `-o` | ✅ | — | Path to the output `.txt` file |
| `--column-name` | `-c` | ❌ | `filled_text` | Column to read from CSV/TSV; ignored for `.txt` input |
 

**Example:**
 
```bash
python generic_preproc.py \
  -i data_generic/Train_BTCatex_de.txt \
  -o data_generic/lemmatized_corpus.txt
```
 
---
 
### `generic_term_finder.py` — Count term frequencies in a corpus
 
Searches a lemmatized corpus for a list of terms extracted from a CSV file and outputs their frequency counts. Matching is case-insensitive and uses spaCy's `PhraseMatcher`. Large corpora are processed in chunks to manage memory.
 
```bash
python generic_term_finder.py \
  --corpus-txt <corpus.txt> \
  --terms-csv <terms.csv> \
  --term-column <column_name> \
  --output <output_file>
```
 
**Arguments:**
 
| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--corpus-txt` | `-c` | ✅ | — | Path to the lemmatized corpus `.txt` file |
| `--terms-csv` | `-t` | ✅ | — | Path to a semicolon-delimited `.csv` containing the terms |
| `--term-column` | — | ✅ | — | Column name in the CSV from which to read terms |
| `--output` | `-o` | ✅ | — | Output path; `.csv` or `.json` |
| `--encoding` | — | ❌ | `utf-8` | File encoding |
 
**Output format:** if the output path ends in `.csv`, results are saved as a two-column table (`term`, `frequency`), sorted by frequency descending. If it ends in `.json`, results are saved as a key-value mapping.
 
**Example:**
 
```bash
python generic_term_finder.py \
  --corpus-txt data_generic/lemmatized_corpus.txt \
  --terms-csv data/legistyr/hom/LegISTyr__hom.csv \
  --term-column "TARGET HYPOTHESIS (DE SOUTH TYROL)" \
  --output data_generic/count_homonyms.csv
```
 
---
 
### Typical generic workflow
 
```bash
# 1. Lemmatize your corpus
python generic_preproc.py \
  -i my_corpus.txt \
  -o data_generic/lemmatized_corpus.txt
 
# 2. Count how often your terms appear in it
python generic_term_finder.py \
  --corpus-txt data_generic/lemmatized_corpus.txt \
  --terms-csv my_terms.csv \
  --term-column "my_term_column" \
  --output data_generic/term_counts.csv
```