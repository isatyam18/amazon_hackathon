# Amazon ML Challenge 2026: Business Entity Resolution

## Overview

In commercial platforms, business identity data originates from multiple independent, noisy sources without shared primary keys. This project addresses the Business Entity Resolution (ER) Challenge: determining which records across heterogeneous sources refer to the exact same real-world business entity.

Source 1 serves as the deduplicated reference source. The objective is to identify all matching records from Source 2 and Source 3 for every record in Source 1. A Source 1 entity may match zero (singleton), one, or multiple records from Source 2 and Source 3.

---

## Dataset Access

Due to size constraints (exceeding 2.4 GB uncompressed and ~1 GB compressed), the raw dataset files are not tracked in this Git repository.

The complete dataset can be accessed and downloaded from Google Drive:
[Amazon ML Challenge Dataset on Google Drive](https://drive.google.com/drive/folders/1pNh7H29uPlFTPMu6MpHEhcSVW1o7j6Qi?usp=sharing)

After downloading, extract the data so the structure matches:
```text
student_resource/
  └── dataset/
      ├── train/
      │   ├── train_source1.tsv
      │   ├── train_source2.tsv
      │   ├── train_source3.tsv
      │   └── train_ground_truth.tsv
      └── test/
          ├── test_source1.tsv
          ├── test_source2.tsv
          └── test_source3.tsv
```

---

## Data Schema and Scale

All data files are tab-separated (`.tsv`) to prevent conflicts with commas inside addresses and name fields.

### Record Schema

Each source record contains:
1. `entity_id`: Unique record identifier with source prefix (`S1-`, `S2-`, `S3-`).
2. `business_name`: Business name containing abbreviations, legal suffixes, typographical noise, punctuation differences, and transliterations.
3. `business_address`: Business address with variations in abbreviations, missing postal codes/states, landmark references, and municipal formatting.
4. `country`: Country label. Training data contains `US` and `India`. Test data contains an unseen third country, `France`.

### Data Dimensions

- **Training Source 1 (`train_source1.tsv`)**: 2,206,821 records (reference entities)
- **Training Source 2 (`train_source2.tsv`)**: 5,034,616 records
- **Training Source 3 (`train_source3.tsv`)**: 5,285,603 records
- **Training Ground Truth (`train_ground_truth.tsv`)**: 2,206,821 records
- **Test Source 1 (`test_source1.tsv`)**: 1,732,544 records (predictions required for all)
- **Test Source 2 (`test_source2.tsv`)**: 4,887,273 records
- **Test Source 3 (`test_source3.tsv`)**: 5,082,316 records

---

## Evaluation Metric

Submissions are scored using the **macro-averaged F-beta score** with beta = 0.5 across all Source 1 entities:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Key Metric Properties
- **Precision Weighting**: Precision is weighted twice as heavily as recall. False positives (merging two distinct entities) severely penalize the score compared to missed matches.
- **Singletons**: Source 1 entities with no matching records score 1.0 when predicted as empty, and 0.0 if any false match is predicted. Handling singletons correctly is critical for a high macro-average score.
- **Open-Set Country**: The model must dynamically handle any country string (such as `France` in test data) without hardcoded filters or fixed categorical one-hot encodings.

---

## Working Plan and Technical Approach

Given the massive search space (~1.7M test S1 entities against ~10M candidates in S2/S3), an exhaustive pairwise comparison is computationally intractable ($O(N \times M)$ is ~17 trillion pairs). The project follows a multi-stage entity resolution pipeline:

### Stage 1: Data Preprocessing and Standardization
- **Indic Transliteration**: 11–15% of S2/S3 names (about 24% of India names) are written in Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Oriya or Gurmukhi, while S1 is pure ASCII. `src/transliteration.py` romanises all nine scripts with a single offset table: they share the ISCII-derived Unicode layout. This runs *before* de-accenting, which would otherwise strip viramas; without it these names were reduced to empty strings.
- **Text Normalization**: Lowercasing, unicode normalization, stripping non-alphanumeric noise, and standardizing common conjunctions (`&` to `and`).
- **Entity Suffix Extraction**: Normalizing legal entity forms across regions (`pvt ltd`, `inc`, `corp`, `llc`, `sa`, `sarl`).
- **Address Parsing (country-aware, `src/regions.py`)**: Abbreviations are expanded per token from a country-keyed table. The generic/US table has `st`→street, `rd`→road, `ste`→suite. France overrides the entries whose meaning differs: `st`→saint, `ste`→sainte, `r`→rue, `av`→avenue, `all`→allee, `nº`/`no`→number. France is 15% of test S1, and its names are generic ("Nantes Sportive", "Bordeaux Ecole"), so the address decides who is who. First-level regions become one code per country: `north carolina`/`nc`, `maharashtra`/`mh`. Transliterated Indian spellings (`maharashtr`, `karnatak`, `dilli`) are matched by phonetic key (`src/phonetic.py`); US names use exact matching only, so `mountain` never becomes `montana`. Countries without a table get the generic expansions and no region mapping. Postal codes turned out to be almost absent from the data (0% six-digit PINs), so later stages do not rely on them.
- **Transliteration detail**: the anusvara (ं/ം) is written `m` before labials and at word end, and `n` elsewhere (keralam, sampark, hindustan).
- **Language and Country Handling**: Preserving country labels as grouping partitions while ensuring zero data leakage or country-exclusive hardcoding.

### Stage 2: Scalable Candidate Generation (Blocking) — `src/blocking/`

**Key findings that shaped the design (train EDA):**
- No true pair crosses countries (0 of 7.6M), so the search is partitioned by country label. The labels are treated as an open set: France (test-only) gets its own partition, with IDF fitted on its own targets.
- Each S1 entity matches at most 5 S2 and 6 S3 records (mean 3.46 overall). Each S2/S3 record matches at most one S1.
- The corpus reuses a small business vocabulary. Single name words are common (a shared name word covers only 51–64% of true pairs at a posting cap of 3k), but **word pairs and name×address cross keys are rare and identifying**.
- Noise includes legal forms anywhere in the name, leet digits (`C0astal`, `5ervices`), domain/handle names (`kaiagouldreliable.com`), random DBA names at the same address, mangled house numbers, and 3.3% empty addresses.

**Two-step design:**

1. **Retrieval over rare keys (sparse TF-IDF, inverted index).** For each (country, source) partition the targets are hashed into feature views, and IDF is fitted on that partition. Three passes each keep the top-50 targets per query per source. Keys with document frequency above 1,000 are not indexed, which bounds the cost per query:

   | Pass | Views (keys) | Catches |
   |---|---|---|
   | `name` | name tokens, unordered token pairs, phonetic-key pairs, compact-name prefix | typos in one word, word order, transliteration, domain names |
   | `addr` | address tokens, adjacent address bigrams | DBA / random trade names at the same address |
   | `name_addr` | name token × address token cross keys | heavy noise on either side; chains and common names at different locations |

2. **Exact rescoring and budget.** Every retrieved pair gets full TF-IDF cosines on five views: name char-3-grams, phonetic skeleton 3-grams, address tokens, address bigrams, and name×address keys. They are combined with weights learned by logistic regression on validation candidates (true pair vs. not; `scripts/fit_rescore_weights.py`). When either record lacks an address, the pair is scored on name evidence alone, rescaled to the same range. The best **15 per source** (up to 30 per S1 entity) are kept. On validation, 15 keeps 97.8% of true pairs vs 97.8% at 25, with 40% fewer pairs to score.

The top-k engine (`topk.py`) runs the sparse product in chunks sized from the exact posting-length bound, uses threads (scipy releases the GIL), and selects the top k per row with `argpartition`. An optional dense pass (`dense.py`, multilingual MiniLM, Apache-2.0, + FAISS) can be enabled with `--dense`. It is off by default because CPU encoding of about 12M records takes hours.

**Results.** All queries are searched against the *full* train S2/S3 (10.3M records). The rescoring weights were fitted on validation candidates. The full-train universe (all 2.2M train S1, evaluated on a random 50k-query subset) is the large-sample estimate:

| Metric | Val (10k S1) | Full train (2.2M S1; 50k-query sample) |
|---|---|---|
| Pair recall (pair completeness) | **0.978** (US 0.986, India 0.966) | **0.976** (US 0.986, India 0.961) |
| F0.5 ceiling (perfect matcher on these candidates) | **0.9926** | **0.9921** |
| Matched S1 entities with zero true candidates | 0.22% | 0.20% |
| Recall@k per source (k=1/5/10/15) | 0.38 / 0.94 / 0.977 / 0.978 | 0.38 / 0.94 / 0.975 / 0.976 |
| Candidates per S1 entity | 30 (15 per source) | 30 (15 per source) |
| Reduction ratio | 0.999997 | 0.999997 |

The full-train run (2.2M S1, 66.2M pairs) takes about 41 min of blocking on this machine. It is the Stage 3/4 training and evaluation universe.

**Outputs:** `output/candidate_pairs.tsv` (submission format: every test S1 entity has one row, ordered by candidate score) and `dataset/candidates/{split}_candidate_pairs.parquet` (one row per pair). The Parquet carries `candidate_score`, `source_rank`, the retrieval cosines `score_*` and the exact similarities `sim_*`, which are ready-made Stage 3 features. `candidate_pairs.tsv` must list exactly the pairs the matcher scores, so the final one is rewritten by `scripts/run_matching.py --mode test` from the set it actually scored.

### Stage 3: Pair Features — `src/matching/records.py`, `features.py`, `context.py`
52 features per (S1, candidate) pair, computed in bulk: rapidfuzz `cpdist` (multithreaded), and token overlaps as row-wise sparse dot products. Record-level normalisation is done once per record and stored as Arrow strings.
- **Name**: ratio / token-sort / token-set on the normalised name (legal & filler tokens removed, leet digits fixed); partial ratio and Jaro-Winkler on the space-free name (domain names such as `kaiagouldreliable.com`); ratio of phonetic skeletons (transliterated names); token Jaccard; first-token agreement; lengths.
- **Address**: ratio / token-set / token-sort / partial-token-set, token Jaccard, numeric-token overlap and **conflict** (both have numbers, none shared = a different branch), **house number** equality and prefix agreement (`245` vs `2454`), missing-address flags. Address features are NaN, not 0, when a side has no address.
- **Stage 2 signals**: retrieval cosines `score_*`, exact TF-IDF similarities `sim_*`, candidate score, per-source rank, source.
- **Competition (context)**: margin over the best *other* candidate of the same S1 (overall and within its source), rank, number of near-equal candidates and near-duplicate names; and, per S2/S3 record, the number of S1 entities competing for it and this pair's margin over the best competing S1. Each record belongs to at most one S1 in all 7.6M training pairs, so a much better competing S1 is strong evidence against the pair. These use only Stage 2 scores, never labels.

No feature uses the country label, so the matcher applies unchanged to France (absent from training).

### Stage 4: Matcher, Calibration and Honest Evaluation — `model.py`, `decision.py`, `workflow.py`
- **Universe**: all 2.2M train S1 entities with their Stage 2 candidates from the full train S2/S3 (66.2M pairs). Candidate lists and the competition between S1 entities for each record look as they will on test.
- **LightGBM** (binary log-loss, 63 leaves, bagging, early stopping on a held-out 10% of the *training* queries). **Two query folds** (hash of the S1 id): a model trained on 120k fold-0 entities scores fold 1 and vice versa, so every pair gets an out-of-fold probability.
- **Calibration** (isotonic) **and the decision rule are fitted on fold 0 only.** Fold 1 is used for nothing but the reported score.
- **Exact metric** (`threshold.py`): per-S1 F0.5 = 1.25·P·R/(0.25·P+R), averaged over *all* S1 entities. Singletons score 1.0 only when predicted empty, and true matches missed by blocking count against recall. Unit-tested against the reference implementation and the problem-statement example.
- **Memory**: everything streams one country partition at a time. No pair, record or constraint crosses countries, so this is exact. Peak memory is about 5 GB.

### Stage 5: Decision Rule and Submission — `decision.py`, `scripts/run_matching.py`
- **Owner normalisation**: a record's probabilities across competing S1 entities are rescaled to sum to at most 1.
- **Expected-F0.5 set selection**: per S1 entity, keep the top-k candidates that maximise the expected F0.5, using F_β = (1+β²)·TP/(|S|+β²·|T|). Predict nothing when "no match" is the better bet. This handles multi-match entities and singletons with one principled rule instead of a global threshold.
- **Exclusivity**: a record predicted for several S1 entities stays with the most probable one.
- The rule variant (and a global-threshold baseline) is chosen on fold 0.
- **Test**: both fold models are averaged, calibrated, and the rule is applied. `output/matching_results.tsv` and `output/candidate_pairs.tsv` (exactly the scored pairs) are written, and the official validator is run.

__STAGE345_RESULTS__

---

## Project Structure

```text
.
├── .gitignore                      # Excludes raw data, model binaries, and caches
├── README.md                       # Complete project overview and implementation guide
├── requirements.txt                # Pinned dependencies
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Stage 1 walkthrough: samples, normalization, val split, full preprocessing
│   └── 02_blocking.ipynb          # Stage 2 walkthrough: design evidence, keys, val/train/test runs, validation
├── student_resource/
│   ├── Documentation_template.md  # Final methodology write-up template
│   ├── README.md                  # Challenge problem statement documentation
│   ├── dataset/                   # Dataset directory (excluded from git)
│   └── utils/
│       └── validate_submission.py # Submission format verification script
├── scripts/
│   ├── preprocess_dataset.py      # Stage 1: raw TSV -> dataset/preprocessed/ (tsv or parquet)
│   ├── create_val_split.py        # Validation S1 sample + ground truth
│   ├── inspect_samples.py         # Print matched record examples
│   ├── run_blocking.py            # Stage 2 CLI: test / val / train candidate generation
│   ├── evaluate_blocking.py       # Stage 2 metrics for any candidate file vs ground truth
│   └── fit_rescore_weights.py     # Stage 2: refit the candidate-ranking weights on labelled queries
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Text and address cleaning routines
│   ├── transliteration.py         # Indic script -> Latin transliteration
│   ├── blocking/                  # Stage 2: candidate generation
│   │   ├── config.py              #   passes, rescoring weights, budget (JSON-overridable)
│   │   ├── normalize.py           #   legal/filler tokens, leet digits, phonetic keys
│   │   ├── features.py            #   feature views + parallel hashing
│   │   ├── index.py               #   per-(country, source) TF-IDF inverted index
│   │   ├── topk.py                #   chunked multithreaded sparse top-k
│   │   ├── fusion.py              #   union of passes, budgeted top-N selection
│   │   ├── rescore.py             #   exact pair cosines + candidate score
│   │   ├── dense.py               #   optional MiniLM + FAISS pass
│   │   ├── pipeline.py            #   orchestration
│   │   ├── candidates.py          #   CandidateSet + TSV / parquet writers
│   │   ├── data.py                #   loading preprocessed sources / ground truth
│   │   ├── evaluation.py          #   pair recall, F0.5 ceiling, reduction ratio, macro F0.5
│   │   └── tuning.py              #   labelled candidate union + logistic-regression weight fitting
│   ├── features.py                # (Stage 3) Feature extraction for candidate pairs
│   ├── train.py                   # (Stage 4) Model training and hyperparameter tuning
│   └── inference.py               # (Stage 5) Test set scoring
├── tests/
│   ├── test_preprocessing.py
│   └── test_blocking.py           # pytest: normalization, top-k, pipeline, validator format
├── dataset/                       # Generated data (excluded from git)
│   ├── preprocessed/              #   Stage 1 output
│   ├── val_split/                 #   validation ground truth
│   └── candidates/                #   Stage 2 scored pairs + reports
└── output/                        # Generated submission files (excluded from git)
    ├── candidate_pairs.tsv
    └── matching_results.tsv
```

## Running the Pipeline

```bash
# Stage 1: preprocess all splits (parquet is ~4x smaller than TSV; blocking reads either)
python scripts/preprocess_dataset.py --split all --format parquet --n-jobs 8

# Validation split: 10k S1 queries + their ground truth (no distractor files needed)
python scripts/create_val_split.py --num-s1 10000 --num-distractors 0

# Stage 2 on validation: val queries vs the FULL train S2/S3, with recall report (~9 min)
python scripts/run_blocking.py --split val

# Stage 2 full-train universe for Stage 3/4 (all 2.2M train S1; parquet only)
python scripts/run_blocking.py --split train --all-train --no-tsv

# Stage 2 on test -> output/candidate_pairs.tsv + dataset/candidates/test_candidate_pairs.parquet
python scripts/run_blocking.py --split test

# Stages 3-5: out-of-fold training + decision tuning + honest report, then test submission
python scripts/run_matching.py --mode train
python scripts/run_matching.py --mode test
# (re-tune the decision rule from saved OOF arrays without retraining)
python scripts/run_matching.py --mode tune

# Re-evaluate / re-cut a candidate set to a smaller budget
python scripts/evaluate_blocking.py --candidates dataset/candidates/val_candidate_pairs.parquet --max-per-source 15

# Refit the rescoring weights (writes a config usable with --config)
python scripts/fit_rescore_weights.py

# Tests
python -m pytest tests/test_blocking.py -q
```

The notebooks (`notebooks/01_preprocessing.ipynb`, `notebooks/02_blocking.ipynb`) walk through each stage on real data and call these same scripts. All pipeline logic lives in `src/`, and the scripts are the reproducible command-line entry points.

Blocking settings (passes, k, `max_df`, rescoring weights, budget) live in `src/blocking/config.py`. Override them without code changes via `--config my_config.json`: the format is `BlockingConfig.to_dict()`, and every run saves the config it used in its `*_blocking_report.json`.

---

## Constraints and Compliance

1. **Model Parameter Limit**: Maximum 8 Billion parameters.
2. **Permissible Licenses**: Open-source MIT or Apache 2.0 licenses only.
3. **No External Lookups**: External databases, commercial APIs, internet lookups, or third-party geocoding services are strictly prohibited.
4. **Formatting Rules**:
   - Every Source 1 entity in the test set must have exactly one row.
   - Matching IDs must only refer to Source 2 or Source 3 entities.
   - All final matches in `matching_results.tsv` must exist in `candidate_pairs.tsv`.

---

## Environment Setup

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Validation

Before uploading to the competition portal, run the validator:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
A return code of 0 confirms formatting compliance.
