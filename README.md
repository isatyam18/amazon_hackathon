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
- **Text Normalization**: Lowercasing, unicode normalization, stripping non-alphanumeric noise, and standardizing common conjunctions (`&` to `and`).
- **Entity Suffix Extraction**: Normalizing legal entity forms across regions (`pvt ltd`, `inc`, `corp`, `llc`, `sa`, `sarl`).
- **Address Parsing**: Extracting numerical tokens, street keywords (`st`, `street`, `rd`, `road`), landmarks, and PIN/postal codes where available.
- **Language and Country Handling**: Preserving country labels as grouping partitions while ensuring zero data leakage or country-exclusive hardcoding.

### Stage 2: Scalable Candidate Generation (Blocking)
The objective of blocking is to maximize candidate recall while minimizing candidate pair volume:
- **Country Partitioning**: Partition comparisons strictly within matching countries (`country` matches between S1 and S2/S3).
- **Multi-Index Blocking Strategy**:
  1. **Character n-gram TF-IDF & Inverted Index**: Index normalized names using 2-4 character n-grams to handle typos and spelling variations.
  2. **MinHash / Locality Sensitive Hashing (LSH)**: Generate fast Jaccard similarity candidate buckets.
  3. **Address Component Keys**: Block on extracted postal codes or normalized street tokens for dense commercial regions.
  4. **Dense Bi-Encoder Retrieval (Optional/Targeted)**: Utilize compact multilingual sentence transformers (e.g., MiniLM under Apache 2.0) with Approximate Nearest Neighbor (ANN) search via Faiss/HNSW.
- **Candidate Output**: Produce top-k (e.g., $k \in [10, 30]$) candidate pairs per Source 1 entity, exported to `output/candidate_pairs.tsv`.

### Stage 3: Feature Engineering
For each candidate pair $(S_1, S_{2/3})$, construct discriminative similarity features:
- **Name Similarities**:
  - Levenshtein distance ratio, Jaro-Winkler similarity.
  - Token Sort Ratio, Token Set Ratio, and Partial Ratio via RapidFuzz.
  - Exact token overlap and longest common prefix/substring.
- **Address Similarities**:
  - Address token Jaccard similarity and numerical/postal code exact match indicator.
  - Substring match indicators for street and city components.
- **Cross-Source Signals**:
  - Source identifier feature (`S2` vs `S3`).
  - Name length difference, token count ratio.
  - Semantic embedding cosine similarity.

### Stage 4: Machine Learning Classifier and Calibration
- **Model Choice**: Gradient Boosted Decision Trees (LightGBM / CatBoost) trained on balanced positive and hard negative pairs sampled from the candidate generation stage.
- **Objective Function**: Binary cross-entropy with sample weighting, or pairwise ranking objective (LambdaMART).
- **Probability Calibration**: Calibrate predicted probabilities using Platt scaling or isotonic regression.

### Stage 5: Decision Thresholding and Post-Processing
- **F-0.5 Threshold Tuning**: Optimize the classification threshold on a held-out local validation split specifically for the $F_{0.5}$ metric (setting a conservative, high-precision threshold $\tau \approx 0.65\text{--}0.80$).
- **Transitivity and Consistency**: Resolve graph consistency constraints across multiple matches if necessary.
- **Submission Output**: Format final matches into `output/matching_results.tsv` and run validation using `student_resource/utils/validate_submission.py`.

---

## Project Structure

```text
.
├── .gitignore                      # Excludes raw data, model binaries, and caches
├── README.md                       # Complete project overview and implementation guide
├── requirements.txt                # Pinned dependencies
├── student_resource/
│   ├── Documentation_template.md  # Final methodology write-up template
│   ├── README.md                  # Challenge problem statement documentation
│   ├── dataset/                   # Dataset directory (excluded from git)
│   └── utils/
│       └── validate_submission.py # Submission format verification script
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Text and address cleaning routines
│   ├── blocking.py                # Candidate generation and indexing
│   ├── features.py                # Feature extraction for candidate pairs
│   ├── train.py                   # Model training and hyperparameter tuning
│   └── inference.py               # Test set candidate generation and scoring
└── output/                        # Generated submission files (excluded from git)
    ├── candidate_pairs.tsv
    └── matching_results.tsv
```

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
