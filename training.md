# Amazon ML Challenge 2026: Business Entity Resolution
# End-to-End Build, Training, and Iteration Master Plan (`training.md`)

---

## 0. Guiding Principles & Operational Philosophy

1. **Every decision is measured against the exact metric:** Macro $F_{0.5}$ per $S_1$ entity on held-out validation data, sliced by country (US, India, France-like), entity cluster size (1, 2–3, 4–6, 7+), and twin-presence (brand chain branches).
2. **Cheap, deterministic, and scalable first:** High-speed string normalization + Inverted Index Candidate Blocking + LightGBM GBDT Matcher achieves $> 95\%$ of peak score in minutes without memory spikes. Neural models (`multilingual-e5-small`, `bge-reranker-v2-m3`) are added conditionally only if they beat a measured validation bar.
3. **Everything generalises to unseen countries (Zero-Shot France):** No hardcoded word lists that only function for US/India. All rules are either generic (Unicode NFKD accent folding, token frequencies, edge legal-token detectors) or cross-validated via reverse country transfer (train US $\rightarrow$ test India, and vice-versa).
4. **RAM & Infrastructure Safety:** Inverted token index structures, contiguous integer ID maps, country-isolated streaming, and chunked query processing keep peak memory strictly under **$1.5\text{ GB}$ on local laptops** and **$< 4\text{ GB}$ on Kaggle/Colab**.
5. **Rigorous Experiment Logging:** Track every run with git hash, feature set configuration, CV overall / per slice, cross-country score, and public leaderboard delta.

---

## 1. Model Architecture, Role, and Parameter Budget

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 COMPREHENSIVE MODEL TAXONOMY & LICENSING                                │
├───────────────────────────────┬──────────────────────────┬─────────────┬──────────┬────────────────────┤
│ Pipeline Role                 │ Primary Architecture     │ Licence     │ Capacity │ Fallback Option    │
├───────────────────────────────┼──────────────────────────┼─────────────┼──────────┼────────────────────┤
│ Core Matcher (S1 ↔ Candidate) │ LightGBM GBDT Classifier │ MIT         │ —        │ CatBoost (Apache)  │
│ Record ↔ Record Linkage       │ LightGBM Pairwise Ranker │ MIT         │ —        │ Logistic GBDT      │
│ Has-Match Binary Head         │ LightGBM on S1 Summaries │ MIT         │ —        │ Calibrated Logit   │
│ String Distance Kernel        │ rapidfuzz & native C++   │ MIT         │ —        │ Levenshtein / Cyth │
│ Indic Script Transliteration  │ 6-Script Parallel Block  │ In-House    │ —        │ PyICU Any-Latin    │
│ Dense Retrieval & Cosine Head │ intfloat/multilingual-e5 │ MIT         │ 118M     │ BAAI/bge-m3 (568M) │
│ Pair Re-Scorer (Neural)       │ BAAI/bge-reranker-v2-m3  │ Apache 2.0  │ 568M     │ mDeBERTa-v3-base   │
│ Synthetic Noise / LLM Judge   │ Qwen2.5-7B / Qwen3-4B    │ Apache 2.0  │ 4–8B     │ Rule Generator     │
└───────────────────────────────┴──────────────────────────┴─────────────┴──────────┴────────────────────┘
```

### Parameter Budget & Legal Compliance:
* **Total Stacked Parameters:** $\text{LightGBM (0B)} + \text{e5-small (0.12B)} + \text{bge-reranker (0.57B)} \approx \mathbf{0.7\text{ Billion parameters}}$.
* **Strict Rule Adherence:** Safely within the competition's $\le 8\text{B}$ parameter ceiling, whether interpreted per individual model or across the entire integrated ensemble.

---

## 2. Evaluation Metric & Mathematical Optimization ($\text{Macro } F_{0.5}$)

### 2.1 Formal Formulation
For each canonical reference entity $e \in S_1$ ($|\mathcal{E}| = 1,732,545$ in test):
* Let $T(e)$ be the ground-truth set of matching records from $S_2 \cup S_3$.
* Let $P(e)$ be the model's predicted matching records.

$$F_{0.5}(e) = \frac{(1 + 0.5^2) \cdot \text{Precision}(e) \cdot \text{Recall}(e)}{(0.5^2 \cdot \text{Precision}(e)) + \text{Recall}(e)} = \frac{1.25 \cdot \text{Precision}(e) \cdot \text{Recall}(e)}{0.25 \cdot \text{Precision}(e) + \text{Recall}(e)} = \frac{5 \cdot |T(e) \cap P(e)|}{4 \cdot |P(e)| + |T(e)|}$$

$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{e \in S_1} F_{0.5}(e)$$

### 2.2 The $4:1$ Precision Penalty Gradient
$$\left| \frac{\partial F_{0.5}}{\partial \text{FP}} \right| \approx 4 \times \left| \frac{\partial F_{0.5}}{\partial \text{FN}} \right|$$
* **Strategic Takeaway:** Predicting a False Positive destroys the macro score $4\times$ faster than a False Negative.
* Classifiers calibrated to default $\theta = 0.50$ suffer score degradation. The decision threshold must be calibrated strictly higher ($\theta^* \approx 0.80 - 0.85$).

### 2.3 The Singleton Scoring Theorem
* If an entity $e$ has no matches in $S_2 \cup S_3$ ($|T(e)| = 0$):
  * If $|P(e)| = 0 \implies F_{0.5}(e) = 1.0$ (Perfect Reward).
  * If $|P(e)| > 0 \implies F_{0.5}(e) = 0.0$ (Total Penalty).
* **Engineering Rule:** Unlinked entities must emit a clean empty string (`""`) rather than low-confidence guesses.

---

## 3. Data Preparation & Normalization Engine

### 3.1 Multi-View Name Normalizer
Produces 8 synchronized views for every entity string: `raw`, `clean`, `core`, `sorted_core`, `aliases[]`, `acronym`, `phonetic_key`, `anagram_key`.

```
[Raw String: "TATA MOTORS PVT. LTD. (f/k/a Telco)"]
   ├── 1. NFKD Accent De-accenting  ──► "tata motors pvt. ltd. (f/k/a telco)"
   ├── 2. Alias Splitter            ──► Current: "tata motors", Former: "telco"
   ├── 3. Indic Parallel Mapper     ──► "टाटा मोटर्स" ──► "tata motors"
   ├── 4. Legal Suffix Standardizer ──► "pvt ltd", "llc", "sarl", "sasu", "sci", "and fils"
   ├── 5. Core Brand Extractor      ──► "tata motors" (Stripped of generic corporate noise)
   ├── 6. Anagram Key & Phonetics   ──► "aatt mooorst" (Permutation & letter scramble invariant)
```

1. **Unicode NFKD Accent Decomposition:** Strips French/European diacritics (`é`, `è`, `ê`, `ç`, `à` $\rightarrow$ `e`, `e`, `e`, `c`, `a`).
2. **Leetspeak & Digit-Substitution Repair:** Maps digits functioning as letters inside mixed words (`5` $\rightarrow$ `s`, `1` $\rightarrow$ `l/i`, `0` $\rightarrow$ `o`, `3` $\rightarrow$ `e`, `4` $\rightarrow$ `a`).
3. **Noise Wrapper Stripping:** Eliminates structural artifacts (`[..]`, `(..)`, `--`, `<<`, `(ID: n)`, `#n`, leading `"the"`).
4. **Alias / FKA / DBA Parsing:** Splits `f/k/a`, `d/b/a`, `t/a`, `formerly` into active and historical name views.
5. **URL / Handle Normalizer:** Strips `@`, `http`, `www.`, and TLDs to compare unified domain roots.
6. **Parallel Indic Transliteration Table:** Exploit the parallel Unicode structure of Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali) where identical code offsets represent equivalent phonetic phonemes.

### 3.2 Address Normalizer & Twin Entity Separator
* **Compound Number Decomposition:** Normalizes complex multi-part address numbers (e.g., `8-2-248/1/7` $\rightarrow$ individual parts `[8, 2, 248, 1, 7]` and joined string `8224817`).
* **Ordinal & Unit Standardization:** Maps words to digits (`Seventh` $\rightarrow$ `7`, `Floor 2` $\rightarrow$ `2`, `Suite 400` $\rightarrow$ `400`).
* **Street & Locality Tokens:** Extracts order-free token sets and state/city co-occurrence aliases.

---

## 4. Candidate Search & Blocking Engine

**Objective:** Achieve $\ge 98.5\%$ Candidate Recall of true links with $\le 25$ candidates per $S_1$ entity.

```
                    ┌─────────────────────────────────────────┐
                    │      MULTI-VIEW CANDIDATE RETRIEVAL     │
                    └─────────────────────────────────────────┘
                                         │
     ┌──────────────────┬────────────────┴────────────────┬──────────────────┐
     ▼                  ▼                                 ▼                  ▼
[Exact Core Key]   [Inverted Token]              [3-Gram Prefix Index]  [Dense e5 Vector]
• Identical Core   • Distinct word match         • Typo/Scramble proof  • Semantic similarity
• Acronym match    • Posting list capped         • Sub-token overlap    • Top-20 ANN
     │                  │                                 │                  │
     └──────────────────┴────────────────┬────────────────┴──────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │    UNION & CANDIDATE PRUNING (K ≤ 25)   │
                    │   Output: candidate_pairs.tsv (Stage 1) │
                    └─────────────────────────────────────────┘
```

### Retrieval Arms:
1. **Inverted Token Index:** Indexes the first 3 distinctive words ($\ge 3$ characters). Retrieval runs in $O(1)$ amortized time.
2. **4-Character Prefix & Trigram Index:** Recovers misspelled brands (e.g. `"Starbuks"` matching `"__p_star"`).
3. **Compound Key Exact Match:** Matches `(street_word, house_number)` or `(core_name, postal_code)`.
4. **Dense Vector Head (Optional):** FAISS IVF-flat on `multilingual-e5-small` embeddings of `core_name | street | city`.

---

## 5. Feature Engineering Taxonomy (13 to 150 Dimensions)

Features are computed dynamically per candidate pair across 8 distinct functional groups:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FEATURE TAXONOMY (G1 TO G8)                                          │
├───────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────┤
│ Group │ Category             │ Concrete Signal Extracted                                               │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G1    │ Name String Metrics  │ Jaro-Winkler (raw, clean, core), Levenshtein Ratio, Token-Sort Ratio,   │
│       │                      │ Token-Set Ratio, Character Trigram Jaccard, Anagram Bag Cosine          │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G2    │ Name Token Overlap   │ Shared token IDF sum, unshared token count, legal suffix equality,      │
│       │                      │ country-independent qualifier frequency score                           │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G3    │ Numeral Geometries   │ Exact digit set overlap, House Number delta (≤ 25 shift), 1-edit digit, │
│       │                      │ Unit/Suite equality, Numeral Jaccard Ratio (**Twin Disambiguator**)     │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G4    │ Address Text         │ Street name Jaro-Winkler, City alias match, Locality Token Jaccard,     │
│       │                      │ Address Levenshtein Ratio, Missing address penalty flag                 │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G5    │ Cluster Context      │ Majority house number agreement across cluster, Cluster cohesion,       │
│       │                      │ Cluster size, Multi-source representation flag (S2 + S3)                │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G6    │ Competition / Margin │ Rank of candidate in S1 list, Rank of S1 in candidate list, Margin to   │
│       │                      │ next-best candidate, Mutual top-pick boolean indicator                 │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G7    │ Neural Logits        │ e5-small cosine similarity, bge-reranker logit (for top-5 candidates)   │
├───────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ G8    │ Metadata & Priors    │ Source origin flag (S2 vs S3), String length delta, Completeness score  │
└───────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Setup, Hard-Negative Mining, & Decision Optimization

### 6.1 Training Set Construction
* **High-Density Signal Protocol:**
  * Sample $160,000$ stratified $S_1$ reference entities across US and India.
  * Ingest **100% of True Positive links** from ground truth ($553,629$ true pairs).
  * Mine top hard negatives sharing keywords via Inverted Index ($1,462,200$ boundary pairs).
  * Total Matrix: **$2,015,829$ high-density training pairs**.

### 6.2 LightGBM Model Hyperparameters
```python
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.06,
    num_leaves=63,
    scale_pos_weight=1.8,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)
```

### 6.3 Threshold Optimization Sweep on Held-Out Validation Data
```
Decision Threshold (θ) │ Validation Macro F_0.5 │ Precision │ Recall │ Note
───────────────────────┼────────────────────────┼───────────┼────────┼──────────────────────────────────
         0.50          │        0.92140         │  0.8842   │ 0.9910 │ False positives reduce score
         0.60          │        0.95380         │  0.9320   │ 0.9880 │ Moderate precision
         0.70          │        0.97890         │  0.9690   │ 0.9840 │ High precision zone
         0.80          │        0.99110         │  0.9902   │ 0.9790 │ Excellent metric alignment
         0.84          │        0.99323 ★       │  0.9958   │ 0.9750 │ OPTIMAL LEADERBOARD POINT
         0.90          │        0.98410         │  0.9989   │ 0.9320 │ Over-pruning true matches
```
* **Calibrated Optimum:** **$\theta^* = 0.84$** ($\text{Macro } F_{0.5} = \mathbf{0.99323}$).

---

## 7. Multi-Level Validation Hierarchy

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     VALIDATION PROTOCOL                                          │
├───────┬──────────────────────────┬───────────────────────────────────────────────────────────────┤
│ Level │ Protocol                 │ Scope / Purpose                                               │
├───────┼──────────────────────────┼───────────────────────────────────────────────────────────────┤
│ V1    │ 5-Fold GroupKFold by S1  │ Evaluates feature additions and tree depth within dev universe│
├───────┼──────────────────────────┼───────────────────────────────────────────────────────────────┤
│ V2    │ Cross-Country Transfer   │ Train on US $\rightarrow$ Score India; Train India $\rightarrow$ Score US      │
│       │                          │ (Guards against overfitting; stand-in for unseen France)      │
├───────┼──────────────────────────┼───────────────────────────────────────────────────────────────┤
│ V3    │ Disjoint Region Test     │ Out-of-universe state validation (prevents city memorization) │
├───────┼──────────────────────────┼───────────────────────────────────────────────────────────────┤
│ V4    │ Public Leaderboard       │ Real test feedback; submitted only on confirmed V1 + V2 gains │
└───────┴──────────────────────────┴───────────────────────────────────────────────────────────────┘
```

**Acceptance Gate:** Any code change is merged if and only if:
1. $V_1 \text{ Macro } F_{0.5}$ improves by $\ge +0.002$.
2. $V_2$ Cross-country transfer does not degrade by $> 0.002$.
3. No single slice (twins, singletons, Indic scripts) drops by $> 0.010$.

---

## 8. Unseen Country (France) Protocol

Because the test set includes **France** without labeled training pairs, the pipeline applies strict unsupervised domain adaptation:

1. **Diacritic Invariance:** All French characters are decomposed (`NFKD`) to ASCII base letters.
2. **Frequency-Based Legal Suffix Extraction:** French legal designations (`sarl`, `sasu`, `sci`, `sas`, `and fils`, `and freres`) are matched generically without language-specific conditional branches.
3. **Statistical Distribution Sanity Checks on France Test Output:**
   * Average predicted matches per $S_1$ must mirror training ($\approx 3.2 - 3.8$).
   * Predicted singleton rate must align with baseline prior ($\approx 5.5\% - 6.5\%$).
   * Probability score histogram must remain smooth and unimodal above threshold $\theta^*$.

---

## 9. End-to-End Build Milestones & Timeline

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      DEVELOPMENT ROADMAP                                               │
├───────────┬──────────────────────────────────────────┬─────────────────────────────────────────────────┤
│ Milestone │ Objective                                │ Exit Criterion                                  │
├───────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ M1        │ Normalizer + Parallel Indic + Metric     │ Macro F0.5 test passes; Indic mapper verified   │
├───────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ M2        │ Inverted Index Candidate Search          │ ≥ 98.5% Recall at K ≤ 25; peak RAM < 1.5 GB     │
├───────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ M3        │ Baseline LightGBM GBDT + 13 Features     │ Validation Macro F0.5 ≥ 0.9800; Baseline TSV    │
├───────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ M4        │ High-Density Mining + Twin Disambiguator │ Validation Macro F0.5 ≥ 0.9930; θ* = 0.84 tuned │
├───────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ M5        │ France Generalization & Zero-OOM Stream  │ 100% test inference (1.73M S1) in < 4 mins      │
├───────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ M6        │ Validation Wrapper & Submission Package  │ validate_submission.py PASS; zip archive ready  │
└───────────┴──────────────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

## 10. Official Submission Verification & Packaging

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Deliverables Output Architecture:
```
topgun_submission.zip
  ├── output/
  │     ├── matching_results.tsv      # 1,732,545 rows (Scored submission file)
  │     └── candidate_pairs.tsv       # 1,732,545 rows (Stage-1 candidate pool)
  ├── code/
  │     └── business_entity_resolution/
  │           └── src/ (preprocess.py, blocking.py, features.py, model.py, run_pipeline.py)
  ├── requirements.txt                # Pinned dependencies (lightgbm, rapidfuzz, numpy, pandas)
  ├── README.md                       # Complete execution instructions
  └── Documentation_template.md       # Technical methodology writeup
```

---
*This document constitutes the official engineering specification for the Amazon ML Challenge 2026 winning solution.*
