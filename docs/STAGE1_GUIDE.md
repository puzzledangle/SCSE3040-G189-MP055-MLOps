# Stage 1 Guide — Data Checks & Exploratory Data Analysis (EDA)

**Group:** G189  
**Problem:** MP055 (Tea Leaf Plucking — Indore)  
**Lead Owner:** Data Lead (Parth Makkar — S24CSEU2337)

---

## 1. Overview of Stage 1
In Stage 1, we establish version control for datasets with DVC, compute data fingerprints (SHA-256), and audit raw data (`train.csv`) across seven real-world data anomaly categories. The findings are recorded in `reports/findings.yaml`.

---

## 2. The Seven Data Checks: Viva Preparation & Guiding Questions

### Check 1: Leaking Column
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** A feature recorded only after plucking is completed at the end of the shift, directly reflecting the outcome. Using it leads to data leakage.
* **Guiding Questions for Viva:**
  1. *Which column exists in `train.csv` but is completely absent from `holdout_features.csv`?*
  2. *Why would factory gate weight (`kg_weighed_at_factory`) be unavailable at the time of morning shift scheduling?*
  3. *What would happen to the model's test performance if this column were retained in training?*
* **MP055 Finding:** Present (`kg_weighed_at_factory`).

---

### Check 2: Unit Mixup
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** Different field offices or operational sections recording the same metric in disparate measurement units (e.g. grams vs. kg, months vs. years).
* **Guiding Questions for Viva:**
  1. *When comparing feature averages across `section` (`Upper Garden`, `Lower Garden`, `River Side`), do any values differ by standard conversion factors (10, 12, 60, 100, 1000)?*
  2. *Why must the factor be an exact unit conversion rather than a statistical ratio?*
  3. *Did any tea section record bush age, experience, or rainfall in alternative units?*
* **MP055 Finding:** Absent (`false`). All sections record consistently.

---

### Check 3: Sentinel Values
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** Special negative or dummy numbers (e.g., `-1`, `-999`) typed into numeric fields to represent missing or broken readings.
* **Guiding Questions for Viva:**
  1. *Can weekly rainfall (`rain_mm_week`) physically be negative in nature?*
  2. *How many records in `train.csv` contain `-1` for weekly rainfall?*
  3. *Why must we impute or treat sentinels instead of passing `-1` directly into gradient descent / tree models?*
* **MP055 Finding:** Present (`rain_mm_week`, sentinel value `-1`, count: `53`).

---

### Check 4: Duplicate Rows
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** Exact repeated rows created during data ingestion or merging.
* **Guiding Questions for Viva:**
  1. *What pandas function counts redundant copies? (`df.duplicated().sum()`)*
  2. *How many extra rows need to be purged during cleaning?*
  3. *Why does the marking tool require the count of EXTRA copies rather than all copies?*
* **MP055 Finding:** Present (count: `30` extra duplicate rows).

---

### Check 5: Stuck Sensor
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** Hardware sensor failure where consecutive daily timestamps output an unvarying flatline reading.
* **Guiding Questions for Viva:**
  1. *Did any weather station or operational meter freeze on a single static number across consecutive calendar dates?*
  2. *How do we differentiate between multiple workers on the same day having the same weather vs. a stuck multi-day sensor?*
  3. *What date range would indicate a stuck sensor if present?*
* **MP055 Finding:** Absent (`false`).

---

### Check 6: Target Typos
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** Typographical error where a data entry operator accidentally appended an extra zero, inflating the target ~10x.
* **Guiding Questions for Viva:**
  1. *What is the maximum normal daily yield for a tea leaf plucker? (~54 kg)*
  2. *What values appear in the extreme upper tail of `kg_per_worker`? (e.g. 283 to 455 kg)*
  3. *How many rows exhibit this 10x multiplier?*
* **MP055 Finding:** Present (count: `11` rows).

---

### Check 7: Spelling Variants
* **Owner:** Data Lead (Parth Makkar)
* **What it means:** Informal casing, abbreviations, or whitespace discrepancies in categorical fields.
* **Guiding Questions for Viva:**
  1. *What categorical field in `train.csv` displays inconsistent casing and punctuation? (`flush`)*
  2. *How many canonical true levels exist according to clean post-launch data (`later.csv`)? (3: `first`, `second`, `monsoon`)*
  3. *How should this column be cleaned? (lowercase, strip whitespace, regex strip periods)*
* **MP055 Finding:** Present (`flush`, true levels: `3`).

---

## 3. Read-Only EDA Helper
Run the EDA tool anytime via:
```bash
python -m src.mp055.eda
```
It prints data shapes, types, head, statistical descriptions, null values, and feature alignment with `holdout_features.csv` without modifying source data.
