# AI-Driven OPD No-Show Analysis — Capstone Project

**Author:** Surendra Singh Pagaria
**Institution:** Walsh College
**Program:** Data Analytics Capstone
**Year:** 2026

---

## Project Overview

Outpatient department (OPD) appointment no-shows — instances where a scheduled patient fails to attend without prior cancellation — represent one of the most operationally damaging and financially costly inefficiencies in modern healthcare delivery. Across healthcare systems globally, no-show rates consistently range from 15% to 30%, translating into idle clinical staff, wasted infrastructure capacity, delayed care for waiting patients, and estimated revenue losses of 3–14% per facility annually.

This capstone project applies machine learning and statistical analysis to **110,522 real OPD appointment records** from Vitória, Espírito Santo, Brazil (Kaggle, 2016) to:

- Identify the strongest patient-level predictors of no-show behaviour
- Quantify the effect of scheduling lead time (WaitDays) on no-show probability
- Resolve the SMS reminder paradox through confounding analysis
- Segment neighbourhoods by no-show risk using K-Means clustering
- Build a predictive model (XGBoost) achieving AUC = 0.727

---

## Research Questions

| # | Research Question | Key Finding |
|---|---|---|
| RQ1 | What patient-level factors best predict no-show risk? | WaitDays (#1 SHAP), Age, Scholarship are top predictors |
| RQ2 | Is there a WaitDays threshold beyond which risk escalates sharply? | 5.1% (same-day) → 33.0% (31+ days); 15-day cap recommended |
| RQ3 | Does SMS reminder effectiveness vary by scheduling lead time? | SMS reduces risk for short-wait (≤7 days) but not long-wait bookings |
| RQ4 | Can neighbourhoods be segmented by no-show risk for targeted intervention? | 4 K-Means clusters identified; top 10 neighbourhoods = 28% of all no-shows |

---

## Dataset

**Source:** [Kaggle — Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

**File:** `KaggleV2-May-2016.csv` (10.2 MB — download from Kaggle link above)

> ⚠️ The dataset is not included in this repository due to its size. Download it from Kaggle and place it in the root folder before running the notebook.

**Key Statistics:**
- Total records: 110,522 appointments
- No-show rate: 20.19% (22,319 missed appointments)
- Features: 14 variables including Age, Gender, WaitDays, SMS_received, Scholarship, chronic conditions, and Neighbourhood
- Date range: 2015–2016, Vitória, Brazil

---

## Repository Structure

```
AI-Driven-OPD-No-Show-Analysis-Capstone/
│
├── README.md                            ← You are here
├── requirements.txt                     ← Python dependencies
├── OPD_NoShow_Capstone_Local.ipynb      ← Main analysis notebook
│
├── figures/                             ← All generated EDA figures
│   ├── fig5_1.png                       ← Figure 1: Class Distribution
│   ├── fig5_2.png                       ← Figure 2: No-Show Rate by Age
│   ├── fig5_3.png                       ← Figure 3: No-Show Rate by WaitDays
│   ├── fig5_4.png                       ← Figure 4: SMS Reminder Paradox
│   ├── fig5_5.png                       ← Figure 5: Chronic Condition & Scholarship
│   └── fig5_6.png                       ← Figure 6: Geographic Distribution
│
├── report/
│   └── OPD_NoShow_Report_Final.docx     ← Full capstone report (APA7 format)
│
└── data/
    └── README.md                        ← Dataset description and download instructions
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the repository
```bash
git clone https://github.com/surendrasinghpagaria/AI-Driven-OPD-No-Show-Analysis-Capstone.git
cd AI-Driven-OPD-No-Show-Analysis-Capstone
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the dataset
Download `KaggleV2-May-2016.csv` from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments) and place it in the root folder.

### Step 4 — Run the notebook
```bash
jupyter notebook OPD_NoShow_Capstone_Local.ipynb
```
Run all cells top-to-bottom (Kernel → Restart & Run All).

---

## Notebook Structure

| Cell | Description |
|---|---|
| Cell 1 | Install dependencies |
| Cell 2 | Load dataset, preprocessing, SMOTE (prevents data leakage) |
| Cell 3 | Figure 1 — Class distribution & imbalance |
| Cell 4 | Figure 2 — No-show rate by age group |
| Cell 5 | Figure 3 — No-show rate by WaitDays bucket + Kruskal-Wallis test |
| Cell 6 | Figure 4 — SMS reminder paradox (raw vs. adjusted) |
| Cell 7 | Figure 5 — Chronic condition & scholarship subgroup rates |
| Cell 8 | Figure 6 — Geographic K-Means clustering |
| Cell 9 | RQ1: Logistic Regression odds ratios + XGBoost + SHAP |
| Cell 10 | RQ3: SHAP dependence plot (SMS × WaitDays interaction) |
| Cell 11 | Model benchmarking: ROC comparison, cross-validation, fairness audit |
| Cell 12 | Final tuned XGBoost model (target AUC > 0.75) |
| Cell 13 | Results summary table |

---

## Key Results

### Model Performance

| Model | AUC-ROC | Recall | F1 Score | DPD (Fairness) |
|---|---|---|---|---|
| Logistic Regression | 0.586 | 52.1% | 0.325 | +0.510 ❌ |
| Decision Tree | 0.691 | 57.8% | 0.398 | +0.338 ❌ |
| Random Forest | 0.663 | 43.4% | 0.362 | +0.123 ❌ |
| XGBoost | 0.687 | 90.4% | 0.384 | +0.209 ❌ |
| **Tuned XGBoost** | **0.727** | — | — | **+0.034 ✓** |

> ⚠️ All models fail the Demographic Parity Difference fairness threshold (< 0.05). Fairness-aware retraining is recommended before clinical deployment.

### No-Show Rate by WaitDays

| WaitDays Bucket | No-Show Rate |
|---|---|
| 0 (Same-day) | 6.6% |
| 1–7 days | 25.0% |
| 8–15 days | 31.2% |
| 16–30 days | 32.7% |
| 31+ days | 33.0% |

### Top Predictors (XGBoost SHAP)
1. WaitDays (scheduling lead time)
2. Age
3. SMS_received
4. Scholarship
5. Neighbourhood

---

## Methodology

- **Data Leakage Prevention:** SMOTE applied to training set only (after 80/20 stratified split)
- **Class Imbalance:** SMOTE (Synthetic Minority Oversampling Technique) on training set; AUC used as primary metric
- **Statistical Tests:** Chi-Square (categorical predictors), Kruskal-Wallis H-test (WaitDays), interaction logistic regression (SMS paradox)
- **Explainability:** SHAP TreeExplainer for XGBoost feature importance
- **Fairness:** Fairlearn demographic parity difference by gender

---

## Citation

If you use this work, please cite:

```
Pagaria, S. S. (2026). AI-Driven OPD No-Show Analysis Capstone [Source code].
GitHub. https://github.com/surendrasinghpagaria/AI-Driven-OPD-No-Show-Analysis-Capstone
```

**Dataset citation:**
```
Arroba, J. (2016). Medical appointment no-shows [Dataset]. Kaggle.
https://www.kaggle.com/datasets/joniarroba/noshowappointments
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Surendra Singh Pagaria**
Walsh College | Data Analytics Program
GitHub: [@surendrasinghpagaria](https://github.com/surendrasinghpagaria)
