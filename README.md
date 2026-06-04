# AI-Driven Analysis of OPD Appointment No-Shows

## DBA Data Analytics Capstone Project

### Overview

Outpatient Department (OPD) appointment no-shows create significant operational inefficiencies, reduce healthcare access, and result in substantial revenue loss for healthcare providers.

This project applies Artificial Intelligence (AI) and Machine Learning (ML) techniques to predict patient no-show behavior using real-world outpatient appointment data from Brazilian public hospitals.

The study develops and evaluates predictive models capable of identifying high-risk appointments at the time of scheduling, enabling proactive intervention and improved healthcare resource utilization.

---

## Research Objectives

This project addresses four key business questions:

1. Which patient characteristics most strongly predict appointment no-shows?
2. At what scheduling lead-time does no-show risk increase significantly?
3. How effective are SMS reminders across different patient groups?
4. Can geographic patterns be used to improve healthcare resource allocation?

---

## Dataset

**Source:** Kaggle Medical Appointment No-Shows Dataset

* 110,522 appointment records
* 81 neighbourhoods
* Public hospitals in Vitória, Brazil
* Study period: November 2015 – June 2016

### Key Variables

* Age
* Gender
* Scholarship Status
* Hypertension
* Diabetes
* Alcoholism
* Handicap
* SMS Received
* WaitDays (engineered feature)
* Neighbourhood

Target Variable:

* No-Show (1 = Missed Appointment, 0 = Attended)

---

## Methodology

### Data Preparation

* Data cleaning
* Feature engineering
* WaitDays calculation
* SMOTE class balancing
* 80/20 stratified train-test split

### Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Tuned XGBoost (Final Model)

### Explainability & Fairness

* SHAP (SHapley Additive exPlanations)
* Fairlearn Fairness Evaluation

---

## Model Performance

| Model               | AUC-ROC   |
| ------------------- | --------- |
| Logistic Regression | 0.586     |
| Decision Tree       | 0.691     |
| Random Forest       | 0.663     |
| XGBoost             | 0.687     |
| **Tuned XGBoost**   | **0.727** |

### Final Model Results

* AUC-ROC: 0.727
* Sensitivity (Recall): 90.4%
* F1 Score: 0.384
* Demographic Parity Difference: 0.034

The Tuned XGBoost model was the only model meeting the predefined fairness threshold and is recommended for deployment.

---

## Key Findings

### Scheduling Lead Time Matters

No-show rates increase dramatically as appointment lead time increases.

* Same-day appointments: ~6.6%
* 31+ day appointments: ~33%

### SMS Reminder Effect

SMS reminders appear ineffective in raw analysis due to selection bias.

After controlling for patient characteristics and waiting time, reminder effectiveness varies significantly across patient groups.

### Geographic Risk Concentration

The ten highest-risk neighbourhoods account for approximately 28% of all missed appointments, providing opportunities for targeted interventions.

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Fairlearn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Repository Structure

```text
AI-Driven-OPD-No-Show-AnalysisCapstone/

├── QM640_Final_Report.pdf
├── OPD_NoShow_Capstone.ipynb
├── Figures/
├── README.md
├── requirements.txt
└── outputs/
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/surendrasinghpagaria/AI-Driven-OPD-No-Show-AnalysisCapstone.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run all cells.

---

## Business Value

The developed solution demonstrates how machine learning can support:

* Reduced missed appointments
* Improved patient engagement
* Better scheduling efficiency
* Revenue recovery
* Fair and explainable healthcare decision-making

---

## Author

**Surendra Singh Pagaria**

Doctor of Business Administration (DBA)

Walsh College

QM640 – Data Analytics Capstone

June 2026
