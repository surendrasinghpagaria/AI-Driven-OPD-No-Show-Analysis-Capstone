# AI-Driven OPD No-Show Analysis

## 📌 Objective
This project aims to analyze and predict outpatient (OPD) appointment no-shows using machine learning techniques.

## 📊 Dataset
- Source: Kaggle (Medical Appointment No Shows)
- Records: 110,527 appointments
- Features include patient demographics, appointment details, and medical conditions

## 🧹 Data Cleaning Steps
- Removed irrelevant columns (PatientId, AppointmentID)
- Fixed column names
- Converted date columns to datetime
- Created WaitingDays feature
- Removed invalid values (negative waiting days, invalid age)
- Encoded categorical variables

## 📁 Folder Structure
- data/raw → Original dataset
- data/processed → Cleaned dataset
- notebooks → Data cleaning notebook

## ▶️ How to Reproduce the Project
1. Download or clone this repository
2. Open the notebook in Google Colab or Jupyter Notebook
3. Upload the raw dataset (KaggleV2-May-2016.csv)
4. Run all cells in the notebook
5. The cleaned dataset will be generated as output

## 🎯 Outcome
Prepared a clean dataset ready for machine learning models to predict patient no-shows.
