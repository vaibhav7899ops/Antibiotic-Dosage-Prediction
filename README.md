# Antibiotic Dosage Prediction (ICU Vitals, MIMIC-III)

This repository contains my B.Tech project from **IIIT-Delhi** on predicting whether an ICU patient’s **antibiotic dosage should be increased, decreased, or maintained** using **time-series vital signs** and other clinical features.

> ⚠️ **Disclaimer**  
> This is a **research / educational** project only.  
> It is **not** a medical device and **must not** be used for real clinical decision-making.

---

## 1. What this project does

Given recent ICU information for a patient (mainly vital signs), the model tries to answer:

> **Should the current antibiotic dose be _decreased_, _kept same_, or _increased_?**

High-level steps (implemented inside `final.ipynb`):

1. Load preprocessed patient-level data (derived from **MIMIC-III** ICU database).
2. Perform basic cleaning and feature processing.
3. Do k means clustering to generate pseudo lablels
4. Train a **Random Forest classifier** (and optionally baseline models).
5. Evaluate model performance using standard classification metrics.
6. Export final outputs to CSV files for analysis / plotting.

The full pipeline — from loading data to saving results — lives inside the notebook **`final.ipynb`**.

---

## 2. Repository structure

Current layout of this repo:

```text
Antibiotic-Dosage-Prediction/
├── final.ipynb                # Main notebook: preprocessing, modelling, evaluation
├── output_with_is_dead.csv    # Model output with an outcome flag/column (e.g. is_dead)
├── output_with_prediction.csv # Model output with predicted dosage class / probabilities
└── README.md                  # Project documentation (this file)
  
