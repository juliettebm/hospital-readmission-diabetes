# 🏥 Hospital Readmission Prediction — Diabetes 130-US Hospitals

Predictive modelling project for 30-day hospital readmission risk in diabetic patients, built on the [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset (UCI Machine Learning Repository).

---

## Objective

Predict whether a diabetic patient will be readmitted to hospital within 30 days of discharge, using only three clinically interpretable composite dimensions derived from administrative data.

---

## Project Structure

```
hospital-readmission-project/
│
├── data/
│   ├── diabetic_data.csv              # Raw dataset (UCI)
│   ├── diabetic_data_clean.csv        # After cleaning (output of notebook 01)
│   └── features.csv                   # Engineered features (output of notebook 02)
│
├── models/
│   ├── readmission_model.pkl          # Trained Random Forest classifier
│   └── feature_params.pkl             # MinMaxScaler parameters (train set only)
│
├── Plot/                              # All saved visualisations
│
├── notebook/
│   ├── 01_data_preparation.ipynb          # Data loading, cleaning, univariate analysis
│   ├── 02_feature_engineering.ipynb       # Feature construction, correlation analysis
│   ├── 03_modelling.ipynb                 # Model training, evaluation, explainability
│
├──app.py                             # Streamlit clinical decision-support interface
│
├── .gitignore 
│
└── README.md
                            
```

---

## Notebooks

### `01_data_preparation.ipynb`
- Loads the raw dataset (101,766 patients, 50 variables)
- Detects and replaces disguised missing values (`'?'` → `NaN`)
- Drops columns with >60% missing data
- Performs univariate analysis on 8 key clinical variables
- Exports `diabetic_data_clean.csv`

### `02_feature_engineering.ipynb`
- Train/test split performed **before** any feature engineering (no data leakage)
- Correlation analysis on training set only (Pearson / point-biserial)
- Expert-driven dimensionality reduction: 7 variables → 3 clinical dimensions
- MinMaxScaling fitted on training set, applied to test set
- Correlation-based weighting within each dimension
- Exports `features.csv` and `feature_params.pkl`

### `03_modelling.ipynb`
- Reconstructs train/test split from the `split` column in `features.csv`
- Trains a DummyClassifier naive baseline (`strategy='most_frequent'`)
- Trains a Random Forest with 5-fold cross-validation on the training set
- Evaluates on the held-out test set (ROC-AUC, sensitivity, specificity)
- Produces evaluation dashboard: confusion matrix, ROC curve, feature importance
- Exports `readmission_model.pkl`

---

## Clinical Framework

The 7 original numeric variables are reduced to 3 interpretable dimensions:

| Dimension | Variables | Clinical Concept |
|---|---|---|
| **Pathological Terrain** | `number_diagnoses` | Comorbidity burden (cf. Charlson score) |
| **Chronic Instability** | `number_inpatient`, `number_emergency`, `number_outpatient` | Past healthcare utilisation |
| **Episode Severity** | `time_in_hospital`, `num_lab_procedures`, `num_medications` | Current episode intensity |

Each dimension is built by MinMax scaling on the training set, then correlation-weighted aggregation.

---

## Results

| Metric | Value |
|---|---|
| Naive baseline AUC | ≈ 0.500 |
| Random Forest AUC (CV) | ≈ 0.644 |
| Train set | 81,412 patients |
| Test set | 20,354 patients |
| Class balance | 54% not readmitted / 46% readmitted |

**Chronic Instability** (prior healthcare utilisation) is the strongest predictor (~65% feature importance), consistent with established clinical evidence.

---

## Methodological Choices

- **No PCA** — expert-driven reduction preserves clinical interpretability
- **Train/test split before feature engineering** — strict leakage prevention
- **`class_weight='balanced'`** — handles class imbalance without synthetic data generation
- **Correlation weighting** used as a pragmatic signal-strength heuristic, not a causal claim
- **DummyClassifier baseline** — defines a performance floor for clinical relevance assessment

---

## Stack

```
Python 3.x · pandas · numpy · scikit-learn · matplotlib · seaborn · joblib · streamlit
```

---

## Streamlit App

A clinical decision-support interface allows manual input of patient data and returns a readmission risk score with clinical explainability.

```bash
streamlit run app.py
```

The app uses `readmission_model.pkl` if available, or falls back to a heuristic approximation based on the trained model's correlation coefficients.

---

## Data Source

Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.  
Dataset: [UCI Machine Learning Repository #296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
