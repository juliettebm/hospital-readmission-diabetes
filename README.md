# Hospital Readmission Prediction — Diabetes 130-US

A machine learning project predicting 30-day hospital readmission for diabetic patients.  
*Built as part of my application for Master 2 in Data Science 

---

## 👩‍🔬 My Approach: Clinical Sense over "Black Box" AI

Coming from a background in clinical research (former Clinical Research Associate), my main goal for this project was to build a model that remains **interpretable for healthcare professionals**. 

Instead of throwing all the raw data into an algorithm or using mathematical dimensionality reduction like PCA (which creates unexplainable variables), I decided to engineer features based on medical logic.

I grouped the numerical variables into **3 clinical dimensions**:

| Dimension | Raw Variables Used | Clinical Concept |
|---|---|---|
| **1. Pathological Terrain** | `number_diagnoses` | Patient's comorbidity burden (inspired by the Charlson score). |
| **2. Chronic Instability** | `number_inpatient`, `number_emergency`, `number_outpatient` | The patient's recent history and healthcare utilization. |
| **3. Episode Severity** | `time_in_hospital`, `num_lab_procedures`, `num_medications` | The intensity of the current hospital stay. |

*(Note: The placement of ambiguous variables like `num_medications` was guided by empirical correlation testing rather than simple guessing).*

---

## 🛠️ Key Data Science Steps Learned & Applied

As I transition into Data Science, I made sure to apply standard methodological rigor to this dataset:

- **Preventing Data Leakage:** The train/test split was strictly performed *before* any feature engineering or weight calculation.
- **Handling Imbalanced Data:** Since readmissions only represent ~11% of the dataset, I used `class_weight='balanced'` in my Random Forest. In healthcare, missing a high-risk patient (False Negative) is a critical error.
- **Setting a Baseline:** I compared my model to a naive `DummyClassifier` to prove the algorithm actually learned meaningful patterns.

---

## 📊 Results

| Model | ROC-AUC Score |
|---|---|
| Naive baseline (Random chance) | ~ 0.500 |
| Random Forest (with my 3 clinical axes) | **0.644** |

While an AUC of 0.644 is not perfect, it represents a solid improvement over the baseline. More importantly, this score is consistent with published medical literature for models built solely on administrative hospital data (typical range: 0.62–0.70). 

My focus here was on creating an explainable, reliable proof-of-concept rather than overfitting the data to get an artificially high score.

---

## 🚀 How to Run the Notebook

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/hospital-readmission-diabetes.git
cd hospital-readmission-diabetes

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
# Download from: [https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
# Unzip and place `diabetic_data.csv` inside the `data/` folder.

# 4. Open the project
jupyter notebook 01_readmission_diabetiques.ipynb
