# Alzheimer's Disease Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-189FDD?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat&logo=plotly&logoColor=white)

---

## Overview

This project was developed as part of a machine learning course in a Data Science program. It addresses one of the most pressing challenges in modern medicine: the early detection of Alzheimer's disease.

Alzheimer's disease is a progressive neurodegenerative disorder affecting over 55 million people worldwide. Its diagnosis typically occurs at a late stage, when cognitive decline has already significantly impacted the patient's quality of life. Early prediction models, built on clinical and genetic data, offer a promising avenue for timely intervention, better patient management, and more targeted clinical trial recruitment.

The core objective of this project is to build, train, evaluate, and compare four supervised classification models capable of predicting a patient's diagnosis from tabular clinical data. The project follows the full machine learning pipeline — from raw data exploration to an interactive Streamlit dashboard with a patient-facing prediction interface.

---

## Relevance to the Data Science Profession

Data Science applied to healthcare is one of the fastest-growing fields in the industry. This project reflects several competencies that are central to the profession:

**Data literacy and domain understanding.** A data scientist working in health or life sciences must be able to understand the clinical meaning of each variable, not just its statistical properties. In this project, features like the MMSE score or the APOE4 genotype are not just numbers — they carry real biological and diagnostic significance that directly shapes modeling decisions.

**End-to-end pipeline ownership.** In a professional context, a data scientist is rarely handed a clean, ready-to-model dataset. This project covers the full workflow: data cleaning (in R), exploratory analysis, feature engineering, model training, hyperparameter selection, evaluation with multiple metrics, and final deployment through an interactive application.

**Model selection and critical evaluation.** Choosing a model is not about picking the one with the highest accuracy figure. A data scientist must weigh interpretability against performance, understand the cost of false negatives in a medical context (failing to detect a sick patient is more dangerous than a false alarm), and use metrics like AUC-ROC that are more informative than raw accuracy on imbalanced datasets.

**Communication of results.** The Streamlit dashboard is not just a technical deliverable — it is a communication tool. Data scientists must be able to translate complex model outputs into actionable, understandable information for non-technical stakeholders, whether they are clinicians, project managers, or patients.

---

## Dataset

**Source:** ADNI — Alzheimer's Disease Neuroimaging Initiative

The ADNI dataset is a well-established benchmark in the medical machine learning literature. It collects longitudinal clinical, genetic, imaging, and biomarker data from participants across North America.

| Property | Value |
|----------|-------|
| Number of patients | 627 |
| Number of features | 10 |
| Target variable | `DX.bl` — Baseline diagnosis |
| Missing values | 0 (after cleaning) |
| Class distribution | CN: 190 / LMCI: 304 / AD: 133 |

The target variable contains three classes:

- **CN (Cognitively Normal):** No cognitive impairment detected. The patient presents a healthy cognitive profile.
- **LMCI (Late Mild Cognitive Impairment):** An intermediate stage. The patient shows measurable cognitive decline that exceeds normal aging but does not yet meet the criteria for Alzheimer's. LMCI is the most difficult class to predict reliably.
- **AD (Alzheimer's Disease):** The patient has been diagnosed with Alzheimer's. This is the class with the highest clinical priority for correct detection.

### Feature Description

| Variable | Type | Clinical Description |
|----------|------|----------------------|
| `DX.bl` | Categorical | Target — Baseline diagnosis (CN / LMCI / AD) |
| `AGE` | Continuous | Patient age in years |
| `PTGENDER` | Binary | Gender (0 = Female, 1 = Male) |
| `PTEDUCAT` | Integer | Years of formal education |
| `PTETHCAT` | Categorical | Ethnicity |
| `PTRACCAT` | Categorical | Race |
| `APOE4` | Integer | Number of APOE4 risk alleles (0, 1, or 2) |
| `MMSE` | Integer | Mini-Mental State Examination score (0 = severe impairment, 30 = normal) |
| `imputed_genotype` | Boolean | Whether the genotype was statistically imputed rather than directly measured |
| `APOE Genotype` | Integer | Full APOE genotype code — removed due to high collinearity with `APOE4` |

**Note on APOE4:** The APOE4 allele is the strongest known genetic risk factor for late-onset Alzheimer's disease. Carriers of two APOE4 alleles have an estimated 8 to 12 times higher risk compared to non-carriers. Its presence in this dataset as the most predictive genetic feature is consistent with the broader clinical literature.

**Note on MMSE:** The Mini-Mental State Examination is a 30-point questionnaire used extensively in clinical practice to measure cognitive impairment. Scores below 24 generally indicate some level of cognitive deficit. In this dataset, MMSE is the single most predictive feature across all models.

---

## Project Structure

```
alzheimer-prediction/
|
|-- alzheimer_dashboard.py                    # Main Streamlit application
|-- alzheimer_clean.csv                       # Cleaned dataset (627 patients)
|-- alzheimer_nettoyage_version_finale.R      # R script for data cleaning
|-- alzheimer_models_corrige_nouveau2.ipynb   # Jupyter notebook for exploration
|-- README.md                                 # Project documentation
```

---

## Installation

### Requirements

- Python 3.8 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/alzheimer-prediction.git
cd alzheimer-prediction
```

### Step 2 — Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
plotly>=5.13.0
```

---

## Running the Dashboard

```bash
streamlit run alzheimer_dashboard.py
```

The application will open automatically at `http://localhost:8501`.

Both `alzheimer_dashboard.py` and `alzheimer_clean.csv` must be located in the same directory.

---

## Machine Learning Pipeline

The project follows a structured, professional pipeline covering every stage from raw data to deployed model.

### 1. Data Cleaning (R)

The raw ADNI data was processed in R to handle inconsistent encodings, standardize variable types, remove redundant columns, and verify the absence of missing values. The output is `alzheimer_clean.csv`.

### 2. Exploratory Data Analysis

The EDA phase investigates the distribution of each variable, identifies class imbalance, and highlights the most discriminative features. Key observations include:

- The dataset is imbalanced: LMCI represents approximately 48% of cases, making it the majority class and the hardest to distinguish from the others.
- MMSE scores decrease clearly and consistently with disease severity: CN patients average around 28, LMCI around 27, and AD patients around 23. This single feature carries the most predictive signal.
- The number of APOE4 alleles is strongly associated with AD diagnosis, which is consistent with established genetic risk research.

### 3. Correlation Analysis

A full correlation matrix was computed after one-hot encoding of categorical variables and numeric encoding of the target. The most notable finding is the strong correlation between `APOE Genotype` and `APOE4` (r > 0.95), which constitutes a case of multicollinearity. `APOE Genotype` was removed from all models to avoid redundant information and numerical instability.

### 4. Data Preparation

- Categorical variables were encoded using one-hot encoding (`pd.get_dummies`)
- The target variable was encoded ordinally using `LabelEncoder` (CN=0, AD=1, LMCI=2)
- The dataset was split into training (80%) and test (20%) sets using stratified sampling to preserve class proportions in both splits

### 5. Model Training

Four models were trained on the same training split and evaluated on the same held-out test split to ensure a fair comparison.

### 6. Evaluation

Each model was assessed using:

- **Accuracy** — overall proportion of correct predictions
- **Precision, Recall, F1-score** — per-class metrics from the classification report
- **Confusion matrix** — to identify which classes are being confused
- **Feature importance** — to understand which variables drive each model's decisions
- **ROC curve and AUC** — macro one-vs-rest averaging across all three classes, which is more informative than accuracy alone on an imbalanced dataset

---

## Models

### Logistic Regression

Logistic regression serves as the baseline model. It learns a set of linear coefficients, one per feature, and combines them to estimate class probabilities. Despite its simplicity, it achieved the highest accuracy on this dataset (69.84%), which suggests that a significant portion of the decision boundary is approximately linear in the feature space.

- Hyperparameter: `max_iter=1000`
- Strength: highly interpretable, fast to train, reliable baseline
- Limitation: cannot capture non-linear interactions between features

### Decision Tree

A decision tree builds a series of binary splitting rules on individual features, such as "Is MMSE less than 24?" The depth was limited to 4 to prevent overfitting. While accuracy is slightly lower, the tree structure offers a clear and visual explanation of decisions — particularly valuable in medical contexts where interpretability matters to clinicians.

- Hyperparameter: `max_depth=4`
- Strength: fully interpretable, visually communicable, no assumptions on data distribution
- Limitation: sensitive to small changes in data; prone to overfitting without depth constraints

### Random Forest

Random Forest is an ensemble method that trains 100 decision trees, each on a random bootstrap sample of the data and a random subset of features. The final prediction is determined by majority vote. This randomness reduces variance and makes the model robust to overfitting.

- Hyperparameters: `n_estimators=100`, `max_depth=5`
- Strength: strong generalization performance, robust to noise and outliers
- Limitation: less interpretable than a single tree; slower inference than linear models

### XGBoost

XGBoost implements gradient boosting, a sequential ensemble technique in which each new tree is trained to correct the residual errors of all previous trees. It uses regularization (L1 and L2) to prevent overfitting and is typically among the highest-performing models on tabular data.

- Hyperparameters: `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`
- Strength: best AUC on this dataset (0.886); excellent performance on structured data
- Limitation: more hyperparameters to tune; less interpretable than simpler models

---

## Results

### Performance Summary

| Rank | Model | Accuracy | AUC (macro) |
|------|-------|----------|-------------|
| 1 | Logistic Regression | 69.84% | 0.869 |
| 2 | Random Forest | 69.0% | 0.855 |
| 3 | XGBoost | 68.3% | 0.886 |
| 4 | Decision Tree | 67.5% | 0.847 |

All results are computed on the held-out test set (20% of data, 126 patients) using stratified splitting.

### Key Observations

**On accuracy vs. AUC:** XGBoost achieves the highest AUC (0.886) despite not having the highest accuracy. On imbalanced multi-class datasets, AUC is the more meaningful metric because it measures discriminative ability independently of the decision threshold, whereas accuracy is biased toward the majority class (LMCI).

**On feature importance:** MMSE and APOE4 consistently rank as the two most important features across all four models. This aligns with established clinical knowledge: cognitive testing and genetic risk profiling are the two primary diagnostic tools in Alzheimer's research.

**On class difficulty:** LMCI is systematically the most difficult class to classify correctly. This is expected, as it represents a transitional state between normal cognition and dementia. Distinguishing LMCI from CN and from early-stage AD is a known challenge even for trained clinicians.

**On the cost of false negatives:** In a clinical context, failing to detect an AD patient (false negative) is generally more costly than a false alarm (false positive). A production-grade model for this use case would likely be calibrated to optimize recall for the AD class, potentially at the cost of some precision.

---

## Dashboard Description

The Streamlit dashboard is organized into six pages accessible from the sidebar:

| Page | Content |
|------|---------|
| Home | Project overview, KPI metrics, class distribution, summary table of model results |
| Data Exploration | Distribution of the target variable, MMSE boxplot, age distribution, APOE4 analysis, categorical variable explorer |
| Correlation Analysis | Full correlation heatmap, ranked bar chart of feature-to-target correlations, multicollinearity discussion |
| Model Training | Per-model classification report, confusion matrix, top-10 feature importance, model explanation card |
| Model Comparison | Accuracy comparison chart, ROC curves (macro and per-class), AUC summary, full comparative table |
| Patient Prediction | User-facing form to input patient characteristics and receive a predicted diagnosis with class probabilities |

The dashboard supports both dark and light modes and is fully interactive using Plotly charts.

---

## Technologies Used

| Technology | Role |
|------------|------|
| Python 3.x | Core programming language |
| R | Data cleaning and preprocessing |
| Streamlit | Interactive web dashboard |
| Pandas / NumPy | Data manipulation and numerical computing |
| Scikit-learn | Model training, evaluation, and preprocessing utilities |
| XGBoost | Gradient boosting classifier |
| Plotly | Interactive data visualization |
| Jupyter Notebook | Exploratory analysis and prototyping |

---

## Authors

This project was developed as part of a Machine Learning course in a Data Science program.

---

## Disclaimer

The predictions generated by this application are based on statistical models trained on a limited clinical dataset. They are intended for educational and research purposes only. They do not constitute a medical diagnosis and should not be used as a substitute for professional clinical evaluation.
