# Titanic Survival Prediction — Model Comparison

A binary classification project on the classic Titanic dataset that trains
and benchmarks five different algorithms — Logistic Regression, KNN,
Naive Bayes, Decision Tree, and SVM — to predict passenger survival, then
validates the best-performing model with 5-fold cross-validation.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## Overview

This project uses the built-in Seaborn Titanic dataset to predict whether a
passenger survived, based on demographic and travel details such as class,
sex, age, and fare. Rather than settling for a single algorithm, five
classifiers are trained on identical train/test splits so their performance
can be compared directly and fairly.

> **Headline result:** Logistic Regression and SVM (linear kernel) tie for
> the best test accuracy at **80.3%**, and SVM's performance holds up under
> 5-fold cross-validation (mean accuracy **78.7%**), confirming it isn't
> just a lucky split.

## Dataset

| Property | Value |
|---|---|
| Source | `seaborn.load_dataset("titanic")` |
| Rows (raw) | 891 |
| Rows (after cleaning) | 889 |
| Features used | 8 |
| Target | `survived` (0 = died, 1 = survived) |

**Features retained for modeling:**

| Column | Type | Description |
|---|---|---|
| `pclass` | numeric | Ticket/passenger class (1st, 2nd, 3rd) |
| `sex` | categorical → encoded | Passenger sex |
| `age` | numeric | Passenger age (missing values imputed) |
| `sibsp` | numeric | Number of siblings/spouses aboard |
| `parch` | numeric | Number of parents/children aboard |
| `fare` | numeric | Ticket fare |
| `embarked` | categorical → encoded | Port of embarkation (`S`, `C`, `Q`) |
| `alone` | boolean → encoded | Whether the passenger traveled alone |
| `survived` | binary | **Target** |

**Columns dropped:** `deck`, `embark_town`, `alive`, `class`, `who`,
`adult_male` — these were either redundant duplicates of retained columns
(e.g. `alive` mirrors `survived`, `class` mirrors `pclass`) or had excessive
missing data (`deck` had only 203 non-null values out of 891).

## Methodology

**1. Data Cleaning**
Dropped 6 redundant/sparse columns. Imputed missing `age` values with the
column mean. Dropped 2 rows with missing `embarked`, leaving 889 clean
records.

**2. Encoding**
Label-encoded `sex` and `embarked` with `LabelEncoder`, then cast the
entire dataframe to integer type.

**3. Train/Test Split**
Split into features (`X`) and target (`y`), then an 80/20 train-test split
(`random_state=42`).

**4. Model Training — Five Algorithms**
Trained and evaluated five classifiers on the same split, using
`StandardScaler`-scaled features for the distance/margin-based models
(KNN, Decision Tree, SVM) and raw features for Logistic Regression and
Naive Bayes:

| Model | Scaling used |
|---|---|
| Logistic Regression | Raw features |
| K-Nearest Neighbors (k=5) | Scaled |
| Gaussian Naive Bayes | Raw features |
| Decision Tree | Scaled |
| SVM (linear kernel) | Scaled |

**5. Cross-Validation**
Re-evaluated the best model (SVM) using 5-fold cross-validation on the
full, scaled dataset to check that its accuracy generalizes beyond a
single train/test split.

## Results

| Model | Accuracy | Precision (survived) | Recall (survived) | F1 (survived) |
|---|---|---|---|---|
| **Logistic Regression** | **0.803** | 0.74 | 0.77 | 0.75 |
| **SVM (linear)** | **0.803** | 0.74 | 0.75 | 0.75 |
| K-Nearest Neighbors | 0.775 | 0.71 | 0.71 | 0.71 |
| Gaussian Naive Bayes | 0.775 | 0.68 | 0.78 | 0.73 |
| Decision Tree | 0.770 | 0.70 | 0.71 | 0.71 |

**5-fold cross-validation (SVM, linear kernel):**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 0.803 | 0.809 | 0.787 | 0.753 | 0.785 | **0.787** |

## Key Insights

- **Logistic Regression and linear SVM are the top performers**, both
  reaching 80.3% test accuracy — a strong sign that survival is largely
  linearly separable given these features.
- **The cross-validated SVM accuracy (78.7%) closely matches its single-split
  accuracy (80.3%)**, indicating the model's performance is stable and not
  an artifact of a favorable train/test split.
- **Naive Bayes achieves the highest recall on survivors (0.78)** despite
  middling overall accuracy — useful if catching survivors matters more
  than overall correctness.
- **KNN and Decision Tree lag slightly behind** the linear models here,
  suggesting the decision boundary between survived/did-not-survive is
  fairly linear rather than highly non-linear or locally clustered.
- All five models are much better at identifying non-survivors (class 0,
  precision ~0.81–0.85) than survivors (class 1, precision ~0.68–0.74),
  reflecting the class imbalance in the dataset (more non-survivors than
  survivors).

## Tech Stack
Python · Pandas · NumPy · Seaborn · scikit-learn

## Project Structure
```
.
├── TitanicModels_ClassificationProject.ipynb   # Full analysis + modeling notebook
└── README.md                                    # This file
```

## Setup & Usage

```bash
pip install pandas numpy seaborn scikit-learn
jupyter notebook TitanicModels_ClassificationProject.ipynb
```

No external file needed — the dataset loads directly via
`seaborn.load_dataset("titanic")`. Running all cells reproduces the
cleaning, encoding, all five trained models with their accuracy/confusion
matrix/classification report, and the final cross-validation check.
