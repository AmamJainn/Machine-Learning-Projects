# Insurance Charges Prediction — EDA, Feature Engineering & Linear Regression

End-to-end pipeline on the **Medical Cost Personal Insurance** dataset
(`insurance.csv`): exploratory data analysis, data cleaning, feature
engineering, statistical feature selection, and a **Linear Regression model**
to predict individual medical insurance `charges`.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Source file | `insurance.csv` |
| Rows | 1,338 |
| Columns | 7 |
| Missing values | 0 |
| Target variable | `charges` (medical insurance cost, continuous) |

**Raw columns:**

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age of the primary beneficiary |
| `sex` | categorical | `male` / `female` (675 male, 662 female) |
| `bmi` | numeric | Body Mass Index |
| `children` | numeric | Number of dependents covered |
| `smoker` | categorical | `yes` / `no` |
| `region` | categorical | Residential area (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | numeric | Individual medical costs billed by insurance (target) |

---

## 2. Project Workflow

The notebook (`Insurance_Charges_Prediction.ipynb`) is organized into four
stages: **EDA → Cleaning/Encoding → Feature Engineering & Selection →
Model Building**.

### Stage 1 — Exploratory Data Analysis (EDA)
- Loaded the data and printed row/column counts (1,338 rows, 7 columns).
- Reviewed `describe()`, `info()`, and confirmed **no missing values**.
- Plotted **histograms with KDE** for numeric columns (`age`, `bmi`,
  `children`, `charges`).
- Plotted **count plots** for `children`, `sex`, `smoker`.
- Plotted **box plots** for the numeric columns to check outliers
  (`charges` shows a long right tail / outliers).
- Plotted a **correlation heatmap** of raw numeric features.

### Stage 2 — Data Cleaning & Encoding
- Created a working copy, `cleaned_dataset`, and removed duplicate rows
  with `drop_duplicates()`.
- Verified no missing values and reviewed dtypes post-cleaning.
- Encoded categorical variables:
  - `sex` → mapped to binary and renamed **`is_female`** (`male`=0, `female`=1)
  - `smoker` → mapped to binary and renamed **`is_smoker`** (`yes`=1, `no`=0)
  - `region` → **one-hot encoded** (`drop_first=True`) → `region_northwest`,
    `region_southeast`, `region_southwest` (`northeast` = baseline)
- Cast the full dataframe to integer type.

### Stage 3 — Feature Engineering & Statistical Selection
- Engineered **`bmi_category`** by binning `bmi` into standard clinical
  ranges (`Underweight` ≤18.5, `Normal` 18.5–24.9, `Overweight` 24.9–29.9,
  `Obese` >29.9), then one-hot encoded it (`drop_first=True`, `Underweight`
  = baseline).
- Scaled `age`, `bmi`, and `children` using **StandardScaler**.
- **Pearson correlation** of each feature against `charges`:

  | Feature | Correlation |
  |---|---|
  | `is_smoker` | **0.787** |
  | `age` | 0.298 |
  | `bmi_category_Obese` | 0.200 |
  | `bmi` | 0.196 |
  | `region_southeast` | 0.074 |
  | `children` | 0.067 |
  | `region_northwest` | -0.039 |
  | `region_southwest` | -0.044 |
  | `is_female` | -0.058 |
  | `bmi_category_Normal` | -0.104 |
  | `bmi_category_Overweight` | -0.121 |

- **Chi-squared test** (categorical features vs. `charges` binned into
  quartiles via `pd.qcut`, α = 0.05):

  | Feature | p-value | Decision |
  |---|---|---|
  | `is_smoker` | ~0.0 | Keep |
  | `region_southeast` | 0.0011 | Keep |
  | `is_female` | 0.0165 | Keep |
  | `bmi_category_Obese` | 0.0365 | Keep |
  | `region_southwest` | 0.165 | Drop |
  | `bmi_category_Overweight` | 0.236 | Drop |
  | `bmi_category_Normal` | 0.295 | Drop |
  | `region_northwest` | 0.769 | Drop |

- **Final feature set** (`final_df`) used for modeling:
  ```
  age, is_female, bmi, children, is_smoker, charges,
  region_southeast, bmi_category_Obese, region_northwest
  ```
  > Note: `region_northwest` was statistically insignificant per the
  > Chi-squared test (p = 0.77) but was still included in `final_df` —
  > worth revisiting if you want a strictly statistically-justified
  > feature set (see *Next Steps*).

### Stage 4 — Model Building
- Split data with `train_test_split` (80% train / 20% test, `random_state=42`).
- Trained a **Linear Regression** model (`sklearn.linear_model.LinearRegression`)
  on the selected features to predict `charges`.
- Evaluated on the held-out test set:

  | Metric | Score |
  |---|---|
  | **R²** | **0.804** |
  | **Adjusted R²** | **0.798** |

---

## 3. Key Findings

- **Smoking status is by far the strongest predictor** of insurance charges
  (Pearson r ≈ 0.79) — this dominates the model's predictive power.
- **Age** and **obesity** (`bmi_category_Obese`) are moderate, statistically
  significant contributors to higher charges.
- A simple **Linear Regression model explains ~80% of the variance**
  (R² = 0.804, Adjusted R² = 0.798) in insurance charges using just 8
  features — a strong result for a linear baseline.
- Region generally has a weak effect; only `region_southeast` was
  statistically significant in the Chi-squared test.
- Sex has a small but statistically significant association with charges
  despite a low linear correlation, suggesting a possible non-linear or
  interaction effect (e.g., combined with smoking).

---

## 4. Project / File Structure

```
.
├── Insurance_Charges_Prediction.ipynb   # Main analysis + modeling notebook
├── insurance.csv                        # Raw dataset (must be placed alongside notebook)
└── README.md                            # This file
```

---

## 5. Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## 6. How to Run

1. Place `insurance.csv` in the same directory as the notebook.
2. Launch Jupyter:
   ```bash
   jupyter notebook Insurance_Charges_Prediction.ipynb
   ```
3. Run all cells top to bottom. The notebook will produce:
   - Distribution, count, box, and correlation plots
   - A cleaned, encoded, and scaled dataframe (`cleaned_dataset`)
   - Pearson correlation and Chi-squared significance tables
   - A final model-ready dataframe (`final_df`)
   - A trained Linear Regression model with **R² ≈ 0.80** on the test set

---

## 7. Possible Next Steps

- Re-run the model **excluding `region_northwest`** (which was statistically
  insignificant, p = 0.77) to see if it changes performance — a cleaner,
  statistically-justified feature set would drop it.
- Apply a **log-transform to `charges`** before modeling, since it is
  right-skewed with outliers — this typically improves linear model fit.
- Try non-linear models (Random Forest, Gradient Boosting, XGBoost) to
  capture potential interaction effects (e.g., `is_smoker × bmi_category_Obese`)
  that a linear model can't represent.
- Check residual plots and multicollinearity (VIF) to validate the linear
  regression's assumptions.
- Use k-fold cross-validation instead of a single train/test split for a
  more robust performance estimate.
