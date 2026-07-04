# Ford Car Price Prediction

A regression project that predicts the resale price of used Ford vehicles
from real listing data — and along the way, runs a controlled experiment
comparing **one-hot encoding vs. label encoding** to show, empirically, why
encoding choice matters for linear models.

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

Used car pricing depends on a mix of numeric factors (mileage, engine size,
age) and categorical ones (model, transmission, fuel type). This project
builds a full pipeline — from raw listings to a trained regression model —
to predict resale `price`, and specifically isolates **how much the choice
of categorical encoding affects model quality**, by training two otherwise
identical Linear Regression models that differ only in that one decision.

> **Headline result:** the one-hot encoded model explains **84% of price
> variance** (R² = 0.840), beating the label-encoded model by a full
> 10 points (R² = 0.731) — using the exact same features, split, and
> algorithm.

## Dataset

| Property | Value |
|---|---|
| File | `ford.csv` |
| Rows | 17,966 |
| Predictors | 8 |
| Target | `price` (continuous, £) |
| Missing values | None |
| Price range | £495 – £54,995 (mean ≈ £12,280) |

| Column | Type | Description |
|---|---|---|
| `model` | categorical | Ford model name (Fiesta, Focus, EcoSport, B-MAX, KA, ...) |
| `year` | numeric | Manufacture / registration year |
| `transmission` | categorical | `Manual`, `Automatic`, `Semi-Auto` |
| `mileage` | numeric | Total mileage driven |
| `fuelType` | categorical | `Petrol`, `Diesel`, etc. |
| `tax` | numeric | Annual road tax |
| `mpg` | numeric | Fuel efficiency (miles per gallon) |
| `engineSize` | numeric | Engine displacement (litres) |
| `price` | numeric | **Target** — vehicle resale price |

**Data-quality notes worth flagging:**
- `year` contains a value of **2060** — an impossible future date, almost
  certainly a data-entry typo.
- `engineSize` contains entries of **0.0**, which isn't physically
  meaningful for a combustion vehicle and likely reflects missing data.

These weren't corrected in this pass, so they should be treated as a
caveat on the reported model scores below.

## Methodology

**1. Exploratory Data Analysis**
Examined the distribution of `price`, plotted a correlation heatmap of
numeric features, and visualized `price` against `year`, `mileage`,
`engineSize`, `transmission`, `fuelType`, and `model` using box and scatter
plots to surface obvious pricing patterns before modeling.

**2. Encoding — Two Competing Strategies**
Built two independent feature sets from the same raw data to isolate the
effect of encoding choice:

| Strategy | How it works | Risk |
|---|---|---|
| **One-Hot Encoding** | Each category becomes its own binary column (`pd.get_dummies`, `drop_first=True`) | None for nominal data |
| **Label Encoding** | Each category mapped to an integer (`LabelEncoder`) | Implies a false numeric order between categories |

Numeric columns were standardized with `StandardScaler` in both pipelines.

**3. Model Training**
Trained two identical `LinearRegression` models — one per encoding
strategy — on the same 67/33 train-test split (`random_state=42`), so the
only variable between them is the encoding method.

## Results

| Model | Encoding | R² | Adjusted R² |
|---|---|---|---|
| Model 1 | One-Hot | **0.840** | **0.839** |
| Model 2 | Label | 0.731 | — |

The one-hot model explains meaningfully more variance in price using the
exact same underlying information — the only difference is how
`model`, `transmission`, and `fuelType` were represented numerically.

## Key Insights

- **Encoding strategy is not a minor implementation detail — it's worth
  ~10 points of R².** Label encoding forces a linear model to treat
  `Fiesta`, `Focus`, and `EcoSport` as if they sit on a numeric scale,
  which actively misleads the regression. One-hot encoding avoids this by
  treating each category independently.
- **Mileage shows the expected negative relationship with price** —
  higher-mileage cars consistently sell for less.
- **Model and engine size both produce visibly distinct price bands**,
  confirming they carry real predictive signal.
- The unresolved `year = 2060` and `engineSize = 0` anomalies are a good
  reminder that a high R² doesn't guarantee clean input data — those rows
  are worth revisiting before treating this model as production-ready.

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn

## Project Structure
```
.
├── Ford_Car_Price_Prediction_Project.ipynb   # Full analysis + modeling notebook
├── ford.csv                                   # Raw dataset
└── README.md                                  # This file
```

## Setup & Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Ford_Car_Price_Prediction_Project.ipynb
```

Place `ford.csv` in the same directory as the notebook before running.
Executing all cells top to bottom reproduces the EDA visuals, both encoded
feature sets (`X_one_encode` and `Xlable`), and both trained regression
models with their R² scores.
