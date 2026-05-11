# Titanic Survival Prediction

Predict passenger survival on the Titanic using logistic regression, based on the classic Kaggle competition dataset.

## Dataset

Standard Kaggle Titanic split: `train.csv` (labelled) and `test.csv`. Features used: passenger class (`Pclass`), sex, and age. Rows with missing `Age` are dropped.

## Methods

- EDA: survival rate by class, sex, and age
- t-test confirming a statistically significant age difference between survivors and non-survivors
- Logistic regression (`glm` with `family = binomial`) trained on a 70/30 split via `caret`

## Tech stack

R · tidyverse · caret

## Report

https://tokarskipatryk.github.io/data-analysis/titanic-survival-prediction/analysis.html
