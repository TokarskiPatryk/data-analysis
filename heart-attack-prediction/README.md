# Heart Attack Prediction

Predict whether a patient experienced a heart attack using clinical measurements collected during ambulance transport.

## Dataset

1,319 patient records with 9 features collected en route to hospital. Source: [Kaggle – Medical Dataset](https://www.kaggle.com/).

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Gender | 0 = female, 1 = male |
| Heart Rate | Maximum heart rate achieved (bpm) |
| Systolic BP | Resting systolic blood pressure (mmHg) |
| Diastolic BP | Resting diastolic blood pressure (mmHg) |
| Blood Sugar | Blood glucose level (mg/dl) |
| CK-MB | Creatine kinase enzyme (ng/mL) |
| Troponin | Troponin enzyme (µg) |
| Result | Target — 0 = no heart attack, 1 = heart attack |

## Methods

- Exploratory data analysis and distribution visualisation
- Decision tree classification model
- Comparison of model-selected features (CK-MB, Troponin) against clinical reference values

## Tech stack

R · tidyverse · foreign · plotly · corrplot

## Authors

Tymoteusz Romanowicz, Patryk Tokarski

## Report

https://tokarskipatryk.github.io/data-analysis/heart-attack-prediction/HeartAttactPrediction.html
