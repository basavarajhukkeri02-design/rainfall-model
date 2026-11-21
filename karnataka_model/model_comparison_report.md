# Karnataka rainfall model comparison

Dataset: 980,554 valid records; target: rain tomorrow (0 = no rain, 1 = rain).
Chronological splits by whole date: train 736,728 rows (2023-01-08 to 2025-02-06), validation 85,909 rows (2025-02-07 to 2025-05-26), test 157,917 rows (2025-05-27 to 2025-12-30).
Rain prevalence: train 20.1%, validation 14.4%, test 34.8% (54,933 of 157,917 test rows).
Validation period alone selected the threshold and the saved model. Test scores below were calculated once on the later held-out period.

## Leakage review

The trainer uses only prior-rainfall lags/rolling summaries, district and calendar fields. `Rainfall_Tomorrow` (the source used to make the label) is excluded. All rows for a date stay in one split, and imputers/encoders are fitted within each training pipeline on training rows only. The feature preparation script shifts rainfall backward for input features and forward for the target. This checks the code path; it cannot rule out errors or duplication in the source observations.
The source table has many station rows per date; splitting by unique date keeps all rows for each date together.

## Test results

Reference supplied before this run: accuracy 65.21%, No Rain precision/recall 0.90/0.52, Rain precision/recall 0.50/0.89. These imply balanced accuracy about 0.705. Neither model below exceeds that reference on this later test period, so this run does not confirm an improvement.

Confusion matrix rows are actual [No Rain, Rain], columns are predicted [No Rain, Rain].
### Random Forest

Threshold (validation tuned): 0.1752
Accuracy: 0.4807; balanced accuracy: 0.5998; ROC-AUC: 0.8289; PR-AUC: 0.7509
Confusion matrix: `[[21459, 81525], [479, 54454]]`

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Rain | 0.9782 | 0.2084 | 0.3436 | 102,984 |
| Rain | 0.4005 | 0.9913 | 0.5705 | 54,933 |

### HistGradientBoosting

Threshold (validation tuned): 0.0590
Accuracy: 0.4980; balanced accuracy: 0.6123; ROC-AUC: 0.8251; PR-AUC: 0.7479
Confusion matrix: `[[24389, 78595], [674, 54259]]`

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Rain | 0.9731 | 0.2368 | 0.3809 | 102,984 |
| Rain | 0.4084 | 0.9877 | 0.5779 | 54,933 |

## Saved model

Selected by validation balanced accuracy: **Random Forest** (validation balanced accuracy 0.7418).
Model file: `karnataka_model\rainfall_model.pkl`. It is stored separately because the root `rainfall_model.pkl` is loaded by the existing app and uses a different feature schema.
