# ============================================================
# DIABETES HEALTH INDICATORS
# PAPER REPRODUCTION - DECISION TREE & RANDOM FOREST
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    "diabetes_binary_health_indicators_BRFSS2015 (1).csv"
)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["Diabetes_binary"].value_counts())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Diabetes_binary",
    axis=1
)

y = df["Diabetes_binary"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing = X.isna().sum().sum()

print("Total missing values:", missing)


# ============================================================
# 5. MEDIAN IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)

print(
    "Missing values after imputation:",
    np.isnan(X_imputed).sum()
)


# ============================================================
# 6. MIN-MAX NORMALIZATION
# ============================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

print("\nMin-Max normalization completed.")


# ============================================================
# 7. SMOTE
# ============================================================

print("\n" + "=" * 60)
print("SMOTE")
print("=" * 60)

smote = SMOTE(
    random_state=42
)

X_smote, y_smote = smote.fit_resample(
    X_scaled,
    y
)

print("Shape after SMOTE:")
print(X_smote.shape)

print("\nClass distribution after SMOTE:")
print(
    pd.Series(y_smote).value_counts()
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_smote,
    y_smote,
    test_size=0.30,
    random_state=42,
    stratify=y_smote
)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts())


# ============================================================
# 9. DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE - MAX DEPTH 5")
print("=" * 60)

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

# Predictions
dt_pred = dt.predict(X_test)

# Probabilities for ROC-AUC
dt_prob = dt.predict_proba(X_test)[:, 1]


# Metrics
dt_precision = precision_score(
    y_test,
    dt_pred
)

dt_recall = recall_score(
    y_test,
    dt_pred
)

dt_f1 = f1_score(
    y_test,
    dt_pred
)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

dt_auc = roc_auc_score(
    y_test,
    dt_prob
)

dt_hamming = hamming_loss(
    y_test,
    dt_pred
)


# Display results

print("\nRESULTS")

print(f"Precision : {dt_precision:.4f}")
print(f"Recall    : {dt_recall:.4f}")
print(f"F1 Score  : {dt_f1:.4f}")
print(f"Accuracy  : {dt_accuracy:.4f}")
print(f"ROC-AUC   : {dt_auc:.4f}")
print(f"Hamming   : {dt_hamming:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        dt_pred
    )
)


print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        dt_pred
    )
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST - MAX DEPTH 10")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

# Predictions
rf_pred = rf.predict(X_test)

# Probabilities for ROC-AUC
rf_prob = rf.predict_proba(X_test)[:, 1]


# Metrics
rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

rf_hamming = hamming_loss(
    y_test,
    rf_pred
)


# Display results

print("\nRESULTS")

print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"ROC-AUC   : {rf_auc:.4f}")
print(f"Hamming   : {rf_hamming:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)


print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ============================================================
# 11. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL RESULTS COMPARISON")
print("=" * 75)

results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],

    "Precision": [
        dt_precision,
        rf_precision
    ],

    "Recall": [
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        dt_f1,
        rf_f1
    ],

    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ],

    "ROC-AUC": [
        dt_auc,
        rf_auc
    ],

    "Hamming Loss": [
        dt_hamming,
        rf_hamming
    ]
})

print(
    results.to_string(
        index=False
    )
)


