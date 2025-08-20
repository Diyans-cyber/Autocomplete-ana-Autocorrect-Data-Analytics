# fraud_analytics.py
# End-to-end fraud analytics on creditcard.csv (highly imbalanced)
# Maps your brief's steps to a proper ML pipeline with metrics & visuals

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Optional: SMOTE if imblearn is available. If not, we’ll fall back to class_weight.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# -----------------------------
# 1) Load & quick sanity checks
# -----------------------------
CSV_PATH = "creditcard.csv"  # same folder
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("creditcard.csv not found in current folder.")

df = pd.read_csv(CSV_PATH)
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe().T.head(10))

# Target is 'Class' (1 = fraud)
assert 'Class' in df.columns, "Expected 'Class' column."

# -----------------------------
# 2) Basic EDA & class balance
# -----------------------------
class_counts = df['Class'].value_counts().sort_index()
print("\nClass distribution:\n", class_counts)

plt.figure(figsize=(4,3))
class_counts.plot(kind='bar')
plt.title("Class Distribution (0=Legit, 1=Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Train / test split
# -----------------------------
X = df.drop(columns=['Class'])
y = df['Class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# 4) Scaling
# -----------------------------
# The PCA-transformed V1..V28 are already scaled-ish, but Time/Amount benefit from scaling.
# We'll scale all numeric features for fairness across models.
num_cols = X_train.columns.tolist()
scaler = StandardScaler()

# -----------------------------
# 5) Modeling approaches
#    A) Supervised (LogReg, RandomForest) with class imbalance handling
#    B) Unsupervised anomaly detection (IsolationForest)
# -----------------------------

def evaluate_and_show(name, y_true, y_prob, y_pred):
    """Common metrics + plots for classifiers with probability outputs."""
    print(f"\n---- {name} ----")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC (AP): {pr_auc:.4f}")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"Precision-Recall Curve - {name}")
    plt.tight_layout()
    plt.show()

# ---------- 5A.1 Logistic Regression ----------
if HAS_SMOTE:
    # Oversample minority class only on training folds
    logreg_clf = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=500, n_jobs=None, class_weight=None))
    ])
else:
    # No SMOTE available -> use class_weight="balanced"
    logreg_clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

logreg_clf.fit(X_train, y_train)
# Predict proba safely (last step is "clf")
y_prob_lr = logreg_clf.named_steps["clf"].predict_proba(
    logreg_clf.named_steps.get("scaler", scaler).transform(X_test) if HAS_SMOTE else
    logreg_clf.named_steps["scaler"].transform(X_test)
)[:, 1] if not HAS_SMOTE else logreg_clf.predict_proba(X_test)[:, 1]

y_pred_lr = (y_prob_lr >= 0.5).astype(int)
evaluate_and_show("Logistic Regression", y_test, y_prob_lr, y_pred_lr)

# ---------- 5A.2 Random Forest ----------
# Compute class weights if no SMOTE
if not HAS_SMOTE:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
else:
    class_weight_dict = None

rf_steps = []
rf_steps.append(("scaler", StandardScaler()))  # not strictly needed for RF, but ok
if HAS_SMOTE:
    rf_steps.append(("smote", SMOTE(random_state=42)))
rf_steps.append(("clf", RandomForestClassifier(
    n_estimators=300, max_depth=None, random_state=42,
    n_jobs=-1, class_weight=class_weight_dict
)))
rf_clf = ImbPipeline(steps=rf_steps) if HAS_SMOTE else Pipeline(steps=rf_steps)
rf_clf.fit(X_train, y_train)

# probas
if HAS_SMOTE:
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
else:
    Xt = rf_clf.named_steps["scaler"].transform(X_test)
    y_prob_rf = rf_clf.named_steps["clf"].predict_proba(Xt)[:, 1]
y_pred_rf = (y_prob_rf >= 0.5).astype(int)
evaluate_and_show("Random Forest", y_test, y_prob_rf, y_pred_rf)

# Feature importance (top 10)
try:
    importances = rf_clf.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=num_cols).sort_values(ascending=False).head(10)
    plt.figure(figsize=(6,4))
    fi.plot(kind="bar")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# ---------- 5B. Isolation Forest (unsupervised anomaly detection) ----------
# Train only on "normal" transactions to learn normality, then score test set
iso = IsolationForest(
    n_estimators=300, contamination= y_train.mean(), random_state=42, n_jobs=-1
)
# Fit on scaled train data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
iso.fit(X_train_scaled[y_train.values == 0])

# Decision_function: higher is more normal; invert to get anomaly prob-like score
scores = -iso.decision_function(X_test_scaled)
# Calibrate a threshold using the fraud rate in train or choose a quantile
thresh = np.quantile(scores, 1.0 - y_train.mean())  # flag top ~fraud_rate as fraud
y_pred_iso = (scores >= thresh).astype(int)
# Normalize scores to 0..1 for AUC (min-max)
s0, s1 = scores.min(), scores.max()
y_prob_iso = (scores - s0) / (s1 - s0 + 1e-9)
evaluate_and_show("Isolation Forest (Anomaly Detection)", y_test, y_prob_iso, y_pred_iso)

# -----------------------------
# 6) “User experience” – simple alerting rule demo
# -----------------------------
def alert_transaction(prob, amount, prob_threshold=0.5, amount_threshold=2000):
    """
    Simple decision rule combining model probability + business rule.
    Returns label + reason string.
    """
    if prob >= prob_threshold or amount >= amount_threshold:
        reason = []
        if prob >= prob_threshold: reason.append(f"model_prob={prob:.2f}>=thr")
        if amount >= amount_threshold: reason.append(f"amount={amount:.2f}>=₹{amount_threshold}")
        return "ALERT", " & ".join(reason)
    return "OK", "looks normal"

# Example on first 5 test rows using Random Forest probabilities
sample = X_test.copy()
sample["rf_prob"] = y_prob_rf
sample["decision"], sample["reason"] = zip(*[
    alert_transaction(p, amt) for p, amt in zip(sample["rf_prob"], sample["Amount"])
])
print("\nSample alerting (first 5 test rows):")
print(sample[["Amount", "rf_prob", "decision", "reason"]].head())

# -----------------------------
# 7) Save best model (by ROC-AUC here) for reuse
# -----------------------------
roc_lr = roc_auc_score(y_test, y_prob_lr)
roc_rf = roc_auc_score(y_test, y_prob_rf)
best_name = "rf" if roc_rf >= roc_lr else "lr"

import joblib
if best_name == "rf":
    joblib.dump(rf_clf, "best_fraud_model.joblib")
else:
    joblib.dump(logreg_clf, "best_fraud_model.joblib")
print(f"\nSaved best model ({'RandomForest' if best_name=='rf' else 'LogisticRegression'}) -> best_fraud_model.joblib")

print("\nDONE ✅  (You now have: EDA, supervised & unsupervised models, full metrics, curves, feature importances, and a simple alerting rule.)")