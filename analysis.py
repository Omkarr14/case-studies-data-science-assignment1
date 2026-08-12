import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42

def evaluate(name, y_test, y_pred, y_proba=None):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    if auc is not None:
        print(f"ROC-AUC  : {auc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def run_dataset(name, X, y):
    print(f"\n{'='*60}\nDATASET: {name}  (n={len(X)}, features={X.shape[1]})\n{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]
    dt_results = evaluate("Decision Tree", y_test, dt_pred, dt_proba)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_s, y_train)
    knn_pred = knn.predict(X_test_s)
    knn_proba = knn.predict_proba(X_test_s)[:, 1]
    knn_results = evaluate("kNN (k=7)", y_test, knn_pred, knn_proba)

    importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 5 Decision Tree feature importances:")
    print(importances.head(5))

    return {"dataset": name, "decision_tree": dt_results, "knn": knn_results,
            "top_features": importances.head(5).to_dict()}

# ============================================================
# HEART DISEASE (UCI, multi-site, 920 rows, 16 cols)
# ============================================================
heart_raw = pd.read_csv("heart_disease_uci.csv")

# Drop id and columns with >30% missing values (ca, thal, slope)
heart = heart_raw.drop(columns=["id", "ca", "thal", "slope", "dataset"])

# Binarize target: num=0 -> no disease, num>0 -> disease present
heart["target"] = (heart["num"] > 0).astype(int)
heart = heart.drop(columns=["num"])

# Encode categorical columns
cat_cols = heart.select_dtypes(include="object").columns.tolist() + \
           [c for c in heart.columns if heart[c].dtype == "str" or heart[c].dtype == "bool"]
cat_cols = list(set(cat_cols) - {"target"})
for c in heart.columns:
    if heart[c].dtype == object or str(heart[c].dtype) == "str":
        heart[c] = heart[c].astype(str)

for c in ["sex", "cp", "fbs", "restecg", "exang"]:
    heart[c] = LabelEncoder().fit_transform(heart[c].astype(str))

# Impute remaining numeric missing values (trestbps, chol, thalch, oldpeak) with median
num_cols = ["trestbps", "chol", "thalch", "oldpeak"]
imputer = SimpleImputer(strategy="median")
heart[num_cols] = imputer.fit_transform(heart[num_cols])

print("Missing values after cleaning:\n", heart.isnull().sum())

X_heart = heart.drop(columns=["target"])
y_heart = heart["target"]
heart_results = run_dataset("Heart Disease (UCI, multi-site)", X_heart, y_heart)

# ============================================================
# PIMA DIABETES (768 rows, 9 cols)
# ============================================================
pima = pd.read_csv("diabetes.csv")
X_pima = pima.drop(columns=["Outcome"])
y_pima = pima["Outcome"]
pima_results = run_dataset("Pima Indians Diabetes", X_pima, y_pima)

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
summary = pd.DataFrame({
    "Heart-DT": heart_results["decision_tree"],
    "Heart-kNN": heart_results["knn"],
    "Diabetes-DT": pima_results["decision_tree"],
    "Diabetes-kNN": pima_results["knn"],
}).T
print(summary.round(3))
summary.round(3).to_csv("summary_results_v2.csv")
print("\nTop features heart:", heart_results["top_features"])
print("Top features diabetes:", pima_results["top_features"])
print("\nSaved summary_results_v2.csv")
