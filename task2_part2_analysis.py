import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_validate, learning_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, recall_score, precision_score

RANDOM_STATE = 42

# ============================================================
# LOAD & CLEAN (same as Task 1)
# ============================================================
heart_raw = pd.read_csv("heart_disease_uci.csv")
heart = heart_raw.drop(columns=["id", "ca", "thal", "slope", "dataset"])
heart["target"] = (heart["num"] > 0).astype(int)
heart = heart.drop(columns=["num"])

for c in ["sex", "cp", "fbs", "restecg", "exang"]:
    heart[c] = heart[c].astype(str)
    heart[c] = LabelEncoder().fit_transform(heart[c])

num_cols = ["trestbps", "chol", "thalch", "oldpeak"]
heart[num_cols] = SimpleImputer(strategy="median").fit_transform(heart[num_cols])

pima = pd.read_csv("diabetes.csv")

# ============================================================
# PART A: CROSS-VALIDATION vs SINGLE SPLIT
# ============================================================
def compare_cv_vs_single_split(name, X, y, model_builder):
    print(f"\n{'='*60}\n{name}: Single Split vs 5-Fold Cross-Validation\n{'='*60}")

    # Single split (as in Task 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = model_builder()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    single_acc = accuracy_score(y_test, pred)
    print(f"Single 75/25 split accuracy: {single_acc:.3f}")

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_cv = model_builder()
    scores = cross_validate(model_cv, X, y, cv=cv,
                             scoring=["accuracy", "recall", "precision", "f1"])
    print(f"5-fold CV accuracy: mean={scores['test_accuracy'].mean():.3f}, "
          f"std={scores['test_accuracy'].std():.3f}")
    print(f"  per-fold accuracy: {np.round(scores['test_accuracy'], 3)}")
    print(f"5-fold CV recall:   mean={scores['test_recall'].mean():.3f}, "
          f"std={scores['test_recall'].std():.3f}")

    return {"single_split_acc": single_acc,
            "cv_acc_mean": scores["test_accuracy"].mean(),
            "cv_acc_std": scores["test_accuracy"].std()}

def dt_builder():
    return DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)

def knn_builder():
    return Pipeline([("scaler", StandardScaler()),
                      ("knn", KNeighborsClassifier(n_neighbors=7))])

X_heart = heart.drop(columns=["target"])
y_heart = heart["target"]
X_pima = pima.drop(columns=["Outcome"])
y_pima = pima["Outcome"]

results = {}
results["heart_dt"] = compare_cv_vs_single_split("Heart Disease - Decision Tree", X_heart, y_heart, dt_builder)
results["heart_knn"] = compare_cv_vs_single_split("Heart Disease - kNN", X_heart, y_heart, knn_builder)
results["pima_dt"] = compare_cv_vs_single_split("Diabetes - Decision Tree", X_pima, y_pima, dt_builder)
results["pima_knn"] = compare_cv_vs_single_split("Diabetes - kNN", X_pima, y_pima, knn_builder)

pd.DataFrame(results).T.round(3).to_csv("cv_vs_single_split.csv")
print("\nSaved cv_vs_single_split.csv")

# ============================================================
# PART B: LEARNING CURVES
# ============================================================
def plot_learning_curve(name, X, y, model_builder, filename):
    train_sizes, train_scores, test_scores = learning_curve(
        model_builder(), X, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, "o-", label="Training accuracy")
    plt.plot(train_sizes, test_mean, "o-", label="Validation accuracy")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve: {name}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

plot_learning_curve("Heart Disease - Decision Tree", X_heart, y_heart, dt_builder, "learning_curve_heart_dt.png")
plot_learning_curve("Heart Disease - kNN", X_heart, y_heart, knn_builder, "learning_curve_heart_knn.png")
plot_learning_curve("Diabetes - Decision Tree", X_pima, y_pima, dt_builder, "learning_curve_diabetes_dt.png")
plot_learning_curve("Diabetes - kNN", X_pima, y_pima, knn_builder, "learning_curve_diabetes_knn.png")

# ============================================================
# PART C: FAIRNESS ANALYSIS (Fairlearn) — Heart Disease by Sex
# ============================================================
print(f"\n{'='*60}\nFAIRNESS ANALYSIS: Heart Disease model performance by Sex\n{'='*60}")
print("(sex encoded: check original mapping - 0/1 after LabelEncoder)")
print("Original sex value counts:\n", heart_raw["sex"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_heart, y_heart, test_size=0.25, random_state=RANDOM_STATE, stratify=y_heart
)
sex_test = X_test["sex"]  # already label-encoded: 0/1

dt_model = dt_builder()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "recall": recall_score,
             "precision": precision_score, "selection_rate": selection_rate},
    y_true=y_test, y_pred=dt_pred, sensitive_features=sex_test
)
print("\nDecision Tree performance by sex group:")
print(mf.by_group)
print("\nDisparity (max difference between groups):")
print(mf.difference())

mf.by_group.round(3).to_csv("fairness_by_sex_heart_dt.csv")
print("\nSaved fairness_by_sex_heart_dt.csv")

print("\nNOTE: Pima Diabetes dataset contains only female patients, "
      "so a sex-based fairness comparison is not applicable for that dataset. "
      "This itself is a fairness/representativeness limitation worth discussing.")

print("\n\nAll analysis complete.")
