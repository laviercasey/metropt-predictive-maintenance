import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv").to_numpy()
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["target"].to_numpy()
    X_test = pd.read_csv(DATA_DIR / "X_test.csv").to_numpy()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["target"].to_numpy()
    return X_train, y_train, X_test, y_test


def get_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, C=0.1,
                solver="lbfgs", random_state=RANDOM_STATE,
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11, metric="euclidean", n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", class_weight="balanced", probability=True,
                C=1.0, random_state=RANDOM_STATE,
            )),
        ]),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=10, random_state=RANDOM_STATE,
        ),
        "GaussianNB": GaussianNB(),
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        "RidgeClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def get_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    df = model.decision_function(X_test)
    return (df - df.min()) / (df.max() - df.min() + 1e-12)


def evaluate(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_prob = get_scores(model, X_test)
    y_pred = model.predict(X_test)

    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "balanced_acc": round(balanced_accuracy_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "fit_time_s": round(fit_time, 1),
    }, y_prob


def plot_roc_pr(probs, y_test, fname_prefix):
    fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(9, 7))

    for name, y_prob in probs:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} ({auc:.3f})", linewidth=1.2)
        ax_pr.plot(rec, prec, label=f"{name} ({ap:.3f})", linewidth=1.2)

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Случайный классификатор")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("ROC-кривые базовых моделей")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(FIG_DIR / f"{fname_prefix}_roc.png", bbox_inches="tight")
    plt.close(fig_roc)

    baseline = float(np.mean(y_test))
    ax_pr.axhline(baseline, color="k", linestyle="--", linewidth=0.8,
                  label=f"Базовая линия ({baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR-кривые базовых моделей")
    ax_pr.legend(fontsize=8, loc="upper right")
    fig_pr.tight_layout()
    fig_pr.savefig(FIG_DIR / f"{fname_prefix}_pr.png", bbox_inches="tight")
    plt.close(fig_pr)


def plot_metrics_bar(df, title, fname):
    metrics = ["roc_auc", "pr_auc", "f1", "balanced_acc", "recall"]
    x = np.arange(len(df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    for i, (col, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[col], width, label=col, color=color)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(model, X_test, y_test, title, fname):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    fontsize=12, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Норма (0)", "Отказ (1)"])
    ax.set_yticklabels(["Норма (0)", "Отказ (1)"])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинно")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def write_report(df):
    best_auc = df.sort_values("roc_auc", ascending=False).iloc[0]
    best_ap = df.sort_values("pr_auc", ascending=False).iloc[0]
    lines = [
        "=" * 70,
        "ОТЧЁТ: БАЗОВЫЕ МОДЕЛИ",
        "=" * 70,
        "",
        df.to_string(index=False),
        "",
        f"Лучшая по ROC-AUC : {best_auc['model']} ({best_auc['roc_auc']:.4f})",
        f"Лучшая по PR-AUC  : {best_ap['model']} ({best_ap['pr_auc']:.4f})",
        "",
        "=" * 70,
    ]
    text = "\n".join(lines)
    (REPORTS_DIR / "03_base_models_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    models = get_models()
    all_metrics = []
    probs = []

    for name, model in models.items():
        metrics, y_prob = evaluate(name, model, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)
        probs.append((name, y_prob))
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    df = pd.DataFrame(all_metrics)
    df.to_csv(REPORTS_DIR / "base_models_metrics.csv", index=False)

    plot_roc_pr(probs, y_test, "07_base")
    plot_metrics_bar(df, "Сравнение метрик — базовые модели", "09_base_metrics_bar.png")

    best_name = df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = joblib.load(MODELS_DIR / f"{best_name}.pkl")
    plot_confusion(best_model, X_test, y_test,
                   f"Матрица ошибок: {best_name}", "10_base_best_confusion.png")

    write_report(df)
