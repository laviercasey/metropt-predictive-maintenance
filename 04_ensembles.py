import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
STACKING_CV_SPLITS = 3

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


def get_ensembles(pos_weight):
    base_estimators = [
        ("lr", Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=500, C=0.1,
                solver="lbfgs", random_state=RANDOM_STATE,
            )),
        ])),
        ("dt", DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=RANDOM_STATE,
        )),
        ("rf_base", RandomForestClassifier(
            n_estimators=100, class_weight="balanced", max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", max_depth=None,
            min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3, random_state=RANDOM_STATE,
            ),
            n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cpu", eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            scale_pos_weight=pos_weight, random_seed=RANDOM_STATE,
            verbose=0, allow_writing_files=False,
        ),
        "VotingClassifier": VotingClassifier(
            estimators=base_estimators, voting="soft", n_jobs=-1,
        ),
    }

    stacking_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=100, scale_pos_weight=pos_weight,
            tree_method="hist", device="cpu", eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=100, scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )),
    ]
    models["StackingClassifier"] = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(
            class_weight="balanced", C=0.1, max_iter=500,
            solver="lbfgs", random_state=RANDOM_STATE,
        ),
        cv=STACKING_CV_SPLITS,
        n_jobs=-1,
        passthrough=False,
    )

    return models


def evaluate(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
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


def plot_roc_pr(probs, y_test, fname_prefix, title_suffix):
    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 7))

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
    ax_roc.set_title(f"ROC-кривые — {title_suffix}")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(FIG_DIR / f"{fname_prefix}_roc.png", bbox_inches="tight")
    plt.close(fig_roc)

    baseline = float(np.mean(y_test))
    ax_pr.axhline(baseline, color="k", linestyle="--", linewidth=0.8,
                  label=f"Базовая линия ({baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR-кривые — {title_suffix}")
    ax_pr.legend(fontsize=8, loc="upper right")
    fig_pr.tight_layout()
    fig_pr.savefig(FIG_DIR / f"{fname_prefix}_pr.png", bbox_inches="tight")
    plt.close(fig_pr)


def plot_metrics_bar(df, title, fname):
    metrics = ["roc_auc", "pr_auc", "f1", "balanced_acc", "recall"]
    x = np.arange(len(df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 5))
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


def plot_feature_importance(model, feature_names, fname):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(indices)), importances[indices], color="#4C78A8")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Топ-20 важных признаков (лучшая ансамблевая модель)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_combined_comparison(base_df, ens_df, fname):
    combined = pd.concat([base_df, ens_df], ignore_index=True)
    combined = combined.sort_values("roc_auc", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    base_models = set(base_df["model"].values)
    colors = ["#4C78A8" if m in base_models else "#E45756" for m in combined["model"]]

    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        ax.barh(combined["model"], combined[metric], color=colors)
        ax.set_xlabel(metric.replace("_", "-").upper())
        ax.set_title(f"{metric.replace('_', '-').upper()} — все модели")
        ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8)
        for i, v in enumerate(combined[metric]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7)

    legend_elements = [
        Patch(facecolor="#4C78A8", label="Базовые модели"),
        Patch(facecolor="#E45756", label="Ансамблевые модели"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def write_report(df, best_name):
    best = df[df["model"] == best_name].iloc[0]
    lines = [
        "=" * 70,
        "ОТЧЁТ: АНСАМБЛЕВЫЕ МОДЕЛИ",
        "=" * 70,
        "",
        df.to_string(index=False),
        "",
        f"Лучшая по ROC-AUC : {df.sort_values('roc_auc', ascending=False).iloc[0]['model']}",
        f"Лучшая по PR-AUC  : {df.sort_values('pr_auc', ascending=False).iloc[0]['model']}",
        "",
        f"Детали лучшей ({best_name}):",
        f"  ROC-AUC        : {best['roc_auc']}",
        f"  PR-AUC         : {best['pr_auc']}",
        f"  F1             : {best['f1']}",
        f"  Recall         : {best['recall']}",
        f"  Precision      : {best['precision']}",
        f"  Balanced Acc   : {best['balanced_acc']}",
        "",
        "=" * 70,
    ]
    text = "\n".join(lines)
    (REPORTS_DIR / "04_ensembles_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    feature_names = pd.read_csv(DATA_DIR / "X_train.csv").columns.tolist()

    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("В y_train нет положительных примеров — обучение невозможно.")
    pos_weight = int((y_train == 0).sum() / n_pos)

    ensembles = get_ensembles(pos_weight)
    all_metrics = []
    probs = []

    for name, model in ensembles.items():
        metrics, y_prob = evaluate(name, model, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)
        probs.append((name, y_prob))
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    df = pd.DataFrame(all_metrics)
    df.to_csv(REPORTS_DIR / "ensemble_metrics.csv", index=False)

    plot_roc_pr(probs, y_test, "11_ensemble", "ансамблевые модели")
    plot_metrics_bar(df, "Сравнение метрик — ансамблевые модели", "13_ensemble_metrics_bar.png")

    best_name = df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = joblib.load(MODELS_DIR / f"{best_name}.pkl")
    plot_confusion(best_model, X_test, y_test,
                   f"Матрица ошибок: {best_name}", "14_ensemble_best_confusion.png")
    plot_feature_importance(best_model, feature_names, "15_feature_importance.png")

    if (REPORTS_DIR / "base_models_metrics.csv").exists():
        base_df = pd.read_csv(REPORTS_DIR / "base_models_metrics.csv")
        plot_combined_comparison(base_df, df, "16_all_models_comparison.png")

    write_report(df, best_name)
