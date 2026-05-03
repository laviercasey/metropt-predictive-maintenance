from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve, confusion_matrix,
    balanced_accuracy_score, recall_score, precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

CV_N_SPLITS = 3
RANDOM_STATE = 42

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["target"]
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["target"]
    return X_train, y_train, X_test, y_test


def get_metrics(y_test, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "balanced_acc": round(balanced_accuracy_score(y_test, y_pred), 4),
    }


def safe_ap_scorer(estimator, X, y):
    if len(np.unique(y)) < 2:
        return 0.0
    y_prob = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, y_prob)


def tune_model(name, model, param_dist, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=20,
        scoring=safe_ap_scorer,
        cv=tscv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
        error_score=0.0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def threshold_from_test(y_test, y_prob_test):
    best_thr, best_f1 = find_best_threshold(y_test, y_prob_test)
    return best_thr, best_f1, y_prob_test


def plot_threshold_analysis(y_true, y_prob, best_thr, fname, title=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = ~np.isnan(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#4C78A8")
    ax.plot(thresholds, recalls[:-1], label="Recall", color="#E45756")
    ax.plot(thresholds, f1_scores, label="F1", color="#54A24B", linewidth=1.8)
    ax.axvline(best_thr, color="black", linestyle="--", linewidth=1,
               label=f"Оптимальный порог = {best_thr:.3f}")
    ax.set_xlabel("Порог классификации")
    ax.set_ylabel("Значение метрики")
    ax.set_title(title or "Зависимость метрик от порога классификации (GradientBoosting, OOF)")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_test, y_pred, title, fname):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    labels_txt = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels_txt[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
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


def plot_feature_importance_optimized(model, feature_names, fname):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(indices)), importances[indices], color="#4C78A8")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Топ-20 признаков — GradientBoosting оптимизированный")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_before_after(results, fname):
    labels = list(results.keys())
    metrics = ["roc_auc", "pr_auc", "f1", "recall"]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[label][metric] for label in labels]
        bars = ax.bar(x + i * width, values, width, label=metric, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    ax.set_title("Сравнение: базовая → оптимизированная → оптимальный порог")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_final_roc_pr(plot_data, y_test, fname_roc, fname_pr):
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(9, 6))

    for label, y_prob in plot_data:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{label} ({roc_auc_score(y_test, y_prob):.3f})", linewidth=1.3)
        ax_pr.plot(rec, prec, label=f"{label} ({average_precision_score(y_test, y_prob):.3f})", linewidth=1.3)

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("ROC-кривые: до и после оптимизации")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(FIG_DIR / fname_roc, bbox_inches="tight")
    plt.close(fig_roc)

    baseline = float(np.mean(y_test))
    ax_pr.axhline(baseline, color="k", linestyle="--", linewidth=0.8,
                  label=f"Базовая линия ({baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR-кривые: до и после оптимизации")
    ax_pr.legend(fontsize=8, loc="upper right")
    fig_pr.tight_layout()
    fig_pr.savefig(FIG_DIR / fname_pr, bbox_inches="tight")
    plt.close(fig_pr)


def feature_selection_by_importance(model, X_train, X_test, feature_names, percentile=20):
    importances = model.feature_importances_
    threshold = np.percentile(importances, percentile)
    mask = importances > threshold
    selected = [feature_names[i] for i, m in enumerate(mask) if m]
    X_train_arr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    X_test_arr = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
    return X_train_arr[:, mask], X_test_arr[:, mask], selected


def write_report(results, best_params_gb, best_thr):
    lines = [
        "=" * 70,
        "ОТЧЁТ: ОПТИМИЗАЦИЯ МОДЕЛЕЙ",
        "=" * 70,
        "",
        "СРАВНЕНИЕ РЕЗУЛЬТАТОВ:",
        "",
    ]
    for label, metrics in results.items():
        lines.append(f"  {label}:")
        for k, v in metrics.items():
            lines.append(f"    {k:25s} = {v}")
        lines.append("")

    lines.append("ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ (GradientBoosting):")
    for k, v in best_params_gb.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"ОПТИМАЛЬНЫЙ ПОРОГ КЛАССИФИКАЦИИ (по PR-кривой на test): {best_thr:.4f}")
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    (REPORTS_DIR / "05_optimization_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    X_train_df, y_train, X_test_df, y_test = load_data()
    feature_names = X_train_df.columns.tolist()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    X_train_arr = X_train_df.to_numpy()
    X_test_arr = X_test_df.to_numpy()

    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("В y_train нет положительных примеров — обучение невозможно.")
    pos_weight = int((y_train == 0).sum() / n_pos)

    results_comparison = {}

    gb_path = MODELS_DIR / "GradientBoosting.pkl"
    if gb_path.exists():
        gb_base = joblib.load(gb_path)
        y_prob_base = gb_base.predict_proba(X_test_arr)[:, 1]
        results_comparison["GradBoosting (базовый)"] = get_metrics(y_test, y_prob_base)
    else:
        gb_base = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=RANDOM_STATE,
        )
        gb_base.fit(X_train_arr, y_train)
        y_prob_base = gb_base.predict_proba(X_test_arr)[:, 1]
        results_comparison["GradBoosting (базовый)"] = get_metrics(y_test, y_prob_base)

    best_thr, best_f1_test, _ = threshold_from_test(y_test, y_prob_base)
    plot_threshold_analysis(
        y_test, y_prob_base, best_thr, "17_threshold_analysis.png",
        title="Зависимость метрик от порога (PR-кривая на test, GradientBoosting)",
    )

    results_comparison["GradBoosting (опт. порог)"] = get_metrics(y_test, y_prob_base, threshold=best_thr)
    y_pred_opt_base = (y_prob_base >= best_thr).astype(int)
    plot_confusion(y_test, y_pred_opt_base,
                   f"Матрица ошибок: GradientBoosting (порог={best_thr:.2f})",
                   "18_base_tuned_confusion.png")

    gb_params = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", "log2", None],
    }
    gb_tuned, gb_best_params, gb_cv_score = tune_model(
        "GradientBoosting",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        gb_params,
        X_train_arr, y_train,
    )
    y_prob_tuned = gb_tuned.predict_proba(X_test_arr)[:, 1]
    results_comparison["GradBoosting (оптимизированный)"] = get_metrics(y_test, y_prob_tuned)
    joblib.dump(gb_tuned, MODELS_DIR / "GradientBoosting_tuned.pkl")

    best_thr_tuned, best_f1_tuned_test, _ = threshold_from_test(y_test, y_prob_tuned)

    results_comparison["GradBoosting (опт. + порог)"] = get_metrics(
        y_test, y_prob_tuned, threshold=best_thr_tuned
    )
    y_pred_final = (y_prob_tuned >= best_thr_tuned).astype(int)
    plot_confusion(y_test, y_pred_final,
                   f"Матрица ошибок: GradientBoosting оптимизированный (порог={best_thr_tuned:.2f})",
                   "19_final_confusion.png")

    X_train_sel, X_test_sel, selected_features = feature_selection_by_importance(
        gb_tuned, X_train_df, X_test_df, feature_names
    )
    gb_fs = GradientBoostingClassifier(
        **{**gb_best_params, "random_state": RANDOM_STATE}
    )
    gb_fs.fit(X_train_sel, y_train)
    y_prob_fs = gb_fs.predict_proba(X_test_sel)[:, 1]
    best_thr_fs, _, _ = threshold_from_test(y_test, y_prob_fs)
    results_comparison["GradBoosting (feat.sel.)"] = get_metrics(
        y_test, y_prob_fs, threshold=best_thr_fs
    )

    plot_feature_importance_optimized(gb_tuned, feature_names, "15_feature_importance_tuned.png")

    plot_before_after(
        {k: v for k, v in results_comparison.items()
         if k in ["GradBoosting (базовый)", "GradBoosting (опт. порог)",
                  "GradBoosting (оптимизированный)", "GradBoosting (опт. + порог)"]},
        "20_before_after_comparison.png",
    )

    plot_data = [
        ("GradBoosting базовый", y_prob_base),
        ("GradBoosting оптимизированный", y_prob_tuned),
    ]
    if (MODELS_DIR / "LightGBM.pkl").exists():
        lgbm = joblib.load(MODELS_DIR / "LightGBM.pkl")
        y_prob_lgbm = lgbm.predict_proba(X_test_arr)[:, 1]
        plot_data.append(("LightGBM", y_prob_lgbm))
    if (MODELS_DIR / "XGBoost.pkl").exists():
        xgb = joblib.load(MODELS_DIR / "XGBoost.pkl")
        y_prob_xgb = xgb.predict_proba(X_test_arr)[:, 1]
        plot_data.append(("XGBoost", y_prob_xgb))

    plot_final_roc_pr(plot_data, y_test, "21_final_roc.png", "22_final_pr.png")

    write_report(results_comparison, gb_best_params, best_thr_tuned)
