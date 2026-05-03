import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SENSOR_LABELS = {
    "DV_pressure": "Давление на выпускном клапане (DV)",
    "TP2": "Давление компрессора (TP2)",
    "TP3": "Давление пневмопанели (TP3)",
    "H1": "Влажность воздуха (H1)",
    "Reservoirs": "Давление в резервуарах",
    "Oil_temperature": "Температура масла",
    "Motor_current": "Ток двигателя",
    "COMP": "Компрессор (вкл/выкл)",
    "DV_eletric": "Электроклапан (DV)",
    "Towers": "Башни осушителя",
    "MPG": "Электропневмоклапан",
    "LPS": "Датчик низкого давления",
    "Pressure_switch": "Реле давления",
    "Oil_level": "Уровень масла",
    "Caudal_impulses": "Импульсы расходомера",
}

STAT_LABELS = {
    "mean": "среднее", "std": "ст.откл.", "min": "мин",
    "max": "макс", "median": "медиана", "skew": "асимметрия",
}


def readable_feat(name: str) -> str:
    for sensor, label in SENSOR_LABELS.items():
        if name.startswith(sensor):
            stat = name[len(sensor):].lstrip("_")
            stat_ru = STAT_LABELS.get(stat, stat)
            return f"{label}\n[{stat_ru}]"
    return name


def explain_binary(model, X) -> shap.Explanation:
    explainer = shap.Explainer(model, feature_names=list(X.columns))
    explanation = explainer(X)
    if explanation.values.ndim == 3:
        explanation = explanation[..., 1]
    return explanation


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["target"].to_numpy()
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["target"].to_numpy()
    ts_test = pd.read_csv(DATA_DIR / "test_timestamps.csv", parse_dates=["window_start", "window_end"])
    return X_train, y_train, X_test, y_test, ts_test


def plot_shap_summary(explanation: shap.Explanation, fname):
    top_n = 20
    mean_abs = np.abs(explanation.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:]

    explanation_top = explanation[:, top_idx]
    explanation_top.feature_names = [readable_feat(n) for n in np.array(explanation.feature_names)[top_idx]]

    plt.figure(figsize=(11, 8))
    shap.plots.beeswarm(
        explanation_top, max_display=top_n, show=False,
        color_bar=True, alpha=0.7, s=10,
    )
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_xlabel("SHAP-значение (влияние на предсказание модели)", fontsize=10)
    ax.set_title(
        "SHAP-анализ: влияние признаков на предсказание GradientBoosting\n"
        "Красный — высокое значение признака, синий — низкое",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight", dpi=150)
    plt.close("all")


def plot_shap_bar(explanation: shap.Explanation, fname):
    feature_names = list(explanation.feature_names)
    mean_abs = np.abs(explanation.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:]

    labels = [readable_feat(feature_names[i]) for i in top_idx]
    values = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = matplotlib.colormaps["RdYlGn_r"]
    colors = cmap(np.linspace(0.2, 0.8, len(values)))
    bars = ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Среднее |SHAP|-значение (средний вклад в предсказание)", fontsize=10)
    ax.set_title("Топ-15 признаков по важности (SHAP)\nGradientBoosting оптимизированный", fontsize=11)

    for bar, v in zip(bars, values):
        ax.text(v + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_shap_waterfall(explanation: shap.Explanation, X_test, y_test, ts_test, fname):
    failure_indices = np.where(y_test == 1)[0]
    if len(failure_indices) == 0:
        return
    idx = int(failure_indices[0])

    feature_names = list(explanation.feature_names)
    sv = explanation.values[idx]
    base_val = float(np.atleast_1d(explanation.base_values[idx]).reshape(-1)[-1])

    abs_sv = np.abs(sv)
    top_n = 10
    top_idx = np.argsort(abs_sv)[-top_n:]

    values_top = sv[top_idx]
    labels_top = [readable_feat(feature_names[i]).replace("\n", " ") for i in top_idx]
    feat_values = X_test.iloc[idx, top_idx].to_numpy()
    pred_val = base_val + sv.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in values_top]
    y_pos = range(len(values_top))

    ax.barh(y_pos, values_top, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{l}\n(знач.={v:.3f})" for l, v in zip(labels_top, feat_values)], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP-значение (вклад в отклонение от базового предсказания)", fontsize=9)

    window_time = ts_test.iloc[idx]["window_start"] if ts_test is not None else ""
    ax.set_title(
        f"Объяснение предсказания: окно отказа\n"
        f"{window_time}  |  базовое значение={base_val:.3f}  →  предсказание={pred_val:.3f}",
        fontsize=10,
    )

    legend_elements = [
        Patch(facecolor="#E45756", label="Увеличивает вероятность отказа"),
        Patch(facecolor="#4C78A8", label="Снижает вероятность отказа"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence(explanation: shap.Explanation, X_test, fname):
    feature_names = list(explanation.feature_names)
    shap_values = explanation.values
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feat_idx = int(np.argmax(mean_abs))
    second_idx = int(np.argsort(mean_abs)[-2])
    top_feat_name = feature_names[top_feat_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        X_test.iloc[:, top_feat_idx],
        shap_values[:, top_feat_idx],
        c=X_test.iloc[:, second_idx],
        cmap="RdYlBu_r",
        s=8, alpha=0.5,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(readable_feat(feature_names[second_idx]).replace("\n", " "), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(readable_feat(top_feat_name).replace("\n", " "), fontsize=10)
    ax.set_ylabel("SHAP-значение признака", fontsize=10)
    ax.set_title(
        f"SHAP dependence plot: {readable_feat(top_feat_name).replace(chr(10), ' ')}\n"
        "Влияние значения признака на предсказание модели",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(models_probs, y_test, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_cal, ax_hist = axes
    n_bins = 10

    for label, y_prob in models_probs:
        fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_test, y_prob)
        ax_cal.plot(mean_pred, fraction_pos, marker="o", markersize=4,
                    label=f"{label} (Brier={brier:.4f})", linewidth=1.5)

    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1, label="Идеальная калибровка")
    ax_cal.set_xlabel("Средняя предсказанная вероятность", fontsize=10)
    ax_cal.set_ylabel("Доля истинных позитивов", fontsize=10)
    ax_cal.set_title("Кривые калибровки вероятностей\n(ближе к диагонали — лучше)", fontsize=11)
    ax_cal.legend(fontsize=8, loc="upper left")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)

    for label, y_prob in models_probs:
        ax_hist.hist(y_prob, bins=50, alpha=0.5, label=label, density=True)
    ax_hist.set_xlabel("Предсказанная вероятность отказа", fontsize=10)
    ax_hist.set_ylabel("Плотность", fontsize=10)
    ax_hist.set_title("Распределение предсказанных вероятностей", fontsize=11)
    ax_hist.legend(fontsize=8)
    ax_hist.set_yscale("log")

    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def write_shap_report(explanation: shap.Explanation):
    feature_names = list(explanation.feature_names)
    shap_values = explanation.values
    base_val = float(np.mean(explanation.base_values))

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]

    lines = [
        "=" * 70,
        "ОТЧЁТ: SHAP-АНАЛИЗ (GradientBoosting оптимизированный)",
        "=" * 70,
        "",
        f"Базовое значение (expected value): {base_val:.6f}",
        f"Среднее |SHAP| по всем признакам: {mean_abs.mean():.6f}",
        "",
        "ТОП-15 ПРИЗНАКОВ ПО SHAP:",
    ]
    for rank, i in enumerate(top_idx, 1):
        lines.append(f"  {rank:2d}. {feature_names[i]:35s}  mean|SHAP|={mean_abs[i]:.6f}")

    lines.append("")
    lines.append("ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
    top1_name = feature_names[top_idx[0]]
    top1_human = readable_feat(top1_name).replace("\n", " ")
    lines.append(f"  Топ-1 признак по SHAP — {top1_name} ({top1_human}).")
    lines.append("  При воздушной утечке режим работы компрессора меняется: интенсивнее")
    lines.append("  включаются циклы, растёт нагрузка на двигатель, изменяется")
    lines.append("  давление в пневматической системе и температура масла. Среди")
    lines.append("  топ-15 признаков по SHAP доминируют статистики DV_pressure,")
    lines.append("  Oil_temperature, Motor_current и Reservoirs — это согласуется")
    lines.append("  с физикой утечки (Air Leak).")
    lines.append("  Порядок признаков в SHAP в целом отличается от ранжирования по")
    lines.append("  Pearson-корреляции: SHAP учитывает нелинейные взаимодействия")
    lines.append("  и контекст, в котором отдельный признак влияет на предсказание.")
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    (REPORTS_DIR / "06_shap_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, ts_test = load_data()

    model = joblib.load(MODELS_DIR / "GradientBoosting_tuned.pkl")

    explanation = explain_binary(model, X_test)

    plot_shap_summary(explanation, "23_shap_summary.png")
    plot_shap_bar(explanation, "24_shap_bar.png")
    plot_shap_waterfall(explanation, X_test, y_test, ts_test, "25_shap_waterfall_failure.png")
    plot_shap_dependence(explanation, X_test, "26_shap_dependence.png")

    X_test_arr = X_test.to_numpy()
    y_prob_tuned = model.predict_proba(X_test_arr)[:, 1]

    models_cal = [("GradBoosting оптимизированный", y_prob_tuned)]
    if (MODELS_DIR / "GradientBoosting.pkl").exists():
        gb_base = joblib.load(MODELS_DIR / "GradientBoosting.pkl")
        y_prob_base = gb_base.predict_proba(X_test_arr)[:, 1]
        models_cal.insert(0, ("GradBoosting базовый", y_prob_base))
    if (MODELS_DIR / "GaussianNB.pkl").exists():
        gnb = joblib.load(MODELS_DIR / "GaussianNB.pkl")
        y_prob_gnb = gnb.predict_proba(X_test_arr)[:, 1]
        models_cal.append(("GaussianNB", y_prob_gnb))

    plot_calibration(models_cal, y_test, "27_calibration.png")

    write_shap_report(explanation)
