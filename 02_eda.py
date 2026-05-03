from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = ROOT / "data" / "MetroPT3_AirCompressor_.csv"

RANDOM_STATE = 42

ANALOG_SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current",
]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig1_target_distribution():
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["target"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["target"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, y, name in zip(axes, [y_train, y_test], ["Train", "Test"]):
        counts = y.value_counts().sort_index()
        counts_arr = counts.to_numpy()
        bars = ax.bar(["Норма (0)", "Отказ/предотказ (1)"], counts_arr,
                      color=["#4C78A8", "#E45756"])
        for bar, v in zip(bars, counts_arr):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n({100 * v / len(y):.2f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{name}: {len(y):,} окон")
        ax.set_ylabel("Количество окон")
        ax.set_ylim(0, counts_arr.max() * 1.15)
    fig.suptitle("Баланс классов в обучающей и тестовой выборках", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig2_target_timeline():
    feats = pd.read_csv(DATA_DIR / "features_aggregated.csv", parse_dates=["window_start", "window_end"])
    periods = pd.read_csv(DATA_DIR / "labeling_periods.csv", parse_dates=["start", "end"])

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.scatter(feats["window_start"], feats["target"], s=3, alpha=0.5, c=feats["target"],
               cmap="coolwarm")
    ax.axvline(pd.Timestamp("2020-06-01"), color="black", linestyle="--", alpha=0.7,
               label="Граница train/test")
    for _, row in periods.iterrows():
        ax.axvspan(row["start"], row["end"], color="red", alpha=0.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Норма", "Отказ/предотказ"])
    ax.set_title("Временная шкала: положительные окна и зарегистрированные отказы")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_target_timeline.png", bbox_inches="tight")
    plt.close(fig)


def fig3_feature_distributions():
    feats = pd.read_csv(DATA_DIR / "features_aggregated.csv")
    key = ["TP2_mean", "TP3_mean", "H1_mean", "DV_pressure_mean",
           "Reservoirs_mean", "Oil_temperature_mean", "Motor_current_mean"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for ax, col in zip(axes, key):
        for label, color in [(0, "#4C78A8"), (1, "#E45756")]:
            data = feats.loc[feats["target"] == label, col]
            ax.hist(data, bins=50, alpha=0.55, color=color,
                    label=f"target={label} (n={len(data)})", density=True)
        ax.set_title(col)
        ax.legend(fontsize=7)
    for ax in axes[len(key):]:
        ax.axis("off")
    fig.suptitle("Распределения средних значений сенсоров по классам", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_feature_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig4_correlation_with_target():
    feats = pd.read_csv(DATA_DIR / "features_aggregated.csv")
    feat_cols = [c for c in feats.columns if c not in ("window_start", "window_end", "target")]
    corr = feats[feat_cols + ["target"]].corr(numeric_only=True)["target"].drop("target")
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)

    top = corr.head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Коэффициент корреляции с target")
    ax.set_title("Топ-20 признаков по корреляции с целевой переменной")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    return corr


def fig5_pairplot():
    feats = pd.read_csv(DATA_DIR / "features_aggregated.csv")
    feat_cols = [c for c in feats.columns if c not in ("window_start", "window_end", "target")]
    corr = feats[feat_cols + ["target"]].corr(numeric_only=True)["target"].drop("target")
    top4 = corr.abs().sort_values(ascending=False).head(4).index.tolist()

    sample = feats.sample(min(5000, len(feats)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, fi in enumerate(top4):
        for j, fj in enumerate(top4):
            ax = axes[i, j]
            if i == j:
                for label, color in [(0, "#4C78A8"), (1, "#E45756")]:
                    data = sample.loc[sample["target"] == label, fi]
                    ax.hist(data, bins=30, alpha=0.55, color=color, density=True)
                ax.set_ylabel("density") if j == 0 else None
            else:
                pos = sample[sample["target"] == 1]
                neg = sample[sample["target"] == 0]
                ax.scatter(neg[fj], neg[fi], s=3, alpha=0.25, color="#4C78A8", label="0")
                ax.scatter(pos[fj], pos[fi], s=8, alpha=0.7, color="#E45756", label="1")
            if i == 3:
                ax.set_xlabel(fj, fontsize=8)
            if j == 0:
                ax.set_ylabel(fi, fontsize=8)
    fig.suptitle("Совместные распределения топ-4 признаков по корреляции с target", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_pairplot_key_features.png", bbox_inches="tight")
    plt.close(fig)


def fig6_raw_around_failure():
    failure_start = pd.Timestamp("2020-06-05 10:00:00")
    if Path(RAW_CSV).exists():
        df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"],
                         usecols=["timestamp"] + ANALOG_SENSORS)
        mask = (df["timestamp"] >= failure_start - pd.Timedelta(hours=3)) & \
               (df["timestamp"] <= failure_start + pd.Timedelta(hours=3))
        seg = df.loc[mask].copy()
        x_col, title_suffix = "timestamp", "Аналоговые сенсоры (raw 10 сек)"
        ycols = ANALOG_SENSORS
    else:
        feats = pd.read_csv(DATA_DIR / "features_aggregated.csv", parse_dates=["window_start"])
        mask = (feats["window_start"] >= failure_start - pd.Timedelta(hours=3)) & \
               (feats["window_start"] <= failure_start + pd.Timedelta(hours=3))
        seg = feats.loc[mask].copy()
        x_col = "window_start"
        title_suffix = "Аналоговые сенсоры (mean окна 10 мин)"
        ycols = [f"{s}_mean" for s in ANALOG_SENSORS]

    fig, axes = plt.subplots(len(ycols), 1, figsize=(12, 1.3 * len(ycols)), sharex=True)
    for ax, col, sensor in zip(axes, ycols, ANALOG_SENSORS):
        ax.plot(seg[x_col], seg[col], linewidth=0.8, color="#4C78A8")
        ax.axvline(failure_start, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(sensor, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    axes[-1].tick_params(axis="x", labelsize=8)
    axes[0].set_title(f"{title_suffix} за 3 ч до и 3 ч после старта отказа #3 (5 июня 10:00)",
                      fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_sensor_raw_around_failure.png", bbox_inches="tight")
    plt.close(fig)


def write_eda_report(corr):
    feats = pd.read_csv(DATA_DIR / "features_aggregated.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["target"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["target"]

    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЁТ EDA: MetroPT-3 (подготовленный табличный датасет)")
    lines.append("=" * 70)
    lines.append("")
    feat_cols_for_count = [c for c in feats.columns if c not in ("window_start", "window_end", "target")]
    n_analog = sum(any(c.startswith(s + "_") for s in ANALOG_SENSORS) for c in feat_cols_for_count)
    n_digital = len(feat_cols_for_count) - n_analog
    lines.append("РАЗМЕРНОСТИ")
    lines.append(f"   Полный датасет: {feats.shape[0]:,} окон × {feats.shape[1]} колонок")
    lines.append(f"   Признаков: {len(feat_cols_for_count)} ({n_analog} аналоговых + {n_digital} цифровых статистик)")
    lines.append(f"   Train: {len(y_train):,} | pos={int(y_train.sum())} ({100 * y_train.mean():.2f}%)")
    lines.append(f"   Test:  {len(y_test):,} | pos={int(y_test.sum())} ({100 * y_test.mean():.2f}%)")
    lines.append("")
    lines.append("ПРОПУСКИ И АНОМАЛИИ")
    na = feats.isna().sum().sum()
    lines.append(f"   Всего NaN в признаках: {na}")
    lines.append("")
    lines.append("ТОП-15 ПРИЗНАКОВ ПО |corr(target)|")
    for name, v in corr.head(15).items():
        lines.append(f"   {name:30s}  {v:+.4f}")
    lines.append("")
    lines.append("НЕИНФОРМАТИВНЫЕ ПРИЗНАКИ (|corr| < 0.01)")
    weak = corr[corr.abs() < 0.01]
    if len(weak):
        for name in weak.index:
            lines.append(f"   {name}  ({corr[name]:+.4f})")
    else:
        lines.append("   нет")
    lines.append("")
    lines.append("ПОСТОЯННЫЕ ПРИЗНАКИ (std == 0)")
    feat_cols = [c for c in feats.columns if c not in ("window_start", "window_end", "target")]
    stds = feats[feat_cols].std()
    const_feats = stds[stds < 1e-12].index.tolist()
    if const_feats:
        for c in const_feats:
            lines.append(f"   {c}  (value={feats[c].iloc[0]:.4f})")
    else:
        lines.append("   нет")
    lines.append("")
    lines.append("ОПИСАТЕЛЬНАЯ СТАТИСТИКА ТОП-10 ПРИЗНАКОВ")
    lines.append("")
    top10 = corr.head(10).index.tolist()
    lines.append(feats[top10].describe().round(3).to_string())
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    (REPORTS_DIR / "02_eda_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    fig1_target_distribution()
    fig2_target_timeline()
    fig3_feature_distributions()
    corr = fig4_correlation_with_target()
    fig5_pairplot()
    fig6_raw_around_failure()
    write_eda_report(corr)
