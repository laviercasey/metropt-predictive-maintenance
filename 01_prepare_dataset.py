from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "MetroPT3_AirCompressor_.csv"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_MIN = 10
STEP_MIN = 5
HORIZON_MIN = 30
SAMPLE_RATE_SEC = 10

WINDOW_SIZE_ROWS = WINDOW_MIN * 60 // SAMPLE_RATE_SEC
STEP_SIZE_ROWS = STEP_MIN * 60 // SAMPLE_RATE_SEC
HORIZON_ROWS = HORIZON_MIN * 60 // SAMPLE_RATE_SEC

TRAIN_END = pd.Timestamp("2020-06-01 00:00:00")

FAILURE_PERIODS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00", "Air Leak"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00", "Air Leak"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00", "Air Leak"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "Air Leak"),
]

ANALOG_SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current",
]
DIGITAL_SENSORS = [
    "COMP", "DV_eletric", "Towers", "MPG",
    "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses",
]


def load_raw_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    df["is_failure"] = 0
    df["is_failure_soon"] = 0

    horizon = pd.Timedelta(minutes=HORIZON_MIN)
    for start, end, _ in FAILURE_PERIODS:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        mask_fail = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
        df.loc[mask_fail, "is_failure"] = 1

        pre_start = start_ts - horizon
        mask_pre = (df["timestamp"] >= pre_start) & (df["timestamp"] < start_ts)
        df.loc[mask_pre, "is_failure_soon"] = 1

    df["target_label"] = ((df["is_failure"] == 1) | (df["is_failure_soon"] == 1)).astype(int)
    return df


def aggregate_windows(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    starts = np.arange(0, n_rows - WINDOW_SIZE_ROWS + 1, STEP_SIZE_ROWS)

    analog_arr = df[ANALOG_SENSORS].to_numpy()
    digital_arr = df[DIGITAL_SENSORS].to_numpy()
    target_arr = df["target_label"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    records = []
    for start_idx in starts:
        end_idx = start_idx + WINDOW_SIZE_ROWS
        a_win = analog_arr[start_idx:end_idx]
        d_win = digital_arr[start_idx:end_idx]

        row = {
            "window_start": timestamps[start_idx],
            "window_end": timestamps[end_idx - 1],
        }
        for j, name in enumerate(ANALOG_SENSORS):
            col = a_win[:, j]
            row[f"{name}_mean"] = col.mean()
            row[f"{name}_std"] = col.std()
            row[f"{name}_min"] = col.min()
            row[f"{name}_max"] = col.max()
            row[f"{name}_median"] = np.median(col)
            mean = col.mean()
            std = col.std()
            if std > 1e-9:
                row[f"{name}_skew"] = ((col - mean) ** 3).mean() / (std ** 3)
            else:
                row[f"{name}_skew"] = 0.0

        for j, name in enumerate(DIGITAL_SENSORS):
            col = d_win[:, j]
            row[f"{name}_mean"] = col.mean()
            row[f"{name}_std"] = col.std()

        row["target"] = int(target_arr[start_idx:end_idx].max())
        records.append(row)

    return pd.DataFrame(records)


def temporal_split(features: pd.DataFrame) -> tuple:
    train_mask = features["window_end"] < TRAIN_END
    train_df = features[train_mask].reset_index(drop=True)
    test_df = features[~train_mask].reset_index(drop=True)
    return train_df, test_df


def save_artifacts(features: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    features.to_csv(DATA_DIR / "features_aggregated.csv", index=False)

    feature_cols = [c for c in features.columns if c not in ("window_start", "window_end", "target")]
    train_df[feature_cols].to_csv(DATA_DIR / "X_train.csv", index=False)
    train_df[["target"]].to_csv(DATA_DIR / "y_train.csv", index=False)
    test_df[feature_cols].to_csv(DATA_DIR / "X_test.csv", index=False)
    test_df[["target"]].to_csv(DATA_DIR / "y_test.csv", index=False)

    train_df[["window_start", "window_end", "target"]].to_csv(
        DATA_DIR / "train_timestamps.csv", index=False
    )
    test_df[["window_start", "window_end", "target"]].to_csv(
        DATA_DIR / "test_timestamps.csv", index=False
    )

    periods_df = pd.DataFrame(FAILURE_PERIODS, columns=["start", "end", "failure_type"])
    periods_df.to_csv(DATA_DIR / "labeling_periods.csv", index=False)


def write_report(df_raw, features, train_df, test_df) -> None:
    feature_cols = [c for c in features.columns if c not in ("window_start", "window_end", "target")]
    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЁТ О ПОДГОТОВКЕ ДАННЫХ MetroPT-3")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. ИСХОДНЫЕ ДАННЫЕ")
    lines.append(f"   Строк: {len(df_raw):,}")
    lines.append(f"   Колонок (сенсоров): {len(ANALOG_SENSORS) + len(DIGITAL_SENSORS)} "
                 f"(аналоговых: {len(ANALOG_SENSORS)}, цифровых: {len(DIGITAL_SENSORS)})")
    lines.append(f"   Период: {df_raw['timestamp'].min()} — {df_raw['timestamp'].max()}")
    lines.append(f"   Пропусков: {df_raw.isna().sum().sum()}")
    lines.append(f"   Фактический шаг дискретизации: {SAMPLE_RATE_SEC} сек")
    lines.append("")
    lines.append("2. РАЗМЕТКА")
    lines.append(f"   Периодов отказов: {len(FAILURE_PERIODS)}")
    for start, end, ftype in FAILURE_PERIODS:
        lines.append(f"     - {start} — {end} ({ftype})")
    lines.append(f"   Горизонт упреждения: {HORIZON_MIN} мин")
    lines.append(f"   Целевая переменная (гибрид): отказ ИЛИ будет в ближайшие {HORIZON_MIN} мин")
    lines.append(f"   Точек в самом отказе:      {int(df_raw['is_failure'].sum()):,}")
    lines.append(f"   Точек в окне пред-отказа:  {int(df_raw['is_failure_soon'].sum()):,}")
    lines.append(f"   Точек в target (гибрид):   {int(df_raw['target_label'].sum()):,}")
    lines.append("")
    lines.append("3. АГРЕГАЦИЯ")
    lines.append(f"   Размер окна: {WINDOW_MIN} мин ({WINDOW_SIZE_ROWS} точек)")
    lines.append(f"   Шаг окна: {STEP_MIN} мин ({STEP_SIZE_ROWS} точек)")
    lines.append(f"   Всего окон: {len(features):,}")
    lines.append(f"   Признаков: {len(feature_cols)}")
    lines.append(f"   Положительных окон: {int(features['target'].sum())} "
                 f"({100 * features['target'].mean():.3f}%)")
    lines.append("")
    lines.append("4. TRAIN/TEST SPLIT (временной)")
    lines.append(f"   Граница: {TRAIN_END}")
    lines.append(f"   Train: {len(train_df):,} окон, "
                 f"положительных {int(train_df['target'].sum())} "
                 f"({100 * train_df['target'].mean():.3f}%)")
    lines.append(f"   Test:  {len(test_df):,} окон, "
                 f"положительных {int(test_df['target'].sum())} "
                 f"({100 * test_df['target'].mean():.3f}%)")
    lines.append("")
    lines.append("5. ФАЙЛЫ")
    for f in sorted(DATA_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        lines.append(f"   {f.name}: {size_mb:.2f} MB")
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    (REPORTS_DIR / "01_preparation_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    df_raw = load_raw_data(RAW_CSV)
    df_raw = label_data(df_raw)
    features = aggregate_windows(df_raw)
    train_df, test_df = temporal_split(features)
    save_artifacts(features, train_df, test_df)
    write_report(df_raw, features, train_df, test_df)
