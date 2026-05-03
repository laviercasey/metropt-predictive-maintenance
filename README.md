# Predictive Maintenance для компрессора поезда метро

Курсовая работа по дисциплине «Машинное обучение и анализ данных».
Применение ансамблевых методов для раннего детектирования утечек воздуха
в воздушном компрессоре (APU) поезда метро Порто.

Финальная модель — оптимизированный GradientBoosting — на отложенной
тестовой выборке достигает **ROC-AUC = 0,9934, PR-AUC = 0,8893,
F1 = 0,878 (precision 0,873 / recall 0,882)**.

## Содержание

- [Датасет](#датасет)
- [Задача](#задача)
- [Пайплайн](#пайплайн)
- [Результаты](#результаты)
- [Структура проекта](#структура-проекта)
- [Зависимости](#зависимости)
- [Лицензия](#лицензия)
- [Источники](#источники)

## Датасет

[MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
(UCI Machine Learning Repository, 2023):
- 1 516 948 точек time series, 15 сенсоров, шаг 10 секунд
- Период сбора: 1 февраля — 31 августа 2020 года
- 4 зарегистрированных отказа типа Air Leak (утечка воздуха)
- Источник: Veloso B., Ribeiro R.P., Pereira P.M., Gama J. (2022)
- Лицензия датасета: CC BY 4.0

### Скачивание датасета

Сырой CSV (~200 MB) не включён в репозиторий — скачайте с UCI:

```bash
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; ds = fetch_ucirepo(id=791); ds.data.features.to_csv('data/MetroPT3_AirCompressor_.csv', index=False)"
```

Альтернативно: `https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset`

Положите файл в `data/MetroPT3_AirCompressor_.csv` перед запуском `01_prepare_dataset.py`.

> Подготовленные обучающая и тестовая выборки (`X_train.csv`, `X_test.csv`,
> `y_train.csv`, `y_test.csv` и др.) **уже лежат в `data/`** — можно сразу
> запускать модели, не пересчитывая агрегацию.

## Задача

Бинарная классификация состояний компрессора с горизонтом упреждения 30 минут:
по показаниям 15 сенсоров в скользящем 10-минутном окне определить, находится
ли компрессор в отказе или отказ начнётся в ближайшие полчаса.
`target = 1` если окно содержит точку из периода отказа **или** из 30-минутного
окна перед его началом.

Главная сложность — сильный дисбаланс классов (~2% позитивов) и временная
структура данных (соседние окна автокоррелированы).

## Пайплайн

```bash
# 1. Подготовка окружения
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/Mac

# 2. Подготовка данных (нужен сырой CSV в data/MetroPT3_AirCompressor_.csv)
python 01_prepare_dataset.py

# 3. Исследовательский анализ (EDA)
python 02_eda.py

# 4. Базовые модели (7 классических алгоритмов)
python 03_train_models.py

# 5. Ансамблевые модели (9 ансамблей: RF, ExtraTrees, AdaBoost, GB, XGB, LGBM, CatBoost, Voting, Stacking)
python 04_ensembles.py

# 6. Оптимизация: RandomizedSearch + threshold tuning + feature selection
python 05_optimization.py

# 7. Интерпретация финальной модели через SHAP
python 06_shap_analysis.py

# 8. Методы обучения без учителя (PCA, K-Means, Isolation Forest)
python 07_unsupervised.py
```

Каждый скрипт сохраняет графики в `reports/figures/` и текстовый отчёт в `reports/`.

## Результаты

### Подготовка данных

| Метрика | Значение |
|---|---|
| Точек сырых данных | 1 516 948 |
| Пропусков | 0 |
| Окон после агрегации (10-мин окно, шаг 5 мин) | 50 563 |
| Признаков (6 статистик × 7 аналог. + 2 × 8 цифр.) | 58 |
| Train (до 01.06.2020) | 28 560 окон, 1,34% позитивов |
| Test (с 01.06.2020) | 22 003 окон, 2,94% позитивов |
| Метод split | Временной (без перемешивания) |
| Валидация в train | TimeSeriesSplit, 3 фолда |

### Метрики моделей на тесте

| Модель | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| GaussianNB (базовая) | 0,946 | 0,807 | 0,635 | 0,921 | 0,485 |
| LDA (базовая) | 0,929 | 0,788 | 0,911 | 0,892 | 0,931 |
| RandomForest | 0,877 | 0,230 | 0,000* | 0,000* | 0,000* |
| GradientBoosting | 0,982 | 0,585 | 0,231* | 0,131* | 0,966 |
| **GradientBoosting опт. + порог** | **0,993** | **0,889** | **0,878** | **0,882** | **0,873** |

\* при пороге 0,5 — после подбора порога recall и F1 принципиально другие.

### Применение методов без учителя

| Метод | Параметр | Результат |
|---|---|---|
| PCA | 58 → компоненты для 95% дисперсии | 11 компонент |
| K-Means | оптимум по силуэту | k = 2, silhouette = 0,637 |
| K-Means | концентрация позитивов в "плохом" кластере | 24,1% (×18 от базовой 1,34%) |
| Isolation Forest | recall / precision на test | 0,195 / 0,395 |

### SHAP-разбор финальной модели

Главный признак — `Oil_temperature_min` (среднее |SHAP| 0,361),
а не `DV_pressure_mean` (как подсказывала линейная корреляция).
Корреляция Пирсона не учитывает взаимодействия признаков, SHAP — учитывает.

## Структура проекта

```
.
├── README.md                          # этот файл
├── LICENSE                            # MIT для кода (датасет — CC BY 4.0)
├── requirements.txt                   # фиксированные версии зависимостей
├── .gitignore
│
├── 01_prepare_dataset.py              # Сырой CSV → обучающие выборки
├── 02_eda.py                          # Разведочный анализ + графики 1–6
├── 03_train_models.py                 # 7 базовых моделей
├── 04_ensembles.py                    # 9 ансамблевых моделей
├── 05_optimization.py                 # RandomizedSearch + threshold tuning
├── 06_shap_analysis.py                # SHAP, калибровка, feature importance
├── 07_unsupervised.py                 # PCA, K-Means, Isolation Forest
│
├── data/                              # Подготовленные выборки
│   ├── X_train.csv, y_train.csv       # 28 560 × 58
│   ├── X_test.csv, y_test.csv         # 22 003 × 58
│   ├── train_timestamps.csv, test_timestamps.csv
│   └── labeling_periods.csv           # 4 периода отказов с метками времени
│
├── reports/                           # Артефакты пайплайна
│   ├── figures/                       # 29 графиков, использованных в курсовой
│   ├── 02_eda_report.txt … 07_unsupervised_report.txt
│   ├── base_models_metrics.csv
│   └── ensemble_metrics.csv
│
└── notebooks/
    └── full_analysis.ipynb            # Интерактивный прогон пайплайна
```

## Зависимости

```bash
pip install -r requirements.txt
```

Версия Python: **3.10+**. Основные пакеты:
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`,
`matplotlib`, `seaborn`, `shap`.

## Лицензия

- **Код проекта** — MIT (см. [LICENSE](LICENSE))
- **Датасет MetroPT-3** — CC BY 4.0 (© авторы датасета, INESC TEC / Porto Metro)

## Источники

- Davari N., Veloso B., Ribeiro R.P., Pereira P.M., Gama J. Predictive
  maintenance based on anomaly detection using deep learning for air production
  unit in the railway industry // DSAA 2021.
  DOI: [10.1109/DSAA53316.2021.9564181](https://doi.org/10.1109/DSAA53316.2021.9564181)
- Veloso B., Ribeiro R.P., Pereira P.M., Gama J. The MetroPT dataset for
  predictive maintenance // Scientific Data 9, 764 (2022).
  DOI: [10.1038/s41597-022-01877-3](https://doi.org/10.1038/s41597-022-01877-3)
