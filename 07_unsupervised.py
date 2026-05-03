from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PCA_N_COMPONENTS = 10
KMEANS_PC = 5
SILHOUETTE_SAMPLE = 5000
ISO_N_ESTIMATORS = 200


def main():
    X_train = pd.read_csv(DATA / "X_train.csv").to_numpy()
    y_train = pd.read_csv(DATA / "y_train.csv").to_numpy().ravel()
    X_test = pd.read_csv(DATA / "X_test.csv").to_numpy()
    y_test = pd.read_csv(DATA / "y_test.csv").to_numpy().ravel()

    n_pos_train = int(y_train.sum())
    if n_pos_train == 0:
        raise ValueError("В y_train нет положительных примеров — расчёт contamination невозможен.")
    contamination = n_pos_train / len(y_train)

    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    t = time.time()
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE).fit(Xs_train)
    pca_time = time.time() - t
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)
    n95 = int(np.searchsorted(cum_var, 0.95) + 1)
    Xp_train = pca.transform(Xs_train)
    Xp_test = pca.transform(Xs_test)

    sil_results = {}
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(Xs_train), size=min(SILHOUETTE_SAMPLE, len(Xs_train)), replace=False)
    for k in (2, 3, 4, 5, 6):
        km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE).fit(Xp_train[:, :KMEANS_PC])
        sil = silhouette_score(Xp_train[sample_idx, :KMEANS_PC], km.labels_[sample_idx])
        sil_results[k] = sil
    best_k = max(sil_results, key=sil_results.get)
    km_best = KMeans(n_clusters=best_k, n_init="auto", random_state=RANDOM_STATE).fit(Xp_train[:, :KMEANS_PC])
    cluster_labels = km_best.labels_

    cluster_pos_rate = {}
    for c in range(best_k):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_pos_rate[c] = float(y_train[mask].mean())
    pos_cluster = max(cluster_pos_rate, key=cluster_pos_rate.get)
    pos_cluster_rate = cluster_pos_rate[pos_cluster]
    overall_pos_rate = float(y_train.mean())
    enrichment = pos_cluster_rate / overall_pos_rate if overall_pos_rate > 0 else 0.0

    t = time.time()
    iso = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(Xs_train)
    iso_time = time.time() - t
    anom_train = (iso.predict(Xs_train) == -1).astype(int)
    anom_test = (iso.predict(Xs_test) == -1).astype(int)

    iso_recall_train = recall_score(y_train, anom_train, zero_division=0)
    iso_precision_train = precision_score(y_train, anom_train, zero_division=0)
    iso_recall_test = recall_score(y_test, anom_test, zero_division=0)
    iso_precision_test = precision_score(y_test, anom_test, zero_division=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    neg = y_train == 0
    pos = y_train == 1
    ax.scatter(Xp_train[neg, 0], Xp_train[neg, 1], s=4, alpha=0.25,
               color="#9aa6b2", label="норма")
    ax.scatter(Xp_train[pos, 0], Xp_train[pos, 1], s=10, alpha=0.85,
               color="#d9534f", label="отказ / pre-fail")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("(а) PCA: проекция на 2 главные компоненты\nцвет — истинная метка")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    cmap = matplotlib.colormaps["tab10"].resampled(best_k)
    for c in range(best_k):
        mask = cluster_labels == c
        ax.scatter(Xp_train[mask, 0], Xp_train[mask, 1], s=4, alpha=0.35,
                   color=cmap(c), label=f"кластер {c} (pos {cluster_pos_rate[c]*100:.1f}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"(б) K-Means k={best_k}\nцвет — кластер")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ks = list(sil_results.keys())
    sils = [sil_results[k] for k in ks]
    bars = ax.bar(ks, sils, color="#5b9bd5", edgecolor="#1f3864")
    for k, s, _b in zip(ks, sils, bars):
        ax.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)
    ax.axvline(best_k, color="#d9534f", linestyle="--", alpha=0.5)
    ax.set_xlabel("k (число кластеров)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("(в) Silhouette score: подбор k")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    n_show = len(var_explained)
    ax.plot(range(1, n_show + 1), cum_var * 100, marker="o", color="#1f3864")
    ax.axhline(95, color="#d9534f", linestyle="--", alpha=0.5, label="95%")
    ax.axvline(n95, color="#d9534f", linestyle="--", alpha=0.5)
    ax.set_xlabel("Число главных компонент")
    ax.set_ylabel("Накопленная дисперсия, %")
    ax.set_title(f"(г) PCA: для 95% дисперсии нужно {n95} компонент")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG / "28_unsupervised_overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

    n_features_in = Xs_train.shape[1]
    rep_lines = [
        "=== Применение методов обучения без учителя ===",
        "",
        f"PCA: {n_features_in} признаков → {PCA_N_COMPONENTS} компонент за {pca_time:.2f} с",
        f"  PC1 объясняет {var_explained[0]*100:.2f}% дисперсии",
        f"  PC2 объясняет {var_explained[1]*100:.2f}% дисперсии",
        f"  для 95% дисперсии достаточно {n95} компонент",
        "",
        f"K-Means (по PC1..PC{KMEANS_PC}):",
    ]
    for k, s in sil_results.items():
        rep_lines.append(f"  k={k}: silhouette = {s:.4f}")
    rep_lines += [
        f"  лучший k = {best_k}",
        f"  кластер {pos_cluster} концентрирует "
        f"{pos_cluster_rate*100:.2f}% позитивов "
        f"(базовая частота — {overall_pos_rate*100:.2f}%); "
        f"enrichment = ×{enrichment:.2f}",
        "",
        f"Isolation Forest (n_est={ISO_N_ESTIMATORS}, contamination={contamination:.4f}):",
        f"  обучение за {iso_time:.2f} с",
        f"  на train: recall={iso_recall_train:.3f}, precision={iso_precision_train:.3f}",
        f"  на test:  recall={iso_recall_test:.3f}, precision={iso_precision_test:.3f}",
    ]
    (REP / "07_unsupervised_report.txt").write_text(
        "\n".join(rep_lines), encoding="utf-8"
    )
    print("\n".join(rep_lines))


if __name__ == "__main__":
    main()
