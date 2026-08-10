"""Clustering / embedding-quality metrics used to evaluate contrastive embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def embedding_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    max_cells: int | None = 20_000,
) -> dict[str, float]:
    """Silhouette (using true labels directly) + ARI/NMI (KMeans clustering vs true labels).

    Silhouette needs >=2 classes each with >=1 sample and <= n_samples-1 clusters; degenerate
    cases (single class) return NaN instead of raising.
    """
    n = embeddings.shape[0]
    if max_cells is not None and n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_cells, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return {"silhouette": float("nan"), "ari": float("nan"), "nmi": float("nan")}

    sil = silhouette_score(embeddings, labels)

    km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed)
    pred = km.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)

    return {"silhouette": float(sil), "ari": float(ari), "nmi": float(nmi)}
