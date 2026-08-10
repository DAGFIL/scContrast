"""Matplotlib figures for TensorBoard logging: UMAP scatter of embeddings by label."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import umap


def umap_figure(
    embeddings: np.ndarray,
    label_names: list[str],
    title: str,
    metrics: dict[str, float] | None = None,
    max_cells: int = 5000,
    seed: int = 0,
):
    n = embeddings.shape[0]
    if n > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_cells, replace=False)
        embeddings = embeddings[idx]
        label_names = [label_names[i] for i in idx]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed)
    coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(label_names))
    n_show = 20
    top_labels = set(
        sorted(unique_labels, key=lambda t: -label_names.count(t))[:n_show]
    ) if len(unique_labels) > n_show else set(unique_labels)
    plot_labels = [lbl if lbl in top_labels else "other" for lbl in label_names]

    fig, ax = plt.subplots(figsize=(7, 6))
    cats = sorted(set(plot_labels))
    cmap = plt.get_cmap("tab20", len(cats))
    for i, cat in enumerate(cats):
        mask = np.array([p == cat for p in plot_labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, color=cmap(i), label=cat, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=3, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

    if metrics:
        text = "\n".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        ax.text(
            0.02, 0.02, text, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    fig.tight_layout()
    return fig


def similarity_histogram_figure(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    max_pairs: int = 200_000,
    seed: int = 0,
):
    """Distribution of pairwise cosine similarity, split into same-class (positive) vs
    different-class (negative) pairs. The gap between the two is the clearest read on
    whether the contrastive objective is actually separating classes in embedding space."""
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    sims = np.sum(embeddings[i] * embeddings[j], axis=1)
    same = labels[i] == labels[j]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1, 1, 60)
    ax.hist(sims[~same], bins=bins, alpha=0.6, density=True, label=f"different class (n={(~same).sum():,})", color="#d62728")
    ax.hist(sims[same], bins=bins, alpha=0.6, density=True, label=f"same class (n={same.sum():,})", color="#1f77b4")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def class_distribution_figure(labels: list[str], title: str, max_classes: int = 30):
    """Bar chart of cell counts per class — surfaces class imbalance in a split at a glance."""
    from collections import Counter

    counts = Counter(labels)
    items = counts.most_common(max_classes)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(names))))
    ax.barh(range(len(names)), vals, color="#4c72b0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("cell count")
    ax.set_title(title)
    fig.tight_layout()
    return fig
