"""Data loading, preprocessing, and seen/unseen splitting for contrastive training.

Pipeline:
    raw h5ad (already protein-coding only) -> CPM normalize -> cache processed h5ad
             -> split cell types into seen / unseen
             -> split seen cells into train / val / test_seen (stratified by cell_type)
             -> unseen-type cells all go to test_unseen

"Unseen" cell types are held out of train/val entirely, so metrics on them measure
whether the embedding space generalizes to novel classes it never saw a label for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class SplitConfig:
    min_cells_per_type: int = 50  # drop cell types too rare to split reliably
    unseen_frac: float = 0.2  # fraction of (kept) cell types held out entirely
    val_frac: float = 0.1  # of seen cells
    test_frac: float = 0.2  # of seen cells
    seed: int = 0


def preprocess(
    raw_path: Path,
    cache_path: Path | None = None,
    force: bool = False,
) -> sc.AnnData:
    """CPM-normalize only. Raw file is already restricted to protein-coding genes
    (see scripts/download_census.py). Cached to disk since it's deterministic."""
    if cache_path is not None and cache_path.exists() and not force:
        return sc.read_h5ad(cache_path)

    adata = sc.read_h5ad(raw_path)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)  # CPM

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(cache_path)
    return adata


def make_splits(obs: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Return a `split` label per cell: one of train/val/test_seen/test_unseen/dropped."""
    rng = np.random.default_rng(cfg.seed)

    counts = obs["cell_type"].value_counts()
    kept_types = counts[counts >= cfg.min_cells_per_type].index.to_numpy()
    dropped_types = set(counts.index) - set(kept_types)

    kept_types_sorted = np.sort(kept_types)
    n_unseen = max(1, int(round(len(kept_types_sorted) * cfg.unseen_frac)))
    unseen_types = set(rng.choice(kept_types_sorted, size=n_unseen, replace=False))
    seen_types = set(kept_types_sorted) - unseen_types

    split = pd.Series("dropped", index=obs.index, dtype=object)
    split.loc[obs["cell_type"].isin(dropped_types)] = "dropped"
    split.loc[obs["cell_type"].isin(unseen_types)] = "test_unseen"

    seen_mask = obs["cell_type"].isin(seen_types)
    seen_idx = obs.index[seen_mask]
    seen_labels = obs.loc[seen_idx, "cell_type"]

    train_idx, rest_idx = train_test_split(
        seen_idx,
        test_size=cfg.val_frac + cfg.test_frac,
        stratify=seen_labels,
        random_state=cfg.seed,
    )
    rest_labels = obs.loc[rest_idx, "cell_type"]
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac),
        stratify=rest_labels,
        random_state=cfg.seed,
    )

    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test_seen"

    return split


def build_and_cache_splits(
    processed_path: Path,
    split_cache_path: Path,
    cfg: SplitConfig,
    force: bool = False,
) -> pd.DataFrame:
    if split_cache_path.exists() and not force:
        df = pd.read_csv(split_cache_path, index_col=0)
        df.index = df.index.astype(str)
        return df

    adata = sc.read_h5ad(processed_path)
    split = make_splits(adata.obs, cfg)
    df = pd.DataFrame({"cell_type": adata.obs["cell_type"].to_numpy(), "split": split.to_numpy()}, index=adata.obs_names)

    split_cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_cache_path)
    return df


class ContrastiveDataset(Dataset):
    """Wraps a dense expression matrix + integer labels for a given split."""

    def __init__(self, x: np.ndarray, labels: np.ndarray, label_names: list[str]):
        self.x = x.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.label_names = label_names

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.labels[idx]


def load_split_datasets(
    processed_path: Path,
    splits_df: pd.DataFrame,
) -> dict[str, ContrastiveDataset]:
    adata = sc.read_h5ad(processed_path)
    x_full = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    # shared label encoding across seen splits so class indices are consistent;
    # unseen-type cells get labels too (only used for eval metrics, never for loss).
    all_types = sorted(adata.obs["cell_type"].unique().tolist())
    type_to_idx = {t: i for i, t in enumerate(all_types)}

    datasets = {}
    for split_name in ["train", "val", "test_seen", "test_unseen"]:
        mask = (splits_df.loc[adata.obs_names, "split"] == split_name).to_numpy()
        x = x_full[mask]
        labels = adata.obs["cell_type"].to_numpy()[mask]
        label_idx = np.array([type_to_idx[t] for t in labels])
        datasets[split_name] = ContrastiveDataset(x, label_idx, all_types)
    return datasets
