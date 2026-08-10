"""Train a SupCon contrastive encoder on scRNA data with seen/unseen cell-type splits.

Testing tissue: brain (transcriptionally distinct cell types — neurons, astrocytes,
oligodendrocytes, microglia — used as a cleaner-separation reference point next to blood).

Usage:
    python experiments/contrastive_supcon_brain/main.py --run-name baseline --epochs 50

Pipeline: raw h5ad -> CPM normalize/log1p/HVG/scale (cached) -> seen/unseen split (cached)
        -> SupCon training (two augmented views per cell, label-aware positives)
        -> TensorBoard scalars (loss, silhouette, ARI, NMI) + UMAP figures per split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations import default_augmentation
from dataset import PROCESSED_DIR, RAW_DIR, REPO_ROOT, SplitConfig, build_and_cache_splits, load_split_datasets, preprocess
from losses import SupConLoss
from metrics import embedding_metrics
from model import SupConModel
from viz import umap_figure

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--raw-path", default=str(RAW_DIR / "brain_homo_sapiens.h5ad"))
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--min-cells-per-type", type=int, default=50)
    p.add_argument("--unseen-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)

    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--mask-p", type=float, default=0.2)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--scale-std", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--force-splits", action="store_true")
    p.add_argument("--debug", action="store_true", help="subsample cells for a fast smoke test")
    p.add_argument("--debug-n-cells", type=int, default=4000)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, augment, criterion, optimizer, device, writer, epoch, global_step):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        view1, view2 = augment(x), augment(x)
        views = torch.cat([view1, view2], dim=0)
        labels = torch.cat([y, y], dim=0)

        _, z = model(views)
        loss = criterion(z, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        writer.add_scalar("loss/train_step", loss.item(), global_step)
        global_step += 1

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def compute_embeddings(model, dataset, device, batch_size=1024):
    """Encoder embeddings, L2-normalized to match the cosine geometry SupCon trains in
    (raw pre-projection features are otherwise on an arbitrary, unconstrained scale)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs, labels = [], []
    for x, y in loader:
        h, _ = model(x.to(device))
        h = torch.nn.functional.normalize(h, dim=-1)
        embs.append(h.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def evaluate_split(model, dataset, device, label_names, writer, tag, step, log_figure, seed):
    if len(dataset) == 0:
        return {}
    embs, labels = compute_embeddings(model, dataset, device)
    m = embedding_metrics(embs, labels, seed=seed)
    for k, v in m.items():
        writer.add_scalar(f"metrics/{tag}_{k}", v, step)

    if log_figure:
        names = [label_names[i] for i in labels]
        fig = umap_figure(embs, names, title=f"{tag} embeddings (epoch {step})", metrics=m, seed=seed)
        writer.add_figure(f"umap/{tag}", fig, step)
    return m


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or time.strftime("supcon_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    raw_stem = Path(args.raw_path).stem
    processed_path = PROCESSED_DIR / f"{raw_stem}_hvg{args.n_top_genes}_scaled.h5ad"
    adata = preprocess(Path(args.raw_path), n_top_genes=args.n_top_genes, cache_path=processed_path, force=args.force_preprocess)
    del adata  # only needed to materialize the cache; splits/datasets re-read from disk

    split_cfg = SplitConfig(
        min_cells_per_type=args.min_cells_per_type,
        unseen_frac=args.unseen_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    split_cache_path = PROCESSED_DIR / f"{raw_stem}_splits_seed{args.seed}_unseen{args.unseen_frac}.csv"
    splits_df = build_and_cache_splits(processed_path, split_cache_path, split_cfg, force=args.force_splits)

    print(splits_df["split"].value_counts())
    n_unseen_types = splits_df.loc[splits_df["split"] == "test_unseen", "cell_type"].nunique()
    n_seen_types = splits_df.loc[splits_df["split"] == "train", "cell_type"].nunique()
    print(f"seen cell types: {n_seen_types}, unseen cell types: {n_unseen_types}")

    datasets = load_split_datasets(processed_path, splits_df)

    if args.debug:
        rng = np.random.default_rng(args.seed)
        for name, ds in datasets.items():
            if len(ds) > args.debug_n_cells:
                idx = rng.choice(len(ds), size=args.debug_n_cells, replace=False)
                ds.x, ds.labels = ds.x[idx], ds.labels[idx]

    label_names = datasets["train"].label_names
    in_dim = datasets["train"].x.shape[1]
    print(f"input dim: {in_dim}, train/val/test_seen/test_unseen sizes: "
          f"{[len(datasets[s]) for s in ['train', 'val', 'test_seen', 'test_unseen']]}")

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    model = SupConModel(
        in_dim=in_dim,
        hidden_dims=args.hidden_dims,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    ).to(device)

    augment = default_augmentation(args.mask_p, args.noise_std, args.scale_std)
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    writer.add_text("config", json.dumps(vars(args), indent=2))

    global_step = 0
    best_val_silhouette = -np.inf
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, global_step = run_epoch(model, train_loader, augment, criterion, optimizer, device, writer, epoch, global_step)
        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  ({time.time() - t0:.1f}s)")

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val_metrics = evaluate_split(model, datasets["val"], device, label_names, writer, "val", epoch, log_figure=True, seed=args.seed)
            print(f"  val: {val_metrics}")
            if val_metrics.get("silhouette", -np.inf) > best_val_silhouette:
                best_val_silhouette = val_metrics["silhouette"]
                torch.save(model.state_dict(), run_dir / "best_model.pt")

    torch.save(model.state_dict(), run_dir / "final_model.pt")

    print("Final test evaluation...")
    test_seen_metrics = evaluate_split(model, datasets["test_seen"], device, label_names, writer, "test_seen", args.epochs, log_figure=True, seed=args.seed)
    test_unseen_metrics = evaluate_split(model, datasets["test_unseen"], device, label_names, writer, "test_unseen", args.epochs, log_figure=True, seed=args.seed)
    print(f"test_seen: {test_seen_metrics}")
    print(f"test_unseen: {test_unseen_metrics}")

    (run_dir / "test_metrics.json").write_text(json.dumps({
        "test_seen": test_seen_metrics,
        "test_unseen": test_unseen_metrics,
    }, indent=2))

    writer.close()
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
