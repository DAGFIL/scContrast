# scContrast

## What this project does

Single-cell RNA sequencing (scRNA-seq) measures, for each individual cell in a sample,
how active every gene is. A dataset is essentially a big table: rows are cells, columns
are genes, values are expression levels. The goal is to group cells by type (T-cell,
neuron, astrocyte, ...) using only this expression data — but raw expression vectors
are noisy and high-dimensional, so cell types don't separate cleanly in that raw space.

This project trains a neural network to transform each cell's expression vector into a
new, lower-dimensional "embedding" where cells of the same type end up close together
and cells of different types end up far apart — a much better space for clustering,
visualization, and downstream analysis than the raw data.

The specific technique is **supervised contrastive learning (SupCon)**: instead of just
telling the network "this cell is a T-cell," we show it two slightly corrupted versions
of the same cell (see [Augmentations](#augmentations) below) and train it to recognize
that both versions — and any other cell sharing the same label — belong together, while
pushing every other cell in the batch away. The network used to do this has two
identical, weight-sharing halves that each process one view and get compared against
each other — this "twin network" setup is what's meant by **siamese** in this context.

We also deliberately hide some cell types from the model entirely during training (see
[Seen vs. unseen splits](#seen-vs-unseen-splits)), so we can check whether the learned
embedding space generalizes to cell types it has never been told the name of.

## Where the data comes from

Data comes from the [CZ CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/),
a public repository of curated single-cell datasets. `scripts/download_census.py` pulls
a subset of cells for a given tissue (e.g. blood, brain), keeps only protein-coding
genes (the genes that actually produce proteins — the rest is mostly noise for this
purpose), and saves the result as an `.h5ad` file (the standard single-cell data format,
via the [AnnData](https://anndata.readthedocs.io/) library):

```
python scripts/download_census.py --tissue blood --n-cells 200000
python scripts/download_census.py --tissue brain --n-cells 200000
```

Two datasets have been downloaded so far, 200k cells each: `data/raw/blood_homo_sapiens.h5ad`
and `data/raw/brain_homo_sapiens.h5ad` (19,512 protein-coding genes). Raw and processed
data files are gitignored — they're large and easy to regenerate from the script, so
they don't belong in version control.

## Repository layout

- `data/` — raw and processed datasets. Gitignored.
- `scripts/` — `download_census.py`, the data acquisition step above.
- `notebooks/` — exploratory analysis (QC plots, a first UMAP look at the blood data)
  before any modeling starts.
- `experiments/` — one self-contained folder per experiment. Each has its own data
  loading, model, loss, training loop, and plotting code, so experiments can diverge
  (different preprocessing, different tissue) without one breaking another. Run outputs
  (checkpoints, logs, figures) are gitignored; the code that produced them is tracked.

## How a training run works, step by step

Each experiment (`experiments/contrastive_supcon` for blood,
`experiments/contrastive_supcon_brain` for brain) runs the same pipeline:

### 1. Preprocess

Raw counts aren't directly comparable between cells — a cell that was sequenced more
deeply just has bigger numbers across the board, not "more expression." Each
experiment's `dataset.py` defines its own preprocessing to correct for this:

- **Blood** (`contrastive_supcon`): normalize each cell to counts-per-million (CPM) so
  sequencing depth doesn't bias the comparison, log-transform to tame extreme values,
  keep only the 2,000 most variable genes (the ones that actually differ between cell
  types — the rest is mostly flat, uninformative signal), then z-score scale.
- **Brain** (`contrastive_supcon_brain`): CPM normalization only, keeping the full set
  of ~19.5k protein-coding genes — no log transform, no gene filtering, no scaling. A
  deliberately simpler pipeline than blood's, to see how much (if any) of that extra
  preprocessing was actually necessary.

The result is cached to `data/processed/` so this only runs once per configuration.

### 2. Split cell types into seen vs. unseen

This is the core experimental design choice, worth explaining in full. Cell types are
split two ways:

- **Unseen types** (20% of types by default) are removed from training entirely — the
  model never sees a single labeled example of them. They only appear later, at
  evaluation time, in a `test_unseen` set.
- **Seen types** (the remaining 80%) are split, per cell type, into `train` / `val` /
  `test_seen` (70% / 10% / 20%).

Evaluating on `test_seen` answers "did the model learn to separate the types it was
trained on?" Evaluating on `test_unseen` answers the more interesting question: "does
the geometry the model learned transfer to types it never saw a label for?" A model that
only memorizes its training labels will do fine on `test_seen` and fall apart on
`test_unseen`; a model that learned something general about what makes cell types
different should hold up reasonably well on both.

### 3. Train

- **Augmentations** (`augmentations.py`): for every cell in a batch, two corrupted
  copies are generated — random genes zeroed out, random per-cell rescaling, and
  Gaussian noise added. These two views are the positive pair the network learns to
  pull together.
- **Model** (`model.py`): a shared-weight MLP encoder turns each view into a 128-dim
  embedding; a small projection head further maps that down to 64 dims, purely for
  computing the loss (standard practice — keeps the loss geometry decoupled from the
  embedding actually used downstream).
- **Loss** (`losses.py`): Supervised Contrastive Loss (Khosla et al., 2020). Within a
  batch, all views of all cells sharing a label are pulled together, everything else is
  pushed apart — a generalization of the more common "only my own two augmented views
  are positives" contrastive setup, made possible by having the true label available.
- **Metrics** (`metrics.py`), computed periodically on val/test: silhouette score
  (how well-separated the true classes are in embedding space) and ARI/NMI (how well an
  unsupervised KMeans clustering of the embeddings agrees with the true labels).
- **Logging**: everything is written to TensorBoard — loss curves, the metrics above per
  split, UMAP visualizations of the embedding space per split/epoch, weight/gradient
  histograms, class balance per split, and a histogram comparing cosine similarity for
  same-class vs. different-class pairs (the clearest single read on whether the loss is
  doing its job).

Run it:

```
python experiments/contrastive_supcon/main.py --run-name <name> --epochs 50
python experiments/contrastive_supcon_brain/main.py --run-name <name> --epochs 50
```

Add `--debug` to subsample every split down to a few thousand cells first — useful for
catching bugs in seconds instead of minutes before committing to a full run.

## Results so far

**Blood** (`contrastive_supcon`, run `supcon_baseline`, 60 epochs, HVG-2000 preprocessing):

| split       | silhouette | ARI   | NMI   |
|-------------|-----------:|------:|------:|
| test_seen   | 0.294      | 0.497 | 0.747 |
| test_unseen | 0.173      | 0.380 | 0.643 |

As expected, performance drops from seen to unseen types, but the drop isn't
catastrophic — the model retains a meaningful chunk of its clustering quality on cell
types it never saw a label for.

**Brain** (`contrastive_supcon_brain`, run `cpm_baseline`, CPM-only preprocessing, all
~19.5k protein-coding genes): training in progress (50 epochs on CPU, ~65s/epoch —
the machine's GPU is currently unavailable at the driver level, unrelated to this
project). Latest validation checkpoint (epoch 25/50):

| split | silhouette | ARI   | NMI   |
|-------|-----------:|------:|------:|
| val   | 0.598      | 0.790 | 0.830 |

Already well ahead of blood's final numbers at the halfway point of training. This is
expected more than it is a sign of a better setup: brain cell types (neurons,
astrocytes, oligodendrocytes, microglia, ...) are more transcriptionally distinct from
each other than most blood cell types are, so they're inherently easier to separate —
brain was picked partly as this kind of "should be easy" reference point. Final
test_seen / test_unseen numbers will be added here once training finishes
(`experiments/contrastive_supcon_brain/runs/cpm_baseline/test_metrics.json`).

## Getting set up

```
conda env create -f environment.yml
conda activate sccontrast
```

Uses PyTorch 2.5.1 + CUDA 12.1, and falls back to CPU automatically if no GPU is
available.
