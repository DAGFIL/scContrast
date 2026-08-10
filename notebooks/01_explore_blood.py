# %% [markdown]
# # Explore blood dataset
#
# Basic QC and UMAP visualization of `data/raw/blood_homo_sapiens.h5ad`
# (200k cells, protein-coding genes, CZ CELLxGENE Census).

# %%
from pathlib import Path

import scanpy as sc

sc.settings.verbosity = 1
sc.settings.figdir = Path("figures")
sc.settings.figdir.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path("..").resolve()
DATA_PATH = REPO_ROOT / "data" / "raw" / "blood_homo_sapiens.h5ad"

# %%
adata = sc.read_h5ad(DATA_PATH)
adata

# %% [markdown]
# ## Overview

# %%
print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"tissue: {adata.obs['tissue'].nunique()} unique")
print(f"donor_id (samples): {adata.obs['donor_id'].nunique()} unique")
print(f"cell_type: {adata.obs['cell_type'].nunique()} unique")
print(f"assay: {adata.obs['assay'].nunique()} unique")
print(f"disease: {adata.obs['disease'].nunique()} unique")

# %%
adata.obs["tissue"].value_counts()

# %%
adata.obs["assay"].value_counts()

# %%
adata.obs["disease"].value_counts()

# %%
top_cell_types = adata.obs["cell_type"].value_counts().head(20)
top_cell_types

# %% [markdown]
# ## QC metrics

# %%
adata.var["mt"] = adata.var["feature_name"].str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# %%
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    save="_qc_violin.png",
)

# %% [markdown]
# ## Preprocessing (normalize, HVGs, PCA)

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="donor_id")
sc.pl.highly_variable_genes(adata, save="_hvg.png")

# %%
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50, svd_solver="arpack")
sc.pl.pca_variance_ratio(adata_hvg, n_pcs=50, save="_pca_variance.png")

# %% [markdown]
# ## Neighbors + UMAP

# %%
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_hvg)

# carry embeddings back onto full adata for plotting with all obs columns
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]

# %% [markdown]
# ## UMAP plots

# %%
top20_types = adata.obs["cell_type"].value_counts().head(20).index
cell_type_top20 = adata.obs["cell_type"].astype(str)
cell_type_top20[~cell_type_top20.isin(top20_types)] = "other"
adata.obs["cell_type_top20"] = cell_type_top20.astype("category")

# %%
sc.pl.umap(
    adata,
    color="cell_type_top20",
    legend_loc="right margin",
    legend_fontsize=6,
    title="UMAP colored by cell type (top 20)",
    save="_cell_type.png",
)

# %%
sc.pl.umap(
    adata,
    color="tissue",
    title="UMAP colored by tissue",
    save="_tissue.png",
)

# %%
sc.pl.umap(
    adata,
    color="assay",
    title="UMAP colored by assay",
    save="_assay.png",
)

# %%
sc.pl.umap(
    adata,
    color="total_counts",
    title="UMAP colored by total counts",
    save="_total_counts.png",
)

# %% [markdown]
# ## Donor / sample composition

# %%
cells_per_donor = adata.obs["donor_id"].value_counts()
cells_per_donor.describe()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(cells_per_donor.values, bins=50)
ax.set_xlabel("cells per donor")
ax.set_ylabel("number of donors")
ax.set_title("Cells per donor distribution")
fig.savefig(sc.settings.figdir / "cells_per_donor_hist.png", dpi=150, bbox_inches="tight")
plt.show()
