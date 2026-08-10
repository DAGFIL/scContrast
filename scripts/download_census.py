"""Download a cell subset from the CZ CELLxGENE Census and save as AnnData.

Usage:
    python scripts/download_census.py --tissue blood --n-cells 200000
"""

import argparse
from pathlib import Path

import anndata as ad
import cellxgene_census
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

OBS_COLUMNS = [
    "assay",
    "cell_type",
    "tissue",
    "tissue_general",
    "disease",
    "sex",
    "suspension_type",
    "donor_id",
]


def fetch_cell_metadata(census, organism: str, tissue_general: str) -> pd.DataFrame:
    """Read obs metadata (no expression data) for all cells matching the tissue filter."""
    obs_value_filter = f"tissue_general == '{tissue_general}' and is_primary_data == True"
    obs = (
        census["census_data"][organism]
        .obs.read(value_filter=obs_value_filter, column_names=["soma_joinid", *OBS_COLUMNS])
        .concat()
        .to_pandas()
    )
    return obs


def subsample_joinids(obs: pd.DataFrame, n_cells: int, seed: int) -> list[int]:
    if len(obs) <= n_cells:
        return obs["soma_joinid"].tolist()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(obs["soma_joinid"].to_numpy(), size=n_cells, replace=False)
    return sampled.tolist()


def fetch_protein_coding_ids(feature_ids: list[str], cache_path: Path) -> set[str]:
    """Look up gene biotypes via mygene.info and return the protein-coding subset.

    Results are cached to disk since Census gene sets are stable and re-querying
    tens of thousands of IDs on every run is slow and unnecessary.
    """
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        known = set(cached["feature_id"])
        if set(feature_ids) <= known:
            return set(cached.loc[cached["type_of_gene"] == "protein-coding", "feature_id"])

    import mygene

    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        feature_ids,
        scopes="ensembl.gene",
        fields="type_of_gene",
        species="human",
        as_dataframe=True,
    )
    results = results.reset_index().rename(columns={"query": "feature_id"})
    results = results[["feature_id", "type_of_gene"]].drop_duplicates("feature_id")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cache_path, index=False)

    return set(results.loc[results["type_of_gene"] == "protein-coding", "feature_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--tissue", default="blood", help="tissue_general value to filter on")
    parser.add_argument("--n-cells", type=int, default=200_000)
    parser.add_argument("--census-version", default="stable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protein-coding-only", action="store_true", default=True)
    parser.add_argument("--output", default=None, help="output .h5ad path")
    args = parser.parse_args()

    organism_key = args.organism.lower().replace(" ", "_")
    output = Path(args.output) if args.output else RAW_DIR / f"{args.tissue}_{organism_key}.h5ad"

    print(f"Opening Census (version={args.census_version})...")
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        print(f"Querying cell metadata: tissue_general == '{args.tissue}', is_primary_data == True")
        obs = fetch_cell_metadata(census, organism_key, args.tissue)
        print(f"Matched {len(obs):,} cells")
        print(obs["cell_type"].value_counts().head(20))

        joinids = subsample_joinids(obs, args.n_cells, args.seed)
        print(f"Downloading {len(joinids):,} cells (all genes)...")

        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism_key,
            obs_coords=joinids,
            obs_column_names=OBS_COLUMNS,
        )

    print(f"Downloaded AnnData: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if args.protein_coding_only:
        print("Resolving protein-coding genes via mygene.info (cached)...")
        cache_path = PROCESSED_DIR / f"gene_biotypes_{organism_key}.csv"
        protein_coding_ids = fetch_protein_coding_ids(adata.var["feature_id"].tolist(), cache_path)
        keep = adata.var["feature_id"].isin(protein_coding_ids)
        print(f"Keeping {keep.sum():,} / {len(keep):,} protein-coding genes")
        adata = adata[:, keep].copy()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    print(f"Saved to {output} ({adata.shape[0]:,} cells x {adata.shape[1]:,} genes)")


if __name__ == "__main__":
    main()
