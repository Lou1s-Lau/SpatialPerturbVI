"""Compute counterfactual perturbation signatures from a trained model.

Example:
    python compute_signatures.py --run results/d1
"""

import argparse
import os

import scanpy as sc

from spatialperturbvi import SpatialPerturbVI, counterfactual_signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="output directory of train.py")
    parser.add_argument("--out", default=None, help="CSV file (default: <run>/signatures.csv)")
    parser.add_argument("--min-controls", type=int, default=50)
    args = parser.parse_args()

    adata = sc.read_h5ad(os.path.join(args.run, "adata.h5ad"))
    model = SpatialPerturbVI.load(os.path.join(args.run, "model"), adata=adata)
    signatures = counterfactual_signatures(model, min_controls=args.min_controls)
    out = args.out or os.path.join(args.run, "signatures.csv")
    signatures.to_csv(out)
    print(f"{signatures.shape[0]} signatures over {signatures.shape[1]} genes -> {out}")


if __name__ == "__main__":
    main()
