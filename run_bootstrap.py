"""
run_bootstrap.py
-------------------
Standalone cluster (crash-level) bootstrap for the primary model,
provided as an independent check against the BFGS approximate
inverse-Hessian standard errors used elsewhere. Also demonstrates
the method used inside 03_heterogeneity/fit_heterogeneity.py.

Usage:
    python run_bootstrap.py --outdir ../01_data_pipeline/outputs --n_boot 300
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clogit import build_strata, fit_clogit, bootstrap_se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_boot", type=int, default=300)
    args = ap.parse_args()

    cas = pd.read_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"))
    strata = build_strata(cas, "fatal", ["ped_", "pax_"])

    beta, _ = fit_clogit(strata, 2)
    se, boots = bootstrap_se(strata, 2, n_boot=args.n_boot, seed=42, chunk_print=50)

    lo, hi = np.percentile(np.exp(boots), [2.5, 97.5], axis=0)
    print(f"\nBootstrap ({args.n_boot} resamples), percentile 95% CI:")
    for name, b, l, h in zip(["pedestrian", "passenger"], beta, lo, hi):
        print(f"  OR {name}: {np.exp(b):.2f}  bootstrap 95% CI ({l:.2f}-{h:.2f})")

    pd.DataFrame(np.exp(boots), columns=["OR_ped", "OR_pax"]).to_csv(
        os.path.join(args.outdir, "primary_model_bootstrap_distribution.csv"), index=False
    )
    print("Saved primary_model_bootstrap_distribution.csv")


if __name__ == "__main__":
    main()
