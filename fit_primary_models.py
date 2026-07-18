"""
fit_primary_models.py
----------------------
Model 1 (primary, casualties-only) and Model 2 (secondary, all
recorded persons) -- the headline within-crash odds ratios reported
in Table 4 / Section 4.2 of the manuscript.

Usage:
    python fit_primary_models.py --outdir ../01_data_pipeline/outputs
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clogit import build_strata, fit_clogit


def summarize(beta, res, strata, label):
    # SEs from BFGS inverse-Hessian approximation -- adequate for this
    # two-parameter model (verified against statsmodels.ConditionalLogit)
    se = np.sqrt(np.abs(np.diag(res.hess_inv)))
    print(f"\n{label}: {len(strata)} strata")
    for name, b, s in zip(["pedestrian", "passenger"], beta, se):
        lo, hi = np.exp(b - 1.96 * s), np.exp(b + 1.96 * s)
        print(f"  OR {name} vs driver: {np.exp(b):.2f} (95% CI {lo:.2f}-{hi:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cas = pd.read_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"))
    persons = pd.read_csv(os.path.join(args.outdir, "persons_clean.csv"))

    # --- Model 1: primary, casualties only ---
    strata1 = build_strata(cas, "fatal", ["ped_", "pax_"])
    beta1, res1 = fit_clogit(strata1, 2)
    summarize(beta1, res1, strata1, "MODEL 1 (primary, casualties only)")

    # --- Model 2: secondary, all recorded persons ---
    per2 = persons.copy()
    per2["fatal"] = (per2["Injury"] == "F").astype(int)
    per2["ped_"] = (per2["role"] == "pedestrian").astype(int)
    per2["pax_"] = (per2["role"] == "passenger").astype(int)
    strata2 = build_strata(per2, "fatal", ["ped_", "pax_"])
    beta2, res2 = fit_clogit(strata2, 2)
    summarize(beta2, res2, strata2, "MODEL 2 (secondary, all recorded persons)")

    pd.DataFrame({
        "model": ["primary_casualties", "primary_casualties",
                  "secondary_all_persons", "secondary_all_persons"],
        "term": ["pedestrian", "passenger", "pedestrian", "passenger"],
        "OR": [np.exp(beta1[0]), np.exp(beta1[1]), np.exp(beta2[0]), np.exp(beta2[1])],
    }).to_csv(os.path.join(args.outdir, "table4_primary_secondary_models.csv"), index=False)
    print("\nSaved table4_primary_secondary_models.csv")


if __name__ == "__main__":
    main()
