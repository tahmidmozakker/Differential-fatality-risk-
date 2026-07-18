"""
fit_pooled_comparison.py
--------------------------
The pooled-vs-within-crash comparison (Table 4, Figure 3): fits an
ordinary person-level logistic regression with crash-clustered
standard errors -- the design family used by every prior analysis
of this database -- and reports it beside the conditional (within-
crash fixed effects) estimates for the same casualties.

Requires statsmodels (only place in this repository that does).

Usage:
    python fit_pooled_comparison.py --outdir ../01_data_pipeline/outputs
"""

import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cas = pd.read_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"))

    X = sm.add_constant(cas[["ped_", "pax_"]].astype(float))
    pooled = sm.GLM(cas["fatal"], X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": cas["Acc_ID"]}
    )

    comp = pd.DataFrame({
        "pooled_OR": np.exp(pooled.params[["ped_", "pax_"]]),
        "pooled_CI_low": np.exp(pooled.conf_int().loc[["ped_", "pax_"], 0]),
        "pooled_CI_high": np.exp(pooled.conf_int().loc[["ped_", "pax_"], 1]),
        # from 02_primary_models/fit_primary_models.py, Model 1
        "within_crash_OR": [5.90, 1.35],
    })
    print(comp.round(2))
    comp.to_csv(os.path.join(args.outdir, "table4_pooled_vs_within.csv"))
    print("Saved table4_pooled_vs_within.csv")
    print("\nNote the passenger sign reversal: pooled OR < 1, within-crash OR > 1.")


if __name__ == "__main__":
    main()
