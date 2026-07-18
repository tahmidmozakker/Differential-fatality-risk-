"""
fit_heterogeneity.py
---------------------
Table 3: role x crash-context interaction models (heavy vehicle,
rural, night, national highway).

NOTE ON VARIANCE ESTIMATION (peer-review comment M5): the four
interaction terms in these models are correlated (~0.8-0.9 with
their corresponding main effects), which makes the BFGS approximate
inverse Hessian numerically unreliable for standard errors here --
unlike the two-parameter primary model, where it was verified against
statsmodels. This script therefore uses a cluster (crash-level)
bootstrap for all Table 3 standard errors and p-values instead. See
07_bootstrap_variance/ for the standalone bootstrap script; this file
calls the same function.

Usage:
    python fit_heterogeneity.py --outdir ../01_data_pipeline/outputs --n_boot 300
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clogit import build_strata, fit_clogit, bootstrap_se

MODIFIERS = ["heavy", "rural", "night", "national"]


def fit_one_modifier(cas, modifier, n_boot, seed):
    inf = cas.groupby("Acc_ID")["fatal"].transform("nunique") > 1
    dfm = cas[inf & cas[modifier].notna()].copy()
    dfm["pedM"] = dfm["ped_"] * dfm[modifier]
    dfm["paxM"] = dfm["pax_"] * dfm[modifier]

    strata = build_strata(dfm, "fatal", ["ped_", "pax_", "pedM", "paxM"])
    beta, _ = fit_clogit(strata, 4)
    se, _ = bootstrap_se(strata, 4, n_boot=n_boot, seed=seed)

    z_ped = beta[2] / se[2]
    z_pax = beta[3] / se[3]
    return {
        "modifier": modifier,
        "strata": len(strata),
        "OR_ped_absent": np.exp(beta[0]),
        "OR_ped_present": np.exp(beta[0] + beta[2]),
        "p_interact_ped": 2 * (1 - norm.cdf(abs(z_ped))),
        "OR_pax_absent": np.exp(beta[1]),
        "OR_pax_present": np.exp(beta[1] + beta[3]),
        "p_interact_pax": 2 * (1 - norm.cdf(abs(z_pax))),
        "n_bootstrap": n_boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_boot", type=int, default=300)
    args = ap.parse_args()

    cas = pd.read_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"))

    rows = []
    for i, m in enumerate(MODIFIERS):
        print(f"Fitting {m} ({i+1}/{len(MODIFIERS)})...")
        rows.append(fit_one_modifier(cas, m, args.n_boot, seed=1000 + i))

    table3 = pd.DataFrame(rows)
    print(table3.round(4).to_string(index=False))
    table3.to_csv(os.path.join(args.outdir, "table3_heterogeneity_bootstrap.csv"), index=False)
    print("Saved table3_heterogeneity_bootstrap.csv")


if __name__ == "__main__":
    main()
