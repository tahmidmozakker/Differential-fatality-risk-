"""
fit_sex_model.py
------------------
Table 5: adds a female indicator to the primary within-crash model,
restricted to casualties with a valid sex code and identified by the
strata containing casualties of both sexes.

Usage:
    python fit_sex_model.py --outdir ../01_data_pipeline/outputs
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clogit import build_strata, fit_clogit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cas = pd.read_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"))
    cas["SexN"] = pd.to_numeric(cas["Sex"].astype(str).str.strip(), errors="coerce")
    cs = cas[cas["SexN"].isin([1, 2])].copy()
    cs["female"] = (cs["SexN"] == 2).astype(int)

    inf_s = cs.groupby("Acc_ID")["fatal"].transform("nunique") > 1
    dfs = cs[inf_s]
    n_sexvar = dfs[dfs.groupby("Acc_ID")["female"].transform("nunique") > 1]["Acc_ID"].nunique()
    print("Strata:", dfs["Acc_ID"].nunique(), "| identifying (sex varies):", n_sexvar)

    strata = build_strata(dfs, "fatal", ["female", "ped_", "pax_"])
    beta, res = fit_clogit(strata, 3)
    se = np.sqrt(np.abs(np.diag(res.hess_inv)))

    from scipy.stats import norm
    result = pd.DataFrame({
        "OR": np.exp(beta),
        "CI_low": np.exp(beta - 1.96 * se),
        "CI_high": np.exp(beta + 1.96 * se),
        "p": 2 * (1 - norm.cdf(np.abs(beta / se))),
    }, index=["female", "ped_", "pax_"])
    print(result.round(3))
    result.to_csv(os.path.join(args.outdir, "table5_sex_model.csv"))
    print("Saved table5_sex_model.csv")


if __name__ == "__main__":
    main()
