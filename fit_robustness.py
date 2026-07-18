"""
fit_robustness.py
--------------------
Table 6: nine pre-specified robustness checks -- alternative outcome
definition, mega-crash exclusion, temporal split, and division
leave-one-out, plus leave-one-role-out and E-value sensitivity.

Usage:
    python fit_robustness.py --outdir ../01_data_pipeline/outputs
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clogit import build_strata, fit_clogit


def run(df, label, ycol="fatal"):
    strata = build_strata(df, ycol, ["ped_", "pax_"])
    beta, _ = fit_clogit(strata, 2)
    return {"check": label, "strata": len(strata),
            "OR_ped": np.exp(beta[0]), "OR_pax": np.exp(beta[1])}


def e_value(or_estimate, ci_bound):
    """VanderWeele & Ding (2017) E-value, common-outcome (sqrt) approximation."""
    rr = np.sqrt(or_estimate)
    rr_ci = np.sqrt(ci_bound)
    ev = rr + np.sqrt(rr * (rr - 1))
    ev_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
    return ev, ev_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cas = pd.read_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"))
    inf = cas.groupby("Acc_ID")["fatal"].transform("nunique") > 1

    rows = []

    # (a) alternative outcome: fatal-or-grievous vs simple
    cas["sev"] = cas["Injury"].isin(["F", "G"]).astype(int)
    inf_sev = cas.groupby("Acc_ID")["sev"].transform("nunique") > 1
    rows.append(run(cas[inf_sev], "outcome F/G vs S", ycol="sev"))

    # (b) exclude strata with more than 10 casualties
    size = cas.groupby("Acc_ID")["fatal"].transform("size")
    rows.append(run(cas[inf & (size <= 10)], "strata <= 10 persons"))

    # (c) temporal split
    rows.append(run(cas[inf & cas["Year"].between(6, 10)], "years 2006-2010"))
    rows.append(run(cas[inf & cas["Year"].between(11, 15)], "years 2011-2015"))

    # (d) division leave-one-out
    for dv in ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Rangpur", "Barisal", "Sylhet"]:
        rows.append(run(cas[inf & (cas["division"] != dv)], f"drop {dv}"))

    # (e) leave-one-role-out
    nopax = cas[cas["pax_"] == 0]
    noped = cas[cas["ped_"] == 0]
    r1 = run(nopax[nopax.groupby("Acc_ID")["fatal"].transform("nunique") > 1], "driver-pedestrian strata only")
    r2 = run(noped[noped.groupby("Acc_ID")["fatal"].transform("nunique") > 1], "driver-passenger strata only")
    rows.append(r1); rows.append(r2)

    rob = pd.DataFrame(rows)
    print(rob.round(2).to_string(index=False))
    rob.to_csv(os.path.join(args.outdir, "table6_robustness.csv"), index=False)
    print("Saved table6_robustness.csv")

    # E-values for the two primary estimates
    ev_ped, ev_ped_ci = e_value(5.90, 4.86)
    ev_pax, ev_pax_ci = e_value(1.35, 1.23)
    print(f"\nE-value pedestrian: {ev_ped:.2f} (CI bound {ev_ped_ci:.2f})")
    print(f"E-value passenger:  {ev_pax:.2f} (CI bound {ev_pax_ci:.2f})")


if __name__ == "__main__":
    main()
