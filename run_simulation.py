"""
run_simulation.py
--------------------
Monte Carlo evaluation of the within-crash design under four
reporting-selection scenarios, with the Form-34 casualty-only
recording rule applied to the simulated data BEFORE fitting -- i.e.
this simulates the estimator actually used for the primary analysis
(peer-review comment M4), not a mismatched all-persons version.

Scenarios:
  S0 -- no selection (every crash reported). Sanity check: both
        estimators should recover the true parameters here.
  S1 -- crash-level selection driven by fatality count only
        (Assumption A1 holds).
  S2 -- crash-level selection driven by fatality + severe count
        (A1 still holds -- selection remains a function of crash
        aggregates, not individual identities).
  S3 -- selection additionally depends on WHETHER THE DRIVER DIED
        specifically (a violation of A1 -- selection now depends on
        the configuration of who died, not just how many).

Usage:
    python run_simulation.py --outdir ../01_data_pipeline/outputs \\
        --n_crash 3000 --reps_per_scenario 80
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clogit import build_strata, fit_clogit

TRUE_OR_PED, TRUE_OR_PAX = 6.0, 1.35
B_PED, B_PAX = np.log(TRUE_OR_PED), np.log(TRUE_OR_PAX)


def gen_and_fit(rng, n_crash, scenario):
    rows = []
    for c in range(n_crash):
        alpha = rng.normal(-1.0, 1.3)
        n_drv = 1 + rng.binomial(1, 0.05)
        n_pax = rng.poisson(1.2)
        has_ped = rng.random() < 0.45
        roles = [(0, 0)] * n_drv + [(0, 1)] * n_pax + ([(1, 0)] if has_ped else [])
        for pd_, px_ in roles:
            eta = alpha + B_PED * pd_ + B_PAX * px_
            # Step 1: injured at all vs uninjured
            injured = rng.random() < 1 / (1 + np.exp(-(eta - (-1.2))))
            if not injured:
                rows.append((c, pd_, px_, "N"))
                continue
            # Step 2: fatal vs non-fatal, given injured (the step the
            # real primary model estimates)
            fatal = rng.random() < 1 / (1 + np.exp(-(eta - 1.5)))
            if fatal:
                rows.append((c, pd_, px_, "F"))
                continue
            # Step 3: grievous vs simple (cosmetic, not used in the fit)
            gflag = rng.random() < 1 / (1 + np.exp(-(eta - 0.3)))
            rows.append((c, pd_, px_, "G" if gflag else "S"))

    df = pd.DataFrame(rows, columns=["c", "ped", "pax", "sev"])

    fatal_count = df.groupby("c")["sev"].apply(lambda s: (s == "F").sum())
    severe_count = df.groupby("c")["sev"].apply(lambda s: s.isin(["F", "G"]).sum())
    is_driver = (df["ped"] == 0) & (df["pax"] == 0)
    driver_died = df[is_driver].groupby("c")["sev"].apply(lambda s: (s == "F").any())

    if scenario == "S0":
        p_report = pd.Series(1.0, index=fatal_count.index)
    elif scenario == "S1":
        p_report = pd.Series(np.where(fatal_count > 0, 0.90, 0.05), index=fatal_count.index)
    elif scenario == "S2":
        z = -3 + 1.2 * fatal_count.values + 0.5 * severe_count.values
        p_report = pd.Series(1 / (1 + np.exp(-z)), index=fatal_count.index)
    elif scenario == "S3":
        dd = driver_died.reindex(fatal_count.index).fillna(False)
        p_report = pd.Series(np.where(dd, 0.95, np.where(fatal_count > 0, 0.40, 0.05)), index=fatal_count.index)
    else:
        raise ValueError(scenario)

    keep_ids = set(p_report.index[rng.random(len(p_report)) < p_report.values])
    d = df[df["c"].isin(keep_ids)].copy()

    # Form-34 recording rule: uninjured non-drivers are never recorded
    drop = (d["sev"] == "N") & ((d["ped"] + d["pax"]) > 0)
    d = d[~drop]

    cas_ = d[d["sev"].isin(["F", "G", "S"])].copy()
    cas_["fatal"] = (cas_["sev"] == "F").astype(int)
    strata = build_strata(cas_, "fatal", ["ped", "pax"], group_col="c")
    if len(strata) < 20:
        return None, len(strata)

    beta, _ = fit_clogit(strata, 2, warmup=False)
    return beta, len(strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_crash", type=int, default=3000)
    ap.add_argument("--reps_per_scenario", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    summary_rows = []

    for scen in ["S0", "S1", "S2", "S3"]:
        results = []
        for r in range(args.reps_per_scenario):
            beta, ns = gen_and_fit(rng, args.n_crash, scen)
            if beta is not None:
                results.append(list(beta) + [ns])
        arr = np.array(results)
        or_ped = np.exp(arr[:, 0])
        or_pax = np.exp(arr[:, 1])
        summary_rows.append({
            "scenario": scen,
            "n_reps": len(arr),
            "mean_OR_ped": or_ped.mean(),
            "bias_ped_logscale": arr[:, 0].mean() - np.log(TRUE_OR_PED),
            "mean_OR_pax": or_pax.mean(),
            "bias_pax_logscale": arr[:, 1].mean() - np.log(TRUE_OR_PAX),
            "mean_strata": arr[:, 2].mean(),
        })
        print(f"{scen}: {len(arr)} reps, mean OR_ped={or_ped.mean():.3f}, mean OR_pax={or_pax.mean():.3f}")

    sim_summary = pd.DataFrame(summary_rows)
    print("\n" + sim_summary.round(3).to_string(index=False))
    sim_summary.to_csv(os.path.join(args.outdir, "table7_simulation.csv"), index=False)
    print("Saved table7_simulation.csv")
    print("\nSanity check: S0 (no selection) should show near-zero bias for both terms.")


if __name__ == "__main__":
    main()
