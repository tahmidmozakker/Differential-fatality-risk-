"""
build_dataset.py
-----------------
Loads the raw ARI/BUET crash database (Excel), cleans it, and produces
the canonical person-level and casualty-level analysis files used by
every downstream script in this repository.

CRITICAL FIX EMBEDDED HERE (do not remove): Acc_ID must be converted
to numeric BEFORE deduplication on the General sheet. Deduplicating
on the raw string column leaves ~112 duplicate crash records whose
Acc_ID values differ only by whitespace/formatting but collapse to
the same integer once converted -- this silently duplicates rows in
every downstream merge and was caught and fixed during development
(see git history / project notes for the diagnostic that found it).

Usage:
    python build_dataset.py --input "2006 to 2015.xlsx" --outdir ./outputs
"""

import argparse
import os
import numpy as np
import pandas as pd


def load_and_clean(xlsx_path):
    xl = pd.ExcelFile(xlsx_path)

    veh = xl.parse("Veh")
    veh["Acc_ID"] = pd.to_numeric(veh["Acc_ID"], errors="coerce")
    veh = veh.drop_duplicates()

    pas = xl.parse("Pass")
    pas["Acc_ID"] = pd.to_numeric(pas["Acc_ID"], errors="coerce")
    pas = pas.drop_duplicates()

    ped = xl.parse("Ped")
    ped["Acc_ID"] = pd.to_numeric(ped["Acc_ID"], errors="coerce")
    ped = ped.drop_duplicates()

    d = veh[["Acc_ID", "Injury", "Sex", "Age"]].copy(); d["role"] = "driver"
    p = pas[["Acc_ID", "Injury", "Sex", "Age"]].copy(); p["role"] = "passenger"
    q = ped[["Acc_ID", "Injury", "Sex", "Age"]].copy(); q["role"] = "pedestrian"
    persons = pd.concat([d, p, q], ignore_index=True)

    persons["Injury"] = persons["Injury"].astype(str).str.strip().str.upper()
    persons = persons[persons["Injury"].isin({"F", "G", "S", "N"})].copy()

    # General sheet: numeric key BEFORE dedup (see module docstring)
    gen = xl.parse("General")
    gen.columns = gen.iloc[0]
    gen = gen.iloc[1:].reset_index(drop=True)
    gen["Acc_ID"] = pd.to_numeric(gen["Acc_ID"], errors="coerce")
    gen = gen.drop_duplicates(subset=["Acc_ID"], keep="first")
    assert gen["Acc_ID"].duplicated().sum() == 0, "General sheet still has duplicate Acc_ID after fix"

    return persons, gen, veh, pas, ped


def build_context(gen, veh):
    HEAVY = {8, 9, 13, 14, 15, 16, 17}  # minibus, bus, trucks, tanker, tractor
    vv = veh.copy()
    vv["vt"] = pd.to_numeric(vv["Veh Type"], errors="coerce")
    heavy = vv.groupby("Acc_ID")["vt"].agg(lambda s: s.isin(HEAVY).any()).rename("heavy")

    ctx = gen[["Acc_ID", "Location Type", "Light", "Road Class", "Year", "District"]].copy()
    for c in ctx.columns[1:]:
        ctx[c] = pd.to_numeric(ctx[c], errors="coerce")
    ctx["rural"] = np.where(ctx["Location Type"].isin([1, 2]), (ctx["Location Type"] == 2).astype(float), np.nan)
    ctx["night"] = np.where(ctx["Light"].isin([1, 2, 3, 4]), ctx["Light"].isin([3, 4]).astype(float), np.nan)
    ctx["national"] = np.where(ctx["Road Class"].isin([1, 2, 3, 4, 5]), (ctx["Road Class"] == 1).astype(float), np.nan)

    DIV = {}
    for k in range(1, 9): DIV[k] = "Rangpur"
    for k in range(9, 18): DIV[k] = "Rajshahi"
    for k in list(range(18, 27)) + [28, 29]: DIV[k] = "Khulna"
    for k in [27] + list(range(30, 35)) + [69]: DIV[k] = "Barisal"
    for k in range(35, 53): DIV[k] = "Dhaka"
    for k in [53, 54, 55, 56, 70]: DIV[k] = "Sylhet"
    for k in range(57, 69): DIV[k] = "Chittagong"
    ctx["division"] = ctx["District"].map(DIV)

    return ctx, heavy


def build_casualty_file(persons, ctx, heavy):
    cas = persons[persons["Injury"].isin(["F", "G", "S"])].copy()
    cas["fatal"] = (cas["Injury"] == "F").astype(int)
    cas["ped_"] = (cas["role"] == "pedestrian").astype(int)
    cas["pax_"] = (cas["role"] == "passenger").astype(int)

    n0 = len(cas)
    cas = cas.merge(ctx[["Acc_ID", "rural", "night", "national", "Year", "division"]], on="Acc_ID", how="left")
    assert len(cas) == n0, f"merge duplicated rows: {len(cas)} vs {n0}"
    cas = cas.merge(heavy, on="Acc_ID", how="left")
    assert len(cas) == n0, f"merge duplicated rows: {len(cas)} vs {n0}"
    cas["heavy"] = cas["heavy"].astype(float)

    return cas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to '2006 to 2015.xlsx'")
    ap.add_argument("--outdir", default="./outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    persons, gen, veh, pas, ped = load_and_clean(args.input)
    ctx, heavy = build_context(gen, veh)
    cas = build_casualty_file(persons, ctx, heavy)

    inf = cas.groupby("Acc_ID")["fatal"].transform("nunique") > 1
    print("persons:", len(persons))
    print("casualties:", len(cas), "| informative strata:", cas[inf]["Acc_ID"].nunique())
    # Expected (canonical pipeline): persons ~66,888; casualties ~40,853; informative strata 3,496

    persons.to_csv(os.path.join(args.outdir, "persons_clean.csv"), index=False)
    cas.to_csv(os.path.join(args.outdir, "casualties_analysis_file.csv"), index=False)
    print("Saved persons_clean.csv and casualties_analysis_file.csv to", args.outdir)


if __name__ == "__main__":
    main()
