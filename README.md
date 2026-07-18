# Within-Crash Differential Vulnerability — Analysis Code

Analysis code accompanying the manuscript submitted to *Accident Analysis &
Prevention*: a within-crash conditional (fixed-effects) logistic regression
design for estimating differential fatality risk by road-user role
(pedestrian, passenger, driver), using the ARI/BUET Bangladesh crash database
(2006–2015).

## Data availability

The underlying crash database is maintained by the Accident Research
Institute (ARI), Bangladesh University of Engineering and Technology, and is
available from ARI subject to institutional permission. It is **not**
included in this repository. All scripts expect the raw file
`2006 to 2015.xlsx` as input.

## Repository structure

```
clogit.py                       Shared conditional-logit fitter (used by everything below)
01_data_pipeline/                Load, clean, and build canonical analysis files
02_primary_models/               Model 1 (casualties-only) and Model 2 (all persons)
03_heterogeneity/                Table 3: role x crash-context interaction models
04_pooled_comparison/            Table 4: pooled logistic vs within-crash comparison
05_sex_model/                    Table 5: female indicator sub-model
06_robustness/                   Table 6: robustness battery + leave-one-role-out + E-values
07_bootstrap_variance/           Standalone cluster bootstrap (independent SE check)
08_simulation/                   Table 7: four-scenario Monte Carlo evaluation
```

## Why a custom conditional-logit implementation

`statsmodels.discrete.conditional_models.ConditionalLogit` fails with a
Hessian-inversion error on interaction models where regressors are strongly
correlated with their corresponding main effects (~0.8–0.9 correlation
between e.g. `ped_` and `ped_ x heavy`), which is exactly the situation in
Table 3. `clogit.py` implements the exact conditional likelihood directly via
the Gail, Lubin & Rubinstein (1981, *Biometrika*) recursion and was verified
to match `ConditionalLogit`'s point estimates and standard errors on the
two-parameter primary model before being used for everything else in this
project.

**Variance estimation note:** the BFGS approximate inverse Hessian is
adequate for the two-parameter models (Model 1, Model 2, sex model,
robustness checks) but is *not* reliable for the four-parameter interaction
models in Table 3, where correlated regressors can make it numerically
unstable. Table 3 standard errors and p-values are therefore computed via a
cluster (crash-level) nonparametric bootstrap (`clogit.bootstrap_se`,
300 resamples), not the Hessian approximation.

## Running the full pipeline

```bash
pip install -r requirements.txt

cd 01_data_pipeline
python build_dataset.py --input "/path/to/2006 to 2015.xlsx" --outdir ./outputs

cd ../02_primary_models
python fit_primary_models.py --outdir ../01_data_pipeline/outputs

cd ../03_heterogeneity
python fit_heterogeneity.py --outdir ../01_data_pipeline/outputs --n_boot 300

cd ../04_pooled_comparison
python fit_pooled_comparison.py --outdir ../01_data_pipeline/outputs

cd ../05_sex_model
python fit_sex_model.py --outdir ../01_data_pipeline/outputs

cd ../06_robustness
python fit_robustness.py --outdir ../01_data_pipeline/outputs

cd ../07_bootstrap_variance
python run_bootstrap.py --outdir ../01_data_pipeline/outputs --n_boot 300

cd ../08_simulation
python run_simulation.py --outdir ../01_data_pipeline/outputs --reps_per_scenario 80
```

Expect the heterogeneity bootstrap (Cell 03) and the simulation (08) to be
the slowest steps (minutes, not seconds) since both refit the conditional
likelihood hundreds of times. If `numba` is installed, `clogit.py` uses a
JIT-compiled inner recursion automatically, which is substantially faster
than the pure-Python fallback.

## Expected headline numbers (canonical pipeline)

- Persons with valid injury codes: 66,888
- Casualties (F/G/S): 40,853; informative strata: 3,496
- Model 1 (primary): OR pedestrian = 5.90 (4.86–7.17), OR passenger = 1.35 (1.23–1.49)
- Model 2 (secondary, all persons): OR pedestrian ≈ 117, OR passenger ≈ 12.9
- Pooled comparison: OR pedestrian = 3.61, OR passenger = 0.79 (sign reversal vs. within-crash)
- Sex model: OR female = 1.18 (1.03–1.34)
- Simulation S0 (null check): bias ≈ 0 for both scenarios; S3 (A1 violated): substantial bias

If a rerun does not reproduce these to within simulation/bootstrap noise,
stop and check the data pipeline before trusting any downstream result — see
`01_data_pipeline/build_dataset.py`'s embedded assertions.

## Citation

If you use this code, please cite the manuscript (full citation to be added
on acceptance) and this repository's Zenodo DOI (badge/DOI to be inserted
after archiving).

## License

MIT License (or your preferred choice — update before publishing).
