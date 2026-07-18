"""
clogit.py
---------
Exact conditional (fixed-effects) logistic regression via the
Gail-Lubin-Rubinstein (1981) recursion, used throughout this project
in place of statsmodels.ConditionalLogit (which fails on interaction
models with correlated regressors -- see 03_heterogeneity/README.md).

Optionally JIT-compiled with numba for speed; falls back to pure
Python/NumPy automatically if numba is not installed.
"""

import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def build_strata(df, outcome_col, x_cols, group_col="Acc_ID"):
    """
    Convert a long-format person-level DataFrame into a list of
    (X, y) arrays, one per crash, keeping only "informative" strata
    (at least one but not all outcomes = 1), since uninformative
    strata contribute nothing to the conditional likelihood.
    """
    strata = []
    for _, g in df.groupby(group_col, sort=False):
        y = g[outcome_col].values
        if 0 < y.sum() < len(y):
            strata.append((
                g[x_cols].values.astype(np.float64),
                y.astype(np.int64),
            ))
    return strata


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _stratum_ll_grad(X, y, beta):
        n = X.shape[0]
        P = X.shape[1]
        k = 0
        for i in range(n):
            k += y[i]
        if k == 0 or k == n:
            return 0.0, np.zeros(P), False

        w = np.empty(n)
        for i in range(n):
            s = 0.0
            for p in range(P):
                s += X[i, p] * beta[p]
            if s > 30.0:
                s = 30.0
            if s < -30.0:
                s = -30.0
            w[i] = np.exp(s)

        f = np.zeros(k + 1)
        f[0] = 1.0
        g = np.zeros((k + 1, P))
        for j in range(n):
            hi = j + 1
            if hi > k:
                hi = k
            for kk in range(hi, 0, -1):
                for p in range(P):
                    g[kk, p] = g[kk, p] + w[j] * g[kk - 1, p] + w[j] * X[j, p] * f[kk - 1]
                f[kk] = f[kk] + w[j] * f[kk - 1]

        num = np.zeros(P)
        for i in range(n):
            if y[i] == 1:
                for p in range(P):
                    num[p] += X[i, p]

        ll = 0.0
        for p in range(P):
            ll += num[p] * beta[p]
        ll -= np.log(f[k])

        grad = np.zeros(P)
        for p in range(P):
            grad[p] = num[p] - g[k, p] / f[k]

        return ll, grad, True
else:
    def _stratum_ll_grad(X, y, beta):
        n, P = X.shape
        k = int(y.sum())
        if k == 0 or k == n:
            return 0.0, np.zeros(P), False
        w = np.exp(np.clip(X @ beta, -30, 30))
        f = np.zeros(k + 1)
        f[0] = 1.0
        g = np.zeros((k + 1, P))
        for j in range(n):
            for kk in range(min(j + 1, k), 0, -1):
                g[kk] = g[kk] + w[j] * g[kk - 1] + w[j] * X[j] * f[kk - 1]
                f[kk] = f[kk] + w[j] * f[kk - 1]
        num = (X[y == 1]).sum(axis=0)
        ll = num @ beta - np.log(f[k])
        grad = num - g[k] / f[k]
        return ll, grad, True


def fit_clogit(strata, n_params, gtol=1e-8, maxiter=1000, warmup=True):
    """
    Fit the exact conditional logit by maximizing the conditional
    log-likelihood with BFGS. Returns (beta, converged_object).
    Standard errors from res.hess_inv are ADEQUATE for the
    two-parameter primary model (verified against statsmodels) but
    should NOT be trusted for interaction models with correlated
    regressors -- use bootstrap_se() instead (07_bootstrap_variance/).
    """
    if NUMBA_AVAILABLE and warmup:
        # trigger JIT compilation once, outside the timed region
        _wx = np.zeros((2, n_params))
        _wx[1, 0] = 1.0
        _wy = np.array([0, 1])
        _stratum_ll_grad(_wx, _wy, np.zeros(n_params))

    def negll_grad(beta):
        total_ll = 0.0
        total_grad = np.zeros(n_params)
        for X, y in strata:
            ll, grad, ok = _stratum_ll_grad(X, y, beta)
            if ok:
                total_ll += ll
                total_grad += grad
        return -total_ll, -total_grad

    res = minimize(
        negll_grad, np.zeros(n_params), jac=True, method="BFGS",
        options={"gtol": gtol, "maxiter": maxiter},
    )
    return res.x, res


def bootstrap_se(strata, n_params, n_boot=300, seed=0, chunk_print=None):
    """
    Cluster (crash-level) nonparametric bootstrap standard errors.
    Use this for any model where the BFGS approximate inverse Hessian
    is not trustworthy (e.g. 4-parameter interaction models with
    correlated regressors -- see peer-review comment M5).
    """
    rng = np.random.default_rng(seed)
    n = len(strata)
    boots = np.zeros((n_boot, n_params))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_strata = [strata[i] for i in idx]
        beta, _ = fit_clogit(boot_strata, n_params, warmup=False)
        boots[b] = beta
        if chunk_print and (b + 1) % chunk_print == 0:
            print(f"  bootstrap {b+1}/{n_boot}")
    return boots.std(axis=0), boots
