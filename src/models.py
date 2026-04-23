from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.cluster import KMeans
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


Y_COLS = ["fx_ret", "jpy_2y_change", "jpy_10y_change", "eq_ret"]
TRANSMISSION_COLS = ["fx_ret", "jpy_2y_change", "jpy_10y_change", "eq_ret_local"]


def lagged_design(df: pd.DataFrame, p: int, frag_col: str = "fragility_pca"):
    y = df[Y_COLS].to_numpy()
    shock = df["boj_shock"].to_numpy()
    frag = df[frag_col].to_numpy()
    rows, targets, states, dates = [], [], [], []
    for t in range(p, len(df)):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(y[t - lag])
        row.append(shock[t])
        rows.append(row)
        targets.append(y[t])
        states.append(frag[t - 1])
        dates.append(df["date"].iloc[t])
    names = ["const"] + [f"{col}_lag{lag}" for lag in range(1, p + 1) for col in Y_COLS] + ["boj_shock"]
    return np.asarray(rows), np.asarray(targets), np.asarray(states), pd.Series(dates), names


def fit_ols(X: np.ndarray, Y: np.ndarray) -> dict:
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ coef
    ssr = float(np.sum(resid**2))
    n, k = X.shape
    sigma = resid.T @ resid / max(n - k, 1)
    return {"coef": coef, "resid": resid, "ssr": ssr, "sigma": sigma, "n": n, "k": k}


def select_lag(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    rows = []
    m = len(Y_COLS)
    for p in range(1, max_lag + 1):
        X, Y, _, _, _ = lagged_design(df, p)
        fit = fit_ols(X, Y)
        sign, logdet = np.linalg.slogdet(fit["sigma"])
        logdet = logdet if sign > 0 else np.log(np.linalg.det(fit["sigma"] + np.eye(m) * 1e-8))
        k_total = fit["k"] * m
        n = fit["n"]
        rows.append({
            "lag": p,
            "aic": logdet + 2 * k_total / n,
            "bic": logdet + np.log(n) * k_total / n,
            "ssr": fit["ssr"],
        })
    return pd.DataFrame(rows)


def fit_linear_var(df: pd.DataFrame, p: int) -> dict:
    X, Y, states, dates, names = lagged_design(df, p)
    fit = fit_ols(X, Y)
    fit.update({"p": p, "names": names, "states": states, "dates": dates})
    return fit


def fit_interaction(df: pd.DataFrame, p: int, frag_col: str = "fragility_pca") -> dict:
    X, Y, states, dates, names = lagged_design(df, p, frag_col)
    interaction = X[:, -1] * states
    X2 = np.column_stack([X, interaction])
    fit0 = fit_ols(X, Y)
    fit1 = fit_ols(X2, Y)
    lr_stat = fit0["n"] * np.log(fit0["ssr"] / fit1["ssr"])
    pvalue = float(1 - chi2.cdf(lr_stat, len(Y_COLS)))
    fit1.update({"names": names + ["boj_shock_x_fragility_lag1"], "lr_stat": lr_stat, "lr_pvalue": pvalue})
    return fit1


def _split_regime(coef: np.ndarray, p: int) -> dict:
    m = len(Y_COLS)
    return {
        "const": coef[0],
        "a": [coef[1 + lag * m:1 + (lag + 1) * m].T for lag in range(p)],
        "b": coef[1 + p * m],
    }


def regime_irf(regime: dict, p: int, shock_size: float = 10.0, horizon: int = 60) -> pd.DataFrame:
    m = len(Y_COLS)
    history0 = [np.zeros(m) for _ in range(p)]
    history1 = [np.zeros(m) for _ in range(p)]
    rows = []
    for h in range(horizon + 1):
        def step(history, shock):
            yhat = regime["const"].copy()
            for lag in range(p):
                yhat += regime["a"][lag] @ history[lag]
            yhat += regime["b"] * shock
            return yhat
        base = step(history0, 0.0)
        shocked = step(history1, shock_size if h == 0 else 0.0)
        response = shocked - base
        rows.append(dict(horizon=h, **dict(zip(Y_COLS, response))))
        history0 = [base] + history0[:-1]
        history1 = [shocked] + history1[:-1]
    return pd.DataFrame(rows)


def cumulative_irf(irf: pd.DataFrame) -> pd.DataFrame:
    """Convert flow-variable IRFs into persistent level/log-level responses."""
    out = irf.copy()
    out[Y_COLS] = out[Y_COLS].cumsum()
    return out


def fit_stvar(df: pd.DataFrame, p: int, frag_col: str = "fragility_pca") -> dict:
    X, Y, states, dates, names = lagged_design(df, p, frag_col)
    c_bounds = tuple(np.quantile(states, [0.2, 0.8]))
    gamma_bounds = (np.log(0.1), np.log(10.0))

    def objective(theta):
        gamma = np.exp(theta[0])
        c = theta[1]
        g = 1.0 / (1.0 + np.exp(-gamma * (states - c)))
        Xs = np.column_stack([(1.0 - g)[:, None] * X, g[:, None] * X])
        return fit_ols(Xs, Y)["ssr"]

    starts = []
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        for c in np.quantile(states, [0.25, 0.5, 0.75]):
            starts.append([np.log(gamma), c])
    best = None
    for start in starts:
        start = [np.clip(start[0], *gamma_bounds), np.clip(start[1], *c_bounds)]
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[gamma_bounds, c_bounds],
            options={"maxiter": 600, "ftol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res
    gamma = float(np.exp(best.x[0]))
    c = float(best.x[1])
    g = 1.0 / (1.0 + np.exp(-gamma * (states - c)))
    Xs = np.column_stack([(1.0 - g)[:, None] * X, g[:, None] * X])
    fit = fit_ols(Xs, Y)
    k = X.shape[1]
    low = _split_regime(fit["coef"][:k], p)
    high = _split_regime(fit["coef"][k:], p)
    fit.update({
        "p": p, "gamma": gamma, "threshold_c": c, "transition_mean": float(g.mean()),
        "low": low, "high": high, "names": names, "states": states, "dates": dates,
        "success": bool(best.success), "message": best.message,
    })
    return fit


def fit_tvar(df: pd.DataFrame, p: int, frag_col: str = "fragility_pca") -> dict:
    X, Y, states, dates, names = lagged_design(df, p, frag_col)
    best = None
    for tau in np.quantile(states, np.linspace(0.2, 0.8, 25)):
        low_mask = states <= tau
        high_mask = ~low_mask
        if low_mask.sum() < X.shape[1] + 20 or high_mask.sum() < X.shape[1] + 20:
            continue
        low_fit = fit_ols(X[low_mask], Y[low_mask])
        high_fit = fit_ols(X[high_mask], Y[high_mask])
        ssr = low_fit["ssr"] + high_fit["ssr"]
        if best is None or ssr < best["ssr"]:
            best = {"threshold": float(tau), "ssr": ssr, "low_fit": low_fit, "high_fit": high_fit,
                    "low_n": int(low_mask.sum()), "high_n": int(high_mask.sum())}
    best["low"] = _split_regime(best["low_fit"]["coef"], p)
    best["high"] = _split_regime(best["high_fit"]["coef"], p)
    best["p"] = p
    return best


def fit_ms_proxy(df: pd.DataFrame) -> pd.DataFrame:
    stress = df[["fx_ret", "jpy_10y_change", "eq_ret"]].copy()
    stress["eq_ret"] = -stress["eq_ret"]
    z = (stress - stress.mean()) / stress.std(ddof=0)
    score = z.mean(axis=1).dropna()
    try:
        model = MarkovRegression(score, k_regimes=2, trend="c", switching_variance=True)
        res = model.fit(disp=False, maxiter=200)
        probs = res.smoothed_marginal_probabilities
        high_regime = int(res.params.filter(like="const").idxmax()[-1]) if hasattr(res.params, "filter") else 1
        out = pd.DataFrame({"date": df.loc[score.index, "date"].values, "ms_high_prob": probs[high_regime].values})
    except Exception:
        km = KMeans(n_clusters=2, n_init=20, random_state=7).fit(score.to_frame())
        high = int(np.argmax([score[km.labels_ == i].mean() for i in range(2)]))
        out = pd.DataFrame({"date": df.loc[score.index, "date"].values, "ms_high_prob": (km.labels_ == high).astype(float)})
    return out


def fit_event_lp_irf(
    df: pd.DataFrame,
    p: int,
    frag_col: str = "fragility_pca",
    horizon: int = 20,
    shock_size: float = 10.0,
) -> dict:
    work = df.copy()
    if "eq_ret_local" not in work:
        work["eq_ret_local"] = work["eq_ret"] - work["fx_ret"]
    if "boj_tightening_shock" not in work:
        work["boj_tightening_shock"] = work["boj_shock"].clip(lower=0.0)
    if "boj_easing_shock" not in work:
        work["boj_easing_shock"] = (-work["boj_shock"]).clip(lower=0.0)

    state = work[frag_col].shift(1)
    event_locs = [
        t for t in range(p, len(work) - horizon)
        if work["boj_event"].iloc[t] == 1 and pd.notna(state.iloc[t])
    ]
    positive_states = [
        state.iloc[t] for t in event_locs
        if work["boj_tightening_shock"].iloc[t] > 0
    ]
    low_state, high_state = np.quantile(positive_states, [0.25, 0.75])

    low_rows, high_rows, diff_rows, coef_rows = [], [], [], []
    control_cols = ["fx_ret", "jpy_2y_change", "jpy_10y_change", "eq_ret_local"]
    for h in range(horizon + 1):
        low_row = {"horizon": h}
        high_row = {"horizon": h}
        diff_row = {"horizon": h}
        for target in TRANSMISSION_COLS:
            y, rows = [], []
            for t in event_locs:
                tightening = work["boj_tightening_shock"].iloc[t]
                easing = work["boj_easing_shock"].iloc[t]
                f = state.iloc[t]
                row = [1.0, tightening, tightening * f, easing, f]
                for lag in range(1, p + 1):
                    row.extend(work[control_cols].iloc[t - lag])
                rows.append(row)
                y.append(work[target].iloc[t:t + h + 1].sum())
            X = np.asarray(rows, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            coef = np.linalg.lstsq(X, y_arr, rcond=None)[0]
            low = shock_size * (coef[1] + coef[2] * low_state)
            high = shock_size * (coef[1] + coef[2] * high_state)
            low_row[target] = low
            high_row[target] = high
            diff_row[target] = high - low
            coef_rows.append({
                "horizon": h,
                "response": target,
                "tightening_beta": coef[1],
                "tightening_x_fragility_beta": coef[2],
                "easing_beta": coef[3],
                "low_state_response": low,
                "high_state_response": high,
                "high_minus_low": high - low,
                "n_events": len(event_locs),
            })
        low_rows.append(low_row)
        high_rows.append(high_row)
        diff_rows.append(diff_row)

    positive_events = work.loc[[t for t in event_locs if work["boj_tightening_shock"].iloc[t] > 0]]
    return {
        "low": pd.DataFrame(low_rows),
        "high": pd.DataFrame(high_rows),
        "diff": pd.DataFrame(diff_rows),
        "coef": pd.DataFrame(coef_rows),
        "low_state": float(low_state),
        "high_state": float(high_state),
        "n_events": len(event_locs),
        "n_tightening_events": int((positive_events["boj_tightening_shock"] > 0).sum()),
        "mean_tightening_shock": float(positive_events["boj_tightening_shock"].mean()),
    }
