"""Index universe, screening statistics, weighting methods and stochastic simulation for
the notebooks. Builds on pfutils (data loaders, cache, PAC simulator).

All return series are monthly, in EUR, total return (dividends/coupons reinvested).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import pfutils as pf

# ---------------------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------------------

# MSCI index codes (EUR, net total return) verified against MSCI's public index pages.
MSCI_EQUITY = {
    "World": 990100, "ACWI": 892400, "ACWI IMI": 664204, "World IMI": 664185,
    "USA": 984000, "Europe": 990500, "EMU": 106400, "Japan": 939200, "Pacific ex Japan": 991400,
    "World ex USA": 991000, "EM": 891800, "EM IMI": 664220,
    "World Small Cap": 106230, "USA Small Cap": 106229, "Europe Small Cap": 106233,
    "World Momentum": 703755, "World Quality": 702787, "World Quality SN": 705169,
    "World Min Vol": 129896, "World Value": 105868, "World Enhanced Value": 705130, "World Growth": 105867,
    "World Equal Weight": 129857, "World High Dividend": 136064, "World Multi-Factor": 706536,
    "ACWI Min Vol": 700404,
}
EQUITY_GROUP = {
    **{k: "Broad" for k in ["World", "ACWI", "ACWI IMI", "World IMI"]},
    **{k: "Region" for k in ["USA", "Europe", "EMU", "Japan", "Pacific ex Japan", "World ex USA", "EM", "EM IMI"]},
    **{k: "Size" for k in ["World Small Cap", "USA Small Cap", "Europe Small Cap"]},
    **{k: "Factor" for k in ["World Momentum", "World Quality", "World Quality SN", "World Min Vol", "World Value",
                              "World Enhanced Value", "World Growth", "World Equal Weight", "World High Dividend",
                              "World Multi-Factor", "ACWI Min Vol"]},
    **{k: "US headline" for k in ["S&P 500", "Nasdaq-100"]},
}

# Government-bond indexes are rebuilt from official constant-maturity yield curves as
# ladders of par bonds (see pfutils.par_bond_return). Weights for "all maturities"
# approximate the maturity profile of the Bloomberg/ICE euro government indexes. Duration ≈ 6.8:
# the earlier ladder (≈ 6.0) fell 17.1% in 2022 against 18.3% for the real funds (VGEA, PRAR,
# XGLE) and tracked them at 1.1%/yr; this one falls 19.0% and tracks at 0.8%.
EA_ALL_WEIGHTS = {1: .108, 2: .108, 3: .099, 5: .153, 7: .117, 10: .135, 15: .105, 20: .096, 30: .079}
# Duration ≈ 6, as published for the Bloomberg US Treasury index. An earlier ladder at ≈ 5.3
# missed the real hedged fund (VDTE) by 2.1 points in 2022; this one by 0.4.
US_ALL_WEIGHTS = {1: .05, 2: .25, 3: .12, 5: .17, 7: .13, 10: .10, 20: .06, 30: .12}


# Two headline US indexes MSCI does not publish, so they come from Yahoo in USD and are converted
# to EUR. Both are total return (dividends reinvested).
#   S&P 500  -> ^SP500TR, the official total-return index, from 1988.
#   Nasdaq-100 -> QQQ, the oldest continuously traded Nasdaq-100 fund, from 1999-03. Yahoo's
#   ^XNDX total-return index returns a single row and is unusable; ^NDX is price-only, which
#   would understate the return by roughly the dividend yield each year.
# QQQ returns are net of its 0.20% fee, so this is very slightly conservative versus the index.
YAHOO_EQUITY = {"S&P 500": "^SP500TR", "Nasdaq-100": "QQQ"}


def other_indexes(end: pd.Timestamp) -> pd.DataFrame:
    """EUR monthly total returns for the non-MSCI headline US indexes.

    The S&P 500 is put on the same footing as MSCI's *net* indexes: price return plus 70% of
    the dividend (30% US withholding, the rate MSCI assumes for a non-treaty investor). Yahoo's
    ^SP500TR reinvests dividends gross, which overstates the index by ~0.5%/yr against MSCI and
    makes every Irish S&P 500 fund look 0.4%/yr dearer than it is. The Nasdaq-100 is QQQ's
    adjusted close (gross dividends, net of QQQ's 0.20% fee); its ~0.7% yield makes the
    difference small."""
    fx = pf.month_end(pf.ecb_series("EXR/D.USD.EUR.SP00.A"), end)
    tr = pf.month_end(pf.yahoo_close("^SP500TR"), end).pct_change(fill_method=None)
    px = pf.month_end(pf.yahoo_close("^GSPC"), end).pct_change(fill_method=None)
    out = {"S&P 500": pf.to_eur(px + 0.70 * (tr - px), fx)}
    s = pf.drop_listing_period(pf.yahoo_close("QQQ"))
    s = s.drop(pf.find_spikes(s))
    out["Nasdaq-100"] = pf.to_eur(pf.month_end(s, end).pct_change(fill_method=None), fx)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------
# Cost: screening indexes you cannot buy, or buying them at a price you did not count, is
# the main way an index screen misleads. Over ten years a 0.18%/yr fee gap compounds to
# ~1.8% of final wealth, which is larger than the gross differences between the broad
# equity indexes themselves. So costs enter the screen, not the write-up.
# ---------------------------------------------------------------------------------------


def net_of_cost(r: pd.Series, drag_pa: float) -> pd.Series:
    """Apply a constant annual cost drag to a monthly return series, geometrically."""
    return (1 + r) * (1 - drag_pa) ** (1 / 12) - 1


def cost_drag(etf_r: pd.Series | None, index_r: pd.Series, ter: float,
              min_months: int = 36, trust_measured: bool = True) -> tuple[float, str]:
    """The realised annual cost of holding the ETF instead of owning the index free.

    The measured tracking difference is the honest number, not the TER: it nets the fee
    against securities-lending revenue and against withholding tax. An Irish-domiciled
    UCITS holding US equities reclaims to the 15% treaty rate, while MSCI's *net* index
    assumes the higher statutory rate, so a good World ETF can track at or even slightly
    above its net index despite charging a fee.

    Falls back to the TER when there is no fund, too little history to measure, or when
    the benchmark is one of our reconstructed bond indexes (``trust_measured=False``),
    where a ±0.3%/yr modelling error would swamp a ~0.1% fee.
    """
    if etf_r is None or not trust_measured:
        return ter, "TER"
    est = drag_slope(etf_r, index_r)
    if est is None or est["Months"] < min_months:
        return ter, "TER"
    return est["Drag"], f"measured, {est['Months']}m"


def drag_slope(etf_r: pd.Series, index_r: pd.Series) -> dict | None:
    """Annual cost drag as the trend slope of the cumulative log return gap.

    The endpoint tracking difference uses two prices and inherits all their timing noise:
    Xetra closes at 17:30 CET while MSCI World closes after New York, so a monthly gap
    mostly reverses the following month (lag-1 autocorrelation ≈ −0.4 across funds). The
    slope uses every month.

    Its standard error is not the textbook OLS one: the regressand is a *cumulative* sum, so
    the residuals behave like a random walk and an OLS standard error understates the noise
    about 4×. The slope is a weighted sum of the monthly gaps, slope = Σ c_k Δ_k, so its
    variance is c'Γc with Γ the Newey-West (Bartlett, 12 lags) autocovariance of the gaps.
    """
    x = pd.concat([etf_r, index_r], axis=1).dropna()
    if len(x) < 3:
        return None
    d = (np.log1p(x.iloc[:, 0]) - np.log1p(x.iloc[:, 1])).to_numpy()
    gap = d.cumsum()
    t = np.arange(len(gap)) / 12
    slope, icpt = np.polyfit(t, gap, 1)
    resid = pd.Series(gap - (slope * t + icpt))
    tc = t - t.mean()
    c = tc[::-1].cumsum()[::-1] / (tc ** 2).sum()          # weight of each monthly gap in the slope
    u, n, L = d - d.mean(), len(d), 12
    var = (c ** 2).sum() * (u ** 2).mean()
    for k in range(1, min(L, n - 1) + 1):
        var += 2 * (1 - k / (L + 1)) * (c[k:] * c[:-k]).sum() * (u[k:] * u[:-k]).sum() / n
    se = float(np.sqrt(max(var, 0.0)))
    return {"Drag": float(-slope), "SE": se, "Months": len(x), "From": x.index[0].strftime("%Y-%m"),
            "Timing noise /yr": float((x.iloc[:, 0] - x.iloc[:, 1]).std() * np.sqrt(12)),
            "Real deviation": float(resid.std())}


def tracking_safe(r: pd.Series, bench: pd.Series) -> dict | None:
    """pfutils.tracking, but returns None instead of raising on an empty overlap."""
    if r is None or len(pd.concat([r, bench], axis=1).dropna()) < 2:
        return None
    return pf.tracking(r, bench)


def etf_returns(ticker: str, end: pd.Timestamp, start: str | None = None, spike: float = 0.12) -> pd.Series:
    """Monthly returns of an ETF from Yahoo closes, with the three Yahoo defects removed:
    thin quotes in the listing months, isolated bad prints, and unadjusted share splits
    (SPYI.DE went 25:1 on 2015-01-02, 83.06 -> 3.32), which show up as impossible monthly
    moves — only the history after the last one is kept. `start` drops placeholder months
    that survive drop_listing_period (IQQ0.DE prints +30% and +22% vs its index in 2013-03/04)."""
    s = pf.drop_listing_period(pf.yahoo_close(ticker))
    s = s.drop(pf.find_spikes(s, spike))
    r = pf.month_end(s, end).pct_change(fill_method=None)
    if start:
        r = r.loc[start:]
    bad = r[r.abs() >= 0.3]   # a 3:2 split is −33%, 2:1 exactly −50%; no fund here moves 30% in a month
    return (r.loc[bad.index[-1]:].iloc[1:] if len(bad) else r).dropna()


def beta(r: pd.Series, market: pd.Series) -> float:
    x = pd.concat([r, market], axis=1).dropna()
    return float(x.iloc[:, 0].cov(x.iloc[:, 1]) / x.iloc[:, 1].var())


def pairwise_sharpe(rets: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Ledoit-Wolf HAC test on every pair, with Holm's step-down correction for the number
    of pairs. Symmetric: no index is the reference. A pair is 'different' only if it
    survives the correction — testing 300 pairs at 5% produces ~15 false positives."""
    cols, rows = list(rets.columns), []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            d, p = sharpe_hac_test(rets[cols[i]], rets[cols[j]], rf)
            rows.append((cols[i], cols[j], d, p))
    out = pd.DataFrame(rows, columns=["A", "B", "Sharpe A − B", "p"]).sort_values("p").reset_index(drop=True)
    m = len(out)
    out["p (Holm)"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * out["p"].to_numpy()))
    return out


def best_weight_boot(eq_r: pd.Series, bd_r: pd.Series, rf: pd.Series, grid: np.ndarray,
                     n: int = 500, block: float = 12, seed: int = 0) -> pd.DataFrame:
    """How well does history pin down the 'best' equity weight? Resample the joint history
    (stationary bootstrap, same length), and on each resample find the weight that maximises
    Sharpe, Sortino and Calmar. A wide spread means the metric cannot choose the weight."""
    X = pd.concat([eq_r, bd_r, rf], axis=1).dropna()
    P = stationary_bootstrap_paths(X, len(X), n, block, seed)          # (n, T, 3)
    a, b, f = P[..., 0], P[..., 1], P[..., 2]
    port = grid[None, :, None] * a[:, None, :] + (1 - grid[None, :, None]) * b[:, None, :]   # (n, G, T)
    ex = port - f[:, None, :]
    sharpe = ex.mean(-1) / port.std(-1)
    sortino = ex.mean(-1) / np.sqrt((np.minimum(ex, 0) ** 2).mean(-1))
    w = np.cumprod(1 + port, axis=-1)
    mdd = (w / np.maximum.accumulate(w, axis=-1) - 1).min(-1)
    cagr = w[..., -1] ** (12 / port.shape[-1]) - 1
    calmar = cagr / np.abs(mdd)
    return pd.DataFrame({k: grid[v.argmax(axis=1)] for k, v in
                         {"Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar}.items()})


def weight_at_risk(weights: list[float], p5_dd: list[float], target: float) -> float:
    """Equity weight at which a portfolio's bad-case (5th percentile) drawdown equals
    `target` (a negative number), by linear interpolation along a weight sweep.
    NaN if even 100% equity stays above the target."""
    dd = -np.asarray(p5_dd)
    if -target > dd.max():
        return np.nan
    order = np.argsort(dd)
    return float(np.interp(-target, dd[order], np.asarray(weights)[order]))


# MSCI World GICS sector indexes, EUR net total return, all with levels from 2000-12-29.
# Deliberately kept OUT of MSCI_EQUITY: adding ten more candidates would inflate the
# multiple-testing correction applied to the core-index screen, and sectors answer a different
# question (whether to tilt) than the core screen (which index to hold).
# Codes run in GICS sector-code order (10 Energy … 55 Utilities). Identified by correlating each
# EUR series, converted to USD, against the matching US sector SPDR: every assignment is a clear
# argmax (0.89–0.98, second-best 0.15–0.30 lower). Beware 106806/106807, which return series
# identical to Energy/Materials — they are the single-industry-group duplicates, not Utilities.
# Real Estate (132085) is absent on purpose: EUR history starts only 2016-09, because before the
# 2016 GICS reshuffle real estate sat inside Financials.
MSCI_SECTOR = {
    "Energy": 106796, "Materials": 106797, "Industrials": 106798, "Cons Discretionary": 106799,
    "Cons Staples": 106800, "Health Care": 106801, "Financials": 106802, "Info Technology": 106803,
    "Comm Services": 106804, "Utilities": 106805,
}

# Broad GICS grouping used to test the defensive/cyclical split rather than assume it.
SECTOR_KIND = {
    **{k: "Defensive" for k in ["Cons Staples", "Health Care", "Utilities"]},
    **{k: "Cyclical" for k in ["Energy", "Materials", "Industrials", "Cons Discretionary", "Financials"]},
    **{k: "Growth" for k in ["Info Technology", "Comm Services"]},
}


def equity_indexes() -> pd.DataFrame:
    """Monthly EUR net total returns for every MSCI index in the universe."""
    out = {}
    for name, code in MSCI_EQUITY.items():
        lv = pf.msci_index(code=code).resample("ME").last()
        out[name] = lv.pct_change()
    return pd.DataFrame(out)


def sector_indexes() -> pd.DataFrame:
    """Monthly EUR net total returns for the ten MSCI World GICS sector indexes.

    Caveat on Communication Services: the code is unchanged but the index was **redefined** in
    2018-09 (renamed from Telecommunication Services, with Alphabet and Meta moving in from IT
    and Netflix and Disney from Consumer Discretionary). Information Technology and Consumer
    Discretionary carry the mirror-image break on the same date. Before 2018 the series is pure
    telecom (−21% in 2001, −40% in 2002); after it, mega-cap internet. It is not one continuous
    exposure, and no statistic computed over the full window should be read as if it were.
    """
    return pd.DataFrame({name: pf.msci_index(code=code).resample("ME").last().pct_change()
                         for name, code in MSCI_SECTOR.items()})


def ea_curve(end: pd.Timestamp) -> dict[int, pd.Series]:
    """Euro-area government par yields (all issuers, ECB), month end, decimal. From 2004-09."""
    return {m: pf.month_end(pf.ecb_series(f"YC/B.U2.EUR.4F.G_N_C.SV_C_YM.PY_{m}Y"), end) / 100
            for m in EA_ALL_WEIGHTS}


def bund_curve() -> dict[int, pd.Series]:
    """German Bund par yields (Bundesbank), month end, decimal. 1-10y from 1972-09."""
    return {m: pf.bundesbank_monthly_yield(m) for m in range(1, 11)}


def ust_curve(end: pd.Timestamp) -> dict[int, pd.Series]:
    """US Treasury constant-maturity yields (FRED daily, month end), decimal."""
    ids = {1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5", 7: "DGS7", 10: "DGS10", 20: "DGS20", 30: "DGS30"}
    return {m: pf.month_end(pf.fred_series(i), end) / 100 for m, i in ids.items()}


def hedge_rates() -> tuple[pd.Series, pd.Series]:
    """(USD, EUR) short rates for the historical hedge carry, month end, decimal: 3-month
    interbank on both sides (FRED/OECD USD, ECB Euribor). Matching tenor *and* credit matters:
    pairing Euribor (a bank rate) with T-bills (a government rate) adds the USD bank-credit
    spread to the carry — 0.3%/yr on average and 1.4%/yr through 2007-09 — and makes hedged
    Treasuries look best exactly in the crisis they are judged on."""
    usd = pf.fred_series("IR3TIB01USM156N") / 100
    eur = pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA") / 100
    usd.index, eur.index = usd.index + pd.offsets.MonthEnd(0), eur.index + pd.offsets.MonthEnd(0)
    return usd, eur


def hedge_to_eur(r_usd: pd.Series, usd_rate: pd.Series, eur_rate: pd.Series) -> pd.Series:
    """Return of a USD asset fully hedged into EUR with rolling 1-month forwards: the
    local return plus the short-rate differential (covered interest parity)."""
    diff = (eur_rate.reindex(r_usd.index).ffill() - usd_rate.reindex(r_usd.index).ffill()) / 12
    return r_usd + diff.shift(1)


def bond_indexes(end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (bonds_since_2004, bonds_long).

    bonds_since_2004: euro-area govt (all issuers) by maturity bucket, plus US Treasuries
    hedged to EUR and unhedged; built from ECB / FRED curves.
    bonds_long: German Bund ladders from 1972 (the only euro govt curve that long).
    """
    ea = ea_curve(end)
    lad = lambda curve, w: pf.ladder_return({m: curve[m] for m in w}, w, full_curve=curve)
    eq = lambda ms: {m: 1 / len(ms) for m in ms}
    b = {
        "EUR Govt 1-3": lad(ea, eq([1, 2, 3])),
        "EUR Govt 3-7": lad(ea, eq([3, 5, 7])),
        "EUR Govt 7-10": lad(ea, eq([7, 10])),
        "EUR Govt 10+": lad(ea, {15: 1 / 3, 20: 1 / 3, 30: 1 / 3}),
        "EUR Govt All": lad(ea, EA_ALL_WEIGHTS),
    }
    ust = ust_curve(end)
    ust_all = pf.ladder_return({m: ust[m] for m in US_ALL_WEIGHTS}, US_ALL_WEIGHTS, freq=2)
    usd_per_eur = pf.month_end(pf.ecb_series("EXR/D.USD.EUR.SP00.A"), end)
    usd_r, eur_r = hedge_rates()
    b["US Treasury (EUR hedged)"] = hedge_to_eur(ust_all, usd_r, eur_r)
    b["US Treasury (unhedged)"] = pf.to_eur(ust_all, usd_per_eur)
    bonds = pd.DataFrame(b)

    bd = bund_curve()
    long = pd.DataFrame({
        "Bund 1-3": lad(bd, eq([1, 2, 3])),
        "Bund 3-7": lad(bd, eq([3, 4, 5, 6, 7])),
        "Bund 7-10": lad(bd, eq([7, 8, 9, 10])),
        "Bund 1-10": lad(bd, eq(list(range(1, 11)))),
    })
    return bonds, long


def ust_hedged_buckets(end: pd.Timestamp) -> pd.DataFrame:
    """US Treasury ladders by maturity bucket, hedged to EUR with 1-month forwards.
    Same construction as the all-maturity series in bond_indexes, so they compare directly."""
    ust = ust_curve(end)
    usd_r, eur_r = hedge_rates()
    lad = lambda ms: pf.ladder_return({m: ust[m] for m in ms}, {m: 1 / len(ms) for m in ms}, freq=2, full_curve=ust)
    return pd.DataFrame({
        "UST 1-3 (EUR hedged)": hedge_to_eur(lad([1, 2, 3]), usd_r, eur_r),
        "UST 3-7 (EUR hedged)": hedge_to_eur(lad([3, 5, 7]), usd_r, eur_r),
        "UST 7-10 (EUR hedged)": hedge_to_eur(lad([7, 10]), usd_r, eur_r),
    })


def current_yields(end: pd.Timestamp) -> pd.DataFrame:
    """Yield to maturity today, per bucket: euro-area govt, US Treasury, and US Treasury
    after the 1-month hedge carry (EUR short rate minus USD short rate). Over a 10-year
    hold, the starting yield is the best single predictor of a bond fund's return."""
    ea, ust = ea_curve(end), ust_curve(end)
    # today's carry from the two overnight risk-free rates, which forward points track:
    # €STR − SOFR (matched tenor and credit; see hedge_rates for why that matters)
    last_ = lambda k: float(pf.fred_series(k).loc[:end].dropna().iloc[-1]) / 100
    carry = last_("ECBESTRVOLWGTTRMDMNRT") - last_("SOFR")
    last = lambda c, ms: float(np.mean([c[m].loc[:end].dropna().iloc[-1] for m in ms]))
    wlast = lambda c, w: float(sum(c[m].loc[:end].dropna().iloc[-1] * v for m, v in w.items()))
    rows = {}
    for k, ms in {"1-3": [1, 2, 3], "3-7": [3, 5, 7], "7-10": [7, 10]}.items():
        rows[k] = {"Euro govt": last(ea, ms), "US Treasury": last(ust, ms),
                   "US Treasury, EUR hedged": last(ust, ms) + carry}
    rows["All maturities"] = {"Euro govt": wlast(ea, EA_ALL_WEIGHTS), "US Treasury": wlast(ust, US_ALL_WEIGHTS),
                              "US Treasury, EUR hedged": wlast(ust, US_ALL_WEIGHTS) + carry}
    out = pd.DataFrame(rows).T
    out.attrs["hedge carry"] = carry
    return out


def long_equity_eur(end: pd.Timestamp) -> pd.DataFrame:
    """1975+ equity in EUR (DEM before 1999): US market and developed ex-US market from
    Kenneth French's data (MSCI-based until 2006), plus a 60/40 US/ex-US 'World' blend
    (validated against MSCI World EUR in the notebook)."""
    dem = pf.fred_series("EXGEUS")
    dem.index = dem.index + pd.offsets.MonthEnd(0)
    fx = pd.concat([(1.95583 / dem).loc[:"1998-12"],
                    pf.month_end(pf.ecb_series("EXR/D.USD.EUR.SP00.A"), end).loc["1999-01":]])
    ff = pf.french_factors("F-F_Research_Data_5_Factors_2x3")
    us = pf.to_eur(ff["Mkt-RF"] + ff["RF"], fx)
    intl = pf.to_eur(pf.french_international(), fx)
    out = pd.DataFrame({"US (FF)": us, "Dev ex-US (FF)": intl}).dropna()
    out["World blend 60/40"] = 0.6 * out["US (FF)"] + 0.4 * out["Dev ex-US (FF)"]
    return out


# ---------------------------------------------------------------------------------------
# Screening statistics
# ---------------------------------------------------------------------------------------


def sharpe_hac_test(a: pd.Series, b: pd.Series, rf: pd.Series, lags: int = 6) -> tuple[float, float]:
    """Difference in annualised Sharpe ratio (a - b) and its two-sided p-value, using the
    delta method with a HAC (Newey-West) covariance of the moment vector, as in Ledoit &
    Wolf (2008). Robust to fat tails and autocorrelation, unlike Jobson-Korkie."""
    df = pd.concat([a - rf, b - rf], axis=1).dropna().to_numpy()
    T = len(df)
    mu = df.mean(axis=0)
    m2 = (df**2).mean(axis=0)
    # moments: [mu_a, mu_b, E[a^2], E[b^2]]
    V = np.column_stack([df[:, 0] - mu[0], df[:, 1] - mu[1], df[:, 0] ** 2 - m2[0], df[:, 1] ** 2 - m2[1]])
    S = V.T @ V / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = V[L:].T @ V[:-L] / T
        S += w * (G + G.T)
    sa, sb = np.sqrt(m2 - mu**2)
    sr = mu / np.array([sa, sb])
    # gradient of SR_a - SR_b wrt [mu_a, mu_b, m2_a, m2_b]
    g = np.array([
        m2[0] / sa**3, -m2[1] / sb**3,
        -mu[0] / (2 * sa**3), mu[1] / (2 * sb**3),
    ])
    se = np.sqrt(g @ S @ g / T)
    diff = sr[0] - sr[1]
    p = 2 * (1 - stats.norm.cdf(abs(diff) / se))
    return diff * np.sqrt(12), p


def deflated_sharpe(r: pd.Series, rf: pd.Series | float, n_trials: int, sr_var_trials: float) -> float:
    """Bailey & Lopez de Prado (2014): probability that the true Sharpe exceeds the best
    Sharpe expected by luck among `n_trials` independent tries (monthly units). Applied to
    active returns (index minus World, rf=0) it asks: does the outperformance survive the
    fact that we screened many indexes?"""
    rf_ = rf.reindex(r.index).fillna(0) if isinstance(rf, pd.Series) else rf
    x = (r - rf_).dropna()
    T = len(x)
    sr = x.mean() / x.std()
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    emc = 0.5772156649
    sr0 = np.sqrt(sr_var_trials) * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
                                    + emc * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf(z))


def downside_stats(r: pd.Series, rf: pd.Series | float = 0.0) -> dict:
    """Sortino and Calmar on top of pfutils.perf_stats."""
    s = pf.perf_stats(r, rf)
    x = r.dropna()
    ex = x - (rf.reindex(x.index).fillna(0) if isinstance(rf, pd.Series) else rf)
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(12)
    s["Sortino"] = ex.mean() * 12 / dd
    s["Calmar"] = s["CAGR"] / abs(s["Max drawdown"])
    return s


CRISES = {
    "Oil shock 1973-74": ("1973-01", "1974-09"),
    "Volcker 1980-82": ("1980-01", "1982-07"),
    "Crash 1987": ("1987-09", "1987-11"),
    "Gulf/Japan 1990": ("1990-01", "1990-09"),
    "Bond rout 1994": ("1994-01", "1994-12"),
    "Dot-com 2000-03": ("2000-04", "2003-03"),
    "GFC 2007-09": ("2007-11", "2009-02"),
    "Euro crisis 2011": ("2011-05", "2011-09"),
    "COVID 2020": ("2020-02", "2020-03"),
    "Inflation 2022": ("2022-01", "2022-09"),
}


def crisis_table(rets: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    """Cumulative return of each series over each crisis window. Blank when a series
    covers less than `min_coverage` of the window's months."""
    rows = {}
    for name, (a, b) in CRISES.items():
        win = rets.loc[a:b]
        n = len(pd.period_range(a, b, freq="M"))
        rows[name] = {c: ((1 + win[c].dropna()).prod() - 1) if win[c].notna().sum() >= min_coverage * n else np.nan
                      for c in rets}
    return pd.DataFrame(rows).T


def ulcer_index(r: pd.Series) -> float:
    """Root-mean-square drawdown (Martin 1987). Max drawdown is a *single observation* — the
    one worst moment ever recorded. The ulcer index integrates depth **and** duration across
    the whole history, so a brief crash and a decade-long slump stop looking alike."""
    return float(np.sqrt((pf.drawdown(r.dropna()) ** 2).mean()))


def martin_ratio(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    """CAGR divided by the ulcer index: Calmar's duration-aware cousin, and far less hostage
    to the single worst month in the sample."""
    x = r.dropna()
    return float((pf.wealth(x).iloc[-1] ** (12 / len(x)) - 1) / ulcer_index(x))


def capture_ratios(r: pd.Series, bench: pd.Series) -> dict:
    """Behaviour conditional on the benchmark's direction.

    Up/down capture: mean return in months the benchmark rose (fell), over the benchmark's own
    mean in those months. Down capture below 1 means the index cushioned the falls.
    Down/up beta: slope fitted separately on falling and rising months. The dangerous pattern
    is down-beta **above** up-beta — more exposed in crashes than in rallies, the worst of both.
    """
    x = pd.concat([r.rename("a"), bench.rename("b")], axis=1).dropna()
    a, b = x["a"], x["b"]
    out = {}
    for lab, m in (("Up", b > 0), ("Down", b < 0)):
        out[f"{lab} capture"] = a[m].mean() / b[m].mean()
        out[f"{lab} beta"] = float(np.polyfit(b[m], a[m], 1)[0])
    out["Capture spread"] = out["Up capture"] - out["Down capture"]
    out["Beta gap (down − up)"] = out["Down beta"] - out["Up beta"]
    out["Avg in worst 5% months"] = a[b <= b.quantile(0.05)].mean()
    return out


def crisis_detail(rets: pd.DataFrame, crises: dict | None = None, recovery_cap: int = 180) -> pd.DataFrame:
    """Per series and crisis: how far it fell below the level at the window's start, how many
    months the fall took, and how many months from the window's start it needed to climb back.
    Recovery is tracked on the full series *after* the window, so it may run well past the
    window's end; NaN means it had not recovered within `recovery_cap` months or by data end.
    Series that do not cover the window's start are skipped."""
    rows = {}
    for name, (a, b) in (CRISES if crises is None else crises).items():
        for c in rets:
            s = rets[c].dropna()
            if s.empty or s.index[0] > pd.Timestamp(a) or s.loc[a:b].empty:
                continue
            w = (1 + s.loc[a:]).cumprod()
            win = w.loc[:b]
            t = win.idxmin()
            after = w.loc[t:].iloc[:recovery_cap]
            back = after[after >= 1.0]
            rows[(name, c)] = {
                "Fall": win.min() - 1,
                "Months to trough": int(w.index.get_loc(t)) + 1,
                "Months to recover": float(w.index.get_loc(back.index[0]) + 1) if len(back) else np.nan,
            }
    out = pd.DataFrame(rows).T
    out.index.names = ["Crisis", "Index"]
    return out


def metric_ci(r: pd.Series, fn, n: int = 1000, block: float = 12, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """Stationary-bootstrap confidence interval for an arbitrary statistic of a return series.
    Resampling in blocks keeps autocorrelation and crisis clustering intact. Wide intervals are
    the point: they show how much of a ranking is sampling noise."""
    x = r.dropna()
    paths = stationary_bootstrap_paths(x.to_frame(), len(x), n, block, seed)
    vals = np.array([fn(pd.Series(p[:, 0], index=x.index)) for p in paths])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"Estimate": float(fn(x)), "Low": float(lo), "High": float(hi), "Width": float(hi - lo)}


def rank_stability(rets: pd.DataFrame, split: str, metrics: dict, min_months: int = 60) -> pd.DataFrame:
    """Does a metric pick winners that *stay* winners? Rank every index by each metric in the
    early period, rank again in the late period, and report the Spearman correlation between
    the two rankings. Near zero means the metric describes the past rather than predicting the
    future — i.e. it is a description, not a selection rule."""
    early, late = rets.loc[:split], rets.loc[split:]
    out = {}
    for name, fn in metrics.items():
        def side(df):
            return df.apply(lambda s: fn(s.dropna()) if s.notna().sum() >= min_months else np.nan)
        a, b = side(early), side(late)
        ok = a.notna() & b.notna()
        rho, p = stats.spearmanr(a[ok], b[ok])
        top = list(a[ok].nlargest(3).index)
        out[name] = {"N": int(ok.sum()), "Rank corr. (Spearman)": float(rho), "p": float(p),
                     "Best 3 early": ", ".join(top),
                     "Their avg later rank": float(b[ok].rank(ascending=False)[top].mean())}
    return pd.DataFrame(out).T


def walk_forward_pick(rets: pd.DataFrame, score, bench: pd.Series | None = None,
                      train: int = 120, hold: int = 12) -> pd.DataFrame:
    """Each period, rank the candidates on `score` using only trailing data, buy the winner,
    hold, repeat. The honest test of whether a ranking metric is actually tradable, as opposed
    to merely descriptive in hindsight."""
    idx, rows, t = rets.index, [], train
    while t + hold <= len(rets):
        tr, te = rets.iloc[t - train:t], rets.iloc[t:t + hold]
        s = tr.apply(lambda c: score(c.dropna()) if c.notna().sum() >= train * 0.9 else np.nan)
        if s.notna().any():
            pick = s.idxmax()
            row = {"From": idx[t], "Pick": pick, "Pick return": (1 + te[pick]).prod() - 1,
                   "Equal-weight return": (1 + te.mean(axis=1)).prod() - 1}
            if bench is not None:
                row["Benchmark return"] = (1 + bench.reindex(te.index).dropna()).prod() - 1
            rows.append(row)
        t += hold
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Weighting methods (long-only, fully invested)
# ---------------------------------------------------------------------------------------


def ledoit_wolf_cov(r: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf constant-correlation shrinkage of the sample covariance (monthly)."""
    from pypfopt import risk_models
    prices = (1 + r.fillna(0)).cumprod()
    return risk_models.CovarianceShrinkage(prices, frequency=1, returns_data=False).ledoit_wolf().to_numpy()


def w_min_var(cov: np.ndarray, bounds) -> np.ndarray:
    # Monthly covariances are ~1e-3, so SLSQP's default tolerance stops at the start point
    # unless the objective is rescaled (annualised, in % squared).
    n = len(cov)
    c = cov * 12 * 1e4
    res = optimize.minimize(lambda w: w @ c @ w, np.full(n, 1 / n), bounds=bounds,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                            options={"ftol": 1e-12, "maxiter": 500})
    return res.x


def w_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, bounds) -> np.ndarray:
    n = len(mu)
    neg = lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w)
    best = None
    for start in [np.full(n, 1 / n), *np.eye(n)]:
        res = optimize.minimize(neg, start, bounds=bounds,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                                options={"ftol": 1e-12, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def w_erc(cov: np.ndarray, bounds=None) -> np.ndarray:
    """Equal risk contribution (Maillard, Roncalli, Teiletche 2010)."""
    n = len(cov)
    c = cov * 12 * 1e4  # rescale so the tolerance is meaningful (see w_min_var)

    def obj(w):
        rc = w * (c @ w)
        return ((rc[:, None] - rc[None, :]) ** 2).sum()

    res = optimize.minimize(obj, np.full(n, 1 / n), bounds=bounds or [(0, 1)] * n,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                            options={"ftol": 1e-14, "maxiter": 1000})
    return res.x / res.x.sum()


def w_inv_vol(cov: np.ndarray) -> np.ndarray:
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def w_min_cvar(scen: np.ndarray, alpha: float, bounds, target: float | None = None) -> np.ndarray:
    """Rockafellar-Uryasev minimum CVaR over scenario returns (rows = scenarios), optionally
    with a minimum expected return. Solved as an LP."""
    from scipy.optimize import linprog
    S, n = scen.shape
    # variables: w (n), VaR z (1), excess u (S)
    c = np.concatenate([np.zeros(n), [1.0], np.full(S, 1 / ((1 - alpha) * S))])
    A_ub = np.hstack([-scen, -np.ones((S, 1)), -np.eye(S)])  # -scen w - z - u <= 0  <=> u >= -scen w - z
    b_ub = np.zeros(S)
    if target is not None:
        A_ub = np.vstack([A_ub, np.concatenate([-scen.mean(axis=0), [0], np.zeros(S)])])
        b_ub = np.append(b_ub, -target)
    A_eq = np.concatenate([np.ones(n), [0], np.zeros(S)])[None, :]
    bnds = list(bounds) + [(None, None)] + [(0, None)] * S
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1], bounds=bnds, method="highs")
    return res.x[:n]


def resampled_max_sharpe(r: pd.DataFrame, rf: float, bounds, n_draws: int = 200, seed: int = 0) -> np.ndarray:
    """Michaud-style resampling: redraw return histories from the estimated mean/covariance,
    re-optimise max-Sharpe each time, average the weights."""
    rng = np.random.default_rng(seed)
    mu, cov = r.mean().to_numpy(), ledoit_wolf_cov(r)
    T = len(r)
    ws = []
    for _ in range(n_draws):
        sim = rng.multivariate_normal(mu, cov, size=T)
        ws.append(w_max_sharpe(sim.mean(axis=0), np.cov(sim.T), rf, bounds))
    return np.mean(ws, axis=0)


def black_litterman_mu(cov: np.ndarray, w_mkt: np.ndarray, delta: float = 2.5) -> np.ndarray:
    """Equilibrium (reverse-optimised) excess returns implied by market weights."""
    return delta * cov @ w_mkt


# ---------------------------------------------------------------------------------------
# Stochastic simulation
# ---------------------------------------------------------------------------------------


def cma_shift(r: pd.DataFrame, targets: dict[str, float]) -> np.ndarray:
    """Per-asset constant to add to monthly simple returns so that each asset's
    *compound* (geometric) annual return equals the capital-market-assumption target.
    CMAs are quoted as compound returns, so matching the arithmetic mean instead would
    understate the target by about half the variance (~1%/yr for equities)."""
    X = r.dropna()
    out = []
    for c in X.columns:
        goal = np.log1p(targets[c]) / 12
        out.append(optimize.brentq(lambda k: np.log1p(X[c].to_numpy() + k).mean() - goal, -0.05, 0.05))
    return np.array(out)


def optimal_block_length(x: pd.DataFrame) -> float:
    """Politis & White (2004, with Patton-Politis-White 2009 correction) stationary
    bootstrap block length, averaged across columns."""
    from arch.bootstrap import optimal_block_length as obl
    return float(obl(x.dropna())["stationary"].mean())


def stationary_bootstrap_paths(r: pd.DataFrame, horizon: int, n_paths: int, block: float,
                               seed: int = 0, shift: np.ndarray | None = None) -> np.ndarray:
    """Politis-Romano stationary bootstrap of whole cross-sections (keeps cross-asset
    correlation and crisis clustering). Returns array (n_paths, horizon, n_assets).
    `shift` (monthly, per asset) re-centres the means, e.g. on capital-market assumptions."""
    rng = np.random.default_rng(seed)
    X = r.dropna().to_numpy()
    if shift is not None:
        X = X + shift[None, :]
    T = len(X)
    p = 1 / block
    out = np.empty((n_paths, horizon, X.shape[1]))
    for i in range(n_paths):
        idx = np.empty(horizon, dtype=int)
        idx[0] = rng.integers(T)
        jumps = rng.random(horizon) < p
        starts = rng.integers(T, size=horizon)
        for t in range(1, horizon):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
        out[i] = X[idx]
    return out


def bootstrap_on_targets(r: pd.DataFrame, horizon: int, n_paths: int, block: float, seed: int,
                         targets: dict[str, float]) -> np.ndarray:
    """Stationary bootstrap re-centred on compound annual `targets`, then corrected so the
    paths *as drawn* hit the targets exactly. With a finite number of paths, random over- and
    under-drawing of particular months moves each asset's simulated return off its target —
    by ~0.1%/yr for equities at 5,000 paths and 12-month blocks, and in the same direction for
    every equity column, since they share the draws — which shifts the equity-bond gap.
    Matching the moment removes that sampling noise without changing the shapes."""
    P = stationary_bootstrap_paths(r, horizon, n_paths, block, seed, shift=cma_shift(r, targets))
    for i, c in enumerate(r.columns):
        goal = np.log1p(targets[c]) / 12
        x = P[..., i]
        P[..., i] = x + optimize.brentq(lambda k: np.log1p(x + k).mean() - goal, -0.01, 0.01)
    return P


def pac_paths(paths: np.ndarray, weights: np.ndarray, tax_rates: np.ndarray, initial: float = 1000.0,
              monthly: float = 100.0, initial_fee: float = 1.0, stamp_duty: float = 0.002) -> dict:
    """Vectorised PAC over simulated paths with the 'steer' rule (each contribution buys
    the most underweight asset: no selling, no tax drag until the end; fee-free monthly
    savings plan; €1 fee on the one-off initial order per asset bought). Returns final
    pre-tax value, after-tax liquidation value, max drawdown of the account value
    (relative to its running peak), and the path of values."""
    n_paths, H, n = paths.shape
    w = weights
    active = w > 0
    hold = np.tile(initial * w, (n_paths, 1)) - initial_fee * active.sum() * w
    basis = np.tile(initial * w, (n_paths, 1))
    values = np.empty((n_paths, H))
    for t in range(H):
        hold *= 1 + paths[:, t, :]
        basis -= hold * stamp_duty / 12   # the bollo is not deductible: it must not shrink the taxable gain
        hold *= 1 - stamp_duty / 12
        total = hold.sum(axis=1, keepdims=True) + monthly
        gap = np.where(active, w * total - hold, -np.inf)
        j = gap.argmax(axis=1)
        hold[np.arange(n_paths), j] += monthly
        basis[np.arange(n_paths), j] += monthly
        values[:, t] = hold.sum(axis=1)
    gains = np.maximum(hold - basis, 0)
    after_tax = hold.sum(axis=1) - (gains * tax_rates).sum(axis=1)
    peak = np.maximum.accumulate(values, axis=1)
    paid = initial + monthly * np.arange(1, H + 1)
    # drawdown of value relative to contributions-adjusted peak: use value / paid-in ratio
    ratio = values / paid
    dd = (ratio / np.maximum.accumulate(ratio, axis=1) - 1).min(axis=1)
    return {"final": values[:, -1], "after_tax": after_tax, "max_dd": dd, "values": values,
            "paid": paid[-1], "peak": peak}


def irr_annual(final: np.ndarray, initial: float, monthly: float, months: int) -> np.ndarray:
    """Money-weighted annual return of a PAC for each final value (vectorised bisection)."""
    lo, hi = np.full(len(final), -0.1), np.full(len(final), 0.1)
    for _ in range(60):
        m = (lo + hi) / 2
        g = 1 + m
        fv = initial * g**months + monthly * np.where(np.abs(m) < 1e-12, months, (g**months - 1) / np.where(m == 0, 1, m))
        lo = np.where(fv < final, m, lo)
        hi = np.where(fv >= final, m, hi)
    return (1 + (lo + hi) / 2) ** 12 - 1


def pac_paths_rule(paths: np.ndarray, weights: np.ndarray, tax_rates: np.ndarray, rule: str = "steer",
                   initial: float = 1000.0, monthly: float = 100.0, initial_fee: float = 1.0,
                   stamp_duty: float = 0.002, spread: np.ndarray | float = 0.0) -> dict:
    """Like pac_paths, with rule='split' (each contribution split by target weight: one
    fee-free savings plan per ETF) or 'steer' (whole contribution to the most underweight
    asset). Also returns the worst drift from target seen on each path.

    `spread` (split rule only, per asset or scalar) is the half bid-ask spread paid on every
    purchase and again on the final sale."""
    if rule == "steer":
        out = pac_paths(paths, weights, tax_rates, initial, monthly, initial_fee, stamp_duty)
    else:
        n_paths, H, n = paths.shape
        w = weights
        active = w > 0
        buy = 1 - np.broadcast_to(np.asarray(spread, dtype=float), (n,))
        hold = np.tile((initial * w - initial_fee * active.sum() * w) * buy, (n_paths, 1))
        basis = np.tile(initial * w, (n_paths, 1))
        values = np.empty((n_paths, H))
        twr = np.empty((n_paths, H))
        drift = np.zeros(n_paths)
        prev = hold.sum(axis=1)
        for t in range(H):
            hold *= 1 + paths[:, t, :]
            basis -= hold * stamp_duty / 12   # the bollo is not deductible: it must not shrink the taxable gain
            hold *= 1 - stamp_duty / 12
            twr[:, t] = hold.sum(axis=1) / prev - 1          # account return before this month's contribution
            hold += monthly * w * buy
            basis += monthly * w
            values[:, t] = prev = hold.sum(axis=1)
            drift = np.maximum(drift, np.abs(hold / values[:, t:t + 1] - w).max(axis=1))
        hold = hold * buy                     # sell at the bid
        tax = np.maximum(hold - basis, 0) * tax_rates
        paid = initial + monthly * np.arange(1, H + 1)
        ratio = values / paid
        out = {"final": values[:, -1], "after_tax": hold.sum(axis=1) - tax.sum(axis=1),
               "max_dd": (ratio / np.maximum.accumulate(ratio, axis=1) - 1).min(axis=1),
               "values": values, "paid": paid[-1], "worst_drift": drift, "twr": twr,
               "sleeve_value": hold, "sleeve_basis": basis, "sleeve_tax": tax}
    return out


def longest_in_red(values: np.ndarray, paid: np.ndarray) -> np.ndarray:
    """Longest run of consecutive months, per path, in which the account was worth less than
    what had been paid into it — how long a saver would have stared at a loss."""
    red = values < paid
    run = np.zeros(len(red), dtype=int)
    best = np.zeros(len(red), dtype=int)
    for t in range(red.shape[1]):
        run = np.where(red[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def rolldown(curve: dict[int, float], weights: dict[int, float]) -> float:
    """Extra annual return of a constant-maturity ladder if today's curve stays where it is:
    each rung's modified duration times the local slope of the curve below it (the yield it
    'rolls down' to as it ages). Annual-coupon par bonds; the shortest rung uses the slope above."""
    ms = sorted(curve)
    out = 0.0
    for m, w in weights.items():
        i = ms.index(m)
        lo, hi = (ms[i - 1], m) if i > 0 else (m, ms[i + 1])
        slope = (curve[hi] - curve[lo]) / (hi - lo)
        y = curve[m]
        out += w * slope * (1 - (1 + y) ** -m) / y
    return out


def inflated_paid(initial: float, monthly: float, months: int, inflation: float) -> float:
    """What the contributions are worth at the end in today's money terms if prices rise
    by `inflation` a year: the bar the plan must clear to keep its purchasing power."""
    g = (1 + inflation) ** (1 / 12)
    return initial * g ** months + monthly * sum(g ** (months - 1 - t) for t in range(months))


def certainty_equivalent(wealth: np.ndarray, gamma: float) -> float:
    """CRRA certainty equivalent: the sure amount an investor with relative risk aversion
    `gamma` would accept instead of the uncertain outcome. Higher gamma = more weight on
    the bad paths. gamma = 1 is the geometric mean of outcomes."""
    w = np.asarray(wealth, dtype=float)
    if gamma == 1:
        return float(np.exp(np.log(w).mean()))
    return float(np.mean(w ** (1 - gamma)) ** (1 / (1 - gamma)))


def alpha_beta(r: np.ndarray | pd.Series, market: np.ndarray | pd.Series, rf) -> dict:
    """Annualised Jensen alpha and beta of monthly returns against a market, both in excess
    of cash. Works on series (aligned) or on pooled simulated paths (flattened)."""
    if isinstance(r, pd.Series):
        x = pd.concat([r, market, rf.reindex(r.index) if isinstance(rf, pd.Series) else pd.Series(rf, r.index)],
                      axis=1).dropna().to_numpy()
        a, m, f = x[:, 0], x[:, 1], x[:, 2]
    else:
        a, m = np.ravel(r), np.ravel(market)
        f = np.broadcast_to(rf, a.shape)
    ea, em = a - f, m - f
    b = float(np.cov(ea, em, ddof=1)[0, 1] / em.var(ddof=1))
    resid = ea - b * em
    alpha = resid.mean()
    if isinstance(r, pd.Series):
        # intercept standard error of the regression, Newey-West (6 lags): monthly factor
        # returns are mildly autocorrelated, and the residual-mean shortcut drops the
        # intercept's x̄²/Sxx term
        Xm = np.column_stack([np.ones_like(em), em])
        n, L = len(resid), 6
        u = Xm * (resid - alpha)[:, None]
        S = u.T @ u / n
        for k in range(1, L + 1):
            g = u[k:].T @ u[:-k] / n
            S += (1 - k / (L + 1)) * (g + g.T)
        Q = np.linalg.inv(Xm.T @ Xm / n)
        se = float(np.sqrt((Q @ S @ Q)[0, 0] / n))
    else:
        se = resid.std(ddof=2) / np.sqrt(len(resid))
    return {"Beta": b, "Alpha /yr": 12 * alpha, "Alpha t-stat": alpha / se}


def curve_paths(eq_r: pd.Series, curve: dict[int, pd.Series], y0: dict[int, float], horizon: int, n_paths: int,
                seed: int = 41, eq_target: float | None = None, revert: float = 0.005,
                floor: float = -0.01) -> tuple[np.ndarray, np.ndarray]:
    """Joint paths of an equity return and a whole yield curve, for valuing bond funds and bonds
    *as the curve moves* rather than through a fixed return target.

    The monthly curve changes are bootstrapped together with the equity month (so the stock-bond
    correlation and the crises carry over), de-meaned so that the expected curve is today's, and
    pulled back towards today's curve at `revert` a month. Without the pull a random walk spreads
    the 10-year yield over ten years by ±2.5 points (1 s.d.); with 0.5% a month (half-life ≈ 11
    years) it spreads by ±1.7, what Bund yields did over every 10-year window since 1972. `floor`
    stops yields below −1% (the deepest Bund yields reached). Returns (equity returns (n, H),
    yields (n, H+1, maturities)), maturities in the order of `curve`."""
    mats = list(curve)
    dY = pd.DataFrame({m: curve[m].diff() for m in mats})
    J = pd.concat([eq_r.rename("eq"), dY], axis=1).dropna()
    J[mats] = J[mats] - J[mats].mean()
    shift = np.zeros(J.shape[1])
    if eq_target is not None:
        shift[0] = cma_shift(J[["eq"]], {"eq": eq_target})[0]
    P = stationary_bootstrap_paths(J, horizon, n_paths, 12, seed, shift=shift)
    if eq_target is not None:
        x = P[..., 0]
        P[..., 0] = x + optimize.brentq(lambda k: np.log1p(x + k).mean() - np.log1p(eq_target) / 12, -0.01, 0.01)
    Y0 = np.array([y0[m] for m in mats])
    Y = np.empty((n_paths, horizon + 1, len(mats)))
    Y[:, 0] = Y0
    for t in range(horizon):
        Y[:, t + 1] = np.maximum(Y[:, t] + revert * (Y0 - Y[:, t]) + P[:, t, 1:], floor)
    return P[..., 0], Y


def _interp_curve(Yt: np.ndarray, mats: list[int], m: float) -> np.ndarray:
    """Yield at maturity m (years) from curve points Yt (n, len(mats)); flat beyond the ends."""
    if m <= mats[0]:
        return Yt[:, 0]
    j = min(int(np.searchsorted(mats, m)), len(mats) - 1)
    if m >= mats[-1]:
        return Yt[:, -1]
    a, b = mats[j - 1], mats[j]
    return Yt[:, j - 1] + (Yt[:, j] - Yt[:, j - 1]) * (m - a) / (b - a)


def fund_paths(Y: np.ndarray, mats: list[int], rungs: list[int], fee: float) -> tuple[np.ndarray, np.ndarray]:
    """Monthly total return (after fee) and coupon income of a constant-maturity bond fund — an
    equal ladder of `rungs`, rolled monthly, each rung repriced at the yield of its shorter remaining
    maturity (so roll-down is included) — on simulated curves Y (n, H+1, len(mats))."""
    R, inc = 0.0, 0.0
    for m in rungs:
        j = mats.index(m)
        lo, hi = (j - 1, j) if j > 0 else (0, 1)
        slope = (Y[..., hi] - Y[..., lo]) / (mats[hi] - mats[lo])
        c, y1 = Y[:, :-1, j], Y[:, 1:, j] - slope[:, 1:] / 12
        p = pf._bond_dirty_price(c.ravel(), y1.ravel(), m - 1 / 12, 1).reshape(c.shape)
        R, inc = R + (p - 1) / len(rungs), inc + c / 12 / len(rungs)
    return (1 + R) * (1 - fee) ** (1 / 12) - 1, inc


def sleeve_paths(R: np.ndarray, contrib: np.ndarray, tax: float, income: np.ndarray | None = None,
                 stamp_duty: float = 0.002, initial_fee: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One sleeve of a savings plan, on per-path monthly returns R (n, H). `contrib` (H+1) is the
    money added before month 1 (index 0, a one-off order paying `initial_fee`) and at the end of each
    month 1..H (savings plan, free). Accumulating fund when `income` is None; distributing when it is
    the monthly coupon rate: the payout is taxed at `tax` and reinvested through the plan (basis
    up by the net amount). Returns (value after each month (n, H), after-tax value at the end)."""
    n, H = R.shape
    hold = np.full(n, max(contrib[0] - initial_fee, 0.0) if contrib[0] > 0 else 0.0)
    basis = np.full(n, float(contrib[0]))
    vals = np.empty((n, H))
    for t in range(H):
        hold = hold * (1 + R[:, t])
        if income is not None:
            paid = hold * income[:, t] * tax
            hold, basis = hold - paid, basis + hold * income[:, t] * (1 - tax)
        duty = hold * stamp_duty / 12
        hold, basis = hold - duty, basis - duty          # the bollo is not deductible
        hold, basis = hold + contrib[t + 1], basis + contrib[t + 1]
        vals[:, t] = hold
    return vals, hold - np.maximum(hold - basis, 0) * tax


def held_bonds_paths(Y: np.ndarray, mats: list[int], contrib: np.ndarray, every: int = 3, fee: float = 1.0,
                     spread: float = 0.002, tax: float = 0.125, stamp_duty: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Individual government bonds instead of a fund: every `every` months the money saved so far
    buys a bond (a one-off order, €`fee`, half-spread `spread`) that matures at the end of the plan,
    at that day's yield for the remaining maturity, and is held to maturity. Coupons are taxed as
    paid and reinvested at the same yield (an approximation: in practice reinvesting a coupon
    is another order). Value before maturity is marked at the current yield, so the account still
    moves with rates — but the end value does not: it is fixed at purchase."""
    n, H1, _ = Y.shape
    H = H1 - 1
    face, idle = np.zeros(n), 0.0
    vals = np.empty((n, H))
    for t in range(H + 1):
        idle += contrib[t]
        rem = (H - t) / 12
        if rem > 0 and t % every == 0 and idle > fee:
            g = 1 + _interp_curve(Y[:, t], mats, rem) * (1 - tax)
            face = face + (idle - fee) * (1 - spread) * g ** rem * (1 - stamp_duty / 12) ** (H - t)
            idle = 0.0
        if t >= 1:
            g = 1 + _interp_curve(Y[:, t], mats, max(rem, 1e-9)) * (1 - tax)
            vals[:, t - 1] = face / g ** rem / (1 - stamp_duty / 12) ** (H - t) + idle
    return vals, face + idle


def historical_windows(r: pd.DataFrame, years: int = 10, step: int = 1) -> np.ndarray:
    """Every overlapping `years`-long window of actual history as an array shaped like
    bootstrap paths (n_windows, months, n_assets); start months are r.index[::step]."""
    X = r.dropna().to_numpy()
    H = years * 12
    starts = range(0, len(X) - H + 1, step)
    return np.stack([X[s:s + H] for s in starts]), r.dropna().index[list(starts)]


def outcome_summary(res: dict, months: int, initial: float = 1000.0, monthly: float = 100.0,
                    inflation: float = 0.02) -> dict:
    at = res["after_tax"]
    irr = irr_annual(at, initial, monthly, months)
    p5 = np.percentile(at, 5)
    out = {
        "Median €": np.median(at), "Bad case p5 €": p5, "Good case p95 €": np.percentile(at, 95),
        "Mean €": at.mean(), "CVaR 5% €": at[at <= p5].mean(), "P(end < paid in)": (at < res["paid"]).mean(),
        "P(real loss)": (at < inflated_paid(initial, monthly, months, inflation)).mean(),
        "Median IRR": np.median(irr), "p5 IRR": np.percentile(irr, 5),
        "Median max DD": np.median(res["max_dd"]), "p5 max DD": np.percentile(res["max_dd"], 5),
    }
    if "twr" in res:
        twr = res["twr"]
        out["Volatility"] = float(twr.std() * np.sqrt(12))
        out["CAGR"] = float(np.exp(np.log1p(twr).mean() * 12) - 1)     # expected compound, all paths
        uw = longest_in_red(res["values"], initial + monthly * np.arange(1, months + 1))
        out["Longest in red, median (m)"] = float(np.median(uw))
        out["Longest in red, p95 (m)"] = float(np.percentile(uw, 95))
        for g in (2, 4, 8):
            out[f"CE γ={g} €"] = certainty_equivalent(at, g)
    return out
