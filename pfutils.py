"""Data loading (cached) and portfolio maths for the notebooks.

Every download is cached under data/ as parquet, so the notebook re-runs offline and
reproduces the same numbers. Pass refresh=True (or delete data/*.parquet) to pull fresh data.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(DATA / ".yf_cache"))

UA = {"User-Agent": "Mozilla/5.0 (portfolio-analysis notebook)"}


def _cached(name: str, fetch, refresh: bool) -> pd.DataFrame:
    path = DATA / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------


def yahoo_close(ticker: str, start: str = "1985-01-01", refresh: bool = False) -> pd.Series:
    """Daily dividend-adjusted close. One ticker per call: yfinance's threaded multi-ticker
    download contends on its sqlite cache, and Yahoo rate-limits bursts (HTTP 429)."""

    def fetch():
        for attempt in range(5):
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
            if not raw.empty:
                close = raw["Close"]
                return (close.iloc[:, 0] if close.ndim > 1 else close).to_frame("Close")
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Yahoo returned no data for {ticker} (rate limited?) - retry later")

    safe = ticker.replace("^", "IDX_").replace("=", "_")
    return _cached(f"yahoo_{safe}", fetch, refresh)["Close"].rename(ticker)


def yahoo_sector_weights(ticker: str, refresh: bool = False) -> pd.Series:
    """Sector weights (decimal) as reported by Yahoo for a fund."""

    def fetch():
        return pd.Series(yf.Ticker(ticker).funds_data.sector_weightings).to_frame("weight")

    return _cached(f"sectors_{ticker}", fetch, refresh)["weight"].rename(ticker)


def yahoo_top_holdings(ticker: str, refresh: bool = False) -> pd.Series:
    """Top-10 holdings (weight, decimal) as reported by Yahoo for a fund."""

    def fetch():
        th = yf.Ticker(ticker).funds_data.top_holdings
        return th[["Holding Percent"]]

    return _cached(f"holdings_{ticker}", fetch, refresh)["Holding Percent"].rename(ticker)


def msci_index(
    code: int = 990100,
    currency: str = "EUR",
    variant: str = "NETR",
    start: str = "19970101",
    end: str | None = None,
    refresh: bool = False,
) -> pd.Series:
    """MSCI month-end index levels from MSCI's public index-tool endpoint.
    990100 = MSCI World. NETR = net total return (dividends reinvested after withholding
    tax): the benchmark a UCITS world ETF actually tracks. EUR NETR history starts 2000-12."""
    end = end or pd.Timestamp.today().strftime("%Y%m%d")

    def fetch():
        url = "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph"
        params = dict(
            currency_symbol=currency,
            index_variant=variant,
            start_date=start,
            end_date=end,
            data_frequency="END_OF_MONTH",
            index_codes=code,
        )
        js = requests.get(url, params=params, headers=UA, timeout=60).json()
        lv = pd.DataFrame(js["indexes"]["INDEX_LEVELS"])
        lv["date"] = pd.to_datetime(lv["calc_date"].astype(str))
        return lv.set_index("date")[["level_eod"]]

    name = f"msci_{code}_{currency}_{variant}"
    return _cached(name, fetch, refresh)["level_eod"].rename(f"MSCI {code} {currency} {variant}")


def ecb_series(key: str, start: str = "1999-01-01", refresh: bool = False) -> pd.Series:
    """Any ECB Data Portal series, e.g. 'EXR/D.USD.EUR.SP00.A' (USD per 1 EUR, daily)."""

    def fetch():
        url = f"https://data-api.ecb.europa.eu/service/data/{key}"
        txt = requests.get(url, params={"format": "csvdata", "startPeriod": start}, timeout=60).text
        df = pd.read_csv(io.StringIO(txt), usecols=["TIME_PERIOD", "OBS_VALUE"])
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        return df.set_index("TIME_PERIOD").rename(columns={"OBS_VALUE": "value"}).sort_index()

    return _cached("ecb_" + key.replace("/", "_").replace(".", "_"), fetch, refresh)["value"].rename(key)


def french_factors(dataset: str, refresh: bool = False) -> pd.DataFrame:
    """Monthly factor returns (decimals, USD) from Kenneth French's data library, e.g.
    'F-F_Research_Data_5_Factors_2x3' or 'F-F_Momentum_Factor'. Only the first (monthly)
    table of the file is read; the annual table after it is ignored."""

    def fetch():
        url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{dataset}_CSV.zip"
        raw = requests.get(url, headers=UA, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        header = next(i for i, ln in enumerate(lines) if ln.startswith(","))
        rows = []
        for ln in lines[header + 1 :]:
            cells = [c.strip() for c in ln.split(",")]
            if not (cells[0].isdigit() and len(cells[0]) == 6):
                break  # end of the monthly table
            rows.append(cells)
        cols = [c.strip() for c in lines[header].split(",")[1:]]
        df = pd.DataFrame([r[1:] for r in rows], columns=cols, dtype=float)
        df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
        return df.replace([-99.99, -999.0], np.nan) / 100

    return _cached(f"ff_{dataset}", fetch, refresh)


# ---------------------------------------------------------------------------------------
# Cleaning and transforms
# ---------------------------------------------------------------------------------------


def find_spikes(close: pd.Series, threshold: float = 0.12) -> pd.DatetimeIndex:
    """Isolated bad prints: a day more than `threshold` away from BOTH neighbours while the
    two neighbours agree with each other. A genuine crash day (March 2020) is not reverted
    the next day, so it is kept; a one-day fat-finger print is flagged."""
    s = close.dropna()
    prev, nxt = s.shift(1), s.shift(-1)
    bad = (
        ((s / prev - 1).abs() > threshold)
        & ((s / nxt - 1).abs() > threshold)
        & ((nxt / prev - 1).abs() < threshold / 2)
    )
    return s.index[bad]


def month_end(close: pd.Series, last_complete: pd.Timestamp, max_stale_days: int = 7) -> pd.Series:
    """Last available price in each calendar month, stamped at month end, up to the last
    complete month. A month whose last trade is more than `max_stale_days` before month end
    (a trading gap) is set to NaN rather than carrying a stale price into the return."""
    s = close.dropna()
    last = s.resample("ME").last()
    last_day = s.index.to_series().resample("ME").last()
    stale = (last.index.to_series() - last_day.reindex(last.index)).dt.days > max_stale_days
    return last.mask(stale).loc[:last_complete]


def drop_listing_period(close: pd.Series, months: int = 3) -> pd.Series:
    """Drop the first months after a listing: fresh listings often carry placeholder or
    stale quotes (e.g. SPPW.DE printed 20.00 for days, then 'fell' 12% on 2019-03-04)."""
    s = close.dropna()
    return s.loc[s.index[0] + pd.DateOffset(months=months):]


def to_eur(r_usd: pd.Series, usd_per_eur: pd.Series) -> pd.Series:
    """Convert monthly USD returns to EUR returns using month-end USD-per-EUR rates."""
    fx = usd_per_eur.reindex(r_usd.index)
    return (1 + r_usd) * (fx.shift(1) / fx) - 1


def splice(actual: pd.Series, proxy: pd.Series) -> pd.Series:
    """Actual returns where they exist, proxy returns before the first actual month."""
    first = actual.first_valid_index()
    return pd.concat([proxy.loc[: first - pd.offsets.MonthEnd(1)], actual.loc[first:]]).dropna()


def tracking(r: pd.Series, bench: pd.Series) -> dict:
    """Tracking difference (annualised return gap), tracking error (annualised stdev of the
    monthly gap) and correlation between a fund and its benchmark, on common months."""
    df = pd.concat([r, bench], axis=1).dropna()
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    years = len(df) / 12
    td = (1 + a).prod() ** (1 / years) - (1 + b).prod() ** (1 / years)
    return {
        "Months": len(df),
        "From": df.index[0].strftime("%Y-%m"),
        "Tracking difference /yr": td,
        "Tracking error /yr": (a - b).std() * np.sqrt(12),
        "Correlation": a.corr(b),
    }


# ---------------------------------------------------------------------------------------
# Statistics on monthly simple returns
# ---------------------------------------------------------------------------------------


def wealth(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    # the peak starts at 1 (the money before the first month), so a first-month loss counts
    w = wealth(r)
    return w / w.cummax().clip(lower=1.0) - 1


def perf_stats(r: pd.Series, rf: pd.Series | float = 0.0) -> dict:
    """CAGR, volatility, Sharpe (mean excess return over rf / vol), max drawdown, worst
    calendar year, worst rolling 12 months, longest time under water (months), and the
    monthly 95% historical VaR / CVaR."""
    r = r.dropna()
    rf_m = rf.reindex(r.index).ffill().fillna(0) if isinstance(rf, pd.Series) else rf
    years = len(r) / 12
    w = wealth(r)
    under = w < w.cummax().clip(lower=1.0)
    runs = under.groupby((~under).cumsum()).sum()
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    full_years = r.groupby(r.index.year).size() == 12
    roll12 = (1 + r).rolling(12).apply(np.prod, raw=True) - 1
    var95 = r.quantile(0.05)
    return {
        "CAGR": w.iloc[-1] ** (1 / years) - 1,
        "Volatility": r.std() * np.sqrt(12),
        "Sharpe": (r - rf_m).mean() / r.std() * np.sqrt(12),
        "Max drawdown": float(drawdown(r).min()),
        "Worst year": yearly[full_years].min(),
        "Worst 12m": roll12.min(),
        "Longest underwater (m)": int(runs.max()),
        "VaR 95% (1m)": var95,
        "CVaR 95% (1m)": r[r <= var95].mean(),
    }


def mix(rets: pd.DataFrame, weights: dict) -> pd.Series:
    """Monthly-rebalanced mix: a clean benchmark for comparing allocations (a PAC investor
    who never sells holds something slightly different; see simulate_pac)."""
    return sum(rets[k] * w for k, w in weights.items())


# ---------------------------------------------------------------------------------------
# PAC (accumulation plan) simulation with Italian taxation
# ---------------------------------------------------------------------------------------


def simulate_pac(
    rets: pd.DataFrame,
    targets: dict,
    initial: float = 1000.0,
    monthly: float = 100.0,
    rule: str = "steer",
    band: float = 0.05,
    fee_per_order: float = 0.0,
    tax_rates: dict | None = None,
    stamp_duty: float = 0.002,
) -> pd.DataFrame:
    """Simulate €`initial` up front plus €`monthly` at every month end, on monthly returns.

    rule:
      'split' - every contribution is split by the target weights (one order per asset).
      'steer' - the whole contribution buys the most underweight asset (one order) and
                nothing is ever sold: the blueprint's 'tax-free rebalancing'.
      'sell'  - contributions split as in 'split'; whenever any weight drifts more than
                `band` from target, sell the overweights and buy the underweights back to
                target. Realised gains are taxed immediately at the asset's rate.

    Italian rules modelled: capital gains taxed on realisation (26% equity ETFs, 12.5% on
    government-bond ETFs, set via tax_rates); no offset of losses against ETF gains (in Italy
    ETF gains are 'redditi di capitale' and cannot absorb losses); 0.2% annual stamp duty on
    the account value, charged monthly. Cost basis is tracked per asset at average cost.

    Returns one row per month with value, paid-in, cumulative fees/taxes, the after-tax
    liquidation value (everything sold and taxed at that month end), and current weights.
    """
    keys = list(targets)
    w = np.array([targets[k] for k in keys], dtype=float)
    rates = np.array([(tax_rates or {}).get(k, 0.26) for k in keys])
    active = w > 0

    def buy(amounts):
        nonlocal fees
        n = int((amounts > 0).sum())
        fees += fee_per_order * n
        net = np.where(amounts > 0, amounts - fee_per_order, 0.0)
        return np.maximum(net, 0.0)

    fees = 0.0
    taxes = 0.0
    hold = buy(initial * w)
    basis = initial * w  # fees are added to cost basis, as Italian brokers report
    rows = []
    for i, (date, r) in enumerate(rets[keys].iterrows()):
        hold = hold * (1 + r.to_numpy())
        stamp = hold * stamp_duty / 12
        hold -= stamp
        basis -= stamp  # the bollo is not deductible: paying it must not shrink the taxable gain
        fees += stamp.sum()

        if rule == "steer":
            gap = w * (hold.sum() + monthly) - hold
            alloc = np.zeros_like(w)
            alloc[int(np.argmax(np.where(active, gap, -np.inf)))] = monthly
        else:
            alloc = monthly * w
        hold += buy(alloc)
        basis += alloc

        if rule == "sell" and np.abs(hold / hold.sum() - w).max() > band:
            delta = w * hold.sum() - hold
            proceeds = 0.0
            for j in np.where(delta < 0)[0]:
                sold = -delta[j]
                frac = sold / hold[j]
                gain = sold - basis[j] * frac
                tax = max(gain, 0.0) * rates[j]
                taxes += tax
                proceeds += sold - tax - fee_per_order
                fees += fee_per_order
                basis[j] *= 1 - frac
                hold[j] -= sold
            want = np.where(delta > 0, delta, 0.0)
            spend = want / want.sum() * proceeds
            hold += buy(spend)
            basis += spend

        unrealised = np.maximum(hold - basis, 0.0)
        cur = hold / hold.sum()
        rows.append(
            {
                "date": date,
                "value": hold.sum(),
                "paid_in": initial + monthly * (i + 1),
                "fees_cum": fees,
                "tax_paid_cum": taxes,
                "net_liquidation": hold.sum() - (unrealised * rates).sum() - fee_per_order * active.sum(),
                "max_drift": float(np.abs(cur - w)[active].max()),
                **{k: cur[j] for j, k in enumerate(keys)},
            }
        )
    return pd.DataFrame(rows).set_index("date")


def irr_monthly(paid: float, monthly: float, months: int, final: float) -> float:
    """Annualised money-weighted return of a PAC (initial `paid`, then `monthly` at each of
    `months` month ends, the last contribution coinciding with the final valuation)."""
    from scipy.optimize import brentq

    def npv(m):
        g = 1 + m
        flows = paid * g**months + monthly * sum(g**k for k in range(months))
        return flows - final

    m = brentq(npv, -0.5, 0.5)
    return (1 + m) ** 12 - 1


def bundesbank_series(key: str, start: str = "1999-01-01", refresh: bool = False) -> pd.Series:
    """Daily Bundesbank time series, e.g. the 7-year Bund par yield
    'BBSIS/D.I.ZAR.ZI.EUR.S1311.B.A604.R07XX.R.A.A._Z._Z.A' (percent)."""

    def fetch():
        url = f"https://api.statistiken.bundesbank.de/rest/data/{key}"
        txt = requests.get(url, params={"startPeriod": start, "format": "csv", "lang": "en"},
                           headers={"Accept": "text/csv"}, timeout=60).text
        rows = [ln.split(",")[:2] for ln in txt.splitlines() if ln[:4].isdigit() and ln[4] == "-"]
        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna().set_index("date")

    return _cached("bbk_" + key.replace("/", "_").replace(".", "_"), fetch, refresh)["value"].rename(key)


def fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    """A FRED series via the public CSV endpoint (no API key), e.g. 'GS10' (percent)."""

    def fetch():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        for attempt in range(4):  # FRED intermittently stalls; retry with backoff
            try:
                txt = requests.get(url, headers={"User-Agent": "curl/8.7.1"}, timeout=90).text
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"FRED did not respond for {series_id}")
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna().set_index("date")

    return _cached(f"fred_{series_id}", fetch, refresh)["value"].rename(series_id)


def bundesbank_monthly_yield(maturity_years: int, refresh: bool = False) -> pd.Series:
    """Month-average par yield (decimal) of German Federal securities at a constant
    maturity, from the Bundesbank term-structure estimates. 1-10y start 1972-09."""
    key = f"BBSIS/M.I.ZAR.ZI.EUR.S1311.B.A604.R{maturity_years:02d}XX.R.A.A._Z._Z.A"
    s = bundesbank_series(key, start="1960-01", refresh=refresh) / 100
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s.rename(f"Bund {maturity_years}y")


def french_international(file: str = "Ind_all.Dat", table: str = "Dollar", refresh: bool = False) -> pd.Series:
    """Monthly value-weight market return (decimal) from Kenneth French's international
    index files (MSCI data 1975-2006, Bloomberg after). Ind_all = all non-US developed
    countries in the dataset. table='Dollar' for USD returns, 'Local' for local currency."""

    def fetch():
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_International_Indices.zip"
        raw = requests.get(url, headers=UA, timeout=120).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            lines = z.read(file).decode("latin-1").splitlines()
        start = next(i for i, ln in enumerate(lines) if f"Value-Weight {table}" in ln)
        rows = []
        for ln in lines[start + 3 :]:
            cells = ln.split()
            if not cells or not (cells[0].isdigit() and len(cells[0]) == 6):
                break
            rows.append((cells[0], float(cells[1])))
        df = pd.DataFrame(rows, columns=["ym", "Mkt"])
        df.index = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
        return df[["Mkt"]] / 100

    return _cached(f"ffintl_{file.split('.')[0]}_{table}", fetch, refresh)["Mkt"].rename(f"FF intl {file} {table}")


# ---------------------------------------------------------------------------------------
# Bond returns from constant-maturity par yields
# ---------------------------------------------------------------------------------------


def _bond_dirty_price(coupon: np.ndarray, yld: np.ndarray, years: float, freq: int) -> np.ndarray:
    """Dirty price per 1 of face of bonds paying `coupon` (annual rate, per bond) `freq`
    times a year with `years` to maturity, discounted at `yld` (same compounding). Cash
    flows sit at the remaining coupon dates, so accrued interest is included."""
    n = years * freq
    periods = int(np.ceil(n - 1e-9))
    t = np.arange(1, periods + 1) - (periods - n)  # time (in periods) to each cash flow
    disc = (1 + yld[:, None] / freq) ** (-t[None, :])
    return (coupon / freq) * disc.sum(axis=1) + disc[:, -1]


def curve_slope(curve: dict[float, pd.Series], maturity: float) -> pd.Series:
    """Local slope of the curve at `maturity` (yield per year of maturity), from the segment
    below it — the part a bond rolls down as it ages; the shortest point uses the segment above.
    Same rule as design.rolldown."""
    ms = sorted(curve)
    i = ms.index(maturity)
    lo, hi = (ms[i - 1], maturity) if i > 0 else (maturity, ms[i + 1])
    return (curve[hi] - curve[lo]) / (hi - lo)


def par_bond_return(yields: pd.Series, maturity: float, freq: int = 1, slope: pd.Series | None = None) -> pd.Series:
    """Monthly total return of a constant-maturity par bond: at each month end buy a new
    bond at par (coupon = that month's yield), one month later reprice it with maturity
    shortened by 1/12, sell and roll. Exact repricing, so duration and convexity need no
    approximation. freq=1 matches Bunds (annual coupons), 2 matches Treasuries.

    The aged bond is priced at the yield of its *new* maturity, m − 1/12: the new m-year yield
    less `slope`/12 (slope = yield per year of maturity, see curve_slope). Without a slope it
    is priced at the m-year yield, which drops the roll-down — about 0.5–1%/yr on a 7-10 ladder
    when the curve is steep."""
    y = yields.dropna()
    y1s = y if slope is None else y - slope.reindex(y.index).fillna(0) / 12
    y0, y1 = y.shift(1).to_numpy()[1:], y1s.to_numpy()[1:]
    price = _bond_dirty_price(y0, y1, maturity - 1 / 12, freq)
    return pd.Series(price - 1, index=y.index[1:])


def ladder_return(curve: dict[float, pd.Series], weights: dict[float, float] | None = None, freq: int = 1,
                  full_curve: dict[float, pd.Series] | None = None) -> pd.Series:
    """Return of a ladder of constant-maturity par bonds: a weighted average of
    par_bond_return over several maturities, each with its roll-down. Mimics a maturity-bucket
    index (e.g. a 1-3y index ~ ladder of 1,2,3y; an all-maturity govt index ~ ladder across the
    curve). The slope is read off `full_curve` when given (so a 7y rung rolls down the 5-7y
    segment), else off the ladder's own points."""
    weights = weights or {m: 1 / len(curve) for m in curve}
    sc = full_curve or curve
    parts = {m: par_bond_return(curve[m], m, freq, slope=curve_slope(sc, m) if len(sc) > 1 else None) * w
             for m, w in weights.items()}
    return pd.concat(parts, axis=1).dropna().sum(axis=1)
