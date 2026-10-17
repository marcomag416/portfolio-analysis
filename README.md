# Portfolio analysis

Two notebooks on EUR total-return data, for an Italian retail investor paying €1,000 up front
plus €100/month into a Trade Republic account (26% equity / 12.5% white-list government bond
tax, 0.2% *imposta di bollo*, free savings plans, €1 on one-off orders only).

```sh
uv sync                                    # install locked dependencies
uv run jupyter lab portfolio_choice.ipynb
```

## `portfolio_choice.ipynb` — choosing the portfolio (start here)

Six steps, each narrowing what the next considers. No index is treated as the default.

- **§1** Buyable universe: only indexes with a UCITS ETF on Trade Republic. Each fund's index is
  identified from its returns (the "World Quality" and "World Value" ETFs track the Sector Neutral and
  Enhanced variants), and its **real cost is measured** as the trend of its gap to the index, which
  nets the TER against lending and the withholding-tax treaty rate.
- **§2** Screen of 12 buyable indexes after cost; pairwise Sharpe tests with Holm correction; crises;
  which properties persist across 2001–13 / 2014–26 (risk does, risk-adjusted return reverses);
  factor indexes backtest vs live.
- **§3** Shortlist: ACWI IMI, World, World Min Vol, World Momentum, S&P 500 — chosen on persistent
  properties (momentum carried as a bet to be priced, not a finding).
- **§4** Bonds judged as partners to each equity sleeve, on today's yields after cost (with the full curve
  and the roll-down a steep curve adds), the hedging cost of Treasuries stress-tested, and an
  inflation-regime subsample from 1975.
- **§5** Weights: ratio optima under resampling, crisis replay, simulation of the plan after Italian
  tax under three return scenarios (same return / factors keep their live record / priced by beta),
  stress tests, every real 10-year window, a risk-budget table, and a check that the ranking does not
  depend on measuring risk by drawdown (seven other measures and a certainty equivalent).
- **§6** Real ETFs, each cost layer over the plan, the options side by side, CAGR / volatility / alpha /
  beta per portfolio and per sleeve, the outcome split by sleeve, and a bond roll-down sensitivity.
- **§7** The portfolio at the investor's risk level (30% bad-case drawdown): 24 candidates ranked on their
  worst-scenario median; recommended World + euro govt 7-10, 50/50, and how to run it.

## `portfolio_multifactor.ipynb` — World vs a multifactor blend, and JPGL

Same investor and risk level as `portfolio_choice` §7 (30% bad-case drawdown).

- **§1–2** The four buyable MSCI World factor sleeves (momentum, quality SN, enhanced value, min vol),
  JPGL (IE00BJRCLL96) and the S&P 500, each charged its TER less one pooled lending/withholding-tax offset;
  JPGL's pre-2019 history is the MSCI mix that best fits its live returns. History, crises, backtest vs live
  alpha (against MSCI World, Newey-West t-stats).
- **§3** Expected returns: the three scenarios plus a **Monte Carlo with uncertain expected returns**
  (market ± 1.5 pts, each factor alpha shrunk towards zero by its own noise and drawn jointly with the
  others, bonds ± 0.5 pts), drawn per path.
- **§4** World vs the 4 × 25% blend, each dialled to the risk level: medians per scenario, path-by-path
  comparison, every real 10-year window, drift and per-sleeve tax.
- **§5** Moving part of the bond sleeve into JPGL, naively and at constant risk.
- **§6** The answers.
- **§7** Single-fund multifactor ETFs (HSBC, iShares, Invesco, Avantis) vs four savings plans; every factor mix
  at the risk level; sector and region exposure (holdings and returns-based) and whether it protects in crises.
- **§8** The bond sleeve revisited: every euro govt maturity (1-3 … 7-10, middle ladders, barbell), hedged
  Treasuries and cash at the same risk; real funds incl. linkers and aggregates; fixed rate paths inside the plan;
  and on simulated euro yield curves, a constant-duration fund vs bonds held to maturity, accumulating vs
  distributing, and a maturity glide path.

## Files

| File | What |
|---|---|
| `portfolio_choice.ipynb` / `portfolio_multifactor.ipynb` | The analyses, executed, with outputs |
| `build_choice_notebook.py` / `build_multifactor_notebook.py` | **Source of the notebooks.** Edit these, not the `.ipynb` |
| `etfs.py` | Trade Republic ETF universe: index tracked, ticker, ISIN, TER, domicile. Unverified fields are `None` |
| `design.py` | Index universe, screening statistics, cost measurement, weighting methods, bootstrap simulation |
| `pfutils.py` | Data loaders (Yahoo, MSCI, ECB, Bundesbank, FRED, Fama–French) with parquet cache; stats; PAC simulator |
| `dump_outputs.py` | Prints an executed notebook's tables as flat text |
| `data/` | Download cache. Delete `data/*.parquet` to refresh |

Rebuild and re-execute:

```sh
uv run python build_choice_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 portfolio_choice.ipynb
uv run python build_multifactor_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 portfolio_multifactor.ipynb
```

## Caveats

- Bond indexes before 2004-10 and equity before 2001 are **reconstructions and splices**, not
  official index history. JPGL before 2019-11 is a fitted proxy.
- Forward-looking simulations re-centre historical returns on published capital-market
  assumptions. They are scenario generators, not forecasts.

Not financial advice.
