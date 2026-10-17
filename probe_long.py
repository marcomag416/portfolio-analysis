import pandas as pd, numpy as np, pfutils as pf
END = pd.Timestamp("2026-09-30")
dem = pf.fred_series("EXGEUS"); dem.index = dem.index + pd.offsets.MonthEnd(0)
usd_per_eur_pre = 1.95583 / dem
ecb = pf.month_end(pf.ecb_series("EXR/D.USD.EUR.SP00.A"), END)
fx = pd.concat([usd_per_eur_pre.loc[:"1998-12"], ecb.loc["1999-01":]])
ff = pf.french_factors("F-F_Research_Data_5_Factors_2x3")
us = ff["Mkt-RF"] + ff["RF"]; intl = pf.french_international()
us_eur = pf.to_eur(us, fx); intl_eur = pf.to_eur(intl, fx)
msci = pf.msci_index().resample("ME").last().pct_change()
for w in [0.5, 0.55, 0.6, 0.65, 0.7]:
    b = w * us_eur + (1 - w) * intl_eur
    t = pf.tracking(b.loc["2001":"2010"], msci.loc["2001":"2010"]); t2 = pf.tracking(b.loc["2016":], msci.loc["2016":])
    print(w, f"2001-10 TD {t['Tracking difference /yr']:+.2%} TE {t['Tracking error /yr']:.2%}   2016+ TD {t2['Tracking difference /yr']:+.2%} TE {t2['Tracking error /yr']:.2%}")
dev = pf.french_factors("Developed_5_Factors"); devm = pf.to_eur(dev["Mkt-RF"] + dev["RF"], fx)
t = pf.tracking(devm.loc["2001":], msci.loc["2001":]); print("FF Developed vs MSCI World EUR", t)
bund = {m: pf.bundesbank_monthly_yield(m) for m in range(1, 11)}
b13 = pf.ladder_return({m: bund[m] for m in [1, 2, 3]}); b710 = pf.ladder_return({m: bund[m] for m in [7, 8, 9, 10]})
print("Bund ladders from", b13.index[0].date(), "FF intl from", intl.index[0].date(), "fx from", fx.dropna().index[0].date())
eps = {"1973-74": ("1973-01", "1974-09"), "1980-82": ("1980-01", "1982-07"), "1987": ("1987-08", "1987-11"),
       "1990": ("1990-01", "1990-09"), "1994": ("1994-01", "1994-12"), "2000-03": ("2000-04", "2003-03"),
       "2007-09": ("2007-11", "2009-02"), "2022": ("2022-01", "2022-09")}
for n, (a, b) in eps.items():
    def c(r):
        x = r.loc[a:b].dropna()
        return round((1 + x).prod() - 1, 3) if len(x) else None
    print(n, "US€", c(us_eur), "Intl€", c(intl_eur), "Bund1-3", c(b13), "Bund7-10", c(b710))
