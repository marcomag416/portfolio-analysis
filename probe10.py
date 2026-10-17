import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END); L=D.long_equity_eur(END)
# 3-asset candidates since 2001 (MSCI) for satellite: EM IMI, Quality, Momentum, Min Vol, Small cap, Europe; compute historical & CMA-based mean-var with BL
eur1m=pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA")/100; eur1m.index=eur1m.index+pd.offsets.MonthEnd(0)
A=pd.concat([eq[["World","EM IMI","World Quality","World Momentum","World Min Vol","World Small Cap","Europe","USA"]], bonds["EUR Govt 3-7"]],axis=1).loc["2001-01":].dropna()
rf=(eur1m/12).reindex(A.index).ffill()
cov=D.ledoit_wolf_cov(A)
# Black-Litterman equilibrium for World+EM+Bonds with market weights (equity 85/15 World/EM? MSCI ACWI EM weight ~ 11%); bonds 30% of total
cols=["World","EM IMI","EUR Govt 3-7"]; idx=[list(A.columns).index(c) for c in cols]
C=cov[np.ix_(idx,idx)]*12
for wm in [np.array([0.62,0.08,0.30]),np.array([0.71,0.09,0.20])]:
    pi=D.black_litterman_mu(C,wm,delta=2.5)
    print("BL implied excess (annual)",dict(zip(cols,np.round(pi,4))), "-> max-Sharpe recovers", np.round(D.w_max_sharpe(pi,C,0,[(0,1)]*3),3))
# Using CMA (JPM 2026, EUR): World 6.3, EM 7.2, EUR govt 3.4 (3-7 ~3.1), cash 2.3
mu=np.array([0.063,0.072,0.031]); rfa=0.023
print("CMA max-Sharpe (LW cov 2001-26):", np.round(D.w_max_sharpe(mu,C,rfa,[(0,1)]*3),3))
for cap in [0.2,0.3]:
    print("CMA max-Sharpe with EM<=",cap, np.round(D.w_max_sharpe(mu,C,rfa,[(0,1),(0,cap),(0,1)]),3))
# Target-vol / target-drawdown: for the 2-asset world+bonds, what equity weight gives 10%/12% vol
for we in [0.6,0.7,0.8,0.9]:
    w=np.array([we,0,1-we]); print(we, "vol", round(np.sqrt(w@C@w),4))
print("corr matrix:\n", pd.DataFrame(np.corrcoef(A[cols].T),index=cols,columns=cols).round(2))
# EM addition: historic effect
for wem in [0,0.1,0.15]:
    p=(0.8-wem)*A["World"]+wem*A["EM IMI"]+0.2*A["EUR Govt 3-7"]
    s=pf.perf_stats(p,rf); print("EM",wem,{k:round(v,3) for k,v in s.items() if k in["CAGR","Volatility","Sharpe","Max drawdown"]})
print("ACWI IMI vs World dSharpe", D.sharpe_hac_test(eq["ACWI IMI"].loc["2001":],eq["World"].loc["2001":],rf))
