import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END); L=D.long_equity_eur(END)
world=pf.splice(eq["World"].dropna(), L["World blend 60/40"])
S=pd.DataFrame({"Equity":world,"EUR Govt All":pf.splice(bonds["EUR Govt All"].dropna(),blong["Bund 1-10"]),
                "EUR Govt 3-7":pf.splice(bonds["EUR Govt 3-7"].dropna(),blong["Bund 3-7"]),
                "UST hedged":bonds["US Treasury (EUR hedged)"]}).loc[:END]
X=S[["Equity","EUR Govt 3-7"]].dropna()
W,starts=D.historical_windows(X,10)
print("historical windows", len(W), starts[0].date(), starts[-1].date())
for we in [1.0,0.9,0.8,0.7,0.6]:
    r=D.pac_paths_rule(W,np.array([we,1-we]),np.array([0.26,0.125]),"split")
    s=D.outcome_summary(r,120); worst=starts[np.argmin(r["after_tax"])]
    print(we, {k:round(v,3) for k,v in s.items()}, "worst start", worst.date(), "drift", round(np.median(r["worst_drift"]),3))
# split vs steer on bootstrap CMA
target={"Equity":0.063,"EUR Govt 3-7":0.031}
P=D.stationary_bootstrap_paths(X,120,4000,12,seed=5,shift=D.cma_shift(X,target))
for rule in ["split","steer"]:
    r=D.pac_paths_rule(P,np.array([0.8,0.2]),np.array([0.26,0.125]),rule); print(rule,{k:round(v,3) for k,v in D.outcome_summary(r,120).items()}, "median worst drift", round(np.median(r.get("worst_drift",np.array([np.nan]))),3))
# bond mix: EA 3-7 vs 50/50 with UST hedged (sample from 1999-02)
X3=S.dropna()
print("3-asset sample from", X3.index[0].date())
tg={"Equity":0.063,"EUR Govt All":0.034,"EUR Govt 3-7":0.031,"UST hedged":0.034}
P3=D.stationary_bootstrap_paths(X3[["Equity","EUR Govt 3-7","UST hedged"]],120,4000,12,seed=6,shift=D.cma_shift(X3[["Equity","EUR Govt 3-7","UST hedged"]],tg))
for w in [[0.8,0.2,0],[0.8,0.1,0.1],[0.8,0,0.2]]:
    r=D.pac_paths_rule(P3,np.array(w),np.array([0.26,0.125,0.125]),"split"); print(w,{k:round(v,3) for k,v in D.outcome_summary(r,120).items()})
