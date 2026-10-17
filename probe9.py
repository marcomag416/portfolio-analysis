import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END); L=D.long_equity_eur(END)
world=pf.splice(eq["World"].dropna(), L["World blend 60/40"])
S=pd.DataFrame({"World":world,"EUR Govt 1-3":pf.splice(bonds["EUR Govt 1-3"].dropna(),blong["Bund 1-3"]),
 "EUR Govt 3-7":pf.splice(bonds["EUR Govt 3-7"].dropna(),blong["Bund 3-7"]),"EUR Govt 7-10":pf.splice(bonds["EUR Govt 7-10"].dropna(),blong["Bund 7-10"])}).dropna()
H=120;N=5000
target={"World":0.063,"EUR Govt 1-3":0.027,"EUR Govt 3-7":0.031,"EUR Govt 7-10":0.035}
TAX=0.125
# Which bond bucket at 70/30 and 80/20, CMA mode, plus sensitivity by block and 2022-regime only
for bb in ["EUR Govt 1-3","EUR Govt 3-7","EUR Govt 7-10"]:
    X=S[["World",bb]]; P=D.stationary_bootstrap_paths(X,H,N,12,seed=2,shift=D.cma_shift(X,target))
    for we in [0.8,0.7]:
        r=D.pac_paths(P,np.array([we,1-we]),np.array([0.26,TAX])); at=r["after_tax"]
        print(bb,we,"median",round(np.median(at)),"p5",round(np.percentile(at,5)),"P<paid",round((at<r["paid"]).mean(),3),"p5DD",round(np.percentile(r["max_dd"],5),3), "CVaR5 €", round(at[at<=np.percentile(at,5)].mean()))
# sensitivity to block length
X=S[["World","EUR Govt 3-7"]]
for blk in [3,6,12,24]:
    P=D.stationary_bootstrap_paths(X,H,N,blk,seed=3,shift=D.cma_shift(X,target)); r=D.pac_paths(P,np.array([0.8,0.2]),np.array([0.26,TAX])); at=r["after_tax"]
    print("block",blk,"median",round(np.median(at)),"p5",round(np.percentile(at,5)),"P<paid",round((at<r["paid"]).mean(),3))
# inflation regime: bootstrap only from periods with positive stock-bond correlation (36m rolling >0)
rc=S["World"].rolling(36).corr(S["EUR Govt 3-7"])
reg=S[rc>0.2]
print("positive-corr regime months", len(reg), reg.index.min().date(), reg.index.max().date())
X2=reg[["World","EUR Govt 3-7"]]
P=D.stationary_bootstrap_paths(X2,H,N,6,seed=4,shift=D.cma_shift(X2,target))
for we in [1.0,0.8,0.7,0.6]:
    r=D.pac_paths(P,np.array([we,1-we]),np.array([0.26,TAX])); at=r["after_tax"]
    print("pos-corr regime",we,"median",round(np.median(at)),"p5",round(np.percentile(at,5)),"P<paid",round((at<r["paid"]).mean(),3),"p5DD",round(np.percentile(r["max_dd"],5),3))
