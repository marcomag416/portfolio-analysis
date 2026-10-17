import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END)
L=D.long_equity_eur(END)
eur1m=pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA")/100; eur1m.index=eur1m.index+pd.offsets.MonthEnd(0)
# Long sample for bootstrap: equity = World blend (FF) pre-2001 then MSCI World; bonds = Bund pre-2004-10 then EA
world=pf.splice(eq["World"].dropna(), L["World blend 60/40"])
b37=pf.splice(bonds["EUR Govt 3-7"].dropna(), blong["Bund 3-7"]); ball=pf.splice(bonds["EUR Govt All"].dropna(), blong["Bund 1-10"])
S=pd.DataFrame({"World":world,"EUR Govt 3-7":b37,"EUR Govt All":ball}).dropna()
print("bootstrap sample", S.index[0].date(), S.index[-1].date(), len(S))
from arch.bootstrap import optimal_block_length as obl
print(obl(S)); print(obl(S**2))
H=120; N=5000; block=12
TAX={"World":0.26,"EUR Govt 3-7":0.125,"EUR Govt All":0.125}
# CMA re-centring: target annual nominal EUR returns (JPM LTCMA 2026): World eq 6.3%, Euro govt 3.4% ; 3-7 ~3.1%
target={"World":0.063,"EUR Govt 3-7":0.031,"EUR Govt All":0.034}
for mode in ["historical","CMA"]:
    cols=["World","EUR Govt 3-7"]
    X=S[cols]
    shift=None
    if mode=="CMA":
        hist=np.log1p(X).mean().to_numpy()*12
        shift=D.cma_shift(X,target)
    P=D.stationary_bootstrap_paths(X,H,N,block,seed=1,shift=shift)
    out={}
    for we in [1.0,0.9,0.8,0.7,0.6,0.5]:
        w=np.array([we,1-we]); tr=np.array([TAX[c] for c in cols])
        res=D.pac_paths(P,w,tr)
        at=res["after_tax"]; irr=D.irr_annual(at,1000,100,H)
        out[f"{int(we*100)}/{int(round((1-we)*100))}"]={"median €":np.median(at),"p5 €":np.percentile(at,5),"p95 €":np.percentile(at,95),
            "P(<paid)":(at<res["paid"]).mean(),"median IRR":np.median(irr),"p5 IRR":np.percentile(irr,5),"median maxDD":np.median(res["max_dd"]),"p5 maxDD":np.percentile(res["max_dd"],5)}
    print(mode); print(pd.DataFrame(out).T.round(3).to_string())
