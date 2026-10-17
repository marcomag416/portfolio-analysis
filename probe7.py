import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END)
eur1m=pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA")/100; eur1m.index=eur1m.index+pd.offsets.MonthEnd(0)
# Asset set: World (or ACWI IMI), EM IMI, EUR Govt 3-7 , Global govt hedged (blend)
glob_h=0.4*bonds["EUR Govt All"]+0.6*bonds["US Treasury (EUR hedged)"]
bund_ext=lambda ea,bu: pf.splice(bonds[ea], blong[bu])
A=pd.DataFrame({"World":eq["World"],"EM":eq["EM IMI"],"Quality":eq["World Quality"],"Momentum":eq["World Momentum"],
                "EUR Govt 3-7":bund_ext("EUR Govt 3-7","Bund 3-7"),"EUR Govt All":bund_ext("EUR Govt All","Bund 1-10")}).loc["1999-01":END].dropna()
rf=(eur1m/12).reindex(A.index).ffill()
print(A.index[0].date(), len(A))
def walk(assets, method, eq_floor=None, train=120):
    rets=[]; wlog=[]
    sub=A[assets]
    n=len(assets); bnds=[(0,1)]*n
    for i in range(train, len(sub), 12):
        tr=sub.iloc[i-train:i]; te=sub.iloc[i:i+12]
        cov=D.ledoit_wolf_cov(tr); mu=tr.mean().to_numpy(); r_f=rf.iloc[i-train:i].mean()
        if method=="1/N": w=np.full(n,1/n)
        elif method=="MinVar": w=D.w_min_var(cov,bnds)
        elif method=="MaxSharpe": w=D.w_max_sharpe(mu,cov,r_f,bnds)
        elif method=="ERC": w=D.w_erc(cov)
        elif method=="InvVol": w=D.w_inv_vol(cov)
        elif method=="MinCVaR": w=D.w_min_cvar(tr.to_numpy(),0.95,bnds)
        elif method=="Resampled": w=D.resampled_max_sharpe(tr,r_f,bnds,n_draws=60)
        elif method.startswith("Fixed"): w=np.array(FIX[method])
        wlog.append(pd.Series(w,index=assets,name=te.index[0]))
        rets.append((te*w).sum(axis=1))
    return pd.concat(rets), pd.DataFrame(wlog)
FIX={"Fixed 80/20":[0.8,0.2],"Fixed 70/30":[0.7,0.3],"Fixed 60/40":[0.6,0.4]}
res={}
for assets in [["World","EUR Govt All"],["World","EUR Govt 3-7"]]:
    for m in ["1/N","MinVar","MaxSharpe","ERC","InvVol","MinCVaR","Resampled","Fixed 80/20","Fixed 70/30"]:
        r,wl=walk(assets,m)
        s=pf.perf_stats(r,rf); res[(" + ".join(assets),m)]={"CAGR":s["CAGR"],"Vol":s["Volatility"],"Sharpe":s["Sharpe"],"MaxDD":s["Max drawdown"],"avg eq w":wl.iloc[:,0].mean(),"min eq w":wl.iloc[:,0].min(),"max eq w":wl.iloc[:,0].max(),"from":r.index[0].strftime("%Y-%m")}
print(pd.DataFrame(res).T.to_string())
