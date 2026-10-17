import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf, statsmodels.api as sm
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END)
eur1m=pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA")/100; eur1m.index=eur1m.index+pd.offsets.MonthEnd(0)
rf_all=(eur1m/12)
B=bonds.loc["2004-10":END]; W=eq["World"].reindex(B.index); rf=rf_all.reindex(B.index).ffill()
rows={}
for c in B:
    s=D.downside_stats(B[c],rf); s["corr World"]=B[c].corr(W); s["corr World (down mo)"]=B[c][W<0].corr(W[W<0])
    s["avg ret when World<-4%"]=B[c][W<-0.04].mean()
    for wb in [0.2,0.3]:
        p=(1-wb)*W+wb*B[c]; ps=pf.perf_stats(p,rf); s[f"{int((1-wb)*100)}/{int(wb*100)} Sharpe"]=ps["Sharpe"]; s[f"{int((1-wb)*100)}/{int(wb*100)} MaxDD"]=ps["Max drawdown"]
    rows[c]=s
p=0.8*W+0.2*rf; ps=pf.perf_stats(p,rf); rows["Cash (Euribor 1m)"]={"80/20 Sharpe":ps["Sharpe"],"80/20 MaxDD":ps["Max drawdown"],"CAGR":(1+rf).prod()**(12/len(rf))-1}
T=pd.DataFrame(rows).T
print(T[["CAGR","Volatility","Sharpe","Max drawdown","Worst 12m","corr World","corr World (down mo)","avg ret when World<-4%","80/20 Sharpe","80/20 MaxDD","70/30 Sharpe","70/30 MaxDD"]].round(3).to_string())
print(D.crisis_table(pd.concat([B,W],axis=1)).round(3).dropna(how="all").to_string())
# long history Bunds vs World blend
L=D.long_equity_eur(END); BL=blong.reindex(L.index)
rfL=(pf.bundesbank_monthly_yield(1)/12).reindex(L.index).ffill()
print("long window", L.index[0].date(), L.index[-1].date())
for c in BL:
    print(c, {k:round(v,3) for k,v in pf.perf_stats(BL[c],rfL).items() if k in ["CAGR","Volatility","Sharpe","Max drawdown"]}, "corr", round(BL[c].corr(L["World blend 60/40"]),2))
ct=D.crisis_table(pd.concat([L,BL],axis=1)); print(ct.round(3).to_string())
# rolling 36m correlation stats by decade
rc=L["World blend 60/40"].rolling(36).corr(BL["Bund 7-10"]); print(rc.groupby(rc.index.year//10*10).mean().round(2))
