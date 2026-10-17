import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]
eur1m=pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA")/100; eur1m.index=eur1m.index+pd.offsets.MonthEnd(0); rf=(eur1m/12)
LAUNCH={"World Momentum":"2013-12","World Quality":"2012-12","World Multi-Factor":"2015-03","World Enh Value":"2014-08","ACWI Min Vol":"2009-11","World Min Vol":"2008-06","World SN Quality":"2018-01"}
rows={}
for k,l in LAUNCH.items():
    for lab,(a,b) in {"pre-launch (backtest)":(None,l),"live":(l,None)}.items():
        x=eq[k].loc[a:b].dropna(); w=eq["World"].reindex(x.index); r=rf.reindex(x.index).ffill()
        ex=(1+x).prod()**(12/len(x))-(1+w).prod()**(12/len(w))
        d,p=D.sharpe_hac_test(x,w,r)
        rows[(k,lab)]={"from":x.index[0].strftime("%Y-%m"),"months":len(x),"excess CAGR vs World":ex,"dSharpe":d,"p":p}
print(pd.DataFrame(rows).T.to_string())
