import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
pd.set_option("display.width",250); pd.set_option("display.max_columns",30); pd.set_option("display.max_rows",80)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc["2001-01":END]
eur1m=pf.ecb_series("FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA")/100; eur1m.index=eur1m.index+pd.offsets.MonthEnd(0); rf=(eur1m/12).reindex(eq.index).ffill()
rows={}
for c in eq:
    s=D.downside_stats(eq[c],rf); d,p=D.sharpe_hac_test(eq[c],eq["World"],rf)
    s["dSharpe vs World"]=d; s["p"]=p; s["corr World"]=eq[c].corr(eq["World"])
    rows[c]=s
T=pd.DataFrame(rows).T
# Deflated sharpe: N trials = 28, var of monthly SR across trials
srm=((eq.sub(rf,axis=0)).mean()/eq.std())
for c in eq: T.loc[c,"DSR"]=D.deflated_sharpe(eq[c],rf,len(eq.columns),srm.var())
# subperiods
for a,b in [("2001","2008"),("2009","2017"),("2018","2026")]:
    x=eq.loc[a:b]; T[f"SR {a}-{b[-2:]}"]=(x.sub(rf.loc[a:b],axis=0).mean()/x.std()*np.sqrt(12))
cols=["CAGR","Volatility","Sharpe","Sortino","Max drawdown","Calmar","Worst 12m","Longest underwater (m)","CVaR 95% (1m)","dSharpe vs World","p","DSR","corr World","SR 2001-08","SR 2009-17","SR 2018-26"]
print(T[cols].sort_values("Sharpe",ascending=False).round(3).to_string())
ct=D.crisis_table(eq)
print(ct.loc[["Dot-com 2000-03","GFC 2007-09","Euro crisis 2011","COVID 2020","Inflation 2022"]].T.round(3).sort_values("GFC 2007-09").to_string())
