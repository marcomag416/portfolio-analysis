import pandas as pd, numpy as np, pfutils as pf
END=pd.Timestamp("2026-09-30")
def etf(t):
    s=pf.drop_listing_period(pf.yahoo_close(t)); s=s.drop(pf.find_spikes(s)); return pf.month_end(s,END).pct_change(fill_method=None)
ea={m:pf.month_end(pf.ecb_series(f"YC/B.U2.EUR.4F.G_N_C.SV_C_YM.PY_{m}Y"),END)/100 for m in [1,2,3,5,7,10,15,20,30]}
bd={m:pf.bundesbank_monthly_yield(m) for m in range(1,11)}
def lad(curve, ms, w=None): return pf.ladder_return({m:curve[m] for m in ms}, w)
tests=[("EA 1-3",lad(ea,[1,2,3]),"DBXP.DE"),("EA 3-7",lad(ea,[3,5,7]),"SXRP.DE"),("EA 7-10",lad(ea,[7,10]),"SXRQ.DE"),
       ("EA all (1..30)",lad(ea,[1,2,3,5,7,10,15,20,30],{1:.12,2:.12,3:.11,5:.17,7:.13,10:.15,15:.08,20:.07,30:.05}),"XGLE.DE"),
       ("EA all (1..30)",lad(ea,[1,2,3,5,7,10,15,20,30],{1:.12,2:.12,3:.11,5:.17,7:.13,10:.15,15:.08,20:.07,30:.05}),"VGEA.DE"),
       ("Bund 1-10",lad(bd,range(1,11)),"EXHA.DE"),("Bund 1-10",lad(bd,range(1,11)),"X03G.DE"),("Bund 1-3",lad(bd,[1,2,3]),"DBXP.DE")]
for n,r,t in tests:
    e=etf(t)
    x=pf.tracking(r.loc["2009":],e.loc["2009":])
    df=pd.concat([e,r],axis=1).dropna(); yr=(1+df).groupby(df.index.year).prod()-1; d=((yr.iloc[:,0]-yr.iloc[:,1])*100).round(1)
    print(f"{n:15} vs {t}: TD {x['Tracking difference /yr']:+.2%} TE {x['Tracking error /yr']:.2%} corr {x['Correlation']:.3f} worst yr gap {d.abs().max()} in {d.abs().idxmax()}")
