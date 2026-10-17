import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf
END=pd.Timestamp("2026-09-30")
bonds,blong=D.bond_indexes(END)
def etf(t): s=pf.drop_listing_period(pf.yahoo_close(t)); s=s.drop(pf.find_spikes(s)); return pf.month_end(s,END).pct_change(fill_method=None)
dbzb=etf("DBZB.DE")
for w in [0.3,0.4,0.5]:
    bl=w*bonds["EUR Govt All"]+(1-w)*bonds["US Treasury (EUR hedged)"]
    t=pf.tracking(bl.loc["2009":],dbzb.loc["2009":]); print("blend EUR",w,{k:(round(v,4) if isinstance(v,float) else v) for k,v in t.items()})
# Bund splice check for EA buckets pre-2004: compare EA vs Bund 2004-10..2008-12
for ea,bu in [("EUR Govt 1-3","Bund 1-3"),("EUR Govt 3-7","Bund 3-7"),("EUR Govt 7-10","Bund 7-10"),("EUR Govt All","Bund 1-10")]:
    t=pf.tracking(blong[bu].loc["2004-10":"2008-06"],bonds[ea].loc["2004-10":"2008-06"]); print(ea,bu,{k:(round(v,4) if isinstance(v,float) else v) for k,v in t.items()})
print("annual vol: EA all", bonds["EUR Govt All"].std()*12**.5, "Bund1-10", blong["Bund 1-10"].loc["2004-10":].std()*12**.5)
# FF developed vs world 1990-2000
L=D.long_equity_eur(END)
ff=pf.french_factors("Developed_5_Factors")
