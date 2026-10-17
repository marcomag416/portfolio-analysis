import pandas as pd, numpy as np, design as D, pfutils as pf
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]
bonds,blong=D.bond_indexes(END)
print("eq", eq.shape, eq.index[0].date(), eq.apply(lambda s:s.first_valid_index().date()).value_counts().to_dict())
print("bonds", bonds.apply(lambda s:s.first_valid_index().date()).to_dict())
print("blong", blong.apply(lambda s:s.first_valid_index().date()).to_dict())
# validate US Treasury hedged vs real ETF IBTE? we have IUSM.DE (USD 7-10 unhedged in EUR) & SXRL (USD 3-7 Acc)
def etf(t): s=pf.drop_listing_period(pf.yahoo_close(t)); s=s.drop(pf.find_spikes(s)); return pf.month_end(s,END).pct_change(fill_method=None)
ust=D.ust_curve(END); fx=pf.month_end(pf.ecb_series("EXR/D.USD.EUR.SP00.A"),END)
u710=pf.ladder_return({7:ust[7],10:ust[10]}); u37=pf.ladder_return({3:ust[3],5:ust[5],7:ust[7]})
for n,r,t in [("UST 7-10 unhedged€",pf.to_eur(u710,fx),"IUSM.DE"),("UST 3-7 unhedged€",pf.to_eur(u37,fx),"SXRL.DE")]:
    x=pf.tracking(r.loc["2010":],etf(t).loc["2010":]); print(n,t,{k:(round(v,4) if isinstance(v,float) else v) for k,v in x.items()})
# DBZB global govt EUR hedged vs blend 50% UST hedged + 30% EUR All + 20% JGB? just compare to UST hedged & EUR all
for n in ["US Treasury (EUR hedged)","EUR Govt All"]:
    x=pf.tracking(bonds[n].loc["2010":],etf("DBZB.DE").loc["2010":]); print(n,"vs DBZB",{k:(round(v,4) if isinstance(v,float) else v) for k,v in x.items()})
print(D.optimal_block_length(pd.concat([eq["World"],bonds["EUR Govt All"]],axis=1)))
