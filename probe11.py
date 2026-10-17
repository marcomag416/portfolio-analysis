import warnings, time; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, design as D, pfutils as pf, yfinance as yf
pd.set_option("display.width",250)
END=pd.Timestamp("2026-09-30")
eq=D.equity_indexes().loc[:END]; bonds,blong=D.bond_indexes(END)
def etf(t):
    s=pf.drop_listing_period(pf.yahoo_close(t)); s=s.drop(pf.find_spikes(s))
    r=s.pct_change(); bad=r[r.abs()>0.5]
    if len(bad): s=s.loc[bad.index[-1]:]   # unadjusted split: keep only after it
    return pf.month_end(s,END).pct_change(fill_method=None)
glob=0.4*bonds["EUR Govt All"]+0.6*bonds["US Treasury (EUR hedged)"]
C={ "VWCE.DE":("IE00BK5BQT80","ACWI IMI"),"IUSQ.DE":("IE00B6R52259","ACWI"),"SPYY.DE":("IE00B44Z5B48","ACWI"),"SPYI.DE":("IE00B3YLTY66","ACWI IMI"),
    "WEBN.DE":("IE0003XJA0J9","ACWI"),"EUNL.DE":("IE00B4L5Y983","World"),"XDWD.DE":("IE00BJ0KDQ92","World"),"SPPW.DE":("IE00BFY0GT14","World"),
    "IS3N.DE":("IE00BKM4GZ66","EM IMI"),
    "VGEA.DE":("IE00BH04GL39","EUR Govt All"),"XGLE.DE":("LU0290355717","EUR Govt All"),"SXRP.DE":("IE00B3VTML14","EUR Govt 3-7"),
    "DBXP.DE":("LU0290356871","EUR Govt 1-3"),"SXRQ.DE":("IE00B3VTMJ91? (7-10)","EUR Govt 7-10"),
    "VDTE.DE":("IE00BMX0B631","US Treasury (EUR hedged)"),"IBB1.DE":("IE00BGPP6697","US Treasury (EUR hedged)"),
    "DBZB.DE":("LU0378818131","glob"),"GOVH.PA":("LU1708330235","glob")}
rows={}
for t,(isin,b) in C.items():
    try: r=etf(t)
    except Exception as e: print(t,"ERR",e); continue
    bench=glob if b=="glob" else (eq[b] if b in eq else bonds[b])
    x=pf.tracking(r,bench); rows[t]={"ISIN":isin,"bench":b,**x}
    time.sleep(0.2)
print(pd.DataFrame(rows).T.to_string())
