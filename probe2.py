import pandas as pd, numpy as np, design as D, pfutils as pf
END=pd.Timestamp("2026-09-30")
s=pf.yahoo_close("SXRL.DE"); print(s.pct_change().abs().nlargest(5))
r=pf.month_end(s,END).pct_change(); print(r.abs().nlargest(5))
from arch.bootstrap import optimal_block_length as obl
eq=D.equity_indexes().loc[:END]; bonds,_=D.bond_indexes(END)
x=pd.concat([eq["World"],bonds["EUR Govt All"],bonds["EUR Govt 1-3"]],axis=1).dropna()
print(obl(x)); print(obl(x**2))
