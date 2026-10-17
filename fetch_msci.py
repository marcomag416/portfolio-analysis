import time, pandas as pd, pfutils as pf
CODES = {990100:"World",892400:"ACWI",664204:"ACWI IMI",664185:"World IMI",891800:"EM",664220:"EM IMI",990500:"Europe",106400:"EMU",
 984000:"USA",939200:"Japan",991400:"Pacific ex JP",106230:"World Small Cap",703755:"World Momentum",702787:"World Quality",
 705169:"World SN Quality",129896:"World Min Vol (USD)",700491:"World Min Vol (EUR)",105868:"World Value",705130:"World Enh Value",
 105867:"World Growth",129857:"World Equal Wt",136064:"World High Div",106803:"World IT",106801:"World Health Care",991000:"World ex USA",
 106229:"USA Small Cap",106233:"Europe Small Cap",706536:"World Multi-Factor",700404:"ACWI Min Vol"}
for c,n in CODES.items():
    path = pf.DATA / f"msci_{c}_EUR_NETR.parquet"
    if path.exists(): continue
    try:
        s = pf.msci_index(code=c); print(c, n, s.index[0].date(), round(s.iloc[-1],1))
    except Exception as e:
        print(c, n, "ERR", type(e).__name__, str(e)[:80])
    time.sleep(3)
print("done")
