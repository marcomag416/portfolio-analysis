"""Trade Republic ETF universe for portfolio_choice.ipynb.

One record per fund. `index` is the key of the return series the fund tracks in this project
(design.equity_indexes / other_indexes / the bond rebuilds). The index each fund tracks was
confirmed from its returns and the issuer, not its name: both "World Quality" ETFs track MSCI
World Quality *Sector Neutral* and both "World Value" ETFs track *Enhanced* Value; the
Xtrackers pair switched to those indexes on 2016-11-03, so they are measured from 2016-12.

Static fields from justETF, cross-checked field by field against extraETF; TER conflicts
resolved on the issuer's site (research run 2026-10-09). Every Yahoo ticker was checked to
return the right fund.

`tr_plan` — Trade Republic does not publish its savings-plan list, so this is third-party:
  "Y"   listed by both Italian broker tables (justETF IT, extraETF IT) and the German lists
  "Y*"  listed by justETF Italy only (extraETF Italy omits many newer funds)
Confirm in the app before buying.

FTSE funds are mapped to their nearest MSCI index (All-World ~ ACWI, Global All Cap ~ ACWI
IMI); their measured "cost" then includes the index difference, so it is labelled.

`start` drops the top of a Yahoo history where it is not usable: listing placeholders
(IQQ0 2013-03/04), the pre-2016 index of the Xtrackers factor funds, the pre-2011 S&P 500 /
Nasdaq series, and isolated bad months — single-month fund-vs-index gaps of 25-33%, which no
physically replicated fund produces (XMEU 2009-01, XXSC 2009-01..04, DBXJ 2013-03, SXR1
2010-11, SC0J 2010-05, UETW 2020-02/03). With the start moved past them, correlation with the
index rises to 0.96-0.997.
"""

FIELDS = ["index", "ticker", "isin", "name", "ter", "acc", "dom", "aum_bn", "tr_plan", "start", "repl"]

EQUITY = [
    # Broad
    ("World", "UETW.DE", "IE00BD4TXV59", "UBS Core MSCI World", 0.0006, True, "IE", 11.6, "Y", "2020-06", "Physical"),
    ("World", "XDWD.DE", "IE00BJ0KDQ92", "Xtrackers MSCI World 1C", 0.0012, True, "IE", 22.1, "Y", None, "Physical"),
    ("World", "EUNL.DE", "IE00B4L5Y983", "iShares Core MSCI World", 0.0020, True, "IE", 133.4, "Y", None, "Physical"),
    ("World", "SPPW.DE", "IE00BFY0GT14", "SPDR MSCI World", 0.0012, True, "IE", 18.7, "Y*", None, "Physical"),
    ("World", "MWRE.DE", "IE000BI8OT95", "Amundi Core MSCI World", 0.0012, True, "IE", 15.7, "Y*", None, "Physical"),
    ("World", "SC0J.DE", "IE00B60SX394", "Invesco MSCI World", 0.0005, True, "IE", 9.4, "Y", "2010-06", "Swap"),
    ("ACWI", "SPYY.DE", "IE00B44Z5B48", "SPDR MSCI ACWI", 0.0012, True, "IE", 18.1, "Y", None, "Physical"),
    ("ACWI", "IUSQ.DE", "IE00B6R52259", "iShares MSCI ACWI", 0.0020, True, "IE", 32.2, "Y", None, "Physical"),
    ("ACWI", "VWCE.DE", "IE00BK5BQT80", "Vanguard FTSE All-World (FTSE index)", 0.0014, True, "IE", 55.1, "Y", None, "Physical"),
    ("ACWI", "FWIA.DE", "IE000716YHJ7", "Invesco FTSE All-World (FTSE index)", 0.0015, True, "IE", 4.3, "Y", None, "Physical"),
    ("ACWI", "ALLW.DE", "IE000L6ZMMC4", "Xtrackers FTSE All-World 1C (FTSE index)", 0.0007, True, "IE", 0.15, "Y*", None, "Physical"),
    ("ACWI IMI", "SPYI.DE", "IE00B3YLTY66", "SPDR MSCI ACWI IMI", 0.0017, True, "IE", 8.3, "Y", None, "Physical"),
    ("ACWI IMI", "VGLA.DE", "IE000VAHT5T0", "Vanguard FTSE Global All Cap (FTSE index)", 0.0007, True, "IE", 2.6, "Y*", None, "Physical"),
    ("World IMI", "WIMI.DE", "IE000X1GW0A7", "Xtrackers MSCI World IMI 1C", 0.0015, True, "IE", 0.02, "Y*", None, "Physical"),
    # Regions
    ("USA", "XD9U.DE", "IE00BJ0KDR00", "Xtrackers MSCI USA 1C", 0.0003, True, "IE", 12.3, "Y", None, "Physical"),
    ("USA", "SXR4.DE", "IE00B52SFT06", "iShares MSCI USA", 0.0003, True, "IE", 4.7, "Y", None, "Physical"),
    ("S&P 500", "SXR8.DE", "IE00B5BMR087", "iShares Core S&P 500", 0.0007, True, "IE", 141.4, "Y", "2011-01", "Physical"),
    ("S&P 500", "VUAA.DE", "IE00BFMXXD54", "Vanguard S&P 500 Acc", 0.0007, True, "IE", 33.4, "Y", None, "Physical"),
    ("S&P 500", "SPYL.DE", "IE000XZSV718", "SPDR S&P 500 Acc", 0.0003, True, "IE", 17.6, "Y", None, "Physical"),
    ("S&P 500", "P500.DE", "IE00B3YCGJ38", "Invesco S&P 500", 0.0005, True, "IE", 39.5, "Y", "2011-01", "Swap"),
    ("Nasdaq-100", "SXRV.DE", "IE00B53SZB19", "iShares Nasdaq 100", 0.0030, True, "IE", 27.0, "Y", "2011-01", "Physical"),
    ("Nasdaq-100", "XNAS.DE", "IE00BMFKG444", "Xtrackers Nasdaq 100 1C", 0.0020, True, "IE", 2.7, "Y", None, "Physical"),
    ("Nasdaq-100", "EQQB.DE", "IE00BFZXGZ54", "Invesco EQQQ Nasdaq-100 Acc", 0.0030, True, "IE", 4.5, "Y", None, "Physical"),
    ("Nasdaq-100", "BCFP.DE", "IE000SB4G4I4", "UBS Nasdaq-100", 0.0013, True, "IE", 0.52, "Y*", None, "Physical"),
    ("Europe", "EUNK.DE", "IE00B4K48X80", "iShares Core MSCI Europe", 0.0012, True, "IE", 15.7, "Y", None, "Physical"),
    ("Europe", "XMEU.DE", "LU0274209237", "Xtrackers MSCI Europe 1C", 0.0012, True, "LU", 7.6, "Y", "2009-06", "Physical"),
    ("Europe", "SPYE.DE", "IE00BKWQ0Q14", "SPDR MSCI Europe", 0.0008, True, "IE", 0.39, "Y", None, "Physical"),
    ("EMU", "SXR7.DE", "IE00B53QG562", "iShares Core MSCI EMU", 0.0012, True, "IE", 6.0, "Y", None, "Physical"),
    ("EMU", "ZPRE.DE", "IE00B910VR50", "SPDR MSCI EMU", 0.0008, True, "IE", 0.35, "Y", None, "Physical"),
    ("Japan", "SXR5.DE", "IE00B53QDK08", "iShares MSCI Japan", 0.0012, True, "IE", 1.6, "Y", None, "Physical"),
    ("Japan", "DBXJ.DE", "LU0274209740", "Xtrackers MSCI Japan 1C", 0.0012, True, "LU", 6.5, "Y", "2015-01", "Physical"),
    ("Pacific ex Japan", "SXR1.DE", "IE00B52MJY50", "iShares Core MSCI Pacific ex-Japan", 0.0020, True, "IE", 3.6, "Y", "2011-01", "Physical"),
    ("Pacific ex Japan", "W450.DE", "IE000SGVQIZ9", "HSBC MSCI Pacific ex Japan", 0.0015, True, "IE", 0.52, "Y*", None, "Physical"),
    ("World ex USA", "EXUS.DE", "IE0006WW1TQ4", "Xtrackers MSCI World ex USA 1C", 0.0015, True, "IE", 7.0, "Y", None, "Physical"),
    ("EM", "EUNM.DE", "IE00B4L5YC18", "iShares MSCI EM", 0.0018, True, "IE", 9.6, "Y", None, "Physical"),
    ("EM", "XMME.DE", "IE00BTJRMP35", "Xtrackers MSCI EM 1C", 0.0018, True, "IE", 12.8, "Y", None, "Physical"),
    ("EM", "H4Z3.DE", "IE000KCS7J59", "HSBC MSCI EM", 0.0015, True, "IE", 1.7, "Y", None, "Physical"),
    ("EM", "EMSM.DE", "IE00B3DWVS88", "Invesco MSCI EM", 0.0009, True, "IE", 0.46, "Y", None, "Swap"),
    ("EM IMI", "IS3N.DE", "IE00BKM4GZ66", "iShares Core MSCI EM IMI", 0.0018, True, "IE", 40.9, "Y", None, "Physical"),
    # Size
    ("World Small Cap", "IUSN.DE", "IE00BF4RFH31", "iShares MSCI World Small Cap", 0.0035, True, "IE", 7.9, "Y", None, "Physical"),
    ("World Small Cap", "ZPRS.DE", "IE00BCBJG560", "SPDR MSCI World Small Cap", 0.0045, True, "IE", 1.7, "Y", None, "Physical"),
    ("World Small Cap", "XSWC.DE", "IE000F354Q61", "Xtrackers MSCI World Small Cap 1C", 0.0025, True, "IE", 0.14, "Y*", None, "Physical"),
    ("Europe Small Cap", "XXSC.DE", "LU0322253906", "Xtrackers MSCI Europe Small Cap 1C", 0.0030, True, "LU", 3.2, "Y", "2009-06", "Physical"),
    # Factors
    ("World Min Vol", "XDEB.DE", "IE00BL25JN58", "Xtrackers MSCI World Minimum Volatility 1C", 0.0025, True, "IE", 1.21, "Y", None, "Physical"),
    ("World Min Vol", "IQQ0.DE", "IE00B8FHGS14", "iShares Edge MSCI World Minimum Volatility", 0.0030, True, "IE", 2.34, "Y", "2013-06", "Physical"),
    ("World Quality SN", "IS3Q.DE", "IE00BP3QZ601", "iShares Edge MSCI World Quality Factor", 0.0025, True, "IE", 5.9, "Y", None, "Physical"),
    ("World Quality SN", "XDEQ.DE", "IE00BL25JL35", "Xtrackers MSCI World Quality 1C", 0.0025, True, "IE", 2.6, "Y", "2016-12", "Physical"),
    ("World Momentum", "IS3R.DE", "IE00BP3QZ825", "iShares Edge MSCI World Momentum Factor", 0.0025, True, "IE", 5.8, "Y", None, "Physical"),
    ("World Momentum", "XDEM.DE", "IE00BL25JP72", "Xtrackers MSCI World Momentum 1C", 0.0025, True, "IE", 2.1, "Y", None, "Physical"),
    ("World Enhanced Value", "IS3S.DE", "IE00BP3QZB59", "iShares Edge MSCI World Value Factor", 0.0025, True, "IE", 6.8, "Y", None, "Physical"),
    ("World Enhanced Value", "XDEV.DE", "IE00BL25JM42", "Xtrackers MSCI World Value 1C", 0.0025, True, "IE", 4.1, "Y", "2016-12", "Physical"),
    ("World Equal Weight", "MWEQ.DE", "IE000OEF25S1", "Invesco MSCI World Equal Weight", 0.0020, True, "IE", 1.8, "Y", None, "Physical"),
]

BOND = [
    # tr_plan for bonds comes from German lists only (justETF DE, brokerexperte): "DE2" = both, "DE" = one.
    ("EUR Govt 1-3", "DBXP.DE", "LU0290356871", "Xtrackers II Eurozone Government Bond 1-3 1C", 0.0010, True, "LU", 3.6, "DE2", None, "Physical"),
    ("EUR Govt 1-3", "SXRN.DE", "IE00B3VTMJ91", "iShares € Govt Bond 1-3yr", 0.0010, True, "IE", 2.5, "DE", None, "Physical"),
    ("EUR Govt 3-7", "SXRP.DE", "IE00B3VTML14", "iShares € Govt Bond 3-7yr", 0.0015, True, "IE", 1.37, "DE", None, "Physical"),
    ("EUR Govt 3-7", "DBXQ.DE", "LU0290356954", "Xtrackers II Eurozone Government Bond 3-5 1C (3-5 yr)", 0.0015, True, "LU", 0.56, "DE", None, "Physical"),
    ("EUR Govt 7-10", "X710.DE", "LU0290357259", "Xtrackers II Eurozone Government Bond 7-10 1C", 0.0015, True, "LU", 1.46, "DE", None, "Physical"),
    ("EUR Govt 7-10", "SXRQ.DE", "IE00B3VTN290", "iShares € Govt Bond 7-10yr", 0.0015, True, "IE", 0.87, "DE", None, "Physical"),
    ("EUR Govt All", "VGEA.DE", "IE00BH04GL39", "Vanguard EUR Eurozone Government Bond Acc", 0.0007, True, "IE", 4.05, "DE2", None, "Physical"),
    ("EUR Govt All", "XGLE.DE", "LU0290355717", "Xtrackers II Eurozone Government Bond 1C", 0.0007, True, "LU", 2.49, "DE", None, "Physical"),
    ("EUR Govt All", "PRAR.DE", "LU2089238898", "Amundi Prime Euro Govies Acc", 0.0005, True, "LU", 0.72, "DE", None, "Physical"),
    ("UST 1-3 (EUR hedged)", "2B7S.DE", "IE00BDFK1573", "iShares $ Treasury Bond 1-3yr EUR Hedged Acc", 0.0010, True, "IE", 1.35, "DE", None, "Physical"),
    ("UST 7-10 (EUR hedged)", "7USH.DE", "LU1407888137", "Amundi US Treasury Bond 7-10Y EUR Hedged Acc", 0.0010, True, "LU", 0.65, "DE", None, "Physical"),
    ("UST 7-10 (EUR hedged)", "CEMF.DE", "IE000K1VI152", "iShares $ Treasury Bond 7-10yr EUR Hedged Acc", 0.0010, True, "IE", 0.15, "DE2", None, "Physical"),
    ("UST All (EUR hedged)", "VDTE.DE", "IE00BMX0B631", "Vanguard USD Treasury Bond EUR Hedged Acc", 0.0008, True, "IE", 0.63, "DE", None, "Physical"),
]

# Share of each fund in white-list government bonds, as certified by the issuer for Italian tax
# (Rebalix, data 2026-10-01/02). Only that share of a gain is taxed at the 12.5% rate; the rest
# at 26%. The figure is set per fund, so EUR-hedged share classes carry the same percentage as
# unhedged ones (iShares 1-3: IBTS and IBTE both 92.02%). Missing = not verified.
WHITE_LIST = {
    "LU0290356871": 0.9764, "IE00B3VTML14": 0.9587, "IE00BH04GL39": 0.9954, "LU2089238898": 0.9990,
    "IE00BDFK1573": 0.9202, "LU1407888137": 0.9970, "IE000K1VI152": 0.9413, "IE00BMX0B631": 0.9975,
}

# Exposures in MSCI's catalogue that no UCITS ETF on Trade Republic tracks.
NOT_INVESTABLE = {
    "ACWI Min Vol": "No UCITS ETF. The US-listed ACWV cannot be bought by EU retail (PRIIPs/KID).",
    "World Quality": "Every 'World Quality' ETF tracks the Sector Neutral variant (see World Quality SN).",
    "World Value": "Every 'World Value' ETF tracks Enhanced Value or another variant (see World Enhanced Value).",
    "World Growth": "No UCITS ETF.",
    "World Multi-Factor": "No UCITS ETF since iShares moved IBCZ to a STOXX index on 2025-01-22.",
    "USA Small Cap": "Only screened or optimised variants exist, not the plain index.",
    "World High Dividend": "The only plain-index UCITS ETF (Fineco AM) is not on Trade Republic.",
}


def records(rows):
    return [dict(zip(FIELDS, r)) for r in rows]
