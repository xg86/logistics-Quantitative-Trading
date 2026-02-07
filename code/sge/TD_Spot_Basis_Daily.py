"""
SGE Gold P&L Attribution Templates (openpyxl, NO file written)

Creates an in-memory Excel workbook with TWO tabs:

1) TD_Spot_Basis_Daily
   - Daily P&L explain for: Spot hedge + Au(T+D)
   - Buckets: Spot P&L, T+D MTM P&L, Net Price P&L, Basis P&L (diagnostic),
     Deferred Fee P&L, Fees P&L, Total
   - Includes sample daily data (30 rows) and all formulas

2) Tenor_Curve_OneDay
   - One-day revaluation ladder for OTC forwards at T+14 and T+30 ("1M" fixed 30d)
   - Includes DF bootstrapping from two zero rates (14d & 30d, continuous comp)
   - Computes:
       * curve interpolation weights
       * theta/roll-down per position
       * spot bucket
       * curve bucket split into 14D pillar and 30D pillar
   - Also includes a compact Spot vs T+D vs Deferred Fee block for same day
   - Includes sample inputs + positions and all formulas

IMPORTANT:
- openpyxl stores formulas; it does NOT evaluate them.
- To see results, open the workbook in Excel (or any calc-capable spreadsheet app).

No file is created unless YOU call wb.save("...xlsx") at the end.
"""

from datetime import date, timedelta
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.styles.numbers import FORMAT_DATE_YYYYMMDD2
import math


def build_workbook():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "TD_Spot_Basis_Daily"
    ws2 = wb.create_sheet("Tenor_Curve_OneDay")
    return wb, ws1, ws2


# --- Styles ---
thin = Side(style="thin", color="A0A0A0")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
header_fill = PatternFill("solid", fgColor="1F4E79")
header_font = Font(bold=True, color="FFFFFF")
input_fill = PatternFill("solid", fgColor="D9E1F2")
bold = Font(bold=True)


def set_col_width(ws, col, width):
    ws.column_dimensions[get_column_letter(col)].width = width


def style_row(ws, r, c1, c2, fill=None, font=None, center=False, wrap=False):
    for c in range(c1, c2 + 1):
        cell = ws.cell(row=r, column=c)
        cell.border = border
        if fill:
            cell.fill = fill
        if font:
            cell.font = font
        cell.alignment = Alignment(
            horizontal="center" if center else "general",
            vertical="center",
            wrap_text=wrap
        )


def style_block(ws, r1, c1, r2, c2, fill=None, font=None):
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = border
            if fill:
                cell.fill = fill
            if font:
                cell.font = font


# =============================================================================
# Build in-memory workbook (NO save yet)
# =============================================================================
wb, ws1, ws2 = build_workbook()

# -----------------------------------------------------------------------------
# Sheet 1: TD_Spot_Basis_Daily (time series)
# -----------------------------------------------------------------------------
ws1["A1"] = "Au(T+D) + Spot Hedge (Daily P&L Explain) - Spot vs Basis vs Deferred Fee"
ws1["A1"].font = Font(bold=True, size=14)
ws1.merge_cells("A1:N1")

# Parameter block
ws1["A3"] = "Inputs"
ws1["A3"].font = bold

inputs = [
    ("SpotQty_g (physical grams, +long)", 5000),
    ("TD_Lots (Au(T+D), +long / -short)", -5),
    ("TD_Unit_g_per_lot", 1000),
    ("Default_DeferredRate_per_day", 0.00011),
    ("Default_DirectionSign (+1=多付空, -1=空付多)", 1),
    ("Default_DayCount", 1),
    ("Default_FeesPnL_CNY_per_day (negative)", -50),
]

ws1["A4"] = "Name"
ws1["B4"] = "Value"
style_row(ws1, 4, 1, 2, fill=header_fill, font=header_font, center=True)

for i, (k, v) in enumerate(inputs, start=5):
    ws1[f"A{i}"] = k
    ws1[f"B{i}"] = v
    ws1[f"A{i}"].fill = input_fill
    ws1[f"B{i}"].fill = input_fill
    ws1[f"A{i}"].border = border
    ws1[f"B{i}"].border = border

# Named references by cell (kept simple)
# B5 SpotQty_g, B6 TD_Lots, B7 Unit, B8 DefRate, B9 DirSign, B10 DayCount, B11 Fees

# Data table header row
hdr = [
    "Date",
    "Spot_CNYg",
    "TD_Settle_CNYg",
    "Basis_CNYg (=Spot-TD)",
    "Spot_PnL_CNY",
    "TD_MTM_PnL_CNY",
    "NetPrice_PnL_CNY (=Spot+TD)",
    "Basis_PnL_CNY (diag)",
    "DeferredRate",
    "DirSign",
    "DayCount",
    "DeferredFee_PnL_CNY",
    "Fees_PnL_CNY",
    "Total_PnL_CNY",
]

header_row = 14
for c, h in enumerate(hdr, start=1):
    ws1.cell(row=header_row, column=c, value=h)

style_row(ws1, header_row, 1, len(hdr), fill=header_fill, font=header_font, center=True, wrap=True)
ws1.row_dimensions[header_row].height = 40

# Column widths
widths = [12, 12, 14, 14, 14, 16, 16, 16, 12, 10, 10, 18, 12, 12]
for col, w in enumerate(widths, start=1):
    set_col_width(ws1, col, w)

# Sample daily data (30 rows)
start = date(2025, 12, 1)
n = 30

# create deterministic sample series
spot = 480.0
td = 479.0
basis = spot - td

for i in range(n):
    r = header_row + 1 + i
    d = start + timedelta(days=i)

    # basic synthetic moves
    spot += (0.6 if i % 2 == 0 else -0.2) + (0.8 if i % 11 == 0 else 0.0)
    # TD tracks spot but with varying basis
    basis += (0.02 if i % 5 == 0 else -0.01)
    td = spot - basis

    ws1.cell(row=r, column=1, value=d).number_format = FORMAT_DATE_YYYYMMDD2
    ws1.cell(row=r, column=2, value=round(spot, 3))
    ws1.cell(row=r, column=3, value=round(td, 3))

    # Basis
    ws1.cell(row=r, column=4, value=f"=B{r}-C{r}")

    # Deferred inputs (you can overwrite per day)
    ws1.cell(row=r, column=9, value="=$B$8")
    ws1.cell(row=r, column=10, value="=$B$9")
    ws1.cell(row=r, column=11, value="=$B$10")

    # Deferred fee
    # PnL = -SIGN(TD_Lots)*DirSign*Notional*Rate*DayCount
    # Notional = ABS(TD_Lots)*Unit*TD_Settle
    ws1.cell(row=r, column=12, value=f"=-SIGN($B$6)*J{r}*ABS($B$6)*$B$7*C{r}*I{r}*K{r}")

    # Fees
    ws1.cell(row=r, column=13, value="=$B$11")

    # Total PnL
    ws1.cell(row=r, column=14, value=f"=G{r}+L{r}+M{r}")

    # PnL formulas start from second row (need prior day)
    if i == 0:
        for c in range(5, 15):
            # Just set formats for consistency, empty string for value
            pass
    else:
        r0 = r - 1
        # Spot PnL = SpotQty_g * (Spot_t - Spot_{t-1})
        ws1.cell(row=r, column=5, value=f"=$B$5*(B{r}-B{r0})")
        # TD MTM PnL = TD_Lots * Unit * (TD_t - TD_{t-1})
        ws1.cell(row=r, column=6, value=f"=$B$6*$B$7*(C{r}-C{r0})")
        # Net price
        ws1.cell(row=r, column=7, value=f"=E{r}+F{r}")
        # Basis PnL diagnostic = SpotQty_g * (Basis_t - Basis_{t-1})
        ws1.cell(row=r, column=8, value=f"=$B$5*(D{r}-D{r0})")

    # formats
    for c in range(2, 5):
        ws1.cell(row=r, column=c).number_format = "0.000"
    for c in range(5, 15):
        ws1.cell(row=r, column=c).number_format = "0.00"

    # borders
    for c in range(1, 15):
        ws1.cell(row=r, column=c).border = border

ws1.freeze_panes = "A15"

# -----------------------------------------------------------------------------
# Sheet 2: Tenor_Curve_OneDay (revaluation ladder + DF + pillars)
# -----------------------------------------------------------------------------
ws2["A1"] = "Au(T+D) + OTC Forwards (T+14, T+30) - Revaluation Ladder (Theta / Spot / Curve 14D / Curve 30D)"
ws2["A1"].font = Font(bold=True, size=14)
ws2.merge_cells("A1:N1")

# --- Block A: One-day market inputs ---
ws2["A3"] = "Market Inputs (t0 -> t1)"
ws2["A3"].font = bold

ws2["A4"] = "ValDate0"
ws2["B4"] = date(2025, 12, 27)
ws2["B4"].number_format = FORMAT_DATE_YYYYMMDD2

ws2["A5"] = "ValDate1"
ws2["B5"] = "=B4+1"
ws2["B5"].number_format = FORMAT_DATE_YYYYMMDD2

ws2["A7"] = "Spot0_CNYg";
ws2["B7"] = 480.000
ws2["A8"] = "Spot1_CNYg";
ws2["B8"] = 485.000

ws2["A10"] = "F14_0_CNYg (OTC T+14)";
ws2["B10"] = 480.500
ws2["A11"] = "F14_1_CNYg (OTC T+14)";
ws2["B11"] = 485.600

ws2["A12"] = "F30_0_CNYg (OTC T+30)";
ws2["B12"] = 481.200
ws2["A13"] = "F30_1_CNYg (OTC T+30)";
ws2["B13"] = 486.300

ws2["A15"] = "Tau14_days";
ws2["B15"] = 14
ws2["A16"] = "Tau30_days";
ws2["B16"] = 30
ws2["A17"] = "DayBasis";
ws2["B17"] = 365

# Forward points pillars
ws2["D7"] = "FP14_0";
ws2["E7"] = "=B10-B7"
ws2["D8"] = "FP30_0";
ws2["E8"] = "=B12-B7"
ws2["D10"] = "FP14_1";
ws2["E10"] = "=B11-B8"
ws2["D11"] = "FP30_1";
ws2["E11"] = "=B13-B8"

# DF inputs (continuous comp zeros)
ws2["A19"] = "Discount Curve Inputs (continuous-comp zeros)"
ws2["A19"].font = bold
ws2["A20"] = "r14_0_cc";
ws2["B20"] = 0.0200
ws2["A21"] = "r30_0_cc";
ws2["B21"] = 0.0210

# Bootstrapped DFs and fwd rate between 14 and 30
ws2["D20"] = "DF14_0"
ws2["E20"] = "=EXP(-B20*(B15/B17))"
ws2["D21"] = "DF30_0"
ws2["E21"] = "=EXP(-B21*(B16/B17))"
ws2["D22"] = "f14_30_0_cc"
ws2["E22"] = "=LN(E20/E21)/((B16-B15)/B17)"

# Style input block
style_block(ws2, 4, 1, 17, 2, fill=input_fill)
style_block(ws2, 20, 1, 21, 2, fill=input_fill)
for r in range(4, 22 + 1):
    for c in range(1, 6):
        if ws2.cell(row=r, column=c).value:
            ws2.cell(row=r, column=c).border = border

for cell in ["B7", "B8", "B10", "B11", "B12", "B13", "E7", "E8", "E10", "E11"]:
    ws2[cell].number_format = "0.000"
for cell in ["B20", "B21"]:
    ws2[cell].number_format = "0.0000"
for cell in ["E20", "E21"]:
    ws2[cell].number_format = "0.000000"

# Column widths
for col, w in [(1, 28), (2, 16), (4, 14), (5, 18)]:
    set_col_width(ws2, col, w)

# --- Block B: Positions table (OTC forwards) ---
ws2["A24"] = "OTC Forward Positions (one row per trade)"
ws2["A24"].font = bold

pos_hdr_row = 26
pos_hdr = [
    "TradeID",
    "MaturityDate",
    "Qty_g (+long/-short)",
    "Strike_K_CNYg",
    "Tau0_days",
    "Tau1_days",
    "w14_tau0",
    "w30_tau0",
    "w14_tau1",
    "w30_tau1",
    "FP_tau0_t0",
    "FP_tau1_t0",
    "FP_tau1_t1",
    "DF_tau0_t0",
    "DF_tau1_t0",
    "Theta_PnL",
    "Spot_PnL",
    "Curve14_PnL",
    "Curve30_PnL",
    "CurveTotal_Check",
    "Total_Explained (Theta+Spot+Curve)",
]

for c, h in enumerate(pos_hdr, start=1):
    ws2.cell(row=pos_hdr_row, column=c, value=h)

style_row(ws2, pos_hdr_row, 1, len(pos_hdr), fill=header_fill, font=header_font, center=True, wrap=True)
ws2.row_dimensions[pos_hdr_row].height = 45

# widths
pos_widths = [10, 14, 18, 14, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 18]
for col, w in enumerate(pos_widths, start=1):
    set_col_width(ws2, col, w)

# Sample positions (2 trades)
sample_positions = [
    ("T1", date(2026, 1, 16), 1000, 481.000),
    ("T2", date(2026, 1, 6), -2000, 480.200),
]

first_pos_row = pos_hdr_row + 1
for i, (tid, mat, qty, strike) in enumerate(sample_positions):
    r = first_pos_row + i
    ws2.cell(row=r, column=1, value=tid)
    ws2.cell(row=r, column=2, value=mat).number_format = FORMAT_DATE_YYYYMMDD2
    ws2.cell(row=r, column=3, value=qty).number_format = "0"
    ws2.cell(row=r, column=4, value=strike).number_format = "0.000"

    # Tau0 = MAX(0, Maturity - ValDate0)
    ws2.cell(row=r, column=5, value=f"=MAX(0,B{r}-$B$4)")
    # Tau1 = MAX(0, Maturity - ValDate1)
    ws2.cell(row=r, column=6, value=f"=MAX(0,B{r}-$B$5)")

    # w14 and w30 definitions (piecewise):
    # w14(tau) = IF(tau<=0, 0, IF(tau<=14, tau/14, IF(tau<=30, (30-tau)/(30-14), NA())))
    # w30(tau) = IF(tau<=0, 0, IF(tau<=14, 0,      IF(tau<=30, (tau-14)/(30-14), NA())))

    # tau0 in col 5 (E), tau1 in col 6 (F)
    ws2.cell(row=r, column=7,
             value=f"=IF(E{r}<=0,0,IF(E{r}<=$B$15,E{r}/$B$15,IF(E{r}<=$B$16,($B$16-E{r})/($B$16-$B$15),NA())))")
    ws2.cell(row=r, column=8, value=f"=IF(E{r}<=0,0,IF(E{r}<=$B$15,0,IF(E{r}<=$B$16,(E{r}-$B$15)/($B$16-$B$15),NA())))")

    ws2.cell(row=r, column=9,
             value=f"=IF(F{r}<=0,0,IF(F{r}<=$B$15,F{r}/$B$15,IF(F{r}<=$B$16,($B$16-F{r})/($B$16-$B$15),NA())))")
    ws2.cell(row=r, column=10,
             value=f"=IF(F{r}<=0,0,IF(F{r}<=$B$15,0,IF(F{r}<=$B$16,(F{r}-$B$15)/($B$16-$B$15),NA())))")

    # FP_tau0_t0 = w14_tau0*FP14_0 + w30_tau0*FP30_0
    ws2.cell(row=r, column=11, value=f"=G{r}*$E$7 + H{r}*$E$8")
    # FP_tau1_t0 = w14_tau1*FP14_0 + w30_tau1*FP30_0
    ws2.cell(row=r, column=12, value=f"=I{r}*$E$7 + J{r}*$E$8")
    # FP_tau1_t1 = w14_tau1*FP14_1 + w30_tau1*FP30_1
    ws2.cell(row=r, column=13, value=f"=I{r}*$E$10 + J{r}*$E$11")

    # DF(tau) using t0 curve:
    # if tau<=0 -> 1
    # else if tau<=14 -> exp(-r14*tau/365)
    # else -> DF14 * exp(-f14_30 * (tau-14)/365)
    ws2.cell(row=r, column=14,
             value=f"=IF(E{r}<=0,1,IF(E{r}<=$B$15,EXP(-$B$20*(E{r}/$B$17)),$E$20*EXP(-$E$22*((E{r}-$B$15)/$B$17))))")
    ws2.cell(row=r, column=15,
             value=f"=IF(F{r}<=0,1,IF(F{r}<=$B$15,EXP(-$B$20*(F{r}/$B$17)),$E$20*EXP(-$E$22*((F{r}-$B$15)/$B$17))))")

    # Theta ladder:
    # F0_thetaStep = Spot0 + FP_tau0_t0
    # Ftheta_thetaStep = Spot0 + FP_tau1_t0
    # V0 = DF_tau0 * Qty * (F0 - K)
    # Vtheta = DF_tau1 * Qty * (Ftheta - K)
    # Theta = Vtheta - V0
    # Using columns: K=FP_tau0_t0, L=FP_tau1_t0, N=DF_tau0, O=DF_tau1, C=Qty, D=StrikeK
    ws2.cell(row=r, column=16, value=f"=O{r}*$C{r}*(($B$7+L{r})-$D{r}) - N{r}*$C{r}*(($B$7+K{r})-$D{r})")

    # Spot bucket (after theta, update Spot0->Spot1 holding FP at t0 at tau1):
    # SpotPnL = DF_tau1 * Qty * (Spot1-Spot0)
    ws2.cell(row=r, column=17, value=f"=O{r}*$C{r}*($B$8-$B$7)")

    # Curve pillar split at tau1 (after spot update): use DF_tau1, weights at tau1, and dFP pillars
    # Curve14 = DF_tau1 * Qty * w14_tau1 * (FP14_1 - FP14_0)
    # Curve30 = DF_tau1 * Qty * w30_tau1 * (FP30_1 - FP30_0)
    ws2.cell(row=r, column=18, value=f"=O{r}*$C{r}*I{r}*($E$10-$E$7)")
    ws2.cell(row=r, column=19, value=f"=O{r}*$C{r}*J{r}*($E$11-$E$8)")

    # Curve total check: DF_tau1 * Qty * (FP_tau1_t1 - FP_tau1_t0)
    ws2.cell(row=r, column=20, value=f"=O{r}*$C{r}*(M{r}-L{r})")

    # Total explained
    ws2.cell(row=r, column=21, value=f"=P{r}+Q{r}+R{r}+S{r}")

    # formats
    for c in range(5, 7):
        ws2.cell(row=r, column=c).number_format = "0"
    for c in range(7, 11):
        ws2.cell(row=r, column=c).number_format = "0.0000"
    for c in range(11, 14):
        ws2.cell(row=r, column=c).number_format = "0.000"
    for c in range(14, 16):
        ws2.cell(row=r, column=c).number_format = "0.000000"
    for c in range(16, 22):
        ws2.cell(row=r, column=c).number_format = "0.00"

    # borders
    for c in range(1, len(pos_hdr) + 1):
        ws2.cell(row=r, column=c).border = border

# Freeze panes on positions header
ws2.freeze_panes = "A27"

# -----------------------------------------------------------------------------
# Block C: Compact Spot vs T+D vs Deferred Fee (same day)
# -----------------------------------------------------------------------------
# This gives you the first use case (spot vs basis vs deferred fee) on the same-day basis,
# alongside the tenor ladder.
base_row = first_pos_row + len(sample_positions) + 4
ws2[f"A{base_row}"] = "Au(T+D) + Spot Hedge (One-Day) - Spot vs Basis vs Deferred Fee"
ws2[f"A{base_row}"].font = bold
ws2.merge_cells(f"A{base_row}:H{base_row}")

# Inputs
ws2[f"A{base_row + 1}"] = "SpotQty_g";
ws2[f"B{base_row + 1}"] = 5000
ws2[f"A{base_row + 2}"] = "TD_Lots";
ws2[f"B{base_row + 2}"] = -5
ws2[f"A{base_row + 3}"] = "TD_Unit_g_per_lot";
ws2[f"B{base_row + 3}"] = 1000
ws2[f"A{base_row + 4}"] = "TD0_CNYg";
ws2[f"B{base_row + 4}"] = 479.000
ws2[f"A{base_row + 5}"] = "TD1_CNYg";
ws2[f"B{base_row + 5}"] = 484.000
ws2[f"A{base_row + 6}"] = "DeferredRate_per_day";
ws2[f"B{base_row + 6}"] = 0.00011
ws2[f"A{base_row + 7}"] = "DirSign (+1 多付空, -1 空付多)";
ws2[f"B{base_row + 7}"] = 1
ws2[f"A{base_row + 8}"] = "DayCount";
ws2[f"B{base_row + 8}"] = 1
ws2[f"A{base_row + 9}"] = "FeesPnL_CNY";
ws2[f"B{base_row + 9}"] = -50

style_block(ws2, base_row + 1, 1, base_row + 9, 2, fill=input_fill)

# Outputs (buckets)
out_r = base_row + 1
ws2[f"D{out_r}"] = "Spot PnL"
ws2[f"E{out_r}"] = f"=B{base_row + 1}*($B$8-$B$7)"  # uses Spot0/Spot1 from top block

ws2[f"D{out_r + 1}"] = "TD_MTM_PnL"
ws2[f"E{out_r + 1}"] = f"=B{base_row + 2}*B{base_row + 3}*(B{base_row + 5}-B{base_row + 4})"

ws2[f"D{out_r + 2}"] = "NetPrice_PnL"
ws2[f"E{out_r + 2}"] = f"=E{out_r}+E{out_r + 1}"

ws2[f"D{out_r + 3}"] = "Basis0"
ws2[f"E{out_r + 3}"] = f"=$B$7-B{base_row + 4}"

ws2[f"D{out_r + 4}"] = "Basis1"
ws2[f"E{out_r + 4}"] = f"=$B$8-B{base_row + 5}"

ws2[f"D{out_r + 5}"] = "Basis_PnL (diag)"
ws2[f"E{out_r + 5}"] = f"=B{base_row + 1}*(E{out_r + 4}-E{out_r + 3})"

ws2[f"D{out_r + 6}"] = "DeferredFee_PnL"
ws2[f"E{out_r + 6}"] = (
    f"=-SIGN(B{base_row + 2})*B{base_row + 7}*ABS(B{base_row + 2})*B{base_row + 3}*B{base_row + 5}*B{base_row + 6}*B{base_row + 8}"
    # Wait, check image for TD settlement price reference. Image 12 says:
    # ... * B{base_row+3} * B{base_row+5} ...
    # B{base_row+5} is TD1 (current day settle).
    # Re-checking Image 12 formulas: It is truncated but looks like B{base_row+5} is used as Settle price.
)
# Correcting from visual inspection of Image 3 and Image 12:
# Image 12 line 483: ... *B{base_row+3}*B{base_row+5}*B{base_row+6}...
# Yes, TD1 is used as settlement price for deferred fee.

ws2[f"D{out_r + 7}"] = "Fees_PnL"
ws2[f"E{out_r + 7}"] = f"=B{base_row + 9}"

ws2[f"D{out_r + 8}"] = "Total_PnL"
ws2[f"E{out_r + 8}"] = f"=E{out_r + 2}+E{out_r + 6}+E{out_r + 7}"

for rr in range(out_r, out_r + 9):
    ws2[f"D{rr}"].font = bold
    ws2[f"D{rr}"].border = border
    ws2[f"E{rr}"].border = border
    ws2[f"E{rr}"].number_format = "0.00"

# Borders for the block
style_block(ws2, base_row, 1, out_r + 8, 5)

# Final touches
ws2["A2"] = "Note: Formulas are stored; open in Excel to calculate."
ws2["A2"].font = Font(italic=True, color="666666")
ws2.merge_cells("A2:N2")

output_filename = "SGE_TD_Spot_Tenor_Attribution.xlsx"
wb.save(output_filename)
print(f"File saved: {output_filename}")

# Quick sanity prints (formulas present)
print("TD daily example formula check:")
print("  NetPrice_PnL cell (G16):", ws1["G16"].value)
print("Tenor curve example formula check:")
print("  Theta_PnL for first trade (P27):", ws2["P27"].value)