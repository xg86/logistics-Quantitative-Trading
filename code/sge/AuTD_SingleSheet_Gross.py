"""
SGE Au(T+D) P&L Attribution — Single Sheet (GROSS long/short supported)
(openpyxl, in-memory workbook; NO file written unless you call wb.save())

What's new vs net-only:
- Adds INPUTS for EOD gross long lots and EOD gross short lots (from statement)
- Deferred fee (延期补偿费) is computed on BOTH gross sides:
  DeferredPnL = (-DirSign)*NotionalLong*Rate*DayCount + (DirSign)*NotionalShort*Rate*DayCount
  where DirSign = +1 for 多付空, -1 for 空付多

Also includes sample data:
- PD long/short
- a few trades
- EOD gross long/short

Note:
- openpyxl writes formulas but does not evaluate them. Open in Excel to see results.
"""

from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


def build_sge_autd_single_sheet_gross():
    wb = Workbook()
    ws = wb.active
    ws.title = "AuTD_SingleSheet_Gross"

    # -----------------------------
    # Styling helpers
    # -----------------------------
    thin = Side(style="thin", color="9E9E9E")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    input_fill = PatternFill("solid", fgColor="D9E1F2")
    bold = Font(bold=True)

    def set_col_width(col, width):
        ws.column_dimensions[get_column_letter(col)].width = width

    def style_range(r1, c1, r2, c2, fill=None, font=None, align=None):
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = ws.cell(row=r, column=c)
                cell.border = border
                if fill:
                    cell.fill = fill
                if font:
                    cell.font = font
                if align:
                    cell.alignment = align

    # -----------------------------
    # Title
    # -----------------------------
    ws["A1"] = "SGE Au(T+D) P&L Attribution (Single Sheet, Gross Long/Short) — CNY"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:N1")

    # -----------------------------
    # INPUTS block (A4:C18)
    # -----------------------------
    # Fixed cell map used by formulas:
    # B5  TradeDate
    # B6  Unit_g_per_lot
    # B7  PD_settle_RMB_per_g
    # B8  CD_settle_RMB_per_g
    # B9  PD_long_lots
    # B10 PD_short_lots
    # B11 FeeRate
    # B12 DeferredRate_per_day
    # B13 DeferredDirection (text)
    # B14 DayCount
    # B15 Statement_TotalPnL (optional)
    # B16 EOD_long_lots_gross (input)
    # B17 EOD_short_lots_gross (input)
    # B18 DirectionSign (computed)

    ws["A4"], ws["B4"], ws["C4"] = "Parameter", "Value", "Notes"
    style_range(4, 1, 4, 3, fill=header_fill, font=header_font, align=Alignment(horizontal="center", vertical="center"))

    inputs = [
        ("TradeDate", datetime(2025, 12, 27).date(), "e.g., 2025-12-27"),
        ("Unit_g_per_lot", 1000, "Au(T+D) = 1 kg/lot = 1000 g"),
        ("PD_settle_RMB_per_g", "", "Prior day settlement (元/克)"),
        ("CD_settle_RMB_per_g", 485.50, "Today settlement (元/克)"),
        ("PD_long_lots", 2.0, "Prior day gross long lots"),
        ("PD_short_lots", 1.0, "Prior day gross short lots"),
        ("FeeRate", 0.0002, "Transaction fee rate (万分之二)"),
        ("DeferredRate_per_day", 0.00011, "Deferred fee rate per calendar day"),
        ("DeferredDirection", "多付空", "Enter: 多付空 or 空付多"),
        ("DayCount", 1, "Calendar days applied (1 normally; 3 on Friday, etc.)"),
        ("Statement_TotalPnL", "", "Optional: broker statement total P&L (CNY)"),
        ("EOD_long_lots_gross", 2.5, "End-of-day gross long lots (from statement)"),
        ("EOD_short_lots_gross", 1.0, "End-of-day gross short lots (from statement)"),
        ("DirectionSign", None, "Computed: 多付空=+1, 空付多=-1"),
    ]

    for idx, (k, v, note) in enumerate(inputs, start=5):
        ws[f"A{idx}"] = k
        ws[f"B{idx}"] = v
        ws[f"C{idx}"] = note
        ws[f"A{idx}"].fill = input_fill
        ws[f"B{idx}"].fill = input_fill
        ws[f"A{idx}"].border = border
        ws[f"B{idx}"].border = border
        ws[f"C{idx}"].border = border

    # DirectionSign formula (B18)
    ws["B18"] = '=IF($B$13="多付空",1,IF($B$13="空付多",-1,0))'

    # Formats
    ws["B5"].number_format = "yyyy-mm-dd"
    for addr in ["B7", "B8", "B11", "B12"]:
        ws[addr].number_format = "0.0000"
    for addr in ["B6", "B14", "B18"]:
        ws[addr].number_format = "0"
    for addr in ["B9", "B10", "B16", "B17"]:
        ws[addr].number_format = "0.000"

    set_col_width(1, 30)  # A
    set_col_width(2, 20)  # B
    set_col_width(3, 46)  # C

    # -----------------------------
    # TRADES table (A21:I220)
    # -----------------------------
    trade_hdr_row = 21
    trade_headers = [
        "TradeTime", "Side", "QtyLots", "TradePrice_RMB_per_g",
        "Unit_g", "CD_Settle", "TradeNotional_CNY", "Fee_CNY", "ExecPnL_vs_Settle_CNY"
    ]

    for c, h in enumerate(trade_headers, start=1):
        ws.cell(row=trade_hdr_row, column=c, value=h)

    style_range(trade_hdr_row, 1, trade_hdr_row, len(trade_headers),
                fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center", wrap_text=True))

    ws.row_dimensions[trade_hdr_row].height = 28

    # Column widths for trades
    for col, w in enumerate([12, 8, 10, 18, 10, 12, 18, 12, 20], start=1):
        set_col_width(col, max(ws.column_dimensions[get_column_letter(col)].width or 0, w))

    first_trade_row = trade_hdr_row + 1
    last_trade_row = 220

    for r in range(first_trade_row, last_trade_row + 1):
        # E: unit, F: settle
        ws[f"E{r}"] = "=$B$6"
        ws[f"F{r}"] = "=$B$8"

        # G: notional, H: fee
        ws[f"G{r}"] = f"=$C{r}*$D{r}*$E{r}"
        ws[f"H{r}"] = f"=$G{r}*$B$11"

        # I: execution PnL vs settle
        ws[f"I{r}"] = (
            f'=IF(OR($B{r}="B",$B{r}="BUY",$B{r}="买"),'
            f'($F{r}-$D{r})*$C{r}*$E{r},'
            f'IF(OR($B{r}="S",$B{r}="SELL",$B{r}="卖"),'
            f'($D{r}-$F{r})*$C{r}*$E{r},0))'
        )

        # Formats
        ws[f"D{r}"].number_format = "0.000"
        ws[f"E{r}"].number_format = "0"
        ws[f"F{r}"].number_format = "0.000"
        ws[f"G{r}"].number_format = "0.00"
        ws[f"H{r}"].number_format = "0.00"
        ws[f"I{r}"].number_format = "0.00"

        for c in range(1, 10):
            ws.cell(row=r, column=c).border = border

    # Sample trades (overwrite freely)
    # Prior day: long 2, short 1 (net +1).
    # Today: sell 1 lot, buy 0.5 lot, buy 1 lot => net +0.5; example EOD long 2.5 short 1.0 (net +1.5)
    ws["A22"], ws["B22"], ws["C22"], ws["D22"] = "10:05", "S", 1.0, 484.80
    ws["A23"], ws["B23"], ws["C23"], ws["D23"] = "11:20", "B", 0.5, 485.20
    ws["A24"], ws["B24"], ws["C24"], ws["D24"] = "13:45", "B", 1.0, 485.10

    # -----------------------------
    # SUMMARY / P&L Explain (K4:M24)
    # -----------------------------
    ws["K4"], ws["L4"], ws["M4"] = "Item", "Amount (CNY)", "Meaning"
    style_range(4, 11, 4, 13, fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center"))

    set_col_width(11, 26)  # K
    set_col_width(12, 18)  # L
    set_col_width(13, 40)  # M

    # Summary rows: include both net-derived and gross EOD inputs, plus gross deferred fee
    summary = [
        (5, "PD_NetLots", "=$B$9-$B$10", "Prior net lots (gross long - gross short)"),
        (6, "BuyLots_Today",
         '=SUMIF($B$22:$B$220,"B",$C$22:$C$220)+SUMIF($B$22:$B$220,"BUY",$C$22:$C$220)+SUMIF($B$22:$B$220,"买",$C$22:$C$220)',
         "buys today (lots)"),
        (7, "SellLots_Today",
         '=SUMIF($B$22:$B$220,"S",$C$22:$C$220)+SUMIF($B$22:$B$220,"SELL",$C$22:$C$220)+SUMIF($B$22:$B$220,"卖",$C$22:$C$220)',
         "sells today (lots)"),
        (8, "Trade_NetLots", "=L6-L7", "Net trade lots (buy - sell)"),
        (9, "EOD_NetLots_Derived", "=L5+L8", "Derived EOD net lots (may differ from statement if gross is used)"),
        (10, "EOD_LongLots_Gross", "=$B$16", "EOD gross long lots (statement input)"),
        (11, "EOD_ShortLots_Gross", "=$B$17", "EOD gross short lots (statement input)"),
        (12, "EOD_NetLots_Gross", "=L10-L11", "EOD net lots computed from gross inputs"),
        None,
        (14, "Position_MTM_CNY", "=($B$8-$B$7)*L5*$B$6", "Prior net lots MTM from PD settle to CD settle"),
        (15, "Trade_ExecPnL_CNY", "=SUM($I$22:$I$220)", "Execution P&L vs CD settlement"),
        (16, "Price_Total_CNY", "=L14+L15", "Total price P&L"),
        None,
        (18, "Notional_Long_EOD_CNY", "=L10*$B$8*$B$6", "Gross long notional at CD settle"),
        (19, "Notional_Short_EOD_CNY", "=L11*$B$8*$B$6", "Gross short notional at CD settle"),
        None,
        # Gross deferred fee PnL:
        # DirSign=+1 (多付空): long pays => negative; short receives => positive
        # DirSign=-1 (空付多): long receives => positive; short pays => negative
        (21, "Deferred_PnL_Gross_CNY",
         "=(-$B$18)*L18*$B$12*$B$14 + ($B$18)*L19*$B$12*$B$14",
         "Gross deferred fee P&L (uses EOD gross long/short, direction sign, rate, daycount)"),
        None,
        (23, "Fees_PnL_CNY", "=-SUM($H$22:$H$220)", "Transaction fees (negative)"),
        (24, "Total_Explained_CNY", "=L16+L20+L21", "Explained total (Price + Deferred + Fees)"),
    ]

    for item in summary:
        if item is None:
            continue
        row, item_name, formula, meaning = item
        ws[f"K{row}"] = item_name
        ws[f"L{row}"] = formula
        ws[f"M{row}"] = meaning
        ws[f"K{row}"].border = border
        ws[f"L{row}"].border = border
        ws[f"M{row}"].border = border
        ws[f"K{row}"].font = bold
        ws[f"L{row}"].number_format = "0.00"
        ws[f"M{row}"].alignment = Alignment(wrap_text=True, vertical="center")

    # Residual
    ws["K22"] = "Residual_vs_Statement"
    ws["L22"] = '=IF($B$15="",$B$15-L22)'
    ws["M22"] = "Statement total - explained"
    ws["K22"].border = border
    ws["L22"].border = border
    ws["M22"].border = border
    ws["K22"].font = bold
    ws["L22"].number_format = "0.00"
    ws["M22"].alignment = Alignment(wrap_text=True, vertical="center")

    style_range(5, 11, 23, 13, align=Alignment(vertical="center"))

    # Make it readable
    ws.freeze_panes = "A22"

    return wb


# Build workbook in memory (no file written)
wb = build_sge_autd_single_sheet_gross()
ws = wb["AuTD_SingleSheet_Gross"]

# Quick checks (formulas present)
print("Sample cells:")
print(f"DirectionSign formula (B18): {ws['B18'].value}")
print(f"Sample ExecPnL formula (I22): {ws['I22'].value}")
print(f"Gross Deferred PnL formula (L20): {ws['L20'].value}")

# OPTIONAL: save locally (uncomment)
wb.save("SGE_AuTD_PnL_Attribution_SingleSheet_Gross.xlsx")

print("\nCreated: SGE_AuTD_PnL_Attribution_SingleSheet_Gross.xlsx")