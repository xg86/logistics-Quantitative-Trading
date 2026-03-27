"""
SGE vs COMEX relative-value "decision sheet" generator (single-sheet workbook)
- Uses openpyxl
- Builds layout + sample data + formulas (no file is created unless you call wb.save()).

How to use:
1) pip install openpyxl
2) Run this script.
3) Optionally save: wb.save("SGE_COMEX_Decision_Sheet.xlsx")

Notes:
- This is a *relative value* sheet (not risk-free arb), suitable when you cannot import gold.
- Marks are settlement on both SGE and CME (as you stated).
- Premium is computed in CNY/g vs COMEX parity (GC*USDCNY/31.1035).
- Includes rolling mean/stdev/z-score and a GO/NO-GO rule set.
"""

from datetime import date, timedelta
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, numbers


def build_decision_sheet():
    wb = Workbook()
    ws = wb.active
    ws.title = "DecisionSheet"

    # ----------------------------------
    # Formatting helpers
    # ----------------------------------
    bold = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="1F4E79")  # dark blue
    header_font = Font(bold=True, color="FFFFFF")
    input_fill = PatternFill("solid", fgColor="D9E1F2")  # light blue
    thin = Side(style="thin", color="A0A0A0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_range(r1, c1, r2, c2, fill=None, font=None, align_center=False):
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = ws.cell(row=r, column=c)
                if fill:
                    cell.fill = fill
                if font:
                    cell.font = font
                if align_center:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = border

    def set_col_width(col, width):
        ws.column_dimensions[get_column_letter(col)].width = width

    # ----------------------------------
    # Parameters block
    # ----------------------------------
    ws["A1"] = "SGE vs COMEX Relative Value Decision Sheet (Cannot Import Gold)"
    ws["A1"].font = Font(bold=True, size=14)

    params = [
        ("Lookback_N (days)", 60),
        ("MinEdge_CNYg (must exceed costs)", 0.30),
        ("EntryZ", 2.0),
        ("ExitZ", 0.5),
        ("StopZ", 3.5),
        ("MaxAbsPrem_CNYg (regime filter)", 5.00),
        ("MaxHoldDays", 10),
        ("BaseSize_kg", 5),
        ("MaxSizeMultiple", 2),
        ("UseFXHedge (1=yes,0=no)", 1),
        ("g_per_oz", 31.1035),
        ("GC_oz_per_contract", 100),
        ("SGE_TD_g_per_lot", 1000),
        ("CostBuffer_CNYg (fees+slip est.)", 0.10),
    ]

    start_row = 3
    ws["A2"] = "Parameters"
    ws["A2"].font = bold

    ws["A3"] = "Name"
    ws["B3"] = "Value"
    ws["C3"] = "Notes"
    style_range(3, 1, 3, 3, fill=header_fill, font=header_font, align_center=True)

    for i, (name, val) in enumerate(params, start=4):
        ws[f"A{i}"] = name
        ws[f"B{i}"] = val
        ws[f"C{i}"] = ""
        ws[f"A{i}"].fill = input_fill
        ws[f"B{i}"].fill = input_fill
        ws[f"A{i}"].border = border
        ws[f"B{i}"].border = border
        ws[f"C{i}"].border = border

    # Parameter value cells we reference in formulas
    # B4 Lookback, B5 MinEdge, B6 EntryZ, B7 ExitZ, B8 StopZ, B9 MaxAbsPrem, B10 MaxHoldDays
    # B11 BaseSize_kg, B12 MaxSizeMultiple, B13 UseFXHedge, B14 g_per_oz, B15 GC_oz_per_contract
    # B16 SGE_TD_g_per_lot, B17 CostBuffer
    # Note: because params started at row 4, indexes match above.

    # Set number formats for some parameters
    ws["B4"].number_format = "0"
    ws["B5"].number_format = "0.00"
    ws["B6"].number_format = "0.00"
    ws["B7"].number_format = "0.00"
    ws["B8"].number_format = "0.00"
    ws["B9"].number_format = "0.00"
    ws["B10"].number_format = "0"
    ws["B11"].number_format = "0.0"
    ws["B12"].number_format = "0.0"
    ws["B13"].number_format = "0"
    ws["B14"].number_format = "0.0000"
    ws["B15"].number_format = "0"
    ws["B16"].number_format = "0"
    ws["B17"].number_format = "0.00"

    # Column widths
    set_col_width(1, 38)
    set_col_width(2, 18)
    set_col_width(3, 40)

    # ----------------------------------
    # Daily data table (sample data)
    # ----------------------------------
    data_header_row = 20
    headers = [
        "Date",
        "CME_GC_Settle (USD/oz)",
        "USDCNY",
        "SGE_AuTD_Settle (CNY/g)",
        "SGE_Au9999_Settle (CNY/g)",
        "SGE_Au9995_Settle (CNY/g)",
        "Parity_CNYg (GC*FX/31.1035)",
        "Prem_TD_CNYg",
        "Prem_9999_CNYg",
        "Prem_9995_CNYg",
        "Prem_TD_Gold (Δ prem)",
        "Prem_TD_FX (Δ prem)",
        "Prem_TD_Local (Δ prem)",
        "Prem_TD_MA",
        "Prem_TD_SD",
        "Prem_TD_Z",
        "Signal",
        "SizeMultiple",
        "TradeSize_g",
        "TD_Lots",
        "GC_Contracts",
        "FXHedge_USDNotional",
        "EdgeToMean_CNY",
        "Go_NoGo",
    ]

    for c, h in enumerate(headers, start=1):
        ws.cell(row=data_header_row, column=c, value=h)

    style_range(data_header_row, 1, data_header_row, len(headers), fill=header_fill, font=header_font,
                align_center=True)

    # Set widths for data columns
    widths = {
        1: 12, 2: 20, 3: 10, 4: 18, 5: 20, 6: 20,
        7: 18, 8: 12, 9: 14, 10: 14,
        11: 16, 12: 14, 13: 16,
        14: 12, 15: 12, 16: 10,
        17: 20, 18: 12, 19: 12,
        20: 10, 21: 12, 22: 18,
        23: 14, 24: 10,
    }
    for col, w in widths.items():
        set_col_width(col, w)

    # Generate ~90 business-day-like rows of sample data (no external libs)
    # We'll just create 90 sequential days and let the user replace with real data.
    n_rows = 90
    start_date = date(2025, 9, 1)

    # Basic synthetic series: GC around 2400, FX around 7.20, SGE prices around parity +/- premium
    # We'll introduce a slowly varying premium and occasional shocks to exercise z-score logic.
    gc = 2400.0
    fx = 7.20
    prem = 1.50  # CNY/g premium

    for i in range(n_rows):
        r = data_header_row + 1 + i
        d = start_date + timedelta(days=i)

        # Fill date
        ws.cell(row=r, column=1, value=d).number_format = numbers.FORMAT_DATE_YYYYMMDD2

        # Synthetic daily moves
        # (simple deterministic-ish pattern; replace with real data)
        gc += (0.8 if i % 3 == 0 else -0.4) + (2.0 if i % 17 == 0 else 0.0)
        fx += (0.002 if i % 5 == 0 else -0.001)

        # Premium pattern with occasional shock (to trigger signals)
        prem += (0.01 if i % 4 == 0 else -0.005)
        if i in (35, 36, 37):  # temporary spike
            prem += 0.50
        if i in (70, 71):  # temporary dip
            prem -= 0.60

        # Compute parity in CNY/g for sample price generation (use same constant 31.1035)
        parity = gc * fx / 31.1035
        # SGE T+D settle ~ parity + premium
        td = parity + prem
        # Add small grade spreads for physical
        au9999 = td + 0.08
        au9995 = td - 0.05

        # Write sample inputs
        ws.cell(row=r, column=2, value=round(gc, 2)).number_format = "0.00"
        ws.cell(row=r, column=3, value=round(fx, 4)).number_format = "0.0000"
        ws.cell(row=r, column=4, value=round(td, 3)).number_format = "0.000"
        ws.cell(row=r, column=5, value=round(au9999, 3)).number_format = "0.000"
        ws.cell(row=r, column=6, value=round(au9995, 3)).number_format = "0.000"

        # ----------------------------------
        # Formulas (computed columns)
        # ----------------------------------
        # Column letters for readability
        A = lambda col: get_column_letter(col)

        # Parity_CNYg in col 7: = (B* C) / g_per_oz_param (B14)
        ws.cell(row=r, column=7, value=f"=({A(2)}{r}*{A(3)}{r})/$B$14").number_format = "0.000"

        # Premiums cols 8-10: SGE - Parity
        ws.cell(row=r, column=8, value=f"={A(4)}{r}-{A(7)}{r}").number_format = "0.000"
        ws.cell(row=r, column=9, value=f"={A(5)}{r}-{A(7)}{r}").number_format = "0.000"
        ws.cell(row=r, column=10, value=f"={A(6)}{r}-{A(7)}{r}").number_format = "0.000"

        # ΔPremium decomposition starts from second data row
        if i == 0:
            # leave blank for first row
            ws.cell(row=r, column=11, value="").number_format = "0.000"
            ws.cell(row=r, column=12, value="").number_format = "0.000"
            ws.cell(row=r, column=13, value="").number_format = "0.000"
        else:
            r0 = r - 1
            # Gold bucket: -(FX0/k) * (GC1 - GC0)
            ws.cell(row=r, column=11, value=f"=-({A(3)}{r0}/$B$14)*({A(2)}{r}-{A(2)}{r0})").number_format = "0.000"
            # FX bucket: =(GC1/k) * (FX1 - FX0)
            ws.cell(row=r, column=12, value=f"=({A(2)}{r}/$B$14)*({A(3)}{r}-{A(3)}{r0})").number_format = "0.000"
            # Local bucket: (SGE TD1 - SGE TD0)
            ws.cell(row=r, column=13, value=f"={A(4)}{r}-{A(4)}{r0}").number_format = "0.000"

        # Rolling stats on Prem_TD (col 8)
        prem_col = A(8)
        start_data_row = data_header_row + 1

        # Rolling mean in col 14:
        # =IF(COUNT($H$21:Hrow)<$B$4,"", AVERAGE(INDEX($H$21:Hrow,COUNT($H$21:Hrow)-$B$4+1):Hrow))
        ws.cell(row=r, column=14,
                value=f'=IF(COUNT(${prem_col}${start_data_row}:{prem_col}{r})<$B$4,"",'
                      f'AVERAGE(INDEX(${prem_col}${start_data_row}:{prem_col}{r},'
                      f'COUNT(${prem_col}${start_data_row}:{prem_col}{r})-$B$4+1):{prem_col}{r}))'
                ).number_format = "0.000"

        # Rolling stdev in col 15
        ws.cell(row=r, column=15,
                value=f'=IF(COUNT(${prem_col}${start_data_row}:{prem_col}{r})<$B$4,"",'
                      f'STDEV.S(INDEX(${prem_col}${start_data_row}:{prem_col}{r},'
                      f'COUNT(${prem_col}${start_data_row}:{prem_col}{r})-$B$4+1):{prem_col}{r}))'
                ).number_format = "0.000"

        # Z-score in col 16
        ws.cell(row=r, column=16,
                value=f'=IF(OR({A(14)}{r}="",{A(15)}{r}=""),"",({prem_col}{r}-{A(14)}{r})/{A(15)}{r})'
                ).number_format = "0.00"

        # Signal in col 17 (EntryZ in $B$6)
        ws.cell(row=r, column=17,
                value=(f'=IF({A(16)}{r}="","",IF({A(16)}{r}="","","",'
                       f'IF({A(16)}{r}>=$B$6,"SHORT_SGE_LONG_GC",'
                       f'IF({A(16)}{r}<=-$B$6,"LONG_SGE_SHORT_GC",""))))')
                )

        # SizeMultiple in col 18: MIN(MaxSizeMultiple, ABS(Z)/EntryZ)
        ws.cell(row=r, column=18,
                value=f'=IF({A(16)}{r}="","",MIN($B$12,ABS({A(16)}{r})/$B$6))'
                ).number_format = "0.00"

        # TradeSize_g in col 19: BaseSize_kg*1000*SizeMultiple
        ws.cell(row=r, column=19,
                value=f'=IF({A(16)}{r}="","",$B$11*1000*{A(18)}{r})'
                ).number_format = "0"

        # TD_Lots in col 20: if SHORT_SGE_LONG_GC then -TradeSize_g/1000 else +TradeSize_g/1000
        ws.cell(row=r, column=20,
                value=(f'=IF({A(16)}{r}="","",IF({A(16)}{r}="","","",'
                       f'IF({A(16)}{r}="SHORT_SGE_LONG_GC",-{A(19)}{r}/$B$16,'
                       f'IF({A(16)}{r}="LONG_SGE_SHORT_GC",{A(19)}{r}/$B$16,0))))')
                ).number_format = "0.000"

        # GC_Contracts in col 21: grams / (oz_per_contract*g_per_oz)
        ws.cell(row=r, column=21,
                value=(f'=IF({A(16)}{r}="","",IF({A(16)}{r}="","","",'
                       f'IF({A(16)}{r}="SHORT_SGE_LONG_GC",{A(19)}{r}/($B$15*$B$14),'
                       f'IF({A(16)}{r}="LONG_SGE_SHORT_GC",-{A(19)}{r}/($B$15*$B$14),0))))')
                ).number_format = "0.000"

        # FXHedge_USDNotional in col 22: if UseFXHedge=1 then -(GC_Contracts*100oz*GC_USD/oz)
        ws.cell(row=r, column=22,
                value=f'=IF(OR({A(16)}{r}="",$B$13=0),0,-{A(21)}{r}*$B$15*{A(2)}{r})'
                ).number_format = "0.00"

        # EdgeToMean_CNY in col 23:
        # If LONG_SGE_SHORT_GC then TradeSize_g*(MA - Prem) else if SHORT then -TradeSize_g*(MA - Prem)
        ws.cell(row=r, column=23,
                value=(f'=IF({A(16)}{r}="","",IF({A(16)}{r}="","","",'
                       f'IF({A(16)}{r}="LONG_SGE_SHORT_GC",{A(19)}{r}*({A(14)}{r}-{prem_col}{r}),'
                       f'IF({A(16)}{r}="SHORT_SGE_LONG_GC",-{A(19)}{r}*({A(14)}{r}-{prem_col}{r}),0))))')
                ).number_format = "0"

        # Go_NoGo in col 24:
        # GO if (|Z|>=EntryZ) and (|Prem|>=MinEdge) and (|Prem|<=MaxAbsPrem) and (|Edge| >= TradeSize_g*(MinEdge+CostBuffer))
        ws.cell(row=r, column=24,
                value=(f'=IF({A(16)}{r}="","",IF(AND(ABS({A(16)}{r})>=$B$6,ABS({prem_col}{r})>=$B$5,'
                       f'ABS({prem_col}{r})<=$B$9,ABS({A(23)}{r})>={A(19)}{r}*($B$5+$B$17)),"GO",'
                       f'IF(ABS({A(16)}{r})>=$B$6,"NO_GO","")))'))

    # Borders for row
    for c in range(1, len(headers) + 1):
        ws.cell(row=r, column=c).border = border

    # Freeze panes at the table header
    ws.freeze_panes = ws[f"A{data_header_row + 1}"]

    # Add a small legend for interpretation
    legend_row = 19
    ws[f"A{legend_row}"] = "Legend:"
    ws[f"A{legend_row}"].font = bold
    ws[
        f"B{legend_row}"] = "Premium = SGE - (GC*USDCNY/31.1035). Signals trade premium mean reversion. Not risk-free without import."
    ws[f"B{legend_row}"].alignment = Alignment(wrap_text=True)
    ws.merge_cells(f"B{legend_row}:F{legend_row}")

    # Make the data table easier to read
    for col in range(1, len(headers) + 1):
        ws.cell(row=data_header_row, column=col).alignment = Alignment(wrap_text=True, horizontal="center",
                                                                       vertical="center")

    # Row heights
    ws.row_dimensions[data_header_row].height = 42
    ws.row_dimensions[legend_row].height = 30

    return wb


# Build workbook in memory (no file written)
wb = build_decision_sheet()

# If you want to inspect something quickly (e.g., ensure formulas exist), print a couple of cells:
ws = wb["DecisionSheet"]
print("Sample cells:")
print(f"Parity formula (G21): {ws['G21'].value}")
print(f"Prem_TD (H21): {ws['H21'].value}")
print(f"Z-score (P80): {ws['P80'].value}")
print(f"Go/NoGo (X80): {ws['X80'].value}")

# To save (optional):
wb.save("SGE_COMEX_Decision_Sheet.xlsx")
print("\nCreated: SGE_COMEX_Decision_Sheet workbook in memory")
print("To save, uncomment: wb.save('SGE_COMEX_Decision_Sheet.xlsx')")