from datetime import date
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


def build_sge_tn_td_lending_attribution():
    wb = Workbook()
    ws = wb.active
    ws.title = "SGE_TN_TD_Lending"

    # --------------------------------------------------------
    # Styles
    # --------------------------------------------------------
    thin = Side(style="thin", color="9E9E9E")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    input_fill = PatternFill("solid", fgColor="D9E1F2")
    bold = Font(bold=True)

    def set_col_width(col, width):
        ws.column_dimensions[get_column_letter(col)].width = width

    def style_range(r1, c1, r2, c2, fill=None, font=None, align=None, wrap=False):
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
                else:
                    cell.alignment = Alignment(vertical="center", wrap_text=wrap)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------
    ws["A1"] = "SGE Gold Attribution: Au(T+D) + Au(T+N1/N2) (6M/1Y) + Lending (Objective A)"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:Z1")

    ws["A2"] = "Note: openpyxl writes formulas but does not calculate them; open in Excel to see results."
    ws["A2"].font = Font(italic=True, color="666666")
    ws.merge_cells("A2:Z2")

    # --------------------------------------------------------
    # INPUTS block (A4:B29)
    # --------------------------------------------------------
    ws["A4"], ws["B4"], ws["C4"] = "Input", "Value", "Notes"
    style_range(4, 1, 4, 3, fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center"), wrap=True)

    # Fixed cell map:
    # B5  ValDate0
    # B6  ValDate1
    # B7  DayBasis
    # B8  dt_days
    # B10 TauN1 days (6M node)
    # B11 TauN2 days (1Y node)
    # B13 Au9999_0
    # B14 Au9999_1
    # B16 TD0
    # B17 TD1
    # B19 N1_0
    # B20 N1_1
    # B22 N2_0
    # B23 N2_1
    # B25 TD_unit_g
    # B26 TN_unit_g
    # B28 DeferredFee_TD_CNY
    # B29 DeferredFee_TN_CNY
    # B30 Fees_CNY

    inputs = [
        ("ValDate0", date(2026, 1, 17), "t0"),
        ("ValDate1", None, "t1 (=ValDate0+1)"),
        ("DayBasis", 365, "day count basis for lease accrual"),
        ("dt_days", None, "t1 - t0 (days)"),

        ("TauN1_days (6M node)", 182, "fixed tenor node for N1 (edit if you use other convention)"),
        ("TauN2_days (1Y node)", 365, "fixed tenor node for N2"),

        ("Au99.99_settle_t0 (CNY/g)", 480.000, "spot mark for lending anchor"),
        ("Au99.99_settle_t1 (CNY/g)", 482.000, ""),

        ("Au(T+D)_settle_t0 (CNY/g)", 479.000, "RV anchor spot = TD"),
        ("Au(T+D)_settle_t1 (CNY/g)", 481.500, ""),

        ("Au(T+N1)_price_t0 (CNY/g)", 483.000, "6M node price"),
        ("Au(T+N1)_price_t1 (CNY/g)", 484.600, ""),

        ("Au(T+N2)_price_t0 (CNY/g)", 490.000, "1Y node price"),
        ("Au(T+N2)_price_t1 (CNY/g)", 491.800, ""),

        ("TD_unit_g_per_lot", 1000, "Au(T+D) = 1kg/lot"),
        ("TN_unit_g_per_lot", 100, "Au(T+N1/N2) = 100g/lot (per SGE contract pages)"),

        ("DeferredFee_TD_CNY (statement)", -200.00, "enter your statement cashflow/sign"),
        ("DeferredFee_TN_CNY (statement)", 0.00, "enter your statement cashflow/sign"),
        ("Fees_CNY (statement/est)", -100.00, "all fees/commissions for the day (negative)"),
    ]

    r = 5
    for name, val, note in inputs:
        ws[f"A{r}"] = name
        ws[f"B{r}"] = val
        ws[f"C{r}"] = note
        ws[f"A{r}"].fill = input_fill
        ws[f"B{r}"].fill = input_fill
        ws[f"A{r}"].border = border
        ws[f"B{r}"].border = border
        ws[f"C{r}"].border = border
        r += 1

    # Fill formulas for ValDate1, dt_days
    ws["B6"] = "=B5+1"
    ws["B8"] = "=B6-B5"

    # Number formats
    ws["B5"].number_format = "yyyy-mm-dd"
    ws["B6"].number_format = "yyyy-mm-dd"
    ws["B7"].number_format = "0"
    ws["B8"].number_format = "0"
    ws["B10"].number_format = "0"
    ws["B11"].number_format = "0"
    for addr in ["B13", "B14", "B16", "B17", "B19", "B20", "B22", "B23"]:
        ws[addr].number_format = "0.000"
    ws["B25"].number_format = "0"
    ws["B26"].number_format = "0"
    for addr in ["B28", "B29", "B30"]:
        ws[addr].number_format = "0.00"

    # Column widths
    set_col_width(1, 32)
    set_col_width(2, 18)
    set_col_width(3, 46)

    # --------------------------------------------------------
    # DERIVED block (E4:H15)
    # --------------------------------------------------------
    ws["E4"], ws["F4"] = "Derived (RV anchor = TD)", "Value"
    style_range(4, 5, 4, 6, fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center"))

    # RV anchor spot:
    ws["E5"], ws["F5"] = "S0_RV (=TD0)", "=B16"
    ws["E6"], ws["F6"] = "S1_RV (=TD1)", "=B17"

    # RV forward points pillars relative to TD:
    ws["E8"], ws["F8"] = "FP_N1_0_RV (=N1_0 - S0_RV)", "=B19-F5"
    ws["E9"], ws["F9"] = "FP_N2_0_RV (=N2_0 - S0_RV)", "=B22-F5"
    ws["E10"], ws["F10"] = "FP_N1_1_RV (=N1_1 - S1_RV)", "=B20-F6"
    ws["E11"], ws["F11"] = "FP_N2_1_RV (=N2_1 - S1_RV)", "=B23-F6"

    # Lending anchor derived
    ws["H4"], ws["I4"] = "Derived (LEND anchor = Au99.99)", "Value"
    style_range(4, 8, 4, 9, fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center"))

    ws["H5"], ws["I5"] = "S0_LEND (=Au9999_0)", "=B13"
    ws["H6"], ws["I6"] = "S1_LEND (=Au9999_1)", "=B14"

    ws["H8"], ws["I8"] = "FP_N1_0_LEND (=N1_0 - S0_LEND)", "=B19-I5"
    ws["H9"], ws["I9"] = "FP_N2_0_LEND (=N2_0 - S0_LEND)", "=B22-I5"
    ws["H10"], ws["I10"] = "FP_N1_1_LEND (=N1_1 - S1_LEND)", "=B20-I6"
    ws["H11"], ws["I11"] = "FP_N2_1_LEND (=N2_1 - S1_LEND)", "=B23-I6"

    # style derived cells
    style_range(5, 5, 11, 6, fill=None)
    style_range(5, 8, 11, 9, fill=None)
    for addr in ["F5", "F6", "F8", "F9", "F10", "F11", "I5", "I6", "I8", "I9", "I10", "I11"]:
        ws[addr].number_format = "0.000"

    # Some widths
    set_col_width(5, 34)  # E
    set_col_width(6, 18)  # F
    set_col_width(8, 34)  # H
    set_col_width(9, 18)  # I

    # --------------------------------------------------------
    # SECTION 1: RV Book (TN hedged with TD)
    # --------------------------------------------------------
    rv_top = 18
    ws[f"A{rv_top}"] = "1) Relative Value / Hedge Book - (Au(T+N1), Au(T+N2)) hedged with Au(T+D)"
    ws[f"A{rv_top}"].font = bold
    ws.merge_cells(f"A{rv_top}:Z{rv_top}")

    rv_hdr_row = rv_top + 2
    rv_headers = [
        "TradeID", "Instr", "QtyLots", "Unit_g", "Qty_g",
        "Tau0_days", "Tau1_days",
        "Price0", "Price1",
        "wN1_tau0", "wN2_tau0", "wN1_taul", "wN2_taul",
        "FP_tau0_t0", "FP_tau1_t0", "FP_tau1_t1",
        "F0_model", "Ftheta_model", "F1_model",
        "TotalPnL",
        "ThetaPnL", "SpotPnL", "CurveN1PnL", "CurveN2PnL", "CurveCheck", "ExplainedPnL"
    ]

    for c, h in enumerate(rv_headers, start=1):
        ws.cell(row=rv_hdr_row, column=c, value=h)
    style_range(rv_hdr_row, 1, rv_hdr_row, len(rv_headers),
                fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center"), wrap=True)
    ws.row_dimensions[rv_hdr_row].height = 36

    # Column widths for RV table
    widths = [10, 6, 10, 8, 10, 9, 9, 10, 10, 10, 10, 10, 10, 12, 12, 12, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12]
    for idx, w in enumerate(widths, start=1):
        set_col_width(idx, max(ws.column_dimensions[get_column_letter(idx)].width or 0, w))

    rv_first = rv_hdr_row + 1
    rv_rows = [
        ("RV_N1", "N1", -50.0),  # short 50 lots (100g/lot) => -5000g
        ("RV_N2", "N2", -50.0),  # short 50 lots => -5000g
        ("RV_TD", "TD", None),  # TD hedge lots computed to delta-neutralize grams
    ]

    for i, (tid, instr, lots) in enumerate(rv_rows):
        rr = rv_first + i
        ws[f"A{rr}"] = tid
        ws[f"B{rr}"] = instr

        # QtyLots
        if instr != "TD":
            ws[f"C{rr}"] = lots
        else:
            # TD lots = -(Qty_g(N1)+Qty_g(N2))/TD_unit
            ws[f"C{rr}"] = f"=- (E{rv_first}+E{rv_first + 1})/$B$25"

        # Unit_g
        ws[f"D{rr}"] = f'=IF(B{rr}="TD",$B$25,$B$26)'
        # Qty_g
        ws[f"E{rr}"] = f"=C{rr}*D{rr}"

        # Tau0_days
        ws[f"F{rr}"] = f'=IF(B{rr}="N1",$B$10,IF(B{rr}="N2",$B$11,0))'
        # Taul_days
        ws[f"G{rr}"] = f"=MAX(0,F{rr}-$B$8)"

        # Price0 / Price1 from input prices
        ws[f"H{rr}"] = f'=IF(B{rr}="TD",$B$16,IF(B{rr}="N1",$B$19,IF(B{rr}="N2",$B$22,NA())))'
        ws[f"I{rr}"] = f'=IF(B{rr}="TD",$B$17,IF(B{rr}="N1",$B$20,IF(B{rr}="N2",$B$23,NA())))'

        # Weights at tau0 (relative to tauN1=B10, tauN2=B11)
        ws[f"J{rr}"] = f'=IF(F{rr}<=0,0,IF(F{rr}<=$B$10,F{rr}/$B$10,IF(F{rr}<=$B$11,($B$11-F{rr})/($B$11-$B$10),NA())))'
        ws[f"K{rr}"] = f'=IF(F{rr}<=0,0,IF(F{rr}<=$B$10,0,IF(F{rr}<=$B$11,(F{rr}-$B$10)/($B$11-$B$10),NA())))'

        # Weights at taul
        ws[f"L{rr}"] = f'=IF(G{rr}<=0,0,IF(G{rr}<=$B$10,G{rr}/$B$10,IF(G{rr}<=$B$11,($B$11-G{rr})/($B$11-$B$10),NA())))'
        ws[f"M{rr}"] = f'=IF(G{rr}<=0,0,IF(G{rr}<=$B$10,0,IF(G{rr}<=$B$11,(G{rr}-$B$10)/($B$11-$B$10),NA())))'

        # Forward points along curve (RV anchor = TD, use FP pillars in F8:F11)
        # FP_N1_0_RV = $F$8, FP_N2_0_RV = $F$9, FP_N1_1_RV = $F$10, FP_N2_1_RV = $F$11
        ws[f"N{rr}"] = f"=J{rr}*$F$8 + K{rr}*$F$9"  # FP_tau0_t0
        ws[f"O{rr}"] = f"=L{rr}*$F$8 + M{rr}*$F$9"  # FP_tau1_t0
        ws[f"P{rr}"] = f"=L{rr}*$F$10 + M{rr}*$F$11"  # FP_tau1_t1

        # Model forward prices
        ws[f"Q{rr}"] = f"=$F$5 + N{rr}"  # F0_model = S0_RV + FP_tau0_t0
        ws[f"R{rr}"] = f"=$F$5 + O{rr}"  # Ftheta_model
        ws[f"S{rr}"] = f"=$F$6 + P{rr}"  # F1_model

        # Total PnL (mechanical)
        ws[f"T{rr}"] = f"=E{rr}*(I{rr}-H{rr})"

        # Attribution
        ws[f"U{rr}"] = f"=E{rr}*(R{rr}-Q{rr})"  # ThetaPnL
        ws[f"V{rr}"] = f"=E{rr}*($F$6-$F$5)"  # SpotPnL (RV spot = TD)
        ws[f"W{rr}"] = f"=E{rr}*L{rr}*($F$10-$F$8)"  # CurveN1PnL
        ws[f"X{rr}"] = f"=E{rr}*M{rr}*($F$11-$F$9)"  # CurveN2PnL
        ws[f"Y{rr}"] = f"=E{rr}*(P{rr}-O{rr})"  # CurveCheck (total curve move)
        ws[f"Z{rr}"] = f"=U{rr}+V{rr}+W{rr}+X{rr}"  # ExplainedPnL

        # Formats
        ws[f"C{rr}"].number_format = "0.000"
        ws[f"D{rr}"].number_format = "0"
        ws[f"E{rr}"].number_format = "0"
        ws[f"F{rr}"].number_format = "0"
        ws[f"G{rr}"].number_format = "0"
        ws[f"H{rr}"].number_format = "0.000"
        ws[f"I{rr}"].number_format = "0.000"
        for col in "JKLMNOPQRS":
            ws[f"{col}{rr}"].number_format = "0.0000"
        for col in "TUVWXYZ":
            ws[f"{col}{rr}"].number_format = "0.00"

        # Borders
        for c in range(1, len(rv_headers) + 1):
            ws.cell(row=rr, column=c).border = border

    # RV summary
    rv_sum = rv_first + len(rv_rows) + 2
    ws[f"A{rv_sum}"] = "RV Summary"
    ws[f"A{rv_sum}"].font = bold

    rv_summary_lines = [
        ("RV_TotalPnL_price", f"=SUM(T{rv_first}:T{rv_first + 2})"),
        ("RV_Total_Theta", f"=SUM(U{rv_first}:U{rv_first + 2})"),
        ("RV_Total_Spot", f"=SUM(V{rv_first}:V{rv_first + 2})"),
        ("RV_Total_CurveN1", f"=SUM(W{rv_first}:W{rv_first + 2})"),
        ("RV_Total_CurveN2", f"=SUM(X{rv_first}:X{rv_first + 2})"),
        ("RV_Total_Explained", f"=SUM(Z{rv_first}:Z{rv_first + 2})"),
        ("Add_DeferredFee_TD", "=B28"),
        ("Add_DeferredFee_TN", "=B29"),
        ("Add_Fees", "=B30"),
        ("RV_Total_AllIn", f"=SUM(T{rv_first}:T{rv_first + 2})+B28+B29+B30"),
    ]

    rline = rv_sum + 1
    for label, formula in rv_summary_lines:
        ws[f"A{rline}"] = label
        ws[f"B{rline}"] = formula
        ws[f"A{rline}"].font = bold
        ws[f"A{rline}"].border = border
        ws[f"B{rline}"].border = border
        ws[f"B{rline}"].number_format = "0.00"
        rline += 1

    # --------------------------------------------------------
    # SECTION 2: Lending (Objective A) hedged with TN (N1/N2)
    # --------------------------------------------------------
    lend_top = rv_sum + 14
    ws[f"A{lend_top}"] = "2) Lending (Objective A) - Lease accrual + Forward Hedge Attribution (hedge with N1/N2)"
    ws[f"A{lend_top}"].font = bold
    ws.merge_cells(f"A{lend_top}:AG{lend_top}")

    lend_hdr = lend_top + 2
    lend_headers = [
        "LoanID", "Role", "Qty_g", "LeaseRate_annual", "Term(N1/N2)",
        "RoleSign", "GoldQtySigned_g",
        "LeaseAccrual_CNY", "SpotPnL_Receivable_CNY",
        "HedgeLots_TN", "HedgeQty_g",
        "Tau0_days", "Taul_days",
        "wN1_tau0", "wN2_tau0", "wN1_taul", "wN2_taul",
        "FP_tau0_t0", "FP_tau1_t0", "FP_tau1_t1",
        "F0_model", "Ftheta_model", "F1_model",
        "HedgeTotalPnL", "HedgeThetaPnL", "HedgeSpotPnL", "HedgeCurveN1PnL", "HedgeCurveN2PnL", "HedgeExplainedPnL",
        "NetPricePnL", "DeferredFee_TN_input", "Fees_input", "TotalPnL_CNY"
    ]

    for c, h in enumerate(lend_headers, start=1):
        ws.cell(row=lend_hdr, column=c, value=h)
    style_range(lend_hdr, 1, lend_hdr, len(lend_headers),
                fill=header_fill, font=header_font,
                align=Alignment(horizontal="center", vertical="center"), wrap=True)
    ws.row_dimensions[lend_hdr].height = 42

    lend_first = lend_hdr + 1

    # Data validation for Role and Term
    dv_role = DataValidation(type="list", formula1='"LENDER,BORROWER"', allow_blank=False)
    dv_term = DataValidation(type="list", formula1='"N1,N2"', allow_blank=False)
    ws.add_data_validation(dv_role)
    ws.add_data_validation(dv_term)

    # Sample lending rows: 6M and 1Y
    lend_samples = [
        ("LEND_6M", "LENDER", 5000, 0.0200, "N1"),
        ("LEND_1Y", "LENDER", 5000, 0.0220, "N2"),
    ]

    for i, (loan_id, role, qty_g, lease_rate, term) in enumerate(lend_samples):
        rr = lend_first + i
        ws[f"A{rr}"] = loan_id
        ws[f"B{rr}"] = role
        ws[f"C{rr}"] = float(qty_g)
        ws[f"D{rr}"] = float(lease_rate)
        ws[f"E{rr}"] = term

        dv_role.add(ws[f"B{rr}"])
        dv_term.add(ws[f"E{rr}"])

        # RoleSign: LENDER=+1, BORROWER=-1
        ws[f"F{rr}"] = f'=IF(B{rr}="LENDER",1,IF(B{rr}="BORROWER",-1,0))'
        # GoldQtySigned_g
        ws[f"G{rr}"] = f"=F{rr}*C{rr}"

        # LeaseAccrual (use Spot0_LEND = I5; dt=B8; DayBasis=B7)
        ws[f"H{rr}"] = f"=G{rr}*$B$13*D{rr}*$B$8/$B$7"

        # SpotPnL on gold receivable/payable (mark to Au99.99)
        ws[f"I{rr}"] = f"=G{rr}*($B$14-$B$13)"

        # HedgeLots TN to neutralize gold delta: -GoldQtySigned / TN_unit_g
        ws[f"J{rr}"] = f"=-G{rr}/$B$26"
        ws[f"K{rr}"] = f"=J{rr}*$B$26"

        # # Tau0 / Taul for hedge tenor
        ws[f"L{rr}"] = f'=IF(E{rr}="N1",$B$10,IF(E{rr}="N2",$B$11,0))'
        ws[f"M{rr}"] = f"=MAX(0,L{rr}-$B$8)"

        # # Weights at tau0 (tenor nodes B10/B11)
        ws[f"N{rr}"] = f'=IF(L{rr}<=0,0,IF(L{rr}<=$B$10,L{rr}/$B$10,IF(L{rr}<=$B$11,($B$11-L{rr})/($B$11-$B$10),NA())))'
        ws[f"O{rr}"] = f'=IF(L{rr}<=0,0,IF(L{rr}<=$B$10,0,IF(L{rr}<=$B$11,(L{rr}-$B$10)/($B$11-$B$10),NA())))'

        # # Weights at taul
        ws[f"P{rr}"] = f'=IF(M{rr}<=0,0,IF(M{rr}<=$B$10,M{rr}/$B$10,IF(M{rr}<=$B$11,($B$11-M{rr})/($B$11-$B$10),NA())))'
        ws[f"Q{rr}"] = f'=IF(M{rr}<=0,0,IF(M{rr}<=$B$10,0,IF(M{rr}<=$B$11,(M{rr}-$B$10)/($B$11-$B$10),NA())))'

        # # Forward points along curve (LEND anchor = Au99.99; FP pillars in I8:I11)
        ws[f"R{rr}"] = f"=N{rr}*$I$8 + O{rr}*$I$9"  # FP_tau0_t0
        ws[f"S{rr}"] = f"=P{rr}*$I$8 + Q{rr}*$I$9"  # FP_tau1_t0
        ws[f"T{rr}"] = f"=P{rr}*$I$10 + Q{rr}*$I$11"  # FP_tau1_t1

        # # Model forward prices
        ws[f"U{rr}"] = f"=$I$5 + R{rr}"  # F0
        ws[f"V{rr}"] = f"=$I$5 + S{rr}"  # Ftheta
        ws[f"W{rr}"] = f"=$I$6 + T{rr}"  # F1

        # # Hedge PnL and attribution
        ws[f"X{rr}"] = f"=K{rr}*(W{rr}-U{rr})"  # HedgeTotalPnL
        ws[f"Y{rr}"] = f"=K{rr}*(V{rr}-U{rr})"  # HedgeThetaPnL
        ws[f"Z{rr}"] = f"=K{rr}*($I$6-$I$5)"  # HedgeSpotPnL
        ws[f"AA{rr}"] = f"=K{rr}*P{rr}*($I$10-$I$8)"  # HedgeCurveN1PnL
        ws[f"AB{rr}"] = f"=K{rr}*Q{rr}*($I$11-$I$9)"  # HedgeCurveN2PnL
        ws[f"AC{rr}"] = f"=Y{rr}+Z{rr}+AA{rr}+AB{rr}"  # HedgeExplainedPnL

        # # NetPricePnL = Spot receivable + hedge total
        ws[f"AD{rr}"] = f"=I{rr}+X{rr}"

        # # Deferred fee + fees per loan (input; default 0)
        ws[f"AE{rr}"] = 0.0
        ws[f"AF{rr}"] = 0.0

        # # TotalPnL = LeaseAccrual + NetPricePnL + Deferred + Fees
        ws[f"AG{rr}"] = f"=H{rr}+AD{rr}+AE{rr}+AF{rr}"

        # # Formats
        ws[f"C{rr}"].number_format = "0"
        ws[f"D{rr}"].number_format = "0.0000"
        ws[f"F{rr}"].number_format = "0"
        ws[f"G{rr}"].number_format = "0"
        for col in ["H", "I", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG"]:
            ws[f"{col}{rr}"].number_format = "0.00"
        for col in ["J"]:
            ws[f"{col}{rr}"].number_format = "0.000"
        for col in ["K"]:
            ws[f"{col}{rr}"].number_format = "0"
        for col in ["L", "M"]:
            ws[f"{col}{rr}"].number_format = "0"
        for col in ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]:
            ws[f"{col}{rr}"].number_format = "0.0000"

        # Borders
        for c in range(1, len(lend_headers) + 1):
            ws.cell(row=rr, column=c).border = border

    # Lending summary
    lend_sum = lend_first + len(lend_samples) + 2
    ws[f"A{lend_sum}"] = "Lending Summary"
    ws[f"A{lend_sum}"].font = bold

    lend_summary_lines = [
        ("LeaseAccrual_Total", f"=SUM(H{lend_first}:H{lend_first + len(lend_samples) - 1})"),
        ("NetPricePnL_Total", f"=SUM(AD{lend_first}:AD{lend_first + len(lend_samples) - 1})"),
        ("TotalPnL_Total", f"=SUM(AG{lend_first}:AG{lend_first + len(lend_samples) - 1})"),
    ]

    rline = lend_sum + 1
    for label, formula in lend_summary_lines:
        ws[f"A{rline}"] = label
        ws[f"B{rline}"] = formula
        ws[f"A{rline}"].font = bold
        ws[f"A{rline}"].border = border
        ws[f"B{rline}"].border = border
        ws[f"B{rline}"].number_format = "0.00"
        rline += 1

    # Freeze panes
    ws.freeze_panes = "A5"

    return wb


if __name__ == "__main__":
    wb = build_sge_tn_td_lending_attribution()
    ws = wb["SGE_TN_TD_Lending"]

    # Quick formula sanity checks
    print("RV TD hedge lots formula (C row for RV_TD):", ws["C21"].value)
    print("Lending total PnL formula (AG for first loan):", ws["AG24"].value)

    # OPTIONAL: save locally (uncommented for your convenience)
    wb.save("SGE_AuTD_AuTN_Lending_Attribution.xlsx")
    print("Saved file: SGE_AuTD_AuTN_Lending_Attribution.xlsx")