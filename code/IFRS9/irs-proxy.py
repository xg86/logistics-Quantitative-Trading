# -*- coding: utf-8 -*-
"""
IFRS 9 Hedge Memo Pack (from scratch) - openpyxl
- Builds Inputs, Market_Data, IRS_3566, IRS_3568, MTM_Input, Effectiveness_Regression, CVA, PnL_and_Journals
- Uses Excel 365 formulas: XLOOKUP, LET, FILTER, SLOPE/INTERCEPT/RSQ, WORKDAY.INTL
- Does NOT read any old workbook.
"""

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from datetime import date, timedelta
import math


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # Clamp day to end-of-month if needed (we use 11th, so safe)
    last_day = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return date(y, m, min(d.day, last_day))


def month_end(dt: date) -> date:
    nm = add_months(date(dt.year, dt.month, 1), 1)
    return nm - timedelta(days=1)


def make_workbook(path: str) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    # -----------------------
    # Styles
    # -----------------------
    thin = Side(style='thin', color='D0D0D0')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    input_fill = PatternFill("solid", fgColor="FFF2CC")     # light yellow (editable)
    section_fill = PatternFill("solid", fgColor="E7EFFA")   # light blue (section)
    header_fill = PatternFill("solid", fgColor="1F4E79")    # dark blue
    header_font = Font(color="FFFFFF", bold=True)
    bold = Font(bold=True)

    def set_col_widths(ws, widths):
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

    def add_header(ws, headers):
        for j, h in enumerate(headers, start=1):
            cell = ws.cell(1, j, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
        ws.freeze_panes = "A2"

    # -----------------------
    # Inputs
    # -----------------------
    inputs_ws = wb.create_sheet("Inputs")
    add_header(inputs_ws, ["Field", "Value", "Notes"])
    set_col_widths(inputs_ws, {"A": 34, "B": 26, "C": 60})

    inputs = [
        ("Entity / Trade group", "CITIC Pacific Ltd", ""),
        ("Hedged item (accounting)", "Debt 3567", "Exposure / debt instrument in Reval"),
        ("Hedging instrument", "IRS 3566", "Actual derivative designated"),
        ("Hypothetical derivative (proxy)", "IRS 3568", "Dummy trade used for PV-change proxy"),
        ("Notional (HKD)", 2000000000, ""),
        ("IRS 3566 fixed rate (pay)", 0.0069, "decimal"),
        ("IRS 3568 fixed rate (pay)", 0.017397888, "decimal"),
        ("Proxy margin (bps)", 108, "added to proxy floating"),
        ("Index", "3M HIBOR", ""),
        ("Fixing lag (business days)", 2, "T-2"),
        ("Day count basis", 365, "Act/365F"),
        ("Designation start date", date(2021, 5, 11), ""),
        ("De-designation end date", date(2024, 10, 31), ""),
        ("IRS maturity date (contractual)", date(2025, 5, 11), ""),
        ("Reporting date (valuation)", date(2024, 10, 31), ""),
        ("Regression Min R²", 0.80, ""),
        ("Regression slope LB", 0.90, ""),
        ("Regression slope UB", 1.10, ""),
        ("Retro dollar-offset LB", 0.80, ""),
        ("Retro dollar-offset UB", 1.25, ""),
        ("Counterparty rating", "A", ""),
        ("LGD (fraction)", 0.60, ""),
    ]

    for i, (f, v, n) in enumerate(inputs, start=2):
        inputs_ws.cell(i, 1, value=f).font = bold
        vc = inputs_ws.cell(i, 2, value=v)
        inputs_ws.cell(i, 3, value=n)

        for c in (1, 2, 3):
            cell = inputs_ws.cell(i, c)
            cell.border = border
            cell.alignment = Alignment(vertical="center", wrap_text=True)

        vc.fill = input_fill
        # formats
        if "rate" in f.lower() or "r²" in f.lower() or "lgd" in f.lower():
            vc.number_format = "0.00%"
        if "notional" in f.lower():
            vc.number_format = "#,##0"
        if "bps" in f.lower():
            vc.number_format = "0"
        if "date" in f.lower():
            vc.number_format = "yyyy-mm-dd"

    # Rating dropdown
    rating_list = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
                   "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
    dv = DataValidation(type="list", formula1='"{}"'.format(",".join(rating_list)), allow_blank=False)
    inputs_ws.add_data_validation(dv)
    for r in range(2, 2 + len(inputs)):
        if inputs_ws.cell(r, 1).value == "Counterparty rating":
            dv.add(inputs_ws.cell(r, 2))
            break

    # -----------------------
    # Market_Data
    # -----------------------
    md = wb.create_sheet("Market_Data")
    add_header(md, ["Fixing Date", "3M HIBOR (decimal)", "Valuation Date", "Par Swap Rate (decimal)",
                    "DF_1Y (discount factor)", "HK Holiday Dates (input)"])
    set_col_widths(md, {"A": 14, "B": 18, "C": 16, "D": 20, "E": 22, "F": 22})

    eff = date(2021, 5, 11)
    maturity = date(2025, 5, 11)

    reset_dates = []
    d = eff
    while d <= maturity:
        reset_dates.append(d)
        d = add_months(d, 3)

    # NOTE: This is a SAMPLE holiday list for demo.
    # Replace/extend Market_Data column F with your official HK HKG holiday calendar.
    hk_holidays = [
        date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14),
        date(2024, 4, 1), date(2024, 5, 1),
        date(2024, 12, 25), date(2024, 12, 26),
    ]

    def is_weekend(dt): return dt.weekday() >= 5
    def is_holiday(dt): return dt in hk_holidays

    def prev_biz(dt):
        x = dt - timedelta(days=1)
        while is_weekend(x) or is_holiday(x):
            x -= timedelta(days=1)
        return x

    fixing_dates = []
    for rd in reset_dates:
        fx = rd
        for _ in range(2):  # T-2
            fx = prev_biz(fx)
        fixing_dates.append(fx)

    fixing_dates = sorted(set(fixing_dates))

    # Write dummy fixings
    r = 2
    for fx in fixing_dates:
        md.cell(r, 1, value=fx).number_format = "yyyy-mm-dd"
        t = (fx - fixing_dates[0]).days / 365.0
        hib = 0.002 + 0.005 * math.sin(2 * math.pi * t / 2.5) + 0.0005 * math.cos(2 * math.pi * t / 1.0)
        md.cell(r, 2, value=float(max(0.0001, hib))).number_format = "0.0000%"
        for c in (1, 2):
            md.cell(r, c).fill = input_fill
            md.cell(r, c).border = border
        r += 1
    last_fix_row = r - 1

    # Monthly valuation dates (month-end) with dummy Par and DF_1Y
    val_dates = []
    cur = month_end(date(2021, 5, 1))
    end = month_end(date(2024, 10, 1))
    while cur <= end:
        val_dates.append(cur)
        cur = month_end(add_months(cur, 1))

    r = 2
    for vd in val_dates:
        md.cell(r, 3, value=vd).number_format = "yyyy-mm-dd"
        tt = (vd - val_dates[0]).days / 365.0
        par = 0.006 + 0.012 * (1 / (1 + math.exp(-(tt - 1.5)))) + 0.002 * math.sin(2 * math.pi * tt / 3.2)
        rrate = max(0.001, par)
        df1y = math.exp(-rrate)  # DF_1Y < 1
        md.cell(r, 4, value=float(par)).number_format = "0.0000%"
        md.cell(r, 5, value=float(df1y)).number_format = "0.000000"
        for c in (3, 4, 5):
            md.cell(r, c).fill = input_fill
            md.cell(r, c).border = border
        r += 1
    last_val_row = r - 1

    # Holiday list in column F
    for i, h in enumerate(hk_holidays):
        md.cell(2 + i, 6, value=h).number_format = "yyyy-mm-dd"
        md.cell(2 + i, 6).fill = input_fill
        md.cell(2 + i, 6).border = border

    # Ranges used in formulas
    hol_rng = "Market_Data!$F$2:$F$200"
    fix_rng_dates = f"Market_Data!$A$2:$A${last_fix_row}"
    fix_rng_rates = f"Market_Data!$B$2:$B${last_fix_row}"
    df_rng_dates = f"Market_Data!$C$2:$C${last_val_row}"
    df_rng_df1y = f"Market_Data!$E$2:$E${last_val_row}"
    par_rng = f"Market_Data!$D$2:$D${last_val_row}"

    # -----------------------
    # IRS cashflow sheets (3566 / 3568)
    # -----------------------
    def build_irs(name: str, fixed_rate_ref: str, spread_bps_ref: str, has_spread: bool):
        sh = wb.create_sheet(name)
        add_header(sh, ["Period", "Accrual Start", "Accrual End", "Unadj Pay (11th)", "Pay Date (ModFol)",
                        "Fixing Date (T-2)", "HIBOR Fixing", "Float Rate", "YearFrac",
                        "Fixed CF (pay)", "Float CF (receive)", "Net CF"])
        set_col_widths(sh, {"A": 8, "B": 14, "C": 14, "D": 16, "E": 16, "F": 16,
                            "G": 16, "H": 12, "I": 10, "J": 16, "K": 16, "L": 16})

        # Quarterly schedule from eff to maturity
        starts, ends = [], []
        d0 = eff
        while d0 < maturity:
            starts.append(d0)
            ends.append(add_months(d0, 3))
            d0 = add_months(d0, 3)

        for i, (as_dt, ae_dt) in enumerate(zip(starts, ends), start=1):
            row = 1 + i
            sh.cell(row, 1, value=i).border = border
            sh.cell(row, 2, value=as_dt).number_format = "yyyy-mm-dd"
            sh.cell(row, 3, value=ae_dt).number_format = "yyyy-mm-dd"

            # Unadj pay: 11th of accrual end month
            sh.cell(row, 4, value=f"=DATE(YEAR(C{row}),MONTH(C{row}),11)")

            # Pay date: Modified Following using WORKDAY.INTL + holiday list
            sh.cell(row, 5, value=(
                f"=LET(u,D{row},adj,WORKDAY.INTL(u,0,1,{hol_rng}),"
                f"IF(MONTH(adj)<>MONTH(u),WORKDAY.INTL(u,-1,1,{hol_rng}),adj))"
            ))

            # Fixing date: T-2 business days before Accrual Start
            sh.cell(row, 6, value=f"=WORKDAY.INTL(B{row},-Inputs!$B$11,1,{hol_rng})")

            # Fixing lookup
            sh.cell(row, 7, value=f"=XLOOKUP(F{row},{fix_rng_dates},{fix_rng_rates},\"\",0)")

            # Float rate = fixing (+ spread for proxy)
            if has_spread:
                sh.cell(row, 8, value=f"=G{row}+{spread_bps_ref}/10000")
            else:
                sh.cell(row, 8, value=f"=G{row}")

            # YearFrac Act/365F
            sh.cell(row, 9, value=f"=(C{row}-B{row})/Inputs!$B$12")

            # Fixed cashflow (pay) negative; Float cashflow receive positive
            sh.cell(row, 10, value=f"=-{fixed_rate_ref}*Inputs!$B$5*I{row}")
            sh.cell(row, 11, value=f"=Inputs!$B$5*H{row}*I{row}")

            # Net CF = receive float + pay fixed (fixed is negative)
            sh.cell(row, 12, value=f"=K{row}+J{row}")

            for c in range(1, 13):
                cell = sh.cell(row, c)
                cell.border = border
                cell.alignment = Alignment(vertical="center")

            sh.cell(row, 5).number_format = "yyyy-mm-dd"
            for c in (7, 8):
                sh.cell(row, c).number_format = "0.0000%"
            sh.cell(row, 9).number_format = "0.000000"
            for c in (10, 11, 12):
                sh.cell(row, c).number_format = "#,##0.00"

        return len(starts)

    n3566 = build_irs("IRS_3566", "Inputs!$B$6", "Inputs!$B$8", False)
    n3568 = build_irs("IRS_3568", "Inputs!$B$7", "Inputs!$B$8", True)

    # -----------------------
    # MTM_Input (NPV engine + deltas)
    # -----------------------
    mtm = wb.create_sheet("MTM_Input")
    add_header(mtm, ["Valuation Date", "NPV 3566", "NPV 3568", "ΔNPV 3566", "ΔNPV 3568",
                     "DF_1Y", "Par Swap Rate (info)"])
    set_col_widths(mtm, {"A": 16, "B": 16, "C": 16, "D": 14, "E": 14, "F": 14, "G": 18})

    pay_range_3566 = f"IRS_3566!$E$2:$E${n3566+1}"
    net_range_3566 = f"IRS_3566!$L$2:$L${n3566+1}"
    pay_range_3568 = f"IRS_3568!$E$2:$E${n3568+1}"
    net_range_3568 = f"IRS_3568!$L$2:$L${n3568+1}"

    for i, vd in enumerate(val_dates, start=1):
        row = 1 + i
        mtm.cell(row, 1, value=vd).number_format = "yyyy-mm-dd"

        mtm.cell(row, 6, value=f"=XLOOKUP(A{row},{df_rng_dates},{df_rng_df1y},\"\",-1)")
        mtm.cell(row, 7, value=f"=XLOOKUP(A{row},{df_rng_dates},{par_rng},\"\",-1)")

        # NPV = Σ(NetCF * DF^(Δt)) across future paydates up to De-designation end
        mtm.cell(row, 2, value=(
            f"=SUMPRODUCT(({pay_range_3566}>A{row})*({pay_range_3566}<=Inputs!$B$14)*"
            f"{net_range_3566}*POWER($F{row},({pay_range_3566}-A{row})/Inputs!$B$12))"
        ))
        mtm.cell(row, 3, value=(
            f"=SUMPRODUCT(({pay_range_3568}>A{row})*({pay_range_3568}<=Inputs!$B$14)*"
            f"{net_range_3568}*POWER($F{row},({pay_range_3568}-A{row})/Inputs!$B$12))"
        ))

        if i == 1:
            mtm.cell(row, 4, value="")
            mtm.cell(row, 5, value="")
        else:
            mtm.cell(row, 4, value=f"=B{row}-B{row-1}")
            mtm.cell(row, 5, value=f"=C{row}-C{row-1}")

        for c in range(1, 8):
            mtm.cell(row, c).border = border
            mtm.cell(row, c).alignment = Alignment(vertical="center")

        for c in (2, 3, 4, 5):
            mtm.cell(row, c).number_format = "#,##0.00"
        mtm.cell(row, 6).number_format = "0.000000"
        mtm.cell(row, 7).number_format = "0.0000%"

    # -----------------------
    # Effectiveness_Regression (B71:B74 fixed)
    # -----------------------
    reg = wb.create_sheet("Effectiveness_Regression")
    reg["A1"] = "Effectiveness Regression (ΔNPV 3566 vs ΔNPV 3568)"
    reg["A1"].font = Font(bold=True, size=14)
    reg["A6"] = "Data"
    reg["A6"].font = bold

    # Data headers at row 8
    for j, h in enumerate(["Valuation Date", "ΔNPV 3566 (Y)", "ΔNPV 3568 (X)"], start=1):
        cell = reg.cell(8, j, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.border = border

    set_col_widths(reg, {"A": 16, "B": 18, "C": 18, "D": 14, "E": 14})
    for i in range(1, len(val_dates) + 1):
        row = 8 + i
        reg.cell(row, 1, value=f"=MTM_Input!A{1+i}")
        reg.cell(row, 2, value=f"=MTM_Input!D{1+i}")
        reg.cell(row, 3, value=f"=MTM_Input!E{1+i}")
        for c in (1, 2, 3):
            reg.cell(row, c).border = border
            reg.cell(row, c).alignment = Alignment(vertical="center")
        reg.cell(row, 1).number_format = "yyyy-mm-dd"
        reg.cell(row, 2).number_format = "#,##0.00"
        reg.cell(row, 3).number_format = "#,##0.00"

    reg.freeze_panes = "A9"
    data_end = 8 + len(val_dates)
    x_rng = f"$C$9:$C${data_end}"
    y_rng = f"$B$9:$B${data_end}"

    # B71:B74 as requested
    summary = [("Observations (n)", "0"),
               ("Slope (β)", "0.0000"),
               ("Intercept (α)", "#,##0.00"),
               ("R-squared (R²)", "0.0000")]
    for rr, (lab, fmt) in enumerate(summary, start=71):
        reg.cell(rr, 1, value=lab).font = bold
        reg.cell(rr, 1).border = border
        reg.cell(rr, 2).border = border
        reg.cell(rr, 2).fill = input_fill
        reg.cell(rr, 2).number_format = fmt

    reg["B71"] = f"=LET(x,{x_rng},y,{y_rng},m,(x<>\"\")*(y<>\"\"),IFERROR(ROWS(FILTER(x,m)),0))"
    reg["B72"] = f"=LET(x,{x_rng},y,{y_rng},m,(x<>\"\")*(y<>\"\"),xf,FILTER(x,m),yf,FILTER(y,m),IF(ROWS(xf)<3,\"\",SLOPE(yf,xf)))"
    reg["B73"] = f"=LET(x,{x_rng},y,{y_rng},m,(x<>\"\")*(y<>\"\"),xf,FILTER(x,m),yf,FILTER(y,m),IF(ROWS(xf)<3,\"\",INTERCEPT(yf,xf)))"
    reg["B74"] = f"=LET(x,{x_rng},y,{y_rng},m,(x<>\"\")*(y<>\"\"),xf,FILTER(x,m),yf,FILTER(y,m),IF(ROWS(xf)<3,\"\",RSQ(yf,xf)))"

    reg["A76"] = "Status"
    reg["A76"].font = bold
    reg["A76"].border = border
    reg["B76"] = "=IF(OR(B74=\"\",B72=\"\"),\"\",IF(AND(B74>=Inputs!$B$16,B72>=Inputs!$B$17,B72<=Inputs!$B$18),\"PASS\",\"REVIEW\"))"
    reg["B76"].border = border
    reg["B76"].fill = input_fill

    # -----------------------
    # CVA (bucketed + scorecard)
    # -----------------------
    cva = wb.create_sheet("CVA")
    cva["A1"] = "Bucketed CVA (by pay date buckets)"
    cva["A1"].font = Font(bold=True, size=14)
    set_col_widths(cva, {"A": 18, "B": 20, "C": 16, "D": 16, "E": 10, "F": 10, "G": 10,
                         "H": 12, "I": 12, "J": 12, "K": 12, "L": 16, "M": 16})

    # CVA inputs
    cva["A3"] = "Valuation date"; cva["B3"] = "=Inputs!$B$15"
    cva["A4"] = "Rating";        cva["B4"] = "=Inputs!$B$21"
    cva["A5"] = "LGD";           cva["B5"] = "=Inputs!$B$22"
    cva["A6"] = "PD(1Y)";        cva["B6"] = "=XLOOKUP(B4,$D$4:$D$25,$E$4:$E$25,\"\",0)"
    cva["A7"] = "Hazard λ";      cva["B7"] = "=IF(B6=\"\",\"\",-LN(1-B6))"
    cva["A8"] = "DF_1Y(v)";      cva["B8"] = f"=XLOOKUP(B3,{df_rng_dates},{df_rng_df1y},\"\",-1)"
    cva["A9"] = "Total CVA";     cva["B9"] = "=IFERROR(SUM($M$15:$M$200),\"\")"

    for addr in ["A3", "A4", "A5", "A6", "A7", "A8", "A9"]:
        cva[addr].font = bold
    for addr in ["B3", "B4", "B5", "B6", "B7", "B8", "B9"]:
        cva[addr].fill = input_fill
        cva[addr].border = border
    cva["B5"].number_format = "0.00%"
    cva["B6"].number_format = "0.0000%"
    cva["B8"].number_format = "0.000000"
    cva["B9"].number_format = "#,##0.00"

    # Scorecard table (dummy PDs; replace with your internal PD mapping)
    cva["D3"] = "Rating"; cva["E3"] = "PD_1Y"; cva["F3"] = "LGD override"
    for addr in ["D3", "E3", "F3"]:
        cva[addr].font = bold
        cva[addr].fill = section_fill
        cva[addr].border = border

    pd_map = {
        "AAA": 0.0002, "AA+": 0.0003, "AA": 0.0004, "AA-": 0.0006,
        "A+": 0.0008, "A": 0.0010, "A-": 0.0015,
        "BBB+": 0.0025, "BBB": 0.0035, "BBB-": 0.0050,
        "BB+": 0.0100, "BB": 0.0150, "BB-": 0.0250,
        "B+": 0.0450, "B": 0.0600, "B-": 0.0850,
        "CCC+": 0.1500, "CCC": 0.2200, "CCC-": 0.3000,
        "CC": 0.4000, "C": 0.5500, "D": 1.0
    }
    for i, rat in enumerate(rating_list, start=4):
        cva.cell(i, 4, value=rat).border = border
        cva.cell(i, 5, value=pd_map.get(rat, 0.05)).number_format = "0.0000%"
        cva.cell(i, 5).border = border
        cva.cell(i, 6, value="").border = border
        cva.cell(i, 6).fill = input_fill

    # Bucket table
    headers = ["#", "Start", "End (Pay)", "Mid", "tS", "tE", "tM",
               "SurvS", "SurvE", "dPD", "DF(mid)", "EE(start)", "CVA contrib"]
    for j, h in enumerate(headers, start=1):
        cell = cva.cell(14, j, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.border = border
    cva.freeze_panes = "A15"

    # EE(start) uses positive NPV at bucket start from MTM_Input (robust demo).
    # You can replace this with a true EPE engine if desired.
    for i in range(1, n3566 + 1):
        r = 14 + i
        pay_cell = f"IRS_3566!E{i+1}"

        cva.cell(r, 1, value=i).border = border
        cva.cell(r, 3, value=f"={pay_cell}")
        cva.cell(r, 2, value=f"=IF(C{r}=\"\",\"\",IF(C{r}<= $B$3,\"\",IF({i}=1,$B$3,C{r-1})))")
        cva.cell(r, 4, value=f"=IF(B{r}=\"\",\"\",B{r}+(C{r}-B{r})/2)")
        cva.cell(r, 5, value=f"=IF(B{r}=\"\",\"\",(B{r}-$B$3)/365)")
        cva.cell(r, 6, value=f"=IF(B{r}=\"\",\"\",(C{r}-$B$3)/365)")
        cva.cell(r, 7, value=f"=IF(B{r}=\"\",\"\",(D{r}-$B$3)/365)")
        cva.cell(r, 8, value=f"=IF(B{r}=\"\",\"\",EXP(-$B$7*E{r}))")
        cva.cell(r, 9, value=f"=IF(B{r}=\"\",\"\",EXP(-$B$7*F{r}))")
        cva.cell(r, 10, value=f"=IF(B{r}=\"\",\"\",H{r}-I{r})")
        cva.cell(r, 11, value=f"=IF(B{r}=\"\",\"\",POWER($B$8,G{r}))")
        cva.cell(r, 12, value=(
            f"=IF(B{r}=\"\",\"\",MAX(0,"
            f"XLOOKUP(B{r},MTM_Input!$A$2:$A${len(val_dates)+1},MTM_Input!$B$2:$B${len(val_dates)+1},\"\",-1)))"
        ))
        cva.cell(r, 13, value=f"=IF(B{r}=\"\",\"\",$B$5*L{r}*K{r}*J{r})")

        for cc in range(1, 14):
            cva.cell(r, cc).border = border
        for cc in (2, 3, 4):
            cva.cell(r, cc).number_format = "yyyy-mm-dd"
        for cc in (5, 6, 7, 8, 9, 10, 11):
            cva.cell(r, cc).number_format = "0.000000"
        cva.cell(r, 12).number_format = "#,##0.00"
        cva.cell(r, 13).number_format = "#,##0.00"

    # CVA time series (for P&L) – uses a stable factor Σ(DF(mid)*dPD) from the bucket table and scales by EE(date).
    cva["A210"] = "CVA time series (for P&L)"
    cva["A210"].font = bold
    cva["A211"] = "Date"; cva["B211"] = "CVA"
    for addr in ["A211", "B211"]:
        cva[addr].fill = header_fill
        cva[addr].font = header_font
        cva[addr].border = border

    q_dates = []
    q = month_end(date(2021, 6, 1))
    while q <= date(2024, 10, 31):
        q_dates.append(q)
        q = month_end(add_months(q, 3))

    for idx, qd in enumerate(q_dates, start=1):
        rr = 211 + idx
        cva.cell(rr, 1, value=qd).number_format = "yyyy-mm-dd"
        cva.cell(rr, 1).border = border
        cva.cell(rr, 2, value=(
            f"=LET("
            f"ee,MAX(0,XLOOKUP(A{rr},MTM_Input!$A$2:$A${len(val_dates)+1},MTM_Input!$B$2:$B${len(val_dates)+1},\"\",-1)),"
            f"factor,IFERROR(SUM($K$15:$K$200*$J$15:$J$200),0),"
            f"lgd,$B$5,"
            f"ee*lgd*factor)"
        ))
        cva.cell(rr, 2).number_format = "#,##0.00"
        cva.cell(rr, 2).border = border

    # -----------------------
    # PnL_and_Journals (CVA integrated)
    # -----------------------
    pnl = wb.create_sheet("PnL_and_Journals")
    add_header(pnl, ["Date", "NPV3566", "NPV3568", "ΔNPV3566", "ΔNPV3568",
                     "Effective (OCI)", "Ineffective (P&L)", "CVA Close", "ΔCVA (P&L)",
                     "Total P&L (Ineff+ΔCVA)"])
    set_col_widths(pnl, {"A": 16, "B": 16, "C": 16, "D": 14, "E": 14,
                         "F": 16, "G": 16, "H": 14, "I": 14, "J": 18})

    last_ts_row = 211 + len(q_dates)

    for idx, qd in enumerate(q_dates, start=1):
        r = 1 + idx
        pnl.cell(r, 1, value=qd).number_format = "yyyy-mm-dd"
        pnl.cell(r, 2, value=f"=XLOOKUP(A{r},MTM_Input!$A$2:$A${len(val_dates)+1},MTM_Input!$B$2:$B${len(val_dates)+1},\"\",-1)")
        pnl.cell(r, 3, value=f"=XLOOKUP(A{r},MTM_Input!$A$2:$A${len(val_dates)+1},MTM_Input!$C$2:$C${len(val_dates)+1},\"\",-1)")

        if idx == 1:
            pnl.cell(r, 4, value="")
            pnl.cell(r, 5, value="")
        else:
            pnl.cell(r, 4, value=f"=B{r}-B{r-1}")
            pnl.cell(r, 5, value=f"=C{r}-C{r-1}")

        # Simplified effectiveness split:
        pnl.cell(r, 6, value=f"=IF(OR(D{r}=\"\",E{r}=\"\"),\"\",SIGN(D{r})*MIN(ABS(D{r}),ABS(E{r})))")
        pnl.cell(r, 7, value=f"=IF(F{r}=\"\",\"\",D{r}-F{r})")

        pnl.cell(r, 8, value=f"=XLOOKUP(A{r},CVA!$A$212:$A${last_ts_row},CVA!$B$212:$B${last_ts_row},\"\",0)")
        if idx == 1:
            pnl.cell(r, 9, value=f"=IF(H{r}=\"\",\"\",-H{r})")
        else:
            pnl.cell(r, 9, value=f"=IF(OR(H{r}=\"\",H{r-1}=\"\"),\"\",-(H{r}-H{r-1}))")

        pnl.cell(r, 10, value=f"=IF(OR(G{r}=\"\",I{r}=\"\"),\"\",G{r}+I{r})")

        for c in range(1, 11):
            pnl.cell(r, c).border = border
            pnl.cell(r, c).alignment = Alignment(vertical="center")
        for c in range(2, 11):
            pnl.cell(r, c).number_format = "#,##0.00"

    wb.save(path)


if __name__ == "__main__":
    out = "IFRS9_Hedge_Memo_Pack_from_scratch_openpyxl_v2.xlsx"
    make_workbook(out)
    print(f"Wrote: {out}")