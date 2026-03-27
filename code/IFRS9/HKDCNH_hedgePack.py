
from __future__ import annotations

from datetime import date
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side


def build_workbook(path_xlsx: str) -> None:
    """
    Builds an IFRS 9 hedge accounting workbook for a HKD/CNH CCS from scratch.
    - No reading of existing files
    - No LET/FILTER/XLOOKUP (Excel-compat friendly)
    - All key fields are formula-driven
    """

    # ----------------------------
    # 1) Hard-coded sample schedule (matches your HKD100m sample)
    # ----------------------------
    periods = list(range(1, 17))
    effective = date(2025, 7, 10)

    end_dates = [
        date(2025, 7, 31), date(2025, 8, 29), date(2025, 9, 30), date(2025, 10, 31),
        date(2025, 11, 28), date(2025, 12, 31), date(2026, 1, 30), date(2026, 2, 27),
        date(2026, 3, 31), date(2026, 4, 30), date(2026, 5, 29), date(2026, 6, 30),
        date(2026, 7, 31), date(2026, 8, 31), date(2026, 9, 30), date(2026, 10, 30),
    ]

    start_dates = [effective] + end_dates[:-1]
    pay_dates = end_dates[:]  # copy
    pay_dates[-1] = date(2026, 10, 29)  # final pay date adjustment in sample

    # Market data pillar dates (valuation date + all schedule dates)
    val_date = effective
    md_dates = sorted(set([val_date] + start_dates + end_dates + pay_dates))

    # ----------------------------
    # 2) Workbook + styles
    # ----------------------------
    wb = Workbook()
    wb.remove(wb.active)

    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    input_fill = PatternFill("solid", fgColor="FFF2CC")
    note_fill = PatternFill("solid", fgColor="E2EFDA")
    title_font = Font(bold=True, size=14)
    bold = Font(bold=True)

    def set_col_widths(ws, widths):
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

    def header_row(ws, row, labels, start_col=1):
        for j, lab in enumerate(labels, start=start_col):
            c = ws.cell(row=row, column=j, value=lab)
            c.font = header_font
            c.fill = header_fill
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = border

    def put(ws, cell, value=None, formula=None, fill=None, font=None, fmt=None, align=None):
        c = ws[cell]
        c.value = formula if formula is not None else value
        if fill:
            c.fill = fill
        if font:
            c.font = font
        if fmt:
            c.number_format = fmt
        if align:
            c.alignment = align
        c.border = border
        return c

    # ----------------------------
    # 3) Sheets
    # ----------------------------
    sheet_names = [
        "CTRL","CCS_Terms","Schedule","MD_AsOf","CCS_CFs","Hypo_CFH","PV_AsOf",
        "CVA_Scorecard","Hist_Reg_CFH","Eff_CFH","Hist_Reg_NIH","Eff_NIH",
        "Sensitivities","Journals"
    ]
    ws = {name: wb.create_sheet(name) for name in sheet_names}

    # ----------------------------
    # CTRL
    # ----------------------------
    w = ws["CTRL"]
    set_col_widths(w, {"A":40,"B":24,"C":18,"D":18})
    put(w,"A1",value="IFRS 9 Hedge Accounting Engine (CCS) — Inputs / Controls",font=title_font)

    ctrl_rows = [
        ("Valuation date (t)", val_date, "date"),                    # B3
        ("Prior valuation date (t-1)", date(2025,7,8), "date"),      # B4
        ("Spot FX HKD per CNH (t)", 1.0945, "num"),                  # B5
        ("Spot FX HKD per CNH (t-1)", 1.0945, "num"),                # B6
        ("Counterparty rating (for CVA)", "A", "text"),              # B7
        ("Recovery rate (for CVA)", 0.40, "pct"),                    # B8
        ("Prior CVA (t-1) [HKD]", 0.0, "num"),                       # B9
        ("Prior clean FV of CCS (t-1) [HKD]", 0.0, "num"),           # B10
        ("Prior HKD interest component FV (t-1) [CFH] [HKD]", 0.0, "num"),  # B11
        ("Net investment designated amount (CNH) [NIH]", 100_000_000.0, "num"), # B12
        ("Loan margin (decimal, e.g. 0.007 for 0.7%)", 0.007, "num"), # B13
        ("Include margin in HYPO? (Y/N)", "N", "text"),               # B14
        ("Include principal in HYPO? (Y/N)", "N", "text"),            # B15
        ("CFH dummy regression: PV01 per 1bp (HKD)", 8500.0, "num"),  # B16
        ("Curve bump size (decimal, 1bp=0.0001)", 0.0001, "num"),     # B17
        ("FX bump size (HKD per CNH)", 0.0001, "num"),                # B18
    ]
    start_row = 3
    for i,(lab,val,typ) in enumerate(ctrl_rows):
        r = start_row+i
        put(w,f"A{r}",value=lab,font=bold)
        fmt = None
        if typ=="date": fmt="yyyy-mm-dd"
        elif typ=="pct": fmt="0.00%"
        elif typ=="num": fmt="#,##0.00"
        put(w,f"B{r}",value=val,fill=input_fill,fmt=fmt)

    put(w,"A21",value="Notes:",font=bold)
    put(w,"A22",value="• CFH: hedged risk normally = HIBOR component only (loan margin excluded).",fill=note_fill)
    put(w,"A23",value="• NIH: hedged item = net investment translation (IAS 21) to OCI.",fill=note_fill)
    put(w,"A24",value="• CVA here is simplified. Replace with your bank’s approved methodology.",fill=note_fill)

    # ----------------------------
    # CCS_Terms
    # ----------------------------
    w = ws["CCS_Terms"]
    set_col_widths(w, {"A":36,"B":22,"C":22})
    put(w,"A1",value="CCS / CCIRS Terms",font=title_font)

    terms = [
        ("HKD Notional (receive leg)", 100_000_000.0, "#,##0"),                 # B3
        ("FX rate (HKD per CNH) for notional conversion", 1.0945, "0.0000"),    # B4
        ("CNH Notional (pay leg)", None, "#,##0.00"),                           # B5 formula
        ("CNH Fixed Rate (p.a.)", 0.0147, "0.0000%"),                           # B6
        ("HKD Leg Index", "1M HIBOR", None),                                    # B7
        ("HKD Daycount basis for YEARFRAC", 3, "0"),                            # B8 (ACT/365)
        ("CNH Daycount basis for YEARFRAC", 2, "0"),                            # B9 (ACT/360)
        ("Effective date", effective, "yyyy-mm-dd"),                            # B10
        ("Termination date", end_dates[-1], "yyyy-mm-dd"),                      # B11
    ]
    r0=3
    for i,(lab,val,fmt) in enumerate(terms):
        r=r0+i
        put(w,f"A{r}",value=lab,font=bold)
        if lab=="CNH Notional (pay leg)":
            put(w,f"B{r}",formula="=B3/B4",fill=input_fill,fmt=fmt)
        else:
            put(w,f"B{r}",value=val,fill=input_fill if i<4 else None,fmt=fmt)

    # ----------------------------
    # Schedule
    # ----------------------------
    w = ws["Schedule"]
    set_col_widths(w, {"A":10,"B":14,"C":14,"D":14})
    put(w,"A1",value="Monthly Schedule (sample)",font=title_font)
    header_row(w,3,["Period","Start Date","End Date","Pay Date"])
    sched_start_row=4
    for i,p in enumerate(periods):
        r=sched_start_row+i
        put(w,f"A{r}",value=p)
        put(w,f"B{r}",value=start_dates[i],fmt="yyyy-mm-dd")
        put(w,f"C{r}",value=end_dates[i],fmt="yyyy-mm-dd")
        put(w,f"D{r}",value=pay_dates[i],fmt="yyyy-mm-dd")
    last_sched_row = sched_start_row+len(periods)-1

    # ----------------------------
    # MD_AsOf (dummy flat-rate DF generator, easily replaced)
    # ----------------------------
    w = ws["MD_AsOf"]
    set_col_widths(w, {"A":14,"B":16,"C":16,"D":20})
    put(w,"A1",value="Market Data (As-of) — Discount Factors + Flat-rate generator (dummy)",font=title_font)
    put(w,"A3",value="Valuation date",font=bold)
    put(w,"B3",formula="=CTRL!B3",fill=input_fill,fmt="yyyy-mm-dd")
    put(w,"A4",value="HKD flat rate (dummy)",font=bold)
    put(w,"B4",value=0.03,fill=input_fill,fmt="0.0000%")
    put(w,"A5",value="CNH flat rate (dummy)",font=bold)
    put(w,"B5",value=0.025,fill=input_fill,fmt="0.0000%")

    header_row(w,7,["Date","HKD DF","CNH DF","YearFrac (HKD basis)"])
    md_start_row=8
    for i,dt in enumerate(md_dates):
        r=md_start_row+i
        put(w,f"A{r}",value=dt,fmt="yyyy-mm-dd")
        put(w,f"B{r}",formula=f"=IF(A{r}=$B$3,1,EXP(-$B$4*YEARFRAC($B$3,A{r},3)))",fmt="0.000000")
        put(w,f"C{r}",formula=f"=IF(A{r}=$B$3,1,EXP(-$B$5*YEARFRAC($B$3,A{r},2)))",fmt="0.000000")
        put(w,f"D{r}",formula=f"=YEARFRAC($B$3,A{r},3)",fmt="0.000000")
    md_last_row=md_start_row+len(md_dates)-1

    md_date_rng = f"MD_AsOf!$A${md_start_row}:$A${md_last_row}"
    md_hkd_rng  = f"MD_AsOf!$B${md_start_row}:$B${md_last_row}"
    md_cnh_rng  = f"MD_AsOf!$C${md_start_row}:$C${md_last_row}"

    # ----------------------------
    # CCS_CFs
    # ----------------------------
    w = ws["CCS_CFs"]
    set_col_widths(w, {c:12 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
    put(w,"A1",value="CCS Cashflows + PV (clean) + Exposure proxy for CVA",font=title_font)

    put(w,"A3",value="Spot FX (HKD per CNH)",font=bold); put(w,"B3",formula="=CTRL!B5",fill=input_fill,fmt="0.0000")
    put(w,"A4",value="HKD Notional",font=bold);         put(w,"B4",formula="=CCS_Terms!B3",fill=input_fill,fmt="#,##0")
    put(w,"A5",value="CNH Notional",font=bold);         put(w,"B5",formula="=CCS_Terms!B5",fill=input_fill,fmt="#,##0.00")
    put(w,"A6",value="CNH Fixed Rate",font=bold);       put(w,"B6",formula="=CCS_Terms!B6",fill=input_fill,fmt="0.0000%")

    headers = [
        "Per","Start","End","Pay",
        "AF_HKD","DF_HKD_Start","DF_HKD_End","Fwd_HKD",
        "HKD_IntCF","HKD_PrnCF","HKD_TotCF","DF_HKD_Pay","PV_HKD",
        "AF_CNH","DF_CNH_Pay",
        "CNH_IntCF","CNH_PrnCF","CNH_TotCF","PV_CNH","PV_CNH(HKD)",
        "NetPV(HKD)","PV_Remaining","EE_Positive"
    ]
    header_row(w,8,headers)

    cf_start=9
    nper=len(periods)
    last_cf_row=cf_start+nper-1

    for i in range(nper):
        r=cf_start+i
        put(w,f"A{r}",formula=f"=Schedule!A{sched_start_row+i}")
        put(w,f"B{r}",formula=f"=Schedule!B{sched_start_row+i}",fmt="yyyy-mm-dd")
        put(w,f"C{r}",formula=f"=Schedule!C{sched_start_row+i}",fmt="yyyy-mm-dd")
        put(w,f"D{r}",formula=f"=Schedule!D{sched_start_row+i}",fmt="yyyy-mm-dd")

        put(w,f"E{r}",formula=f"=YEARFRAC(B{r},C{r},CCS_Terms!B8)",fmt="0.000000")
        put(w,f"F{r}",formula=f"=INDEX({md_hkd_rng},MATCH(B{r},{md_date_rng},0))",fmt="0.000000")
        put(w,f"G{r}",formula=f"=INDEX({md_hkd_rng},MATCH(C{r},{md_date_rng},0))",fmt="0.000000")
        put(w,f"H{r}",formula=f"=((F{r}/G{r})-1)/E{r}",fmt="0.000000%")

        put(w,f"I{r}",formula=f"=$B$4*((F{r}/G{r})-1)",fmt="#,##0.00")
        put(w,f"J{r}",formula=f"=IF(A{r}=MAX($A${cf_start}:$A${last_cf_row}),$B$4,0)",fmt="#,##0.00")
        put(w,f"K{r}",formula=f"=I{r}+J{r}",fmt="#,##0.00")

        put(w,f"L{r}",formula=f"=INDEX({md_hkd_rng},MATCH(D{r},{md_date_rng},0))",fmt="0.000000")
        put(w,f"M{r}",formula=f"=K{r}*L{r}",fmt="#,##0.00")

        put(w,f"N{r}",formula=f"=YEARFRAC(B{r},C{r},CCS_Terms!B9)",fmt="0.000000")
        put(w,f"O{r}",formula=f"=INDEX({md_cnh_rng},MATCH(D{r},{md_date_rng},0))",fmt="0.000000")

        put(w,f"P{r}",formula=f"=-$B$5*$B$6*N{r}",fmt="#,##0.00")
        put(w,f"Q{r}",formula=f"=IF(A{r}=MAX($A${cf_start}:$A${last_cf_row}),-$B$5,0)",fmt="#,##0.00")
        put(w,f"R{r}",formula=f"=P{r}+Q{r}",fmt="#,##0.00")

        put(w,f"S{r}",formula=f"=R{r}*O{r}",fmt="#,##0.00")
        put(w,f"T{r}",formula=f"=$B$3*S{r}",fmt="#,##0.00")
        put(w,f"U{r}",formula=f"=M{r}+T{r}",fmt="#,##0.00")

        put(w,f"V{r}",formula=f"=SUM(INDEX($U${cf_start}:$U${last_cf_row},ROW()-{cf_start-1}):$U${last_cf_row})",fmt="#,##0.00")
        put(w,f"W{r}",formula=f"=MAX(V{r},0)",fmt="#,##0.00")

    # Totals
    put(w,f"T{last_cf_row+2}",value="Totals",font=bold)
    put(w,f"M{last_cf_row+2}",formula=f"=SUM($M${cf_start}:$M${last_cf_row})",fill=input_fill,fmt="#,##0.00")
    put(w,f"S{last_cf_row+2}",formula=f"=SUM($S${cf_start}:$S${last_cf_row})",fill=input_fill,fmt="#,##0.00")
    put(w,f"U{last_cf_row+2}",formula=f"=SUM($U${cf_start}:$U${last_cf_row})",fill=input_fill,fmt="#,##0.00")

    put(w,f"M{last_cf_row+3}",value="HKD interest PV (component)",font=bold)
    put(w,f"U{last_cf_row+3}",
        formula=f"=SUM($M${cf_start}:$M${last_cf_row})-($B$4*INDEX($L${cf_start}:$L${last_cf_row},ROWS($L${cf_start}:$L${last_cf_row})))",
        fill=input_fill,fmt="#,##0.00")
    put(w,f"V{last_cf_row+3}",value="(PV HKD total - principal PV)",fill=note_fill)

    # ----------------------------
    # Hypo_CFH
    # ----------------------------
    w = ws["Hypo_CFH"]
    set_col_widths(w, {"A":14,"B":14,"C":14,"D":14,"E":14,"F":16,"G":16,"H":16,"I":16})
    put(w,"A1",value="Hypothetical Derivative (CFH) — HKD loan interest cashflows (benchmark component)",font=title_font)

    put(w,"A3",value="Loan Notional (HKD)",font=bold); put(w,"B3",formula="=CCS_Terms!B3",fill=input_fill,fmt="#,##0")
    put(w,"A4",value="Loan margin (decimal)",font=bold); put(w,"B4",formula="=CTRL!B13",fill=input_fill,fmt="0.0000%")
    put(w,"A5",value="Include margin? (Y/N)",font=bold); put(w,"B5",formula="=CTRL!B14",fill=input_fill)
    put(w,"A6",value="Include principal? (Y/N)",font=bold); put(w,"B6",formula="=CTRL!B15",fill=input_fill)

    header_row(w,8,["Per","Start","End","Pay","AF(HKD)","DF_Start","DF_End","HIBOR_CF","Margin_CF","Principal_CF","Total_CF","DF_Pay","PV"])

    hypo_start=9
    hypo_last=hypo_start+nper-1
    for i in range(nper):
        r=hypo_start+i
        put(w,f"A{r}",formula=f"=Schedule!A{sched_start_row+i}")
        put(w,f"B{r}",formula=f"=Schedule!B{sched_start_row+i}",fmt="yyyy-mm-dd")
        put(w,f"C{r}",formula=f"=Schedule!C{sched_start_row+i}",fmt="yyyy-mm-dd")
        put(w,f"D{r}",formula=f"=Schedule!D{sched_start_row+i}",fmt="yyyy-mm-dd")
        put(w,f"E{r}",formula=f"=YEARFRAC(B{r},C{r},CCS_Terms!B8)",fmt="0.000000")
        put(w,f"F{r}",formula=f"=INDEX({md_hkd_rng},MATCH(B{r},{md_date_rng},0))",fmt="0.000000")
        put(w,f"G{r}",formula=f"=INDEX({md_hkd_rng},MATCH(C{r},{md_date_rng},0))",fmt="0.000000")

        put(w,f"H{r}",formula=f"=$B$3*((F{r}/G{r})-1)",fmt="#,##0.00")
        put(w,f"I{r}",formula=f"=IF($B$5=\"Y\",$B$3*$B$4*E{r},0)",fmt="#,##0.00")
        put(w,f"J{r}",formula=f"=IF(AND($B$6=\"Y\",A{r}=MAX($A${hypo_start}:$A${hypo_last})), $B$3, 0)",fmt="#,##0.00")
        put(w,f"K{r}",formula=f"=H{r}+I{r}+J{r}",fmt="#,##0.00")
        put(w,f"L{r}",formula=f"=INDEX({md_hkd_rng},MATCH(D{r},{md_date_rng},0))",fmt="0.000000")
        put(w,f"M{r}",formula=f"=K{r}*L{r}",fmt="#,##0.00")

    put(w,"A27",value="PV(t) (Hypo)",font=bold)
    put(w,"B27",formula=f"=SUM($M${hypo_start}:$M${hypo_last})",fill=input_fill,fmt="#,##0.00")
    put(w,"A28",value="PV(t-1) input",font=bold)
    put(w,"B28",formula="=CTRL!B11",fill=input_fill,fmt="#,##0.00")
    put(w,"A29",value="ΔPV Hedged item (HD)",font=bold)
    put(w,"B29",formula="=B27-B28",fill=input_fill,fmt="#,##0.00")

    # ----------------------------
    # PV_AsOf
    # ----------------------------
    w = ws["PV_AsOf"]
    set_col_widths(w, {"A":44,"B":22,"C":22})
    put(w,"A1",value="PV / FV Summary (Clean + CVA) and ΔFV",font=title_font)

    lines = [
        ("Clean PV HKD leg (HKD)", f"=CCS_CFs!M{last_cf_row+2}", "#,##0.00"),
        ("Clean PV CNH leg (CNH)", f"=CCS_CFs!S{last_cf_row+2}", "#,##0.00"),
        ("Clean PV CNH leg converted (HKD)", f"=CTRL!B5*CCS_CFs!S{last_cf_row+2}", "#,##0.00"),
        ("Clean total FV of CCS (HKD)", f"=CCS_CFs!U{last_cf_row+2}", "#,##0.00"),
        ("CVA (HKD) [positive = cost]", f"=CVA_Scorecard!B30", "#,##0.00"),
        ("Adjusted FV incl. CVA (HKD)", f"=B6-B7", "#,##0.00"),
        ("Prior clean FV (t-1) (HKD)", "=CTRL!B10", "#,##0.00"),
        ("ΔClean FV (t vs t-1) (HKD)", "=B6-B9", "#,##0.00"),
        ("Prior CVA (t-1) (HKD)", "=CTRL!B9", "#,##0.00"),
        ("ΔCVA (t vs t-1) (HKD)", "=B7-B11", "#,##0.00"),
        ("HKD interest component FV (t) [for CFH]", f"=CCS_CFs!U{last_cf_row+3}", "#,##0.00"),
        ("HKD interest component FV (t-1) input", "=CTRL!B11", "#,##0.00"),
        ("ΔHI designated component (CFH) (HKD)", "=B13-B14", "#,##0.00"),
    ]
    r0=3
    for i,(lab,form,fmt) in enumerate(lines):
        r=r0+i
        put(w,f"A{r}",value=lab,font=bold)
        put(w,f"B{r}",formula=form,fill=input_fill,fmt=fmt)

    # ----------------------------
    # CVA_Scorecard
    # ----------------------------
    w = ws["CVA_Scorecard"]
    set_col_widths(w, {"A":26,"B":18,"C":18,"D":18,"E":18,"F":18})
    put(w,"A1",value="CVA Scorecard (simplified) + CVA calculation",font=title_font)

    put(w,"A3",value="Counterparty rating",font=bold); put(w,"B3",formula="=CTRL!B7",fill=input_fill)
    put(w,"A4",value="Recovery rate",font=bold);      put(w,"B4",formula="=CTRL!B8",fill=input_fill,fmt="0.00%")
    put(w,"A5",value="LGD = 1 - Recovery",font=bold); put(w,"B5",formula="=1-B4",fill=input_fill,fmt="0.00%")

    put(w,"A7",value="Scorecard (dummy PDs)",font=bold)
    header_row(w,8,["Rating","1Y PD","Comment"])
    ratings = [
        ("AAA",0.0002,"dummy"),("AA",0.0005,"dummy"),("A",0.0015,"dummy"),
        ("BBB",0.0040,"dummy"),("BB",0.0150,"dummy"),("B",0.0500,"dummy"),("CCC",0.2000,"dummy")
    ]
    sc_first=9
    for i,(rt,pd1,com) in enumerate(ratings):
        r=sc_first+i
        put(w,f"A{r}",value=rt)
        put(w,f"B{r}",value=pd1,fmt="0.0000%")
        put(w,f"C{r}",value=com)
    sc_last=sc_first+len(ratings)-1

    put(w,"A18",value="1Y PD from scorecard",font=bold)
    put(w,"B18",formula=f"=IFERROR(VLOOKUP($B$3,$A${sc_first}:$B${sc_last},2,FALSE),0.0015)",fill=input_fill,fmt="0.0000%")
    put(w,"A19",value="Hazard rate λ (flat)",font=bold)
    put(w,"B19",formula="=-LN(1-B18)",fill=input_fill,fmt="0.000000")

    put(w,"A21",value="Exposure & CVA by payment date (proxy using remaining PV)",font=bold)
    header_row(w,22,["Pay Date","EE (HKD)","DF (HKD)","t (yrs)","CumPD","ΔPD","CVA contrib"])

    cva_start=23
    for i in range(nper):
        r=cva_start+i
        put(w,f"A{r}",formula=f"=CCS_CFs!D{cf_start+i}",fmt="yyyy-mm-dd")
        put(w,f"B{r}",formula=f"=CCS_CFs!W{cf_start+i}",fmt="#,##0.00")
        put(w,f"C{r}",formula=f"=CCS_CFs!L{cf_start+i}",fmt="0.000000")
        put(w,f"D{r}",formula=f"=YEARFRAC(CTRL!B3,A{r},3)",fmt="0.000000")
        put(w,f"E{r}",formula=f"=1-EXP(-$B$19*D{r})",fmt="0.000000")
        put(w,f"F{r}",formula=f"=E{r}" if i==0 else f"=E{r}-E{r-1}",fmt="0.000000")
        put(w,f"G{r}",formula=f"=$B$5*B{r}*C{r}*F{r}",fmt="#,##0.00")

    cva_last=cva_start+nper-1
    put(w,"A30",value="Total CVA (HKD)",font=bold)
    put(w,"B30",formula=f"=SUM($G${cva_start}:$G${cva_last})",fill=input_fill,fmt="#,##0.00")

    # ----------------------------
    # Hist_Reg_CFH (dummy)
    # ----------------------------
    w = ws["Hist_Reg_CFH"]
    set_col_widths(w, {"A":14,"B":18,"C":18,"D":12,"E":18,"F":18,"G":18,"H":20,"I":20})
    put(w,"A1",value="Historical Series (Dummy) — Cash Flow Hedge (CFH)",font=title_font)
    put(w,"A3",value="Replace with real ΔFV history: hedging instrument component vs hypothetical derivative.",fill=note_fill)

    header_row(w,5,[
        "Obs Date",
        "ΔHI (designated)",
        "ΔHD (hypo)",
        "Include?",
        "x = -ΔHD",
        "y = ΔHI",
        "Dollar offset |HI|/|HD|",
        "Effective (OCI) [lower-of]",
        "Ineffectiveness (P&L)",
    ])
    reg_start=6
    nobs=36
    for i in range(nobs):
        r=reg_start+i
        put(w,f"A{r}",formula=f"=EDATE(CTRL!B3,-{nobs-1-i})",fmt="yyyy-mm-dd")
        idx=i+1
        # Dummy 1M HKD rate path and monthly Δrate
        delta_rate = f"(0.03+0.002*SIN({idx}/4))-(0.03+0.002*SIN({idx-1}/4))"
        # Dummy ΔHD from PV01 per 1bp: ΔPV ≈ -PV01 * (Δrate / bump)
        put(w,f"C{r}",formula=f"=-CTRL!B16*(({delta_rate})/CTRL!B17)",fmt="#,##0.00")
        # Dummy ΔHI set to offset ΔHD with small noise
        put(w,f"B{r}",formula=f"=-C{r}*(1+0.03*COS({idx}/3))",fmt="#,##0.00")
        put(w,f"D{r}",value="Y")
        put(w,f"E{r}",formula=f"=IF(D{r}=\"Y\",-C{r},\"\")")
        put(w,f"F{r}",formula=f"=IF(D{r}=\"Y\",B{r},\"\")")
        put(w,f"G{r}",formula=f"=IF(D{r}=\"Y\",IFERROR(ABS(B{r})/ABS(C{r}),\"\"),\"\" )",fmt="0.0000")
        put(w,f"H{r}",formula=f"=IF(D{r}=\"Y\",SIGN(B{r})*MIN(ABS(B{r}),ABS(C{r})),\"\" )",fmt="#,##0.00")
        put(w,f"I{r}",formula=f"=IF(D{r}=\"Y\",B{r}-H{r},\"\" )",fmt="#,##0.00")
    reg_last=reg_start+nobs-1

    # ----------------------------
    # Eff_CFH (retro + regression with SUMPRODUCT)
    # ----------------------------
    w = ws["Eff_CFH"]
    set_col_widths(w, {"A":44,"B":22,"C":22,"D":22,"E":22})
    put(w,"A1",value="Effectiveness — Cash Flow Hedge (CFH)",font=title_font)

    put(w,"A3",value="Retrospective (t vs t-1) — Dollar offset + lower-of",font=bold)
    put(w,"A4",value="ΔHI (designated component, HKD)",font=bold)
    put(w,"B4",formula="=PV_AsOf!B15",fill=input_fill,fmt="#,##0.00")
    put(w,"A5",value="ΔHD (hypo, HKD)",font=bold)
    put(w,"B5",formula="=Hypo_CFH!B29",fill=input_fill,fmt="#,##0.00")
    put(w,"A6",value="Dollar offset (signed) = -ΔHI/ΔHD",font=bold)
    put(w,"B6",formula="=IFERROR(-B4/B5,\"\")",fill=input_fill)
    put(w,"A7",value="Dollar offset (absolute)",font=bold)
    put(w,"B7",formula="=IFERROR(ABS(B4)/ABS(B5),\"\")",fill=input_fill)
    put(w,"A8",value="Effective portion (OCI) [lower-of]",font=bold)
    put(w,"B8",formula="=SIGN(B4)*MIN(ABS(B4),ABS(B5))",fill=input_fill,fmt="#,##0.00")
    put(w,"A9",value="Ineffectiveness to P&L",font=bold)
    put(w,"B9",formula="=B4-B8",fill=input_fill,fmt="#,##0.00")

    put(w,"A11",value="Prospective regression (dummy history, SUMPRODUCT)",font=bold)
    put(w,"A12",value="n (included obs)",font=bold)
    put(w,"B12",formula=f"=SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"))",fill=input_fill)
    put(w,"A13",value="x̄",font=bold)
    put(w,"B13",formula=f"=IF(B12=0,\"\",SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"),Hist_Reg_CFH!$E${reg_start}:$E${reg_last})/B12)",fill=input_fill)
    put(w,"A14",value="ȳ",font=bold)
    put(w,"B14",formula=f"=IF(B12=0,\"\",SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"),Hist_Reg_CFH!$F${reg_start}:$F${reg_last})/B12)",fill=input_fill)
    put(w,"A15",value="Slope",font=bold)
    put(w,"B15",formula=(
        f"=IF(B12<3,\"\","
        f"SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"),"
        f"(Hist_Reg_CFH!$E${reg_start}:$E${reg_last}-B13),"
        f"(Hist_Reg_CFH!$F${reg_start}:$F${reg_last}-B14))/"
        f"SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"),"
        f"(Hist_Reg_CFH!$E${reg_start}:$E${reg_last}-B13)^2)"
        f")"
    ), fill=input_fill)
    put(w,"A16",value="Intercept",font=bold)
    put(w,"B16",formula="=IF(B15=\"\",\"\",B14-B15*B13)",fill=input_fill)
    put(w,"A17",value="R²",font=bold)
    put(w,"B17",formula=(
        f"=IF(B15=\"\",\"\","
        f"SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"),"
        f"(B16+B15*Hist_Reg_CFH!$E${reg_start}:$E${reg_last}-B14)^2)"
        f"/SUMPRODUCT(--(Hist_Reg_CFH!$D${reg_start}:$D${reg_last}=\"Y\"),"
        f"(Hist_Reg_CFH!$F${reg_start}:$F${reg_last}-B14)^2)"
        f")"
    ), fill=input_fill)
    put(w,"A18",value="Pass (example): R²≥0.80 and slope 0.80–1.25",font=bold)
    put(w,"B18",formula="=IF(AND(B17>=0.8,B15>=0.8,B15<=1.25),\"PASS\",\"REVIEW\")",fill=input_fill)

    # ----------------------------
    # Hist_Reg_NIH (dummy)
    # ----------------------------
    w = ws["Hist_Reg_NIH"]
    set_col_widths(w, {"A":14,"B":18,"C":18,"D":12,"E":18,"F":18,"G":16,"H":16})
    put(w,"A1",value="Historical Series (Dummy) — Net Investment Hedge (NIH)",font=title_font)
    put(w,"A3",value="Replace with real history: ΔFV(hedging instrument) vs net investment translation Δ.",fill=note_fill)

    header_row(w,5,["Obs Date","ΔHI (CCS total)","ΔHD (translation)","Include?","x = -ΔHD","y = ΔHI","Spot","ΔSpot"])
    reg_start2=6
    nobs2=36
    for i in range(nobs2):
        r=reg_start2+i
        idx=i+1
        put(w,f"A{r}",formula=f"=EDATE(CTRL!B3,-{nobs2-1-i})",fmt="yyyy-mm-dd")
        put(w,f"G{r}",formula=f"=CTRL!B6*(1+0.02*SIN({idx}/5))",fmt="0.0000")
        put(w,f"H{r}",formula="=0" if i==0 else f"=G{r}-G{r-1}",fmt="0.0000")
        put(w,f"C{r}",formula=f"=CTRL!B12*H{r}",fmt="#,##0.00")
        put(w,f"B{r}",formula=f"=-CCS_Terms!B5*H{r}*(1+0.02*COS({idx}/4))",fmt="#,##0.00")
        put(w,f"D{r}",value="Y")
        put(w,f"E{r}",formula=f"=IF(D{r}=\"Y\",-C{r},\"\")")
        put(w,f"F{r}",formula=f"=IF(D{r}=\"Y\",B{r},\"\")")
    reg_last2=reg_start2+nobs2-1

    # ----------------------------
    # Eff_NIH
    # ----------------------------
    w = ws["Eff_NIH"]
    set_col_widths(w, {"A":44,"B":22,"C":22,"D":22,"E":22})
    put(w,"A1",value="Effectiveness — Net Investment Hedge (NIH)",font=title_font)

    put(w,"A3",value="Retrospective (t vs t-1) — Dollar offset + lower-of",font=bold)
    put(w,"A4",value="ΔHI (clean CCS FV change, HKD)",font=bold)
    put(w,"B4",formula="=PV_AsOf!B10",fill=input_fill,fmt="#,##0.00")
    put(w,"A5",value="ΔHD (net investment translation, HKD)",font=bold)
    put(w,"B5",formula="=CTRL!B12*(CTRL!B5-CTRL!B6)",fill=input_fill,fmt="#,##0.00")
    put(w,"A6",value="Dollar offset (signed) = -ΔHI/ΔHD",font=bold)
    put(w,"B6",formula="=IFERROR(-B4/B5,\"\")",fill=input_fill)
    put(w,"A7",value="Dollar offset (absolute)",font=bold)
    put(w,"B7",formula="=IFERROR(ABS(B4)/ABS(B5),\"\")",fill=input_fill)
    put(w,"A8",value="Effective portion (OCI – FCTR) [lower-of]",font=bold)
    put(w,"B8",formula="=SIGN(B4)*MIN(ABS(B4),ABS(B5))",fill=input_fill,fmt="#,##0.00")
    put(w,"A9",value="Ineffectiveness to P&L",font=bold)
    put(w,"B9",formula="=B4-B8",fill=input_fill,fmt="#,##0.00")

    put(w,"A11",value="Prospective regression (dummy history, SUMPRODUCT)",font=bold)
    put(w,"A12",value="n (included obs)",font=bold)
    put(w,"B12",formula=f"=SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"))",fill=input_fill)
    put(w,"A13",value="x̄",font=bold)
    put(w,"B13",formula=f"=IF(B12=0,\"\",SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"),Hist_Reg_NIH!$E${reg_start2}:$E${reg_last2})/B12)",fill=input_fill)
    put(w,"A14",value="ȳ",font=bold)
    put(w,"B14",formula=f"=IF(B12=0,\"\",SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"),Hist_Reg_NIH!$F${reg_start2}:$F${reg_last2})/B12)",fill=input_fill)
    put(w,"A15",value="Slope",font=bold)
    put(w,"B15",formula=(
        f"=IF(B12<3,\"\","
        f"SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"),"
        f"(Hist_Reg_NIH!$E${reg_start2}:$E${reg_last2}-B13),"
        f"(Hist_Reg_NIH!$F${reg_start2}:$F${reg_last2}-B14))/"
        f"SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"),"
        f"(Hist_Reg_NIH!$E${reg_start2}:$E${reg_last2}-B13)^2)"
        f")"
    ), fill=input_fill)
    put(w,"A16",value="Intercept",font=bold)
    put(w,"B16",formula="=IF(B15=\"\",\"\",B14-B15*B13)",fill=input_fill)
    put(w,"A17",value="R²",font=bold)
    put(w,"B17",formula=(
        f"=IF(B15=\"\",\"\","
        f"SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"),"
        f"(B16+B15*Hist_Reg_NIH!$E${reg_start2}:$E${reg_last2}-B14)^2)"
        f"/SUMPRODUCT(--(Hist_Reg_NIH!$D${reg_start2}:$D${reg_last2}=\"Y\"),"
        f"(Hist_Reg_NIH!$F${reg_start2}:$F${reg_last2}-B14)^2)"
        f")"
    ), fill=input_fill)
    put(w,"A18",value="Pass (example): R²≥0.80 and slope 0.80–1.25",font=bold)
    put(w,"B18",formula="=IF(AND(B17>=0.8,B15>=0.8,B15<=1.25),\"PASS\",\"REVIEW\")",fill=input_fill)

    # ----------------------------
    # Sensitivities
    # ----------------------------
    w = ws["Sensitivities"]
    set_col_widths(w, {"A":34,"B":18,"C":18,"D":18})
    put(w,"A1",value="Sensitivities (simple)",font=title_font)

    put(w,"A3",value="Base FV (clean, HKD)",font=bold)
    put(w,"B3",formula="=PV_AsOf!B6",fill=input_fill,fmt="#,##0.00")
    put(w,"A4",value="Base PV CNH leg (CNH)",font=bold)
    put(w,"B4",formula="=PV_AsOf!B4",fill=input_fill,fmt="#,##0.00")
    put(w,"A6",value="FX01 (per FX bump in CTRL!B18)",font=bold)
    put(w,"B6",formula="=PV_AsOf!B4*CTRL!B18",fill=input_fill,fmt="#,##0.00")
    put(w,"C6",value="approx dPV/dSpot = PV_CNH",fill=note_fill)

    put(w,"A8",value="HKD curve DV01 (parallel bump, HKD leg only)",font=bold)
    put(w,"B8",formula="=CTRL!B17",fill=input_fill,fmt="0.0000%")

    header_row(w,10,["Pay Date","HKD CF","DF base","DF bumped","PV base","PV bumped"])
    sens_start=11
    for i in range(nper):
        r=sens_start+i
        put(w,f"A{r}",formula=f"=CCS_CFs!D{cf_start+i}",fmt="yyyy-mm-dd")
        put(w,f"B{r}",formula=f"=CCS_CFs!K{cf_start+i}",fmt="#,##0.00")
        put(w,f"C{r}",formula=f"=CCS_CFs!L{cf_start+i}",fmt="0.000000")
        put(w,f"D{r}",formula=f"=C{r}*EXP(-$B$8*YEARFRAC(CTRL!B3,A{r},3))",fmt="0.000000")
        put(w,f"E{r}",formula=f"=B{r}*C{r}",fmt="#,##0.00")
        put(w,f"F{r}",formula=f"=B{r}*D{r}",fmt="#,##0.00")
    sens_last=sens_start+nper-1
    put(w,"A28",value="HKD DV01 = PV bumped - base",font=bold)
    put(w,"B28",formula=f"=SUM(F{sens_start}:F{sens_last})-SUM(E{sens_start}:E{sens_last})",fill=input_fill,fmt="#,##0.00")

    put(w,"A30",value="CNH curve DV01 (parallel bump, converted to HKD)",font=bold)
    put(w,"B30",formula="=CTRL!B17",fill=input_fill,fmt="0.0000%")

    header_row(w,32,["Pay Date","CNH CF","DF base","DF bumped","PV base (CNH)","PV bumped (CNH)","PV base (HKD)","PV bumped (HKD)"])
    sens2_start=33
    for i in range(nper):
        r=sens2_start+i
        put(w,f"A{r}",formula=f"=CCS_CFs!D{cf_start+i}",fmt="yyyy-mm-dd")
        put(w,f"B{r}",formula=f"=CCS_CFs!R{cf_start+i}",fmt="#,##0.00")
        put(w,f"C{r}",formula=f"=CCS_CFs!O{cf_start+i}",fmt="0.000000")
        put(w,f"D{r}",formula=f"=C{r}*EXP(-$B$30*YEARFRAC(CTRL!B3,A{r},2))",fmt="0.000000")
        put(w,f"E{r}",formula=f"=B{r}*C{r}",fmt="#,##0.00")
        put(w,f"F{r}",formula=f"=B{r}*D{r}",fmt="#,##0.00")
        put(w,f"G{r}",formula=f"=CTRL!B5*E{r}",fmt="#,##0.00")
        put(w,f"H{r}",formula=f"=CTRL!B5*F{r}",fmt="#,##0.00")
    sens2_last=sens2_start+nper-1
    put(w,"A51",value="CNH DV01 (HKD) = PV bumped - base",font=bold)
    put(w,"B51",formula=f"=SUM(H{sens2_start}:H{sens2_last})-SUM(G{sens2_start}:G{sens2_last})",fill=input_fill,fmt="#,##0.00")

    # ----------------------------
    # Journals (includes CVA)
    # ----------------------------
    w = ws["Journals"]
    set_col_widths(w, {"A":12,"B":36,"C":34,"D":14,"E":14,"F":44})
    put(w,"A1",value="Journal Entries (IFRS 9) — CFH & NIH + CVA",font=title_font)
    put(w,"A3",value="Convention: Debit/Credit columns show positive numbers; direction driven by sign.",fill=note_fill)

    put(w,"A5",value="Key amounts (HKD)",font=bold)
    put(w,"A6",value="ΔClean FV total (t vs t-1)",font=bold); put(w,"B6",formula="=PV_AsOf!B10",fill=input_fill,fmt="#,##0.00")
    put(w,"A7",value="ΔCVA (t vs t-1)",font=bold);            put(w,"B7",formula="=PV_AsOf!B12",fill=input_fill,fmt="#,##0.00")
    put(w,"A8",value="CFH Effective (OCI)",font=bold);         put(w,"B8",formula="=Eff_CFH!B8",fill=input_fill,fmt="#,##0.00")
    put(w,"A9",value="CFH Ineffectiveness (P&L)",font=bold);   put(w,"B9",formula="=Eff_CFH!B9",fill=input_fill,fmt="#,##0.00")
    put(w,"A10",value="NIH Effective (OCI-FCTR)",font=bold);   put(w,"B10",formula="=Eff_NIH!B8",fill=input_fill,fmt="#,##0.00")
    put(w,"A11",value="NIH Ineffectiveness (P&L)",font=bold);  put(w,"B11",formula="=Eff_NIH!B9",fill=input_fill,fmt="#,##0.00")

    put(w,"A13",value="CFH Journals (designated component approach)",font=bold)
    header_row(w,14,["Date","Journal","Account","Debit","Credit","Narrative"])

    # Post total clean FV movement to derivative
    put(w,"A15",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B15",value="CFH-1")
    put(w,"C15",value="Derivative (asset/liability) — clean FV change")
    put(w,"D15",formula="=MAX(PV_AsOf!B10,0)",fmt="#,##0.00")
    put(w,"E15",formula="=MAX(-PV_AsOf!B10,0)",fmt="#,##0.00")
    put(w,"F15",value="Recognize total clean FV movement on derivative (before hedge split).")

    # Offsets: OCI effective + P&L ineff + P&L undesignated residual
    put(w,"A16",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B16",value="CFH-2")
    put(w,"C16",value="OCI – Cash flow hedge reserve (effective)")
    put(w,"D16",formula="=MAX(-Eff_CFH!B8,0)",fmt="#,##0.00")
    put(w,"E16",formula="=MAX(Eff_CFH!B8,0)",fmt="#,##0.00")
    put(w,"F16",value="Effective portion of designated hedge.")

    put(w,"A17",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B17",value="CFH-3")
    put(w,"C17",value="P&L – Hedge ineffectiveness (designated)")
    put(w,"D17",formula="=MAX(-Eff_CFH!B9,0)",fmt="#,##0.00")
    put(w,"E17",formula="=MAX(Eff_CFH!B9,0)",fmt="#,##0.00")
    put(w,"F17",value="Ineffective portion of designated hedge.")

    put(w,"A18",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B18",value="CFH-4")
    put(w,"C18",value="P&L – Undesignated derivative component")
    put(w,"D18",formula="=MAX(-(PV_AsOf!B10-PV_AsOf!B15),0)",fmt="#,##0.00")
    put(w,"E18",formula="=MAX(PV_AsOf!B10-PV_AsOf!B15,0)",fmt="#,##0.00")
    put(w,"F18",value="Residual (FX/CNH/basis etc.) not in the CFH designation.")

    # CVA: reserve + P&L
    put(w,"A20",value="CVA Journals (simplified)",font=bold)
    put(w,"A21",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B21",value="CVA-1")
    put(w,"C21",value="CVA reserve (contra-derivative)")
    put(w,"D21",formula="=MAX(-PV_AsOf!B12,0)",fmt="#,##0.00")
    put(w,"E21",formula="=MAX(PV_AsOf!B12,0)",fmt="#,##0.00")
    put(w,"F21",value="Recognize change in CVA reserve.")

    put(w,"A22",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B22",value="CVA-2")
    put(w,"C22",value="P&L – CVA expense/(release)")
    put(w,"D22",formula="=MAX(PV_AsOf!B12,0)",fmt="#,##0.00")
    put(w,"E22",formula="=MAX(-PV_AsOf!B12,0)",fmt="#,##0.00")
    put(w,"F22",value="Offsetting P&L impact of CVA change.")

    # NIH: show effective and ineffective postings (translation reserve + P&L)
    put(w,"A24",value="NIH Journals (net investment hedge)",font=bold)
    put(w,"A25",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B25",value="NIH-1")
    put(w,"C25",value="OCI – Foreign currency translation reserve (effective)")
    put(w,"D25",formula="=MAX(-Eff_NIH!B8,0)",fmt="#,##0.00")
    put(w,"E25",formula="=MAX(Eff_NIH!B8,0)",fmt="#,##0.00")
    put(w,"F25",value="Effective portion of NIH (offsets translation differences).")

    put(w,"A26",formula="=CTRL!B3",fmt="yyyy-mm-dd")
    put(w,"B26",value="NIH-2")
    put(w,"C26",value="P&L – Hedge ineffectiveness (NIH)")
    put(w,"D26",formula="=MAX(-Eff_NIH!B9,0)",fmt="#,##0.00")
    put(w,"E26",formula="=MAX(Eff_NIH!B9,0)",fmt="#,##0.00")
    put(w,"F26",value="Ineffective portion of net investment hedge.")

    # Force Excel to recalculate all formulas on open
    wb.calculation.calcMode = 'auto'
    wb.calculation.fullCalcOnLoad = True

    wb.save(path_xlsx)


if __name__ == "__main__":
    build_workbook("IFRS9_CCS_Engine_from_scratch_v2.xlsx")