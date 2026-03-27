# build_IFRS9_HKDUSD_FXForward_CFH_CL_FX001_demo_v6_from_scratch.py
# Build "IFRS9_HKDUSD_FXForward_CFH_CL_FX001_demo_v6_CVA_scorecard.xlsx" FROM SCRATCH
# using openpyxl, including formulas for each computed field.
#
# Usage:
#   python build_IFRS9_HKDUSD_FXForward_CFH_CL_FX001_demo_v6_from_scratch.py output.xlsx
#
# Notes:
# - This is a DEMO workbook for IFRS 9 cash flow hedge (HKD/USD) with:
#   * MTN hedged layer (forecast principal cashflow)
#   * FX Forward designated spot element + excluded forward points (COH)
#   * HKD discount curve demo (continuous compounding, linear interpolation)
#   * CVA module with CSA flag and a rating→PD scorecard
#   * 'CVA booked' calibrated to 0 at designation date (T0)
#
# - Excel will calculate formulas on open (openpyxl does not evaluate formulas).

import sys
from datetime import datetime
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.table import Table, TableStyleInfo


# -----------------------------
# Styles / formats
# -----------------------------
MONEY_FMT = '#,##0;[Red]-#,##0;0'
INT_FMT = '#,##0'
FX4_FMT = '0.0000'
DF6_FMT = '0.000000'
PCT5_FMT = '0.00000%'
PCT3_FMT = '0.000%'
PCT1_FMT = '0.0%'
DATE_FMT = 'yyyy-mm-dd'

thin = Side(style='thin', color='D0D0D0')
GRID_BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

HEADER_FILL = PatternFill('solid', fgColor='1F4E79')
HEADER_FONT = Font(color='FFFFFF', bold=True)
HEADER_ALIGN = Alignment(horizontal='center', vertical='center', wrap_text=True)
DATA_ALIGN = Alignment(vertical='top', wrap_text=True)

SCORE_HEADER_FILL = PatternFill('solid', fgColor='D9E1F2')
SCORE_HEADER_FONT = Font(bold=True)
SCORE_HEADER_ALIGN = Alignment(horizontal='center', vertical='center', wrap_text=True)


def dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d)


def set_col_widths(ws, widths: dict) -> None:
    for col, w in widths.items():
        ws.column_dimensions[col].width = float(w)


def apply_grid(ws, ref: str, header_row: int = 1,
               header_fill=HEADER_FILL, header_font=HEADER_FONT, header_align=HEADER_ALIGN) -> None:
    """Apply border/alignment to a rectangular range ref like 'A1:K7'."""
    start, end = ref.split(':')
    start_col = openpyxl.utils.column_index_from_string(''.join([c for c in start if c.isalpha()]))
    start_row = int(''.join([c for c in start if c.isdigit()]))
    end_col = openpyxl.utils.column_index_from_string(''.join([c for c in end if c.isalpha()]))
    end_row = int(''.join([c for c in end if c.isdigit()]))

    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = GRID_BORDER
            cell.alignment = DATA_ALIGN

    if start_row <= header_row <= end_row:
        for c in range(start_col, end_col + 1):
            cell = ws.cell(row=header_row, column=c)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = GRID_BORDER


def add_table(ws, name: str, ref: str) -> None:
    if name in ws.tables:
        del ws.tables[name]
    tbl = Table(displayName=name, ref=ref)
    tbl.tableStyleInfo = TableStyleInfo(
        name='TableStyleMedium9',
        showRowStripes=True,
        showColumnStripes=False
    )
    ws.add_table(tbl)


def main() -> None:
    # if len(sys.argv) != 2:
    print('Usage: python build_IFRS9_HKDUSD_FXForward_CFH_CL_FX001_demo_v6_from_scratch.py output.xlsx')
    #     raise SystemExit(1)

    #out_path = sys.argv[1]
    out_path = 'IFRS9_HKDUSD_FXForward_MTN_CL_FX001_CVA.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.title = 'README'

    # -----------------------------
    # README
    # -----------------------------
    readme_lines = [
        'IFRS 9 HKD/USD FX Forward CFH Demo (CL_FX001) — Monthly valuation + history + HKD discount curve demo',
        None,
        'Changes per request:',
        '1) Removed future valuation date 2026-02-28 (as of Feb 2026).',
        '2) Added SENSITIVITIES_HISTORY and PROSPECTIVE_HISTORY to show per-valuation-date sensitivities/tests.',
        '3) Added HKD_CURVE_DEMO with a simple zero-rate curve and linear interpolation to compute HKD discount rate in MARKET_DATA.',
        None,
        'Production note (discounting):',
        "- In practice you build/consume an HKD discount curve consistent with the trade's collateral/CSA (often OIS/HONIA).",
        '- This file demonstrates a simplified: (a) curve nodes, (b) linear interpolation in zero rates, (c) DF = EXP(-r*t).',
        None,
        'If you prefer to manually input HKD discount rate per month, overwrite MARKET_DATA column D.',
        None,
        'Added MTN_CASHFLOWS and MTN_PV_MONTHLY: MTN principal cashflow (hedged layer + unhedged remainder) and monthly PV/exposure view. HI PV in VALUATION_MONTHLY now uses DF_to_MTN pay date, while hedge FV uses DF_to_settle (1-day mismatch).',
        None,
        'CVA added: SETUP includes CSA flag (Y/N), threshold, PD_1Y and LGD. CVA_MONTHLY computes simplified CVA and ΔCVA. VALUATION_MONTHLY now shows dirty FV (clean + CVA). JOURNALS_MONTHLY includes ΔCVA as P&L credit adjustment.',
        None,
        "Update v6: CVA uses an Expected Positive Exposure (EPE) proxy based on FX volatility; and 'CVA booked' is calibrated to 0 at designation date (T0) to reflect that XVA is embedded in the traded forward rate.",
    ]
    for i, line in enumerate(readme_lines, start=1):
        ws[f'A{i}'] = line
    ws['A1'].font = Font(bold=True)
    ws.freeze_panes = 'A2'
    set_col_widths(ws, {'A': 120})

    # -----------------------------
    # SETUP
    # -----------------------------
    setup = wb.create_sheet('SETUP')
    setup.freeze_panes = 'A2'
    set_col_widths(setup, {'A': 42, 'B': 22, 'C': 60})

    setup_rows = [
        ['Field', 'Value', 'Units/Notes'],
        ['HedgeRelationship_ID', 'H_FX_CL_FX001', 'Internal hedge relationship ID'],
        ['Ref No', 'CL_FX001', 'FX Forward reference'],
        ['Counterparty', 'XYZ Bank', None],
        ['Hedge type', 'Cash flow hedge', None],
        ['Trade date / Designation date (T0)', dt(2025, 9, 2), None],
        ['Settlement date (forward)', dt(2027, 2, 25), None],
        ['MTN principal pay date', dt(2027, 2, 26), 'Hedged item date (1-day mismatch vs settlement)'],
        ['Buy currency', 'USD', None],
        ['USD notional hedged', 40000000, 'USD amount designated as hedged item'],
        ['Sell currency', 'HKD', None],
        ['Forward rate K (HKD per USD)', 7.71, 'Contract forward rate'],
        ['Spot HKDUSD at T0 (HKD per USD)', 7.805, 'Spot at designation'],
        ['Swap points at T0 (K-Spot0)', '=B12-B13', 'Computed'],
        ['HKD discount rate at T0 (cont, r0)', 0.03, 'Demo input; MARKET_DATA uses curve demo by default'],
        ['Year fraction T0->settle (t0)', '=(B7-B6)/365', 'Computed from dates /365'],
        ['DF at T0 (SETUP!B17) = EXP(-r0*t0)', '=EXP(-B15*B16)', 'Computed - required cell location'],
        ['Total MTN principal (USD)', 750000000, 'Total principal outstanding (demo input).'],
        ['Hedged layer of principal (USD)', '=B10', 'Portion of principal designated as hedged item (layer).'],
        ['CSA in place? (Y/N)', 'Y', 'If Y, exposure reduced by threshold in CVA demo.'],
        ['CSA threshold (HKD)', 0, 'Simplified: collateral fully posted above threshold; ignore MTA/margin period.'],
        ['Counterparty rating (S&P)', 'A', 'For documentation only.'],
        ['Counterparty PD (1Y)', '=IFERROR(XLOOKUP(B22,SCORECARD_PD_LGD!$A$2:$A$20,SCORECARD_PD_LGD!$B$2:$B$20),0.002)', 'Demo 1-year default probability (2bp=0.02%? here 0.20%).'],
        ['LGD', 0.60, 'Loss given default (60% typical).'],
        ['CVA method', 'CVA = -LGD*EPE*PD_rem', 'Simplified single-factor reduced-form demo.'],
        ['Credit dominance screen (optional)', 'CVA/FX_PV < 10%', 'Policy example; adjust per governance.'],
        ['Notes', 'If CSA=Y and threshold=0, CVA is close to 0 when derivatives are collateralised daily.', None],
        # 28-30 intentionally outside tblSetup ref (matches v6 behavior)
        ['FX volatility σ (annual)', 0.05, 'Demo input for EPE proxy (e.g., 5%).'],
        ['CSA margin period of risk (MPOR) (days)', 10, 'If CSA=Y, EPE horizon uses MPOR; otherwise uses time-to-settlement.'],
        ['CVA EPE proxy', '=IF($B$20="Y","Use MPOR horizon","Use full remaining time")', 'This is a simplified demo (not Monte Carlo).'],
    ]
    for r in setup_rows:
        setup.append(r)

    for rr in [6, 7, 8]:
        setup[f'B{rr}'].number_format = DATE_FMT
    setup['B10'].number_format = INT_FMT
    setup['B12'].number_format = FX4_FMT
    setup['B13'].number_format = FX4_FMT
    setup['B14'].number_format = FX4_FMT
    setup['B15'].number_format = FX4_FMT
    setup['B16'].number_format = FX4_FMT
    setup['B17'].number_format = DF6_FMT
    setup['B18'].number_format = INT_FMT
    setup['B19'].number_format = INT_FMT
    setup['B21'].number_format = MONEY_FMT
    setup['B23'].number_format = PCT5_FMT
    setup['B24'].number_format = PCT1_FMT
    setup['B28'].number_format = '0.00%'
    setup['B29'].number_format = '0'

    add_table(setup, 'tblSetup', 'A1:C27')
    apply_grid(setup, 'A1:C27')

    # -----------------------------
    # SCORECARD_PD_LGD
    # -----------------------------
    sc = wb.create_sheet('SCORECARD_PD_LGD')
    sc.freeze_panes = 'A2'
    set_col_widths(sc, {'A': 10, 'B': 16, 'C': 12, 'D': 14, 'E': 55})

    sc.append(['Rating', 'PD_1Y (decimal)', 'PD_1Y (%)', 'LGD (assumption)', 'Notes / Source'])
    score_note = 'PD from Scope idealised default probability table (Year 1); LGD is a policy assumption.'
    pd_rows = [
        ('AAA', 0.00003, 0.00003),
        ('AA+', 0.00003, 0.00003),
        # Note: v6 file contains a tiny float representation difference between col B and C for 'AA' and 'A-'
        ('AA', 7e-05, 7.000000000000001e-05),
        ('AA-', 0.00014, 0.00014),
        ('A+', 0.00024, 0.00024),
        ('A', 0.00041, 0.00041),
        ('A-', 0.00071, 0.0007099999999999999),
        ('BBB+', 0.00122, 0.00122),
        ('BBB', 0.00211, 0.00211),
        ('BBB-', 0.00364, 0.00364),
        ('BB+', 0.01142, 0.01142),
        ('BB', 0.01778, 0.01778),
        ('BB-', 0.02541, 0.02541),
        ('B+', 0.04604, 0.04604),
        ('B', 0.05941, 0.05941),
        ('B-', 0.09232, 0.09232),
        ('CCC', 0.24731, 0.24731),
        ('CC', 0.47076, 0.47076),
        ('C', 0.83919, 0.83919),
    ]
    for rating, pd_b, pd_c in pd_rows:
        sc.append([rating, pd_b, pd_c, 0.60, score_note])

    sc.append([None, None, None, None, None])
    sc.append([None, None, None, None, None])

    sc.append(['Note', 'Scope states it used a static 50% LGD to convert default probabilities to expected loss in its idealised tables. Adjust LGD per your CSA/seniority policy.', None, None, None])

    for r in range(2, 21):
        sc[f'B{r}'].number_format = PCT5_FMT
        sc[f'C{r}'].number_format = PCT3_FMT
        sc[f'D{r}'].number_format = PCT1_FMT

    # light header format
    for c in range(1, 6):
        cell = sc.cell(row=1, column=c)
        cell.fill = SCORE_HEADER_FILL
        cell.font = SCORE_HEADER_FONT
        cell.alignment = SCORE_HEADER_ALIGN
    sc['A23'].font = Font(bold=True)

    # -----------------------------
    # HKD_CURVE_DEMO
    # -----------------------------
    curve = wb.create_sheet('HKD_CURVE_DEMO')
    curve.freeze_panes = 'A2'
    set_col_widths(curve, {'A': 16, 'B': 22, 'C': 18, 'D': 60})
    curve_rows = [
        ['Tenor (years)', 'HKD zero rate (cont)', 'DF = EXP(-r*t)', 'Notes'],
        [0, 0.028, '=EXP(-B2*A2)', 'Zero rates are interpolated linearly in MARKET_DATA based on year fraction to settlement.'],
        [0.25, 0.0285, '=EXP(-B3*A3)', 'Input curve nodes (demo).'],
        [0.5, 0.029, '=EXP(-B4*A4)', 'Input curve nodes (demo).'],
        [1, 0.03, '=EXP(-B5*A5)', 'Input curve nodes (demo).'],
        [2, 0.0315, '=EXP(-B6*A6)', 'Input curve nodes (demo).'],
        [5, 0.033, '=EXP(-B7*A7)', 'Input curve nodes (demo).'],
    ]
    for r in curve_rows:
        curve.append(r)
    for r in range(2, 8):
        curve[f'A{r}'].number_format = '0.0000'
        curve[f'B{r}'].number_format = FX4_FMT
        curve[f'C{r}'].number_format = DF6_FMT
    add_table(curve, 'tblCurve', 'A1:D7')
    apply_grid(curve, 'A1:D7')

    # -----------------------------
    # MARKET_DATA
    # -----------------------------
    md = wb.create_sheet('MARKET_DATA')
    md.freeze_panes = 'A2'
    set_col_widths(md, {'A': 16, 'B': 14, 'C': 24, 'D': 18, 'E': 18, 'F': 12, 'G': 22, 'H': 48, 'I': 18, 'J': 26, 'K': 16})

    interp_settle = (
        '=INDEX(HKD_CURVE_DEMO!$B$2:$B$7,MATCH(E{r},HKD_CURVE_DEMO!$A$2:$A$7,1))'
        '+(INDEX(HKD_CURVE_DEMO!$B$2:$B$7,MATCH(E{r},HKD_CURVE_DEMO!$A$2:$A$7,1)+1)'
        '-INDEX(HKD_CURVE_DEMO!$B$2:$B$7,MATCH(E{r},HKD_CURVE_DEMO!$A$2:$A$7,1)))'
        '*(E{r}-INDEX(HKD_CURVE_DEMO!$A$2:$A$7,MATCH(E{r},HKD_CURVE_DEMO!$A$2:$A$7,1)))'
        '/(INDEX(HKD_CURVE_DEMO!$A$2:$A$7,MATCH(E{r},HKD_CURVE_DEMO!$A$2:$A$7,1)+1)'
        '-INDEX(HKD_CURVE_DEMO!$A$2:$A$7,MATCH(E{r},HKD_CURVE_DEMO!$A$2:$A$7,1)))'
    )
    interp_mtn = (
        '=INDEX(HKD_CURVE_DEMO!$B$2:$B$7,MATCH(I{r},HKD_CURVE_DEMO!$A$2:$A$7,1))'
        '+(INDEX(HKD_CURVE_DEMO!$B$2:$B$7,MATCH(I{r},HKD_CURVE_DEMO!$A$2:$A$7,1)+1)'
        '-INDEX(HKD_CURVE_DEMO!$B$2:$B$7,MATCH(I{r},HKD_CURVE_DEMO!$A$2:$A$7,1)))'
        '*(I{r}-INDEX(HKD_CURVE_DEMO!$A$2:$A$7,MATCH(I{r},HKD_CURVE_DEMO!$A$2:$A$7,1)))'
        '/(INDEX(HKD_CURVE_DEMO!$A$2:$A$7,MATCH(I{r},HKD_CURVE_DEMO!$A$2:$A$7,1)+1)'
        '-INDEX(HKD_CURVE_DEMO!$A$2:$A$7,MATCH(I{r},HKD_CURVE_DEMO!$A$2:$A$7,1)))'
    )

    md.append(['Valuation date', 'Spot HKDUSD', 'Market forward to settlement', 'HKD disc rate (cont)', 'Year frac to settle', 'DF', 'Forward points (Fwd-Spot)', 'Notes', 'Year frac to MTN pay', 'HKD disc rate to MTN (cont)', 'DF_to_MTN pay'])

    valuation_rows = [
        (dt(2025, 9, 2), 7.805, 7.71),
        (dt(2025, 9, 30), 7.82, 7.73),
        (dt(2025, 10, 31), 7.835, 7.745),
        (dt(2025, 11, 30), 7.81, 7.725),
        (dt(2025, 12, 31), 7.79, 7.705),
        (dt(2026, 1, 31), 7.8, 7.712),
    ]
    for i, (dte, spot, fwd) in enumerate(valuation_rows, start=2):
        md[f'A{i}'] = dte
        md[f'B{i}'] = spot
        md[f'C{i}'] = fwd
        md[f'D{i}'] = interp_settle.format(r=i)
        md[f'E{i}'] = f'=(SETUP!$B$7-A{i})/365'
        md[f'F{i}'] = f'=EXP(-D{i}*E{i})'
        md[f'G{i}'] = f'=C{i}-B{i}'
        md[f'H{i}'] = 'Edit spot/forward here. HKD disc rate computed from HKD_CURVE_DEMO; overwrite if needed.'
        md[f'I{i}'] = f'=(SETUP!$B$8-A{i})/365'
        md[f'J{i}'] = interp_mtn.format(r=i)
        md[f'K{i}'] = f'=EXP(-J{i}*I{i})'

        md[f'A{i}'].number_format = DATE_FMT
        md[f'B{i}'].number_format = FX4_FMT
        md[f'C{i}'].number_format = FX4_FMT
        md[f'D{i}'].number_format = FX4_FMT
        md[f'E{i}'].number_format = FX4_FMT
        md[f'F{i}'].number_format = DF6_FMT
        md[f'G{i}'].number_format = FX4_FMT
        md[f'I{i}'].number_format = FX4_FMT
        md[f'J{i}'].number_format = FX4_FMT
        md[f'K{i}'].number_format = DF6_FMT

    add_table(md, 'tblMarketData', 'A1:K7')
    apply_grid(md, 'A1:K7')

    # -----------------------------
    # TRADES_FX
    # -----------------------------
    fx = wb.create_sheet('TRADES_FX')
    fx.freeze_panes = 'A2'
    set_col_widths(fx, {'A': 16, 'B': 10, 'C': 14, 'D': 14, 'E': 14, 'F': 10, 'G': 16, 'H': 10, 'I': 18, 'J': 12, 'K': 12, 'L': 18, 'M': 50})

    fx.append(['Hedge_ID', 'Ref No', 'Counterparty', 'Trade date', 'Settlement date', 'Buy CUR', 'Buy amount (USD)', 'Sell CUR', 'Sell amount (HKD)', 'Spot_T0', 'Forward K', 'Swap points (K-Spot0)', 'Notes'])
    fx.append(['=SETUP!$B$2', '=SETUP!$B$3', '=SETUP!$B$4', '=SETUP!$B$6', '=SETUP!$B$7', '=SETUP!$B$9', '=SETUP!$B$10', '=SETUP!$B$11', '=SETUP!$B$10*SETUP!$B$12', '=SETUP!$B$13', '=SETUP!$B$12', '=SETUP!$B$14', 'Ticket only; valuation uses MARKET_DATA / VALUATION_MONTHLY.'])
    fx['D2'].number_format = DATE_FMT
    fx['E2'].number_format = DATE_FMT
    fx['G2'].number_format = INT_FMT
    fx['I2'].number_format = INT_FMT
    fx['J2'].number_format = FX4_FMT
    fx['K2'].number_format = FX4_FMT
    fx['L2'].number_format = FX4_FMT
    add_table(fx, 'tblTradesFX', 'A1:M2')
    apply_grid(fx, 'A1:M2')

    # -----------------------------
    # MTN_CASHFLOWS
    # -----------------------------
    mtn = wb.create_sheet('MTN_CASHFLOWS')
    mtn.freeze_panes = 'A2'
    set_col_widths(mtn, {'A': 12, 'B': 12, 'C': 10, 'D': 16, 'E': 14, 'F': 14, 'G': 16, 'H': 20, 'I': 22, 'J': 45})

    mtn.append(['Cashflow_ID', 'Type', 'Currency', 'Amount', 'Pay date', 'Hedged? (1/0)', 'Hedge_ID', 'HKD equiv @ Spot_T0', 'HKD equiv @ Forward K', 'Notes'])
    mtn.append(['CF_P1', 'Principal', 'USD', '=SETUP!$B$19', '=SETUP!$B$8', 1, '=SETUP!$B$2', '=$D$2*SETUP!$B$13', '=$D$2*SETUP!$B$12', 'Designated hedged layer of principal.'])
    mtn.append(['CF_P2', 'Principal', 'USD', '=SETUP!$B$18-SETUP!$B$19', '=SETUP!$B$8', 0, None, '=$D$3*SETUP!$B$13', None, 'Unhedged remainder of MTN principal (for exposure disclosure).'])
    for r in [2, 3]:
        mtn[f'D{r}'].number_format = INT_FMT
        mtn[f'E{r}'].number_format = DATE_FMT
        mtn[f'H{r}'].number_format = MONEY_FMT
        if r == 2:
            mtn[f'I{r}'].number_format = MONEY_FMT
    add_table(mtn, 'tblMtnCf', 'A1:J3')
    apply_grid(mtn, 'A1:J3')

    # -----------------------------
    # VALUATION_MONTHLY
    # -----------------------------
    vm = wb.create_sheet('VALUATION_MONTHLY')
    vm.freeze_panes = 'A2'
    set_col_widths(vm, {'A': 16, 'B': 12, 'C': 18, 'D': 10, 'E': 20, 'F': 22, 'G': 24, 'H': 24, 'I': 14, 'J': 20, 'K': 20, 'L': 16, 'M': 40, 'N': 16, 'O': 18, 'Q': 18, 'R': 22, 'S': 14})

    vm_headers = [
        'Valuation date', 'Spot', 'Market forward', 'DF',
        'HI PV (HKD) = -USD*Spot*DF',
        'Derivative total FV = (Fwd-K)*USD*DF',
        'Designated spot element FV = (Spot-Spot0)*USD*DF',
        'Excluded forward element FV = Total - SpotElem',
        'ΔHI', 'ΔHedge (designated)', 'ΔExcluded (COH)', 'ΔTotal FV',
        'Notes', 'DF_to_MTN pay', 'DF_to_MTN pay', 'MTN principal pay date',
        'CVA booked (HKD) (relative to T0)', 'Dirty FV (HKD) = Clean + CVA_booked', 'ΔDirty FV'
    ]
    vm.append(vm_headers)

    for i in range(2, 8):
        vm[f'A{i}'] = f'=MARKET_DATA!A{i}'
        vm[f'B{i}'] = f'=MARKET_DATA!B{i}'
        vm[f'C{i}'] = f'=MARKET_DATA!C{i}'
        vm[f'D{i}'] = f'=MARKET_DATA!F{i}'
        vm[f'N{i}'] = f'=MARKET_DATA!$K${i}'
        vm[f'O{i}'] = '=SETUP!$B$8'

        vm[f'E{i}'] = f'=-SETUP!$B$10*B{i}*N{i}'
        vm[f'F{i}'] = f'=(C{i}-SETUP!$B$12)*SETUP!$B$10*D{i}'
        vm[f'G{i}'] = f'=(B{i}-SETUP!$B$13)*SETUP!$B$10*D{i}'
        vm[f'H{i}'] = f'=F{i}-G{i}'

        if i == 2:
            vm[f'I{i}'] = 0
            vm[f'J{i}'] = 0
            vm[f'K{i}'] = 0
            vm[f'L{i}'] = 0
            vm[f'M{i}'] = 'Designation baseline (Δ=0).'
            vm[f'S{i}'] = 0
        else:
            vm[f'I{i}'] = f'=E{i}-E{i-1}'
            vm[f'J{i}'] = f'=G{i}-G{i-1}'
            vm[f'K{i}'] = f'=H{i}-H{i-1}'
            vm[f'L{i}'] = f'=F{i}-F{i-1}'
            vm[f'M{i}'] = 'Monthly close-to-close changes.'
            vm[f'S{i}'] = f'=R{i}-R{i-1}'

        vm[f'Q{i}'] = f'=CVA_MONTHLY!$T${i}'
        vm[f'R{i}'] = f'=F{i}+Q{i}'

        vm[f'A{i}'].number_format = DATE_FMT
        vm[f'B{i}'].number_format = FX4_FMT
        vm[f'C{i}'].number_format = FX4_FMT
        vm[f'D{i}'].number_format = DF6_FMT
        vm[f'N{i}'].number_format = DF6_FMT
        vm[f'O{i}'].number_format = DATE_FMT
        for col in ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Q', 'R', 'S']:
            vm[f'{col}{i}'].number_format = MONEY_FMT

    add_table(vm, 'tblValMonthly', 'A1:S7')
    apply_grid(vm, 'A1:S7')

    # -----------------------------
    # CVA_MONTHLY
    # -----------------------------
    cva = wb.create_sheet('CVA_MONTHLY')
    cva.freeze_panes = 'A2'
    set_col_widths(cva, {'A': 16, 'B': 20, 'C': 28, 'D': 10, 'E': 18, 'F': 16, 'G': 16, 'H': 8, 'I': 22, 'J': 18, 'K': 45, 'L': 20, 'M': 20, 'N': 20, 'O': 20, 'P': 20, 'Q': 20, 'R': 20, 'S': 20, 'T': 26})

    cva_headers = [
        'Valuation date',
        'Clean FV (HKD) (from VALUATION_MONTHLY)',
        'EPE (HKD PV) = Notional*DF*E[(Fτ-K)+] minus CSA threshold (if CSA=Y)',
        'PD_1Y',
        'Hazard rate λ = -LN(1-PD_1Y)',
        'T_rem to settle (years)',
        'PD_rem = 1-EXP(-λ*T_rem)',
        'LGD',
        'CVA_model (HKD) = -LGD*EPE*PD_rem',
        'ΔCVA_model (close-to-close)',
        'Notes',
        'F (market fwd)',
        'K (contract)',
        'σ (annual)',
        'τ (years)',
        'd1',
        'd2',
        'E[(Fτ-K)+] (HKD per USD)',
        'EPE_model (HKD PV) = Notional*DF*E[...]',
        'CVA booked (HKD) = CVA_model - CVA_model(T0)',
    ]
    cva.append(cva_headers)

    for i in range(2, 8):
        cva[f'A{i}'] = f'=MARKET_DATA!A{i}'
        cva[f'B{i}'] = f'=VALUATION_MONTHLY!F{i}'

        cva[f'L{i}'] = f'=MARKET_DATA!C{i}'
        cva[f'M{i}'] = '=SETUP!$B$12'
        cva[f'N{i}'] = '=SETUP!$B$28'
        cva[f'O{i}'] = f'=IF(SETUP!$B$20="Y",SETUP!$B$29/365,MARKET_DATA!E{i})'
        cva[f'P{i}'] = f'=IF(O{i}<=0,0,(LN(L{i}/M{i})+0.5*N{i}^2*O{i})/(N{i}*SQRT(O{i})))'
        cva[f'Q{i}'] = f'=P{i}-N{i}*SQRT(O{i})'
        cva[f'R{i}'] = f'=IF(O{i}<=0,MAX(L{i}-M{i},0),L{i}*NORM.S.DIST(P{i},TRUE)-M{i}*NORM.S.DIST(Q{i},TRUE))'
        cva[f'S{i}'] = f'=SETUP!$B$10*MARKET_DATA!F{i}*R{i}'

        cva[f'C{i}'] = f'=IF(SETUP!$B$20="Y",MAX(0,S{i}-SETUP!$B$21),S{i})'
        cva[f'D{i}'] = '=SETUP!$B$23'
        cva[f'E{i}'] = f'=-LN(1-D{i})'
        cva[f'F{i}'] = f'=MARKET_DATA!E{i}'
        cva[f'G{i}'] = f'=1-EXP(-E{i}*F{i})'
        cva[f'H{i}'] = '=SETUP!$B$24'
        cva[f'I{i}'] = f'=-H{i}*C{i}*G{i}'
        if i == 2:
            cva[f'J{i}'] = 0
            cva[f'K{i}'] = 'Baseline.'
        else:
            cva[f'J{i}'] = f'=I{i}-I{i-1}'
            cva[f'K{i}'] = 'CVA change posted to P&L (credit adjustment) in this demo.'
        cva[f'T{i}'] = f'=I{i}-$I$2'

        cva[f'A{i}'].number_format = DATE_FMT
        for col in ['B', 'C', 'I', 'J']:
            cva[f'{col}{i}'].number_format = MONEY_FMT
        for col in ['D', 'E', 'F', 'G', 'H']:
            cva[f'{col}{i}'].number_format = DF6_FMT
        cva[f'L{i}'].number_format = '0.00000'
        cva[f'M{i}'].number_format = '0.00000'
        cva[f'N{i}'].number_format = '0.00%'
        cva[f'O{i}'].number_format = '0.0000'
        cva[f'P{i}'].number_format = '0.0000'
        cva[f'Q{i}'].number_format = '0.0000'
        cva[f'R{i}'].number_format = '0.00000'
        cva[f'S{i}'].number_format = INT_FMT
        cva[f'T{i}'].number_format = INT_FMT

    add_table(cva, 'tblCVA', 'A1:K7')
    apply_grid(cva, 'A1:K7')

    # -----------------------------
    # MTN_PV_MONTHLY
    # -----------------------------
    mpv = wb.create_sheet('MTN_PV_MONTHLY')
    mpv.freeze_panes = 'A2'
    set_col_widths(mpv, {'A': 16, 'B': 12, 'C': 14, 'D': 26, 'E': 26, 'F': 20, 'G': 22, 'H': 28, 'I': 28, 'J': 40})

    mpv.append([
        'Valuation date', 'Spot', 'DF_to_MTN pay',
        'PV Hedged principal (HKD) = -USD_hedged*Spot*DF_mtn',
        'PV Unhedged principal (HKD) = -USD_unhedged*Spot*DF_mtn',
        'PV Total principal (HKD)',
        'Undiscounted HKD @ Spot (Total)',
        'Undiscounted HKD locked by hedge (Hedged layer at K)',
        'Undiscounted HKD spot exposure hedged layer',
        'Notes'
    ])
    for i in range(2, 8):
        mpv[f'A{i}'] = f'=MARKET_DATA!A{i}'
        mpv[f'B{i}'] = f'=MARKET_DATA!B{i}'
        mpv[f'C{i}'] = f'=MARKET_DATA!K{i}'
        mpv[f'D{i}'] = f'=-SETUP!$B$19*B{i}*C{i}'
        mpv[f'E{i}'] = f'=-(SETUP!$B$18-SETUP!$B$19)*B{i}*C{i}'
        mpv[f'F{i}'] = f'=D{i}+E{i}'
        mpv[f'G{i}'] = f'=SETUP!$B$18*B{i}'
        mpv[f'H{i}'] = '=SETUP!$B$19*SETUP!$B$12'
        mpv[f'I{i}'] = f'=SETUP!$B$19*B{i}'
        mpv[f'J{i}'] = 'Principal cashflow exposure view; only hedged layer is in hedge relationship.'

        mpv[f'A{i}'].number_format = DATE_FMT
        mpv[f'B{i}'].number_format = FX4_FMT
        mpv[f'C{i}'].number_format = DF6_FMT
        for col in ['D', 'E', 'F', 'G', 'H', 'I']:
            mpv[f'{col}{i}'].number_format = MONEY_FMT

    add_table(mpv, 'tblMtnPv', 'A1:J7')
    apply_grid(mpv, 'A1:J7')

    # -----------------------------
    # RETROSPECTIVE_MONTHLY
    # -----------------------------
    retro = wb.create_sheet('RETROSPECTIVE_MONTHLY')
    retro.freeze_panes = 'A2'
    set_col_widths(retro, {'A': 16, 'B': 16, 'C': 22, 'D': 20, 'E': 18, 'F': 18, 'G': 55})

    retro.append(['Valuation date', 'ΔHI', 'ΔHedge (designated)', 'Offset ratio (-ΔHedge/ΔHI)', 'Screen (0.8–1.25)', 'ΔExcluded (COH)', 'Notes'])
    for i in range(2, 8):
        retro[f'A{i}'] = f'=VALUATION_MONTHLY!A{i}'
        retro[f'B{i}'] = f'=VALUATION_MONTHLY!I{i}'
        retro[f'C{i}'] = f'=VALUATION_MONTHLY!J{i}'
        retro[f'D{i}'] = f'=IF(B{i}=0,"",-C{i}/B{i})'
        retro[f'E{i}'] = f'=IF(D{i}="","",IF(AND(ABS(D{i})>=0.8,ABS(D{i})<=1.25,SIGN(B{i})=-SIGN(C{i})),"PASS","REVIEW"))'
        retro[f'F{i}'] = f'=VALUATION_MONTHLY!K{i}'
        retro[f'G{i}'] = 'Legacy 80–125 is a review screen only under IFRS 9.'

        retro[f'A{i}'].number_format = DATE_FMT
        for col in ['B', 'C', 'F']:
            retro[f'{col}{i}'].number_format = MONEY_FMT
        retro[f'D{i}'].number_format = '0.000'
        retro[f'E{i}'].alignment = Alignment(horizontal='center')

    add_table(retro, 'tblRetroMonthly', 'A1:G7')
    apply_grid(retro, 'A1:G7')

    # -----------------------------
    # JOURNALS_MONTHLY
    # -----------------------------
    j = wb.create_sheet('JOURNALS_MONTHLY')
    j.freeze_panes = 'A2'
    set_col_widths(j, {'A': 16, 'B': 26, 'C': 16, 'D': 26, 'E': 22, 'F': 30, 'G': 14, 'H': 14, 'I': 55, 'J': 22, 'K': 22})

    j.append(['Valuation date', 'ΔHedge (designated spot element)', 'ΔHI', 'OCI (CFH reserve) = SIGN(ΔHedge)*MIN(abs)', 'P&L ineffectiveness = ΔHedge - OCI', 'OCI (COH excluded forward points) = ΔExcluded', 'Total OCI', 'Total P&L', 'Notes', 'ΔCVA (credit adj, P&L)', 'Total P&L incl credit'])
    for i in range(2, 8):
        j[f'A{i}'] = f'=RETROSPECTIVE_MONTHLY!A{i}'
        j[f'B{i}'] = f'=RETROSPECTIVE_MONTHLY!C{i}'
        j[f'C{i}'] = f'=RETROSPECTIVE_MONTHLY!B{i}'
        j[f'D{i}'] = f'=IF(OR(B{i}=0,C{i}=0),0,SIGN(B{i})*MIN(ABS(B{i}),ABS(C{i})))'
        j[f'E{i}'] = f'=B{i}-D{i}'
        j[f'F{i}'] = f'=RETROSPECTIVE_MONTHLY!F{i}'
        j[f'G{i}'] = f'=D{i}+F{i}'
        j[f'H{i}'] = f'=E{i}'
        j[f'I{i}'] = 'Illustrative monthly close-to-close posting.'
        j[f'J{i}'] = f'=CVA_MONTHLY!$J${i}'
        j[f'K{i}'] = f'=H{i}+J{i}'

        j[f'A{i}'].number_format = DATE_FMT
        for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']:
            j[f'{col}{i}'].number_format = MONEY_FMT

    add_table(j, 'tblJournalsMonthly', 'A1:K7')
    apply_grid(j, 'A1:K7')

    # -----------------------------
    # SENSITIVITIES_HISTORY
    # -----------------------------
    sh = wb.create_sheet('SENSITIVITIES_HISTORY')
    sh.freeze_panes = 'A2'
    set_col_widths(sh, {'A': 16, 'B': 12, 'C': 14, 'D': 14, 'E': 22, 'F': 22, 'G': 26, 'H': 24, 'I': 55})

    sh.append(['Valuation date', 'Spot', 'DF_to_settle', 'DF_to_MTN pay', 'HI spot sens (+1bp rel)', 'Hedge spot sens (+1bp rel)', 'COH fwd-pts sens (+1bp abs 0.0001)', 'HI fwd-pts sens (optional, forward-based)', 'Comments'])
    for i in range(2, 8):
        sh[f'A{i}'] = f'=MARKET_DATA!A{i}'
        sh[f'B{i}'] = f'=MARKET_DATA!B{i}'
        sh[f'C{i}'] = f'=MARKET_DATA!F{i}'
        sh[f'D{i}'] = f'=MARKET_DATA!K{i}'
        sh[f'E{i}'] = f'=-SETUP!$B$19*B{i}*D{i}*0.0001'
        sh[f'F{i}'] = f'=SETUP!$B$19*B{i}*C{i}*0.0001'
        sh[f'G{i}'] = f'=SETUP!$B$19*C{i}*0.0001'
        sh[f'H{i}'] = f'=-SETUP!$B$19*D{i}*0.0001'
        sh[f'I{i}'] = 'HI uses MTN pay-date DF; hedge uses forward settlement DF; mismatch yields ineffectiveness.'

        sh[f'A{i}'].number_format = DATE_FMT
        sh[f'B{i}'].number_format = FX4_FMT
        sh[f'C{i}'].number_format = DF6_FMT
        sh[f'D{i}'].number_format = DF6_FMT
        for col in ['E', 'F', 'G', 'H']:
            sh[f'{col}{i}'].number_format = MONEY_FMT

    add_table(sh, 'tblSensHist', 'A1:I7')
    apply_grid(sh, 'A1:I7')

    # -----------------------------
    # PROSPECTIVE_HISTORY
    # -----------------------------
    ph = wb.create_sheet('PROSPECTIVE_HISTORY')
    ph.freeze_panes = 'A2'
    set_col_widths(ph, {'A': 16, 'B': 14, 'C': 14, 'D': 16, 'E': 16, 'F': 16, 'G': 16, 'H': 20, 'I': 18, 'J': 55})
    ph.append(['Valuation date', 'Scenario', 'Shock (units)', 'HI sensitivity', 'Hedge sensitivity', 'ΔHI', 'ΔHedge', 'Offset ratio (-ΔHedge/ΔHI)', 'Screen (0.8–1.25)', 'Notes'])

    out_row = 2
    for i in range(2, 8):
        for scenario, shock in [('Spot +1bp', 1), ('Spot -1bp', -1)]:
            ph[f'A{out_row}'] = f'=SENSITIVITIES_HISTORY!A{i}'
            ph[f'B{out_row}'] = scenario
            ph[f'C{out_row}'] = shock
            ph[f'D{out_row}'] = f'=SENSITIVITIES_HISTORY!E{i}'
            ph[f'E{out_row}'] = f'=SENSITIVITIES_HISTORY!F{i}'
            ph[f'F{out_row}'] = f'=C{out_row}*D{out_row}'
            ph[f'G{out_row}'] = f'=C{out_row}*E{out_row}'
            ph[f'H{out_row}'] = f'=-G{out_row}/F{out_row}'
            ph[f'I{out_row}'] = f'=IF(AND(ABS(H{out_row})>=0.8,ABS(H{out_row})<=1.25,SIGN(F{out_row})=-SIGN(G{out_row})),"PASS","REVIEW")'
            ph[f'J{out_row}'] = 'Prospective screen per valuation date (legacy 80–125 used as review).'

            ph[f'A{out_row}'].number_format = DATE_FMT
            for col in ['D', 'E', 'F', 'G']:
                ph[f'{col}{out_row}'].number_format = MONEY_FMT
            ph[f'H{out_row}'].number_format = '0.000'
            ph[f'I{out_row}'].alignment = Alignment(horizontal='center')

            out_row += 1

    add_table(ph, 'tblProsHist', 'A1:J13')
    apply_grid(ph, 'A1:J13')

    # -----------------------------
    # SENSITIVITIES (single-date)
    # -----------------------------
    s = wb.create_sheet('SENSITIVITIES')
    s.freeze_panes = 'A2'
    set_col_widths(s, {'A': 18, 'B': 18, 'C': 26, 'D': 26, 'E': 28, 'F': 55})

    s.append(['Hedge_ID', 'Risk factor', 'Component', 'Sensitivity (HKD per unit)', 'Shock unit', 'Comments'])
    s.append(['=SETUP!$B$2', 'HKDUSD spot', 'HI (spot-based)', '=-SETUP!$B$19*MARKET_DATA!$B$2*MARKET_DATA!$K$2*0.0001', 'per +1 bp (0.01%)', 'HI PV uses DF_to_MTN pay.'])
    s.append(['=SETUP!$B$2', 'HKDUSD spot', 'Hedge (designated spot element)', '=SETUP!$B$19*MARKET_DATA!$B$2*MARKET_DATA!$F$2*0.0001', 'per +1 bp (0.01%)', 'Hedge spot element discounted to settlement.'])
    s.append(['=SETUP!$B$2', 'Forward points', 'Excluded component (COH)', '=SETUP!$B$19*MARKET_DATA!$F$2*0.0001', 'per +1 bp (abs 0.0001 HKD/USD)', 'Forward points sensitivity (excluded COH).'])
    s.append(['=SETUP!$B$2', 'Forward points', 'HI (forward-based, optional)', '=-SETUP!$B$19*MARKET_DATA!$K$2*0.0001', 'per +1 bp (abs 0.0001 HKD/USD)', 'Only if hedged item is measured using forward rate.'])
    for r in range(2, 6):
        s[f'D{r}'].number_format = MONEY_FMT
    add_table(s, 'tblSens', 'A1:F5')
    apply_grid(s, 'A1:F5')

    # -----------------------------
    # PROSPECTIVE (single-date)
    # -----------------------------
    p = wb.create_sheet('PROSPECTIVE')
    p.freeze_panes = 'A2'
    set_col_widths(p, {'A': 10, 'B': 18, 'C': 14, 'D': 16, 'E': 16, 'F': 16, 'G': 16, 'H': 20, 'I': 18, 'J': 55})

    p.append(['Scenario', 'Risk factor', 'Shock (units)', 'HI sensitivity', 'Hedge sensitivity', 'ΔHI', 'ΔHedge', 'Offset ratio (-ΔHedge/ΔHI)', 'Screen (0.8–1.25)', 'Notes'])
    p.append(['S1', 'HKDUSD spot', 1, '=SENSITIVITIES!$D$2', '=SENSITIVITIES!$D$3', '=C2*D2', '=C2*E2', '=-G2/F2', '=IF(AND(ABS(H2)>=0.8,ABS(H2)<=1.25,SIGN(F2)=-SIGN(G2)),"PASS","REVIEW")', 'Screen only; DF mismatch can cause REVIEW.'])
    p.append(['S2', 'HKDUSD spot', -1, '=SENSITIVITIES!$D$2', '=SENSITIVITIES!$D$3', '=C3*D3', '=C3*E3', '=-G3/F3', '=IF(AND(ABS(H3)>=0.8,ABS(H3)<=1.25,SIGN(F3)=-SIGN(G3)),"PASS","REVIEW")', 'Screen only; DF mismatch can cause REVIEW.'])

    for r in [2, 3]:
        for col in ['D', 'E', 'F', 'G']:
            p[f'{col}{r}'].number_format = MONEY_FMT
        p[f'H{r}'].number_format = '0.000'
        p[f'I{r}'].alignment = Alignment(horizontal='center')

    add_table(p, 'tblPros', 'A1:J3')
    apply_grid(p, 'A1:J3')

    # -----------------------------
    # HEDGE_DOC_CN
    # -----------------------------
    doc = wb.create_sheet('HEDGE_DOC_CN')
    doc.freeze_panes = 'A3'
    set_col_widths(doc, {'A': 120})

    doc_lines = [
        '中信XX — IFRS 9 套期会计套期文件（示例文本，参数见 SETUP / TRADES_FX / MARKET_DATA）',
        '（市场数据按月在 MARKET_DATA 维护；示例为 Spot 风险现金流量套期，远期点作为成本套期处理。）',
        None,
        '描述:',
        '为应对美元中期票据（MTN）到期偿还本金时因港币/美元即期汇率波动产生的风险，本公司采用港币/美元远期外汇合约进行现金流量套期。',
        None,
        '套期类型:',
        '现金流量套期',
        None,
        '指定日期:',
        '2025年9月2日',
        None,
        '套期工具:',
        '远期外汇合约：CL_FX001：交易对手：XYZ Bank',
        '买入美元 40,000,000（远期汇率 7.7100；即期 7.8050；Swap Point -0.0950）',
        '卖出港币 308,400,000',
        '交收日期：2027年2月25日',
        None,
        '被套期项目:',
        '预测的美元本金现金流（USD 40,000,000），支付日期：2027年2月26日（与远期交收日存在1天差异，可能构成无效性来源）',
        None,
        '被套期风险:',
        '港币/美元即期汇率波动（Spot risk）',
        None,
        '预期有效性测试方法:',
        '关键条款匹配的定性评估 + 月度监控（名义金额、币种、到期日/付款日、信用风险是否主导价值变动等）。',
        None,
        '无效性来源:',
        '- 交收日与付款日不一致（时间错配）',
        '- 初始公允价值不为零（非平价远期）',
        '- 信用风险重大变化',
        None,
        '信用风险与CVA（示例处理）:',
        '本示例在公允价值计量中加入简化CVA：CVA=-LGD×EPE×PD_rem。若存在CSA且保证金阈值为0，则EPE显著下降，CVA通常较小。月度ΔCVA作为信用调整计入损益，并持续监控信用风险是否主导价值变动。',
    ]
    for i, line in enumerate(doc_lines, start=1):
        doc[f'A{i}'] = line

    doc['A1'].font = Font(bold=True)
    doc['A33'].font = Font(bold=True)
    for rr in [2, 5, 34]:
        doc[f'A{rr}'].alignment = Alignment(wrap_text=True)

    # Save
    wb.save(out_path)
    print('Wrote:', out_path)


if __name__ == '__main__':
    main()