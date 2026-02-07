import math
import datetime
from datetime import date
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import calendar

# Third-party imports for Excel generation
import openpyxl
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter


# ==========================================
# Date utilities
# ==========================================
def add_months(d: date, months: int) -> date:
    """Add months to a date, clamping day-of-month if needed."""
    y = d.year + (d.month + months - 1) // 12
    m = (d.month + months - 1) % 12 + 1
    d_day = d.day
    last_day = calendar.monthrange(y, m)[1]
    day = min(d_day, last_day)
    return date(y, m, day)


def build_accrual_schedule(effective_date: date,
                           maturity_date: date,
                           freq_months: int) -> List[Tuple[date, date]]:
    """
    Returns a list of (AccStart, PayDate) covering [effective_date, maturity_date].
    Uses simple month stepping; last period stubs to maturity_date if needed.
    """
    if maturity_date <= effective_date:
        return []

    rows: List[Tuple[date, date]] = []
    acc_start = effective_date
    pay_date = add_months(acc_start, freq_months)

    while pay_date < maturity_date:
        rows.append((acc_start, pay_date))
        acc_start = pay_date
        pay_date = add_months(acc_start, freq_months)

    # Final stub
    rows.append((acc_start, maturity_date))
    return rows


def last_pay_date_before(schedule: List[Tuple[date, date]],
                         val_date: date) -> date:
    """Returns the last PayDate <= val_date, otherwise effective_date."""
    # effective date is the start of the first period
    last = schedule[0][0]
    for _, pay in schedule:
        if pay <= val_date and pay > last:
            last = pay
    return last


def future_schedule_from_val(schedule: List[Tuple[date, date]],
                             val_date: date) -> List[Tuple[date, date]]:
    """Filter schedule to keep only periods with PayDate > val_date"""
    return [(acc_start, pay) for (acc_start, pay) in schedule if pay > val_date]


# ==========================================
# Simplified CRMA pricer
# ==========================================
def solve_lambda_flat(val_date: date,
                      future_periods: List[Tuple[date, date]],
                      day_basis: int,
                      r: float,
                      market_spread_bps: float,
                      recovery: float) -> float:
    """
    Solve flat hazard lambda such that model par spread equals market_spread_bps,
    using discrete premium dates and 'remaining' accrual fractions (from val_date).
    """
    lgd = 1.0 - recovery
    S = market_spread_bps / 10000.0

    # Precompute T, DF, alpha_remaining for each pay date
    Ts: List[float] = []
    DFs: List[float] = []
    alphas_rem: List[float] = []

    for acc_start, pay in future_periods:
        T = (pay - val_date).days / day_basis
        if T <= 0:
            continue
        DF = math.exp(-r * T)
        alpha_rem = max(0.0, (pay - max(acc_start, val_date)).days / day_basis)
        Ts.append(T)
        DFs.append(DF)
        alphas_rem.append(alpha_rem)

    def S_model(lam: float) -> float:
        Q_prev = 1.0
        prot_sum = 0.0
        rpv01 = 0.0
        for T, DF, alpha_rem in zip(Ts, DFs, alphas_rem):
            Q = math.exp(-lam * T)
            prot_sum += DF * (Q_prev - Q)
            rpv01 += alpha_rem * DF * Q
            Q_prev = Q

        if rpv01 <= 0:
            return 0.0
        return lgd * prot_sum / rpv01

    # Bisection: find lam where S_model(lam) = S
    lo = 0.0
    hi = max(0.5, (S / max(lgd, 1e-12)) * 5.0 + 1e-6)

    def f(lam: float) -> float:
        return S_model(lam) - S

    # Expand hi until bracketed
    while f(hi) < 0:
        hi *= 2.0
        if hi > 100.0:
            raise RuntimeError("Could not bracket lambda; check inputs.")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-14:
            return mid
        if fm > 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def crma_pv_simplified(notional: float,
                       fee_bps: float,
                       recovery: float,
                       effective_date: date,
                       maturity_date: date,
                       freq_months: int,
                       day_basis: int,
                       val_date: date,
                       r: float,
                       market_spread_bps: float) -> Dict[str, float]:
    """
    Compute simplified CRMA PV (protection buyer):
      - Calibrate lambda to market spread
      - PV Protection = N * LGD * sum(DF * (Q_prev - Q))
      - PV Premium = N * fee * sum(alpha_remaining * DF * Q)
      - Clean = PV Prot - PV Prem
      - Accrued = N * fee * (val_date - last_pay_date) / day_basis
      - Dirty = Clean - Accrued

    Returns dict with MTM components.
    """
    lgd = 1.0 - recovery
    fee = fee_bps / 10000.0

    full_schedule = build_accrual_schedule(effective_date, maturity_date, freq_months)
    last_pay = last_pay_date_before(full_schedule, val_date)
    future_periods = future_schedule_from_val(full_schedule, val_date)

    # If matured (no future pay dates), PV is zero
    if not future_periods:
        return {
            "lambda": 0.0, "RPV01_years": 0.0, "ProtSum": 0.0,
            "PV_protection": 0.0, "PV_premium": 0.0, "clean": 0.0,
            "last_pay_date": float(last_pay.toordinal()),
            "accrued_days": float(max(0, (val_date - last_pay).days)),
            "accrued": 0.0, "dirty": 0.0, "model_spread_bps": 0.0
        }

    lam = solve_lambda_flat(val_date=val_date, future_periods=future_periods,
                            day_basis=day_basis, r=r,
                            market_spread_bps=market_spread_bps, recovery=recovery)

    prot_sum = 0.0
    rpv01 = 0.0
    Q_prev = 1.0

    for acc_start, pay in future_periods:
        T = (pay - val_date).days / day_basis
        DF = math.exp(-r * T)
        Q = math.exp(-lam * T)
        alpha_rem = max(0.0, (pay - max(acc_start, val_date)).days / day_basis)

        prot_sum += DF * (Q_prev - Q)
        rpv01 += alpha_rem * DF * Q
        Q_prev = Q

    pv_prot = notional * lgd * prot_sum
    pv_prem = notional * fee * rpv01
    clean = pv_prot - pv_prem

    accrued_days = max(0, (val_date - last_pay).days)
    accrued = notional * fee * accrued_days / day_basis
    dirty = clean - accrued

    model_spread_bps = (10000.0 * lgd * prot_sum / rpv01) if rpv01 > 0 else 0.0

    return {
        "lambda": lam,
        "RPV01_years": rpv01,
        "ProtSum": prot_sum,
        "PV_protection": pv_prot,
        "PV_premium": pv_prem,
        "clean": clean,
        "last_pay_date": float(last_pay.toordinal()),
        "accrued_days": float(accrued_days),
        "accrued": accrued,
        "dirty": dirty,
        "model_spread_bps": model_spread_bps
    }


# ==========================================
# Workbook builder (openpyxl)
# ==========================================
@dataclass
class DealConfig:
    deal_name: str
    protection_seller: str
    effective_date: date
    maturity_date: date
    notional: float
    fee_bps: float
    pay_freq_months: int
    day_basis: int
    recovery: float


def set_col_widths(ws, widths: Dict[int, float]) -> None:
    for col_idx, w in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = w


def build_pv_sheet_crma(wb: openpyxl.Workbook,
                        sheet_name: str,
                        deal: DealConfig,
                        val_date: date,
                        r: float,
                        market_spread_bps: float,
                        max_rows: int = 80) -> Tuple[str, Dict[str, float]]:
    """
    Build a PV sheet with fixed layout:
      - Inputs in B3:B16
      - Schedule table from row 21 to row 20+max_rows
      - Summary outputs in row 105+
    Returns (sheet_name, computed_numbers).
    """
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)
    ws["A1"] = f"{deal.deal_name} | PV Sheet: {sheet_name}"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A1"].alignment = Alignment(horizontal="left")

    # Build schedule for writing pay dates / accrual start dates
    full_schedule = build_accrual_schedule(deal.effective_date, deal.maturity_date, deal.pay_freq_months)
    last_pay = last_pay_date_before(full_schedule, val_date)
    future_periods = future_schedule_from_val(full_schedule, val_date)

    # Compute PV in python (for lambda + printed results)
    pv = crma_pv_simplified(
        notional=deal.notional,
        fee_bps=deal.fee_bps,
        recovery=deal.recovery,
        effective_date=deal.effective_date,
        maturity_date=deal.maturity_date,
        freq_months=deal.pay_freq_months,
        day_basis=deal.day_basis,
        val_date=val_date,
        r=r,
        market_spread_bps=market_spread_bps
    )

    # ---- Input block (A3:B16)
    inputs = [
        ("ValuationDate", val_date),
        ("DiscountRate_r", r),
        ("MarketSpread_bps (mark)", market_spread_bps),
        ("Notional", deal.notional),
        ("ContractFee_bps", deal.fee_bps),
        ("ContractFee_decimal", "=B7/10000"),
        ("Recovery", deal.recovery),
        ("LGD (=1-Recovery)", "=1-B9"),
        ("SolvedLambda (flat)", pv["lambda"]),
        ("DayBasis", deal.day_basis),
        ("PayFreqMonths", deal.pay_freq_months),
        ("EffectiveDate", deal.effective_date),
        ("MaturityDate", deal.maturity_date),
        ("LastPayDate (<= ValDate)", last_pay)
    ]

    start_row = 3
    for i, (k, v) in enumerate(inputs):
        r_i = start_row + i
        ws[f"A{r_i}"] = k
        ws[f"A{r_i}"].font = bold
        ws[f"B{r_i}"] = v

    # ---- Schedule table (row 20 header; rows 21..)
    header_row = 20
    headers = [
        "PayDate", "AccStart", "AccEnd",
        "AccDays", "AlphaFull",
        "AlphaRem FromVal",
        "DaysToPay", "T_years",
        "DF", "Q", "Q_prev", "dQ",
        "PremFactor (=AlphaRem*DF*Q)",
        "ProtFactor (=DF*dQ)"
    ]

    for c, h in enumerate(headers, start=1):
        ws.cell(row=header_row, column=c, value=h).font = bold

    sched_start = 21
    sched_end = sched_start + max_rows - 1

    # Write pay dates + accrual starts (only for actual future rows)
    for i in range(max_rows):
        row = sched_start + i
        if i < len(future_periods):
            acc_start, pay = future_periods[i]
            ws[f"A{row}"] = pay
            ws[f"B{row}"] = acc_start
            ws[f"C{row}"] = pay
        else:
            ws[f"A{row}"] = None
            ws[f"B{row}"] = None
            ws[f"C{row}"] = None

        # Formulas (guarded by IF PayDate blank)
        ws[f"D{row}"] = f'=IF($A{row}="","", $C{row}-$B{row})'
        ws[f"E{row}"] = f'=IF($A{row}="","", $D{row}/$B$12)'
        ws[f"F{row}"] = f'=IF($A{row}="","", MAX(0, ($C{row}-MAX($B{row}, $B$3))/$B$12))'
        ws[f"G{row}"] = f'=IF($A{row}="","", $A{row}-$B$3)'
        ws[f"H{row}"] = f'=IF($A{row}="","", $G{row}/$B$12)'
        ws[f"I{row}"] = f'=IF($A{row}="","", EXP(-$B$4*$H{row}))'
        ws[f"J{row}"] = f'=IF($A{row}="","", EXP(-$B$11*$H{row}))'

        # Q_prev: first populated row -> 1; otherwise previous row's Q
        if i == 0:
            ws[f"K{row}"] = f'=IF($A{row}="","", 1)'
        else:
            ws[f"K{row}"] = f'=IF($A{row}="","", IF($A{row - 1}="", 1, $J{row - 1}))'

        ws[f"L{row}"] = f'=IF($A{row}="","", $K{row}-$J{row})'
        ws[f"M{row}"] = f'=IF($A{row}="","", $F{row}*$I{row}*$J{row})'
        ws[f"N{row}"] = f'=IF($A{row}="","", $I{row}*$L{row})'

    # ---- Summary block at fixed rows
    sum_rpv01 = 105

    ws[f"A{sum_rpv01}"] = "RPV01_years = SUM(PremFactor)"
    ws[f"A{sum_rpv01}"].font = bold
    ws[f"B{sum_rpv01}"] = f"=SUM(M{sched_start}:M{sched_end})"

    ws[f"A{sum_rpv01 + 1}"] = "ProtSum = SUM(ProtFactor)"
    ws[f"A{sum_rpv01 + 1}"].font = bold
    ws[f"B{sum_rpv01 + 1}"] = f"=SUM(N{sched_start}:N{sched_end})"

    ws[f"A{sum_rpv01 + 2}"] = "PV_Protection = Notional*LGD*ProtSum"
    ws[f"A{sum_rpv01 + 2}"].font = bold
    ws[f"B{sum_rpv01 + 2}"] = f"=$B$6*$B$10*B{sum_rpv01 + 1}"

    ws[f"A{sum_rpv01 + 3}"] = "PV_Premium = Notional*FeeDecimal*RPV01"
    ws[f"A{sum_rpv01 + 3}"].font = bold
    ws[f"B{sum_rpv01 + 3}"] = f"=$B$6*$B$8*B{sum_rpv01}"

    ws[f"A{sum_rpv01 + 4}"] = "Clean MTM (buyer) = PVprot - PVprem"
    ws[f"A{sum_rpv01 + 4}"].font = bold
    ws[f"B{sum_rpv01 + 4}"] = f"=B{sum_rpv01 + 2}-B{sum_rpv01 + 3}"

    ws[f"A{sum_rpv01 + 5}"] = "AccruedDays = ValDate - LastPayDate"
    ws[f"A{sum_rpv01 + 5}"].font = bold
    ws[f"B{sum_rpv01 + 5}"] = "=$B$3-$B$16"

    ws[f"A{sum_rpv01 + 6}"] = "AccruedPremium = Notional*FeeDecimal*AccruedDays/DayBasis"
    ws[f"A{sum_rpv01 + 6}"].font = bold
    ws[f"B{sum_rpv01 + 6}"] = f"=$B$6*$B$8*B{sum_rpv01 + 5}/$B$12"

    ws[f"A{sum_rpv01 + 7}"] = "Dirty MTM (buyer) = Clean - Accrued"
    ws[f"A{sum_rpv01 + 7}"].font = bold
    ws[f"B{sum_rpv01 + 7}"] = f"=B{sum_rpv01 + 4}-B{sum_rpv01 + 6}"

    ws[f"A{sum_rpv01 + 9}"] = "ModelSpread_bps = 10000*LGD*ProtSum/RPV01"
    ws[f"A{sum_rpv01 + 9}"].font = bold
    ws[f"B{sum_rpv01 + 9}"] = f"=IF(B{sum_rpv01}=0,0,10000*$B$10*B{sum_rpv01 + 1}/B{sum_rpv01})"

    ws[f"A{sum_rpv01 + 10}"] = "SpreadDiff_bps = ModelSpread_bps - MarketSpread_bps"
    ws[f"A{sum_rpv01 + 10}"].font = bold
    ws[f"B{sum_rpv01 + 10}"] = f"=B{sum_rpv01 + 9}-$B$5"

    # Formatting
    set_col_widths(ws, {
        1: 16, 2: 14, 3: 14,
        4: 10, 5: 12, 6: 16,
        7: 12, 8: 10, 9: 12,
        10: 12, 11: 12, 12: 10,
        13: 22, 14: 16
    })

    return ws.title, pv


def build_attribution_sheet(wb, pv_sheet_names: Dict[str, str],
                            dirty_cell: str, clean_cell: str,
                            accrued_cell: str) -> None:
    """
    Create Attribution sheet and link to PV sheets.
    pv_sheet_names must include keys: "t-1", "carry", "rates", "t"
    """
    ws = wb.create_sheet("Attribution")
    bold = Font(bold=True)

    ws["A1"] = "P&L Attribution Ladder (Dirty MTM)"
    ws["A1"].font = Font(bold=True, size=13)

    ws["A3"], ws["B3"] = "Scenario", "Dirty MTM (linked)"
    ws["A3"].font = ws["B3"].font = bold

    row_map = {"t-1": 4, "carry": 5, "rates": 6, "t": 7}
    for k, r in row_map.items():
        ws[f"A{r}"] = k
        ws[f"B{r}"] = f"='{pv_sheet_names[k]}'!{dirty_cell}"

    ws["A9"], ws["B9"] = "Carry = carry - (t-1)", "=B5-B4"
    ws["A10"], ws["B10"] = "Rates = rates - carry", "=B6-B5"
    ws["A11"], ws["B11"] = "Credit = t - rates", "=B7-B6"
    ws["A12"], ws["B12"] = "Total = t - (t-1)", "=B7-B4"
    ws["A13"], ws["B13"] = "Residual = Total - (Carry+Rates+Credit)", "=B12-(B9+B10+B11)"

    for rr in range(9, 14):
        ws[f"A{rr}"].font = bold

    # Optional: carry split
    ws["A15"], ws["B15"] = "Clean carry = Clean(carry) - Clean(t-1)", \
        f"='{pv_sheet_names['carry']}'!{clean_cell} - '{pv_sheet_names['t-1']}'!{clean_cell}"
    ws["A16"], ws["B16"] = "Accrual P&L = -(Accrued(carry) - Accrued(t-1))", \
        f"=-('{pv_sheet_names['carry']}'!{accrued_cell} - '{pv_sheet_names['t-1']}'!{accrued_cell})"
    ws["A17"], ws["B17"] = "Check: Clean carry + Accrual P&L", "=B15+B16"
    for rr in range(15, 17):
        ws[f"A{rr}"].font = bold

    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 35


def create_crma_workbook(deal: DealConfig) -> Tuple[openpyxl.Workbook, Dict[str, Dict[str, float]]]:
    """
    Creates one in-memory workbook for a deal with 4 PV sheets + Attribution.
    Returns:
        - Workbook object (NOT saved)
        - Dict of python-computed PV results for each scenario sheet
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Scenario definitions for ladder
    scenarios = {
        "t-1": (deal.t_minus_1, deal.r_prev, deal.spread_prev_bps),
        "carry": (deal.t, deal.r_prev, deal.spread_prev_bps),
        "rates": (deal.t, deal.r_today, deal.spread_prev_bps),
        "t": (deal.t, deal.r_today, deal.spread_today_bps),
    }

    pv_results: Dict[str, Dict[str, float]] = {}
    pv_sheet_names: Dict[str, str] = {}

    for key, (val_date, r, spr) in scenarios.items():
        sh_name = f"PV_{key}"
        sheet_name, pv = build_pv_sheet_crma(
            wb=wb,
            sheet_name=sh_name,
            deal=deal,
            val_date=val_date,
            r=r,
            market_spread_bps=spr,
            max_rows=80
        )
        pv_results[key] = pv
        pv_sheet_names[key] = sheet_name

    # Fixed output cells (by construction)
    # Summary starts at row 105; Dirty is B112, Clean is B109, AccruedPremium is B111.
    dirty_cell = "B112"
    clean_cell = "B109"
    accrued_cell = "B111"

    build_attribution_sheet(
        wb=wb,
        pv_sheet_names=pv_sheet_names,
        dirty_cell=dirty_cell,
        clean_cell=clean_cell,
        accrued_cell=accrued_cell
    )

    # Add an info sheet (optional)
    info = wb.create_sheet("DealInfo")
    bold = Font(bold=True)
    info["A1"] = "Deal"
    info["B1"] = deal.deal_name
    info["A2"] = "Protection Seller"
    info["B2"] = deal.protection_seller
    info["A3"] = "Notional"
    info["B3"] = deal.notional
    info["A4"] = "Fee (bps)"
    info["B4"] = deal.fee_bps
    info["A5"] = "Pay freq (months)"
    info["B5"] = deal.pay_freq_months
    info["A6"] = "Effective"
    info["B6"] = deal.effective_date
    info["A7"] = "Maturity"
    info["B7"] = deal.maturity_date
    for r in range(1, 8):
        info[f"A{r}"].font = bold
    info.column_dimensions["A"].width = 20
    info.column_dimensions["B"].width = 40

    return wb, pv_results


# ==========================================
# Demo: two CRMA deals
# ==========================================
def fmt(x: float) -> str:
    return f"{x:,.2f}"


def run_demo() -> None:
    """
    Builds both workbooks (in memory), prints MTM + attribution buckets numerically.
    Modify the deal configs to match your actual term sheets and market marks.
    """

    # --- Attribution scenario inputs ---
    t_minus_1 = date(2026, 1, 21)
    t = date(2026, 1, 22)
    r_prev = 0.0200
    r_today = 0.0198
    spread_prev_bps = 95.0
    spread_today_bps = 100.0

    # --- Deal 1: CBICL CRMA ---
    deal_cbicl = DealConfig(
        deal_name="CRMA | CBICL (中债信用增进) | Example Template",
        protection_seller="中债信用增进投资股份有限公司 (CBICL)",
        effective_date=date(2021, 7, 26),
        maturity_date=date(2026, 7, 26),
        notional=100_000_000.0,
        fee_bps=60.0,
        pay_freq_months=3,
        day_basis=365,
        recovery=0.40
    )
    # Attach scenario marks to the deal object dynamically for convenience
    deal_cbicl.t_minus_1 = t_minus_1
    deal_cbicl.t = t
    deal_cbicl.r_prev = r_prev
    deal_cbicl.r_today = r_today
    deal_cbicl.spread_prev_bps = spread_prev_bps
    deal_cbicl.spread_today_bps = spread_today_bps

    # --- Deal 2: Henan Zhongyu CRMA ---
    deal_henan = DealConfig(
        deal_name="CRMA | 河南中豫信用增进 | Example Template",
        protection_seller="河南中豫信用增进有限公司",
        effective_date=date(2024, 12, 13),
        maturity_date=date(2027, 12, 13),
        notional=320_000_000.0,
        fee_bps=60.0,
        pay_freq_months=3,
        day_basis=365,
        recovery=0.40
    )
    deal_henan.t_minus_1 = t_minus_1
    deal_henan.t = t
    deal_henan.r_prev = 0.0150
    deal_henan.r_today = 0.0148
    deal_henan.spread_prev_bps = 98.0
    deal_henan.spread_today_bps = 100.0

    # Build in-memory workbooks
    wb_cbicl, pv_cbicl = create_crma_workbook(deal_cbicl)
    wb_henan, pv_henan = create_crma_workbook(deal_henan)

    # --- Print key results ---
    for deal, pv in [(deal_cbicl, pv_cbicl), (deal_henan, pv_henan)]:
        print("\n" + "=" * 90)
        print(deal.deal_name)
        print("Seller:", deal.protection_seller)
        print(f"Valuation: {deal.t_minus_1} -> {deal.t}")
        print("-" * 90)

        # MTM at t
        mtm_t = pv["t"]
        print("MTM as-of t (Dirty):", fmt(mtm_t["dirty"]))
        print("    Clean:", fmt(mtm_t["clean"]))
        print("    Accrued:", fmt(mtm_t["accrued"]), f"(days={int(mtm_t['accrued_days'])})")
        print("    PV_protection:", fmt(mtm_t["PV_protection"]))
        print("    PV_premium:", fmt(mtm_t["PV_premium"]))
        print("    RPV01_years:", f"{mtm_t['RPV01_years']:.6f}")
        print("    lambda:", f"{mtm_t['lambda']:.8f} /yr")
        print("    model spread check (bps):", f"{mtm_t['model_spread_bps']:.4f}")

        # Ladder attribution (Dirty MTM)
        carry = pv["carry"]["dirty"] - pv["t-1"]["dirty"]
        rates = pv["rates"]["dirty"] - pv["carry"]["dirty"]
        credit = pv["t"]["dirty"] - pv["rates"]["dirty"]
        total = pv["t"]["dirty"] - pv["t-1"]["dirty"]
        residual = total - (carry + rates + credit)

        print("\nP&L Attribution (Dirty MTM ladder):")
        print("    Carry :", fmt(carry))
        print("    Rates :", fmt(rates))
        print("    Credit:", fmt(credit))
        print("    Total :", fmt(total))
        print("    Residual (should ~0):", fmt(residual))

    # NOTE: We do not save files.
    # If you ever want to save locally, uncomment:
    wb_cbicl.save("CRMA_CBICL_template.xlsx")
    wb_henan.save("CRMA_HenanZhongyu_template.xlsx")


if __name__ == "__main__":
    run_demo()