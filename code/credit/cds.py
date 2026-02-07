import math
from datetime import date
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

"""
USD CDS example (Protection Buyer) - build an Excel-style schedule + formulas with openpyxl,
AND compute MTM + P&L attribution numerically in Python (since openpyxl does not evaluate formulas).

Trade (sample):
 - Notional: 10,000,000 USD
 - Fixed coupon: 100 bps (1.00%) ACT/360
 - Recovery: 40%
 - Last coupon date: 2009-09-21
 - Remaining pay dates: 2009-12-21 (91/360), 2010-03-22 (90/360)

Valuation / attribution example:
 - t-1: 2009-10-01, r=2.02%, market par spread=390 bps
 - t:   2009-10-02, r=2.00%, market par spread=400 bps

Model (simplified CDS, discrete, flat r and flat hazard lambda):
 - DF(T) = exp(-r*T)
 - Q(T)  = exp(-lambda*T)
 - Par spread condition (solve lambda): S = LGD * Sum_i[ DF_i * (Q_{i-1}-Q_i) ] / Sum_i[ alpha_i*DF_i*Q_i ]
 - Clean PV (buyer) = PV_protection - PV_premium_fixed_coupon
 - Dirty PV = Clean PV - AccruedPremium (since buyer owes accrued)

NOTE: This simplified model ignores accrued-on-default premium and some ISDA timing nuances.
"""


# ---------------------------------------------------------
# 1) Core pricing functions
# ---------------------------------------------------------

def solve_lambda_flat(
        val_date: date,
        pay_dates: list[date],
        accrual_fracs: list[float],
        r: float,
        spread_decimal: float,
        lgd: float,
        day_basis: int = 360
) -> float:
    """
    Solve for a flat hazard lambda so that the model par spread equals spread_decimal.
    """

    # Par spread model (discrete):
    #   S_model(lambda) = LGD * prot_sum(lambda) / rpv01(lambda)
    #
    # where:
    #   prot_sum(lambda) = Sum DF_i * (Q_{i-1} - Q_i)
    #   rpv01(lambda)    = Sum alpha_i * DF_i * Q_i

    # Precompute times and DFs for speed
    Ts = []
    DFs = []
    for t in pay_dates:
        T = (t - val_date).days / day_basis
        Ts.append(T)
        DFs.append(math.exp(-r * T))

    def S_model(lam: float) -> float:
        Q_prev = 1.0
        prot_sum = 0.0
        rpv01 = 0.0
        for T, DF, a in zip(Ts, DFs, accrual_fracs):
            if T <= 0:
                continue

            Q = math.exp(-lam * T)
            prot_sum += DF * (Q_prev - Q)
            rpv01 += a * DF * Q
            Q_prev = Q

        if rpv01 == 0:
            return 0.0
        return lgd * prot_sum / rpv01

    # Bisection (robust). At lam=0, S_model=0. As lam increases, S_model increases.
    S_target = spread_decimal
    lo = 0.0
    hi = max(0.5, (S_target / max(lgd, 1e-12)) * 5.0 + 1e-6)

    def f(lam: float) -> float:
        return S_model(lam) - S_target

    # Ensure bracket
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


def cds_pv_simplified(
        val_date: date,
        last_coupon_date: date,
        pay_dates: list[date],
        accrual_fracs: list[float],
        notional: float,
        coupon_decimal: float,
        r: float,
        market_spread_bps: float,
        recovery: float,
        day_basis: int = 360
) -> dict:
    """
    Return a dictionary with solved lambda, RPV01, protection PV, premium PV, clean/dirty PV.
    market_spread_bps is used ONLY to solve lambda (i.e., represent today's market credit level).
    The contract coupon is coupon_decimal (fixed 100bp here).
    """

    lgd = 1.0 - recovery
    S = market_spread_bps / 10000.0

    lam = solve_lambda_flat(
        val_date=val_date,
        pay_dates=pay_dates,
        accrual_fracs=accrual_fracs,
        r=r,
        spread_decimal=S,
        lgd=lgd,
        day_basis=day_basis
    )

    prot_sum = 0.0
    rpv01 = 0.0
    Q_prev = 1.0

    for t, a in zip(pay_dates, accrual_fracs):
        T = (t - val_date).days / day_basis
        if T <= 0:
            continue

        DF = math.exp(-r * T)
        Q = math.exp(-lam * T)
        prot_sum += DF * (Q_prev - Q)
        rpv01 += a * DF * Q
        Q_prev = Q

    pv_prot = notional * lgd * prot_sum
    pv_prem = notional * coupon_decimal * rpv01
    clean = pv_prot - pv_prem

    accrued_days = (val_date - last_coupon_date).days
    accrued = notional * coupon_decimal * accrued_days / day_basis

    dirty = clean - accrued

    model_spread_bps = 10000.0 * lgd * prot_sum / rpv01 if rpv01 != 0 else 0.0

    return {
        "lambda": lam,
        "RPV01_years": rpv01,
        "PV_protection": pv_prot,
        "PV_premium": pv_prem,
        "Clean": clean,
        "accrued_days": accrued_days,
        "accrued": accrued,
        "dirty": dirty,
        "model_spread_bps": model_spread_bps,
    }


# ---------------------------------------------------------
# 2) Build an in-memory Excel workbook
#    with schedule + formulas (openpyxl)
# ---------------------------------------------------------

def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_bp(x: float) -> str:
    return f"{x:,.2f} bp"


def _sheet_title_safe(s: str) -> str:
    # Excel sheet name max length 31
    return s[:31]


def build_pv_sheet(
        wb: Workbook,
        sheet_name: str,
        val_date: date,
        r: float,
        market_spread_bps: float,
        trade_inputs: dict
) -> tuple[str, dict]:
    """
    Creates one PV sheet with:
      - input block
      - 2-row schedule table
      - formulas for DF/Q/PV legs/MTM
    Also computes numeric values in python and returns them.
    """

    ws = wb.create_sheet(_sheet_title_safe(sheet_name))

    # Unpack trade inputs
    notional = trade_inputs["notional"]
    coupon_bps = trade_inputs["coupon_bps"]
    coupon = coupon_bps / 10000.0
    recovery = trade_inputs["recovery"]
    last_coupon = trade_inputs["last_coupon_date"]
    pay_dates = trade_inputs["pay_dates"]
    accrual_fracs = trade_inputs["accrual_fracs"]
    day_basis = trade_inputs["day_basis"]

    # Compute numbers in python for printing (and to populate lambda)
    pv = cds_pv_simplified(
        val_date=val_date,
        last_coupon_date=last_coupon,
        pay_dates=pay_dates,
        accrual_fracs=accrual_fracs,
        notional=notional,
        coupon_decimal=coupon,
        r=r,
        market_spread_bps=market_spread_bps,
        recovery=recovery,
        day_basis=day_basis
    )

    # Styling helpers
    bold = Font(bold=True)
    ws["A1"] = f"PV Sheet: {sheet_name}"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A1"].alignment = Alignment(horizontal="left")

    # Input block (rows 3-12)
    # We'll keep key inputs in column B, so formulas can anchor to fixed cells.
    ws["A3"], ws["B3"] = "ValuationDate", val_date
    ws["A4"], ws["B4"] = "DiscountRate_r", r
    ws["A5"], ws["B5"] = "MarketSpread_bps", market_spread_bps
    ws["A6"], ws["B6"] = "Notional", notional
    ws["A7"], ws["B7"] = "ContractCoupon_bps", coupon_bps
    ws["A8"], ws["B8"] = "Recovery", recovery
    ws["A9"], ws["B9"] = "LGD (=1-Recovery)", "=1-$B$8"
    ws["A10"], ws["B10"] = "SolvedLambda", pv["lambda"]
    ws["A11"], ws["B11"] = "LastCouponDate", last_coupon
    ws["A12"], ws["B12"] = "DayBasis", day_basis

    for r_i in range(3, 13):
        ws[f"A{r_i}"].font = bold

    # Schedule header row (row 15)
    headers = [
        "PayDate", "Alpha", "DaysToPay", "T_years",
        "DF", "Q", "Q_prev", "dQ",
        "PremFactor (alpha*DF*Q)", "ProtFactor (DF*dQ)"
    ]

    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=15, column=c, value=h)
        cell.font = bold

    # Schedule rows start at 16
    start_row = 16
    for i, (pd, a) in enumerate(zip(pay_dates, accrual_fracs)):
        row = start_row + i
        ws.cell(row=row, column=1, value=pd)  # A: PayDate
        ws.cell(row=row, column=2, value=a)  # B: Alpha

        # C: DaysToPay = PayDate - ValuationDate
        ws.cell(row=row, column=3, value=f"=A{row}-$B$3")

        # D: T_years = DaysToPay / DayBasis
        ws.cell(row=row, column=4, value=f"=C{row}/$B$12")

        # E: DF = EXP(-r*T)
        ws.cell(row=row, column=5, value=f"=EXP(-$B$4*D{row})")

        # F: Q = EXP(-lambda*T)
        ws.cell(row=row, column=6, value=f"=EXP(-$B$10*D{row})")

        # G: Q_prev
        if i == 0:
            ws.cell(row=row, column=7, value=1.0)
        else:
            ws.cell(row=row, column=7, value=f"=F{row - 1}")

        # H: dQ = Q_prev - Q
        ws.cell(row=row, column=8, value=f"=G{row}-F{row}")

        # I: PremFactor = alpha*DF*Q
        ws.cell(row=row, column=9, value=f"=B{row}*E{row}*F{row}")

        # J: ProtFactor = DF*dQ
        ws.cell(row=row, column=10, value=f"=E{row}*H{row}")

    # Summary block (rows 20+)
    sum_row = start_row + len(pay_dates) + 2  # e.g. 20
    ws[f"A{sum_row}"] = "RPV01_years (SUM PremFactor)"
    ws[f"A{sum_row}"].font = bold
    ws[f"B{sum_row}"] = f"=SUM(I{start_row}:I{start_row + len(pay_dates) - 1})"

    ws[f"A{sum_row + 1}"] = "ProtSum (SUM ProtFactor)"
    ws[f"A{sum_row + 1}"].font = bold
    ws[f"B{sum_row + 1}"] = f"=SUM(J{start_row}:J{start_row + len(pay_dates) - 1})"

    ws[f"A{sum_row + 2}"] = "PV Protection"
    ws[f"A{sum_row + 2}"].font = bold
    ws[f"B{sum_row + 2}"] = f"=$B$6*$B$9*B{sum_row + 1}"

    ws[f"A{sum_row + 3}"] = "PV Premium (fixed coupon)"
    ws[f"A{sum_row + 3}"].font = bold
    ws[f"B{sum_row + 3}"] = f"=$B$6*($B$7/10000)*B{sum_row}"

    ws[f"A{sum_row + 4}"] = "Clean MTM (buyer) = PVprot - PVprem"
    ws[f"A{sum_row + 4}"].font = bold
    ws[f"B{sum_row + 4}"] = f"=B{sum_row + 2}-B{sum_row + 3}"

    ws[f"A{sum_row + 5}"] = "AccruedDays = ValDate - LastCouponDate"
    ws[f"A{sum_row + 5}"].font = bold
    ws[f"B{sum_row + 5}"] = "=($B$3-$B$11)"

    ws[f"A{sum_row + 6}"] = "AccruedPremium"
    ws[f"A{sum_row + 6}"].font = bold
    ws[f"B{sum_row + 6}"] = f"=$B$6*($B$7/10000)*B{sum_row + 5}/$B$12"

    ws[f"A{sum_row + 7}"] = "Dirty MTM (buyer) = Clean - Accrued"
    ws[f"A{sum_row + 7}"].font = bold
    ws[f"B{sum_row + 7}"] = f"=B{sum_row + 4}-B{sum_row + 6}"

    # Par spread check (optional)
    ws[f"A{sum_row + 9}"] = "ModelSpread_bps = 10000*LGD*ProtSum/RPV01"
    ws[f"A{sum_row + 9}"].font = bold
    ws[f"B{sum_row + 9}"] = f"=10000*$B$9*B{sum_row + 1}/B{sum_row}"

    ws[f"A{sum_row + 10}"] = "SpreadDiff_bps = ModelSpread_bps - MarketSpread_bps"
    ws[f"A{sum_row + 10}"].font = bold
    ws[f"B{sum_row + 10}"] = f"=B{sum_row + 9}-$B$5"

    # Basic formatting
    ws.column_dimensions["A"].width = 30
    for col in range(2, 11):
        ws.column_dimensions[get_column_letter(col)].width = 18

    return ws.title, pv


def build_attribution_sheet(
        wb: Workbook,
        refs: dict
) -> None:
    """
    refs: dict of {label: (sheet_name, dirty_cell_address, clean_cell_address, accrued_cell_address)}
    """
    ws = wb.create_sheet("Attribution")

    bold = Font(bold=True)
    ws["A1"] = "Daily P&L Attribution Ladder (Dirty MTM)"
    ws["A1"].font = Font(bold=True, size=13)

    ws["A3"], ws["B3"] = "Scenario", "Dirty MTM (formula link)"
    ws["A3"].font = ws["B3"].font = bold

    # Place scenario dirty PV links
    # Expected keys: "t-1", "carry", "rates", "t"
    row_map = {"t-1": 4, "carry": 5, "rates": 6, "t": 7}
    for key, row in row_map.items():
        sh, dirty_cell, clean_cell, acc_cell = refs[key]
        ws[f"A{row}"] = key
        ws[f"B{row}"] = f"='{sh}'!{dirty_cell}"

    # Bucket calcs
    ws["A9"], ws["B9"] = "Carry = carry - (t-1)", "=B5-B4"
    ws["A10"], ws["B10"] = "Rates = rates - carry", "=B6-B5"
    ws["A11"], ws["B11"] = "Credit = t - rates", "=B7-B6"
    ws["A12"], ws["B12"] = "Total = t - (t-1)", "=B7-B4"

    ws["A13"], ws["B13"] = "Residual = Total - (Carry+Rates+Credit)", "=B12-(B9+B10+B11)"
    for r in range(9, 14):
        ws[f"A{r}"].font = bold

    # Optional: split carry into clean carry and accrual P&L
    ws["A15"], ws["B15"] = "Clean carry = Clean(carry) - Clean(t-1)", None
    ws["A16"], ws["B16"] = "Accrual P&L = -(Accrued(carry) - Accrued(t-1))", None
    ws["A17"], ws["B17"] = "Check: Clean carry + Accrual P&L", None
    for r in range(15, 16, 17):
        ws[f"A{r}"].font = bold

    # Link clean and accrued
    sh_t1, dirty_t1, clean_t1, acc_t1 = refs["t-1"]
    sh_c, dirty_c, clean_c, acc_c = refs["carry"]
    ws["B15"] = f"='{sh_c}'!{clean_c} - '{sh_t1}'!{clean_t1}"
    ws["B16"] = f"= - ('{sh_c}'!{acc_c} - '{sh_t1}'!{acc_t1} )"
    ws["B17"] = "=B15+B16"

    ws.column_dimensions["A"].width = 45
    ws.column_dimensions["B"].width = 28


# ---------------------------------------------------------
# 3) Main script
# ---------------------------------------------------------

def main() -> None:
    # Trade constants
    trade = {
        "notional": 10_000_000.0,
        "coupon_bps": 100.0,
        "recovery": 0.40,
        "last_coupon_date": date(2009, 9, 21),
        "pay_dates": [date(2009, 12, 21), date(2010, 3, 22)],
        "accrual_fracs": [91 / 360, 90 / 360],
        "day_basis": 360,
    }

    # Market marks used for this attribution example
    t_minus_1 = date(2009, 10, 1)
    t = date(2009, 10, 2)

    r1 = 0.0202  # 2.02%
    r2 = 0.0200  # 2.00%

    S1_bps = 390.0
    S2_bps = 400.0

    # Build workbook in memory
    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Create PV sheets
    sh_t1, pv_t1 = build_pv_sheet(wb, "PV_2009-10-01", t_minus_1, r1, S1_bps, trade)
    sh_c, pv_c = build_pv_sheet(wb, "PV_Carry", t, r1, S1_bps, trade)
    sh_r, pv_r = build_pv_sheet(wb, "PV_Rates", t, r2, S1_bps, trade)
    sh_t, pv_t = build_pv_sheet(wb, "PV_2009-10-02", t, r2, S2_bps, trade)

    # The summary block location depends on table size; we used a deterministic layout:
    # sum_row = start_row(16) + n_pay(2) + 2 = 20
    # Clean MTM is at B(sum_row+4) => B24
    # AccruedPremium is at B(sum_row+6) => B26
    # Dirty MTM is at B(sum_row+7) => B27
    #
    # Let's keep those cell references consistent for attribution linking.
    clean_cell = "B24"
    acc_cell = "B26"
    dirty_cell = "B27"

    refs = {
        "t-1": (sh_t1, dirty_cell, clean_cell, acc_cell),
        "carry": (sh_c, dirty_cell, clean_cell, acc_cell),
        "rates": (sh_r, dirty_cell, clean_cell, acc_cell),
        "t": (sh_t, dirty_cell, clean_cell, acc_cell),
    }

    build_attribution_sheet(wb, refs)

    # DO NOT SAVE (per your request). If you ever want to save locally, uncomment:
    wb.save("usd_cds_pnl_attribution_2009-10-02.xlsx")

    # ----------------------------------------------------
    # 4) Print the MTM and attribution results (numerically computed)
    # ----------------------------------------------------

    # MTM on 02-Oct-2009
    print("\n=== MTM as-of 2009-10-02 (r=2.00%, market spread=400bp) ===")
    print(f"lambda (solved)        : {pv_t['lambda']:.12f} per year")
    print(f"RPV01 (years)          : {pv_t['RPV01_years']:.12f}")
    print(f"PV (protection leg)    : {_fmt_money(pv_t['PV_protection'])}")
    print(f"PV (premium leg @100bp) : {_fmt_money(pv_t['PV_premium'])}")
    print(f"Clean MTM (buyer)      : {_fmt_money(pv_t['Clean'])}")
    print(f"Accrued days           : {pv_t['accrued_days']} days")
    print(f"Accrued premium (payable): {_fmt_money(pv_t['accrued'])}")
    print(f"Dirty MTM (buyer)      : {_fmt_money(pv_t['dirty'])}")
    print(f"Model par spread check : {_fmt_bp(pv_t['model_spread_bps'])}")

    # Ladder P&L attribution for 01-Oct -> 02-Oct
    carry = pv_c["dirty"] - pv_t1["dirty"]
    rates = pv_r["dirty"] - pv_c["dirty"]
    credit = pv_t["dirty"] - pv_r["dirty"]
    total = pv_t["dirty"] - pv_t1["dirty"]
    residual = total - (carry + rates + credit)

    print("\n=== Daily P&L Attribution (Dirty MTM) : 2009-10-01 -> 2009-10-02 ===")
    print(f"Dirty MTM (t-1) : {_fmt_money(pv_t1['dirty'])}")
    print(f"Dirty MTM (carry): {_fmt_money(pv_c['dirty'])}")
    print(f"Dirty MTM (rates): {_fmt_money(pv_r['dirty'])}")
    print(f"Dirty MTM (t)    : {_fmt_money(pv_t['dirty'])}")
    print("\nBuckets:")
    print(f"Carry   : {_fmt_money(carry)}")
    print(f"Rates   : {_fmt_money(rates)}")
    print(f"Credit  : {_fmt_money(credit)}")
    print(f"Total   : {_fmt_money(total)}")
    print(f"Residual (should ~0): {_fmt_money(residual)}")

    # Optional: show key Excel formulas that were written (illustrative)
    ws = wb[sh_t]
    print("\n=== Example formulas written to PV_2009-10-02 sheet ===")
    print(f"DF (row 16) formula     : {ws['E16'].value}")
    print(f"Q  (row 16) formula     : {ws['F16'].value}")
    print(f"RPV01 formula (B20)     : {ws['B20'].value}")
    print(f"Clean MTM formula (B24) : {ws['B24'].value}")
    print(f"Dirty MTM formula (B27) : {ws['B27'].value}")


if __name__ == "__main__":
    main()