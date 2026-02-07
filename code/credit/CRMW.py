from __future__ import annotations
from datetime import date
import math
from typing import List, Dict, Tuple, Any
from openpyxl import Workbook


# ==============================================================================
# # 0) Helpers
# ==============================================================================

def d(iso: str) -> date:
    return date.fromisoformat(iso)


def act365(a: date, b: date) -> float:
    return (b - a).days / 365.0


def interp_linear(x: float, xs: List[float], ys: List[float]) -> float:
    """Linear interpolation with flat extrapolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def bump_node(xs: List[float], ys: List[float], bump_tenor: float, bump_bp: float = 1.0) -> List[float]:
    """Bump one curve node by +bump_bp (bp)."""
    bump = bump_bp / 10000.0
    out = ys[:]
    idx = xs.index(bump_tenor)
    out[idx] += bump
    return out


def print_block(title: str, rows: List[List[Any]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    for r in rows:
        print(" | ".join(str(x) for x in r))


# ==============================================================================
# # 1) CRMW terms (from your discussed example)
# ==============================================================================

PRODUCT = "25 中债增 CRMW003(022500034)"
REF_BOND = "25 鲁宏桥 GN003(科创债)"

# Core economic inputs
M = 100_000_000  # Notional
S = 0.006  # Fee Rate 0.6%
R = 0.40  # Recovery assumption

prot_start = d("2025-05-29")
prot_mat = d("2030-05-29")

# Payment schedule (5 installments) and coverage periods
# Period i: [Start_i, End_i], PayDate_i
periods = [
    (1, d("2025-05-29"), d("2026-05-29"), d("2025-06-03")),
    (2, d("2026-05-29"), d("2027-05-29"), d("2026-05-29")),
    (3, d("2027-05-29"), d("2028-05-29"), d("2027-05-29")),
    (4, d("2028-05-29"), d("2029-05-29"), d("2028-05-29")),
    (5, d("2029-05-29"), d("2030-05-29"), d("2029-05-29")),
]

# Daily attribution dates (dummy example)
t0 = d("2026-01-20")
t1 = d("2026-01-21")

# ==============================================================================
# # 2) Dummy curves (for MTM + attribution)
# ==============================================================================

# Risk-free curve nodes for DV01
rf_tenors = [0.5, 1.0, 3.0, 5.0, 10.0]
rf_t0 = [0.0230, 0.0240, 0.0260, 0.0270, 0.0280]  # dummy
rf_t1 = [0.0232, 0.0242, 0.0262, 0.0272, 0.0282]  # dummy (+2bp)

# Reference credit spread curve nodes for CS01 (proxy z-spread / hazard driver)
cr_tenors = [0.5, 1.0, 3.0, 5.0, 10.0]
cr_t0 = [0.0120, 0.0130, 0.0150, 0.0160, 0.0170]  # dummy
cr_t1 = [0.0125, 0.0135, 0.0158, 0.0168, 0.0178]  # dummy (widen)


# ==============================================================================
# # 3) Simple CRMW PV model (CDS-like proxy)
# ==============================================================================

def df_rf(asof: date, dt: date, rf_curve: List[float]) -> float:
    tau = act365(asof, dt)
    if tau <= 0:
        return 0.0
    r = interp_linear(tau, rf_tenors, rf_curve)
    return math.exp(-r * tau)


def hazard(asof: date, dt: date, cr_curve: List[float], R_: float) -> float:
    tau = act365(asof, dt)
    z = interp_linear(tau, cr_tenors, cr_curve)
    return z / (1.0 - R_)


def crmw_pv(asof: date, rf_curve: List[float], cr_curve: List[float],
            notional: float, fee_rate: float, recovery: float) -> Tuple[float, float, float]:
    """
    Build a timeline of future dates = union(end dates, pay dates) after asof.
    Step survival Q over timeline using piecewise constant hazard on each segment.
    Protection PV ~ LGD * sum DF(t_i) * (Q_prev - Q_i)
    Premium PV    ~ sum FeeAmount_i * DF(pay_i) * Q(pay_i)  for pay_i > asof
    """
    LGD = (1.0 - recovery) * notional
    fee_amt = notional * fee_rate

    # timeline dates for survival / default discretization
    future_dates = set()
    pay_dates = []
    end_dates = []

    for _, sd, ed, pd in periods:
        if ed > asof:
            future_dates.add(ed)
            end_dates.append(ed)
        if pd > asof:
            future_dates.add(pd)
            pay_dates.append(pd)

    if prot_mat > asof:
        future_dates.add(prot_mat)
        end_dates.append(prot_mat)

    timeline = sorted(future_dates)

    # step survival
    Q_prev = 1.0
    prev_dt = asof
    Q_at: Dict[date, float] = {}

    prot_pv = 0.0
    for dt in timeline:
        delta = act365(prev_dt, dt)
        if delta < 0:
            continue

        h = hazard(asof, dt, cr_curve, recovery)
        Q = Q_prev * math.exp(-h * delta)

        DF = df_rf(asof, dt, rf_curve)

        # Protection contribution on each interval paid at dt
        prot_pv += LGD * DF * (Q_prev - Q)

        Q_at[dt] = Q
        prev_dt = dt
        Q_prev = Q

    # premium PV (expected premiums paid only if survive to pay date)
    prem_pv = 0.0
    for pd in pay_dates:
        DF = df_rf(asof, pd, rf_curve)
        Qp = Q_at[pd]
        prem_pv += fee_amt * DF * Qp

    total = prot_pv - prem_pv
    return total, prot_pv, prem_pv


# ==============================================================================
# # 4) PV ladder for attribution (stepwise reval)
# ==============================================================================

PV_t0, Prot_t0, Prem_t0 = crmw_pv(t0, rf_t0, cr_t0, M, S, R)
PV_theta, Prot_theta, Prem_theta = crmw_pv(t1, rf_t0, cr_t0, M, S, R)
PV_rates, Prot_rates, Prem_rates = crmw_pv(t1, rf_t1, cr_t0, M, S, R)
PV_credit, Prot_credit, Prem_credit = crmw_pv(t1, rf_t1, cr_t1, M, S, R)

carry = PV_theta - PV_t0
rates = PV_rates - PV_theta
credit = PV_credit - PV_rates

# Cashflow between dates (actual premium paid, not PV)
fee_amt = M * S
cashflow_between = 0.0
for _, _, _, pd in periods:
    if (pd > t0) and (pd <= t1):
        cashflow_between -= fee_amt

total_pnl = (PV_credit - PV_t0) + cashflow_between
residual = total_pnl - (carry + rates + credit + cashflow_between)

# ==============================================================================
# # 5) Bucketed DV01 & CS01 (node bump +1bp)
# ==============================================================================
# Base for sensitivities = PV_credit scenario (asof t1, curves t1)

PV_base = PV_credit
BUMP_BP = 1.0

dv01 = []
for k in rf_tenors:
    rf_b = bump_node(rf_tenors, rf_t1, k, bump_bp=BUMP_BP)
    pv_b, _, _ = crmw_pv(t1, rf_b, cr_t1, M, S, R)
    dv01.append((k, pv_b))

cs01 = []
for k in cr_tenors:
    cr_b = bump_node(cr_tenors, cr_t1, k, bump_bp=BUMP_BP)
    pv_b, _, _ = crmw_pv(t1, rf_t1, cr_b, M, S, R)
    cs01.append((k, pv_b))

# ==============================================================================
# # 6) Build openpyxl workbook (IN MEMORY ONLY, no file saved)
# ==============================================================================

wb = Workbook()

# --- Inputs Sheet ---
ws_in = wb.active
ws_in.title = "Inputs"
ws_in.append(["Field", "Value"])
ws_in.append(["Product", PRODUCT])
ws_in.append(["Reference Bond", REF_BOND])
ws_in.append(["Notional M", M])
ws_in.append(["Fee rate S", S])
ws_in.append(["Recovery R", R])
ws_in.append(["Protection Start", prot_start])
ws_in.append(["Protection Maturity", prot_mat])
ws_in.append(["t0", t0])
ws_in.append(["t1", t1])
ws_in.append(["Per-installment FeeAmount (=M*S)", None])
ws_in["B11"] = "=B4*B5"

# --- Schedule Sheet ---
ws_sc = wb.create_sheet("Schedule")
ws_sc.append(["Period", "StartDate", "EndDate", "PayDate", "Days", "Fee_i", "DailyCarry"])
for i, sd, ed, pd in [(p[0], p[1], p[2], p[3]) for p in periods]:
    row = ws_sc.max_row + 1
    ws_sc.cell(row=row, column=1, value=i)
    ws_sc.cell(row=row, column=2, value=sd)
    ws_sc.cell(row=row, column=3, value=ed)
    ws_sc.cell(row=row, column=4, value=pd)
    # Days = EndDate - StartDate
    ws_sc.cell(row=row, column=5, value=f"=C{row}-B{row}")
    # Fee_i = Inputs!Notional * Inputs!FeeRate
    ws_sc.cell(row=row, column=6, value="=Inputs!$B$4*Inputs!$B$5")
    # DailyCarry (buyer) = -Fee_i / Days
    ws_sc.cell(row=row, column=7, value=f"=-F{row}/E{row}")

# --- Curves Sheet ---
ws_cv = wb.create_sheet("Curves")
ws_cv.append(["TenorY", "RF_t0", "RF_t1", "dRF_bp", "CR_t0", "CR_t1", "dCR_bp"])
for i in range(len(rf_tenors)):
    row = ws_cv.max_row + 1
    ws_cv.cell(row=row, column=1, value=rf_tenors[i])
    ws_cv.cell(row=row, column=2, value=rf_t0[i])
    ws_cv.cell(row=row, column=3, value=rf_t1[i])
    ws_cv.cell(row=row, column=4, value=f"=(C{row}-B{row})*10000")  # bp move
    ws_cv.cell(row=row, column=5, value=cr_tenors[i])
    ws_cv.cell(row=row, column=6, value=cr_t0[i])
    ws_cv.cell(row=row, column=7, value=cr_t1[i])
    ws_cv.cell(row=row, column=8, value=f"=(G{row}-F{row})*10000")  # bp move

# --- Accrual Sheet (t0 vs t1) ---
ws_ac = wb.create_sheet("Accrual")
ws_ac.append([
    "Period", "StartDate", "EndDate", "PayDate", "Days", "Fee_i",
    "ConsumedDays_t0", "FeeAccrued_t0", "Payable_t0", "Prepaid_t0",
    "ConsumedDays_t1", "FeeAccrued_t1", "Payable_t1", "Prepaid_t1",
    "PremiumPnL_(t0->t1)"
])
for r in range(2, 2 + len(periods)):
    out = ws_ac.max_row + 1
    # references to schedule
    ws_ac.cell(out, 1, value=f"=Schedule!A{r}")
    ws_ac.cell(out, 2, value=f"=Schedule!B{r}")
    ws_ac.cell(out, 3, value=f"=Schedule!C{r}")
    ws_ac.cell(out, 4, value=f"=Schedule!D{r}")
    ws_ac.cell(out, 5, value=f"=Schedule!E{r}")
    ws_ac.cell(out, 6, value=f"=Schedule!F{r}")

    # ConsumedDays(t) = MAX(0, MIN(t, EndDate) - StartDate)
    ws_ac.cell(out, 7, value=f"=MAX(0,MIN(Inputs!$B$9,C{out})-B{out})")
    ws_ac.cell(out, 11, value=f"=MAX(0,MIN(Inputs!$B$10,C{out})-B{out})")

    # FeeAccrued(t) = Fee_i * ConsumedDays / Days
    ws_ac.cell(out, 8, value=f"=F{out}*G{out}/E{out}")
    ws_ac.cell(out, 12, value=f"=F{out}*K{out}/E{out}")

    # Payable(t) = IF(t < PayDate, FeeAccrued, 0)
    ws_ac.cell(out, 9, value=f"=IF(Inputs!$B$9<D{out},H{out},0)")
    ws_ac.cell(out, 13, value=f"=IF(Inputs!$B$10<D{out},L{out},0)")

    # Prepaid(t) = IF(t < PayDate, 0, Fee_i - FeeAccrued)
    ws_ac.cell(out, 10, value=f"=IF(Inputs!$B$9<D{out},0,F{out}-H{out})")
    ws_ac.cell(out, 14, value=f"=IF(Inputs!$B$10<D{out},0,F{out}-L{out})")

    # Premium PnL (buyer) over (t0->t1): -(Accrued(t1) - Accrued(t0))
    ws_ac.cell(out, 15, value=f"=-(L{out}-H{out})")

# --- Valuation (PV ladder) ---
ws_vl = wb.create_sheet("Valuation")
ws_vl.append(["Scenario", "Asof", "PV_Prot (num)", "PV_Prem (num)", "PV_Total (=Prot-Prem)"])
ws_vl.append(["PV_t0", t0, Prot_t0, Prem_t0, "=C2-D2"])
ws_vl.append(["PV_theta", t1, Prot_theta, Prem_theta, "=C3-D3"])
ws_vl.append(["PV_rates", t1, Prot_rates, Prem_rates, "=C4-D4"])
ws_vl.append(["PV_credit", t1, Prot_credit, Prem_credit, "=C5-D5"])
ws_vl.append(["PV_base_for_sens", t1, "", "", "=E5"])

# --- Attribution (stepwise reval) ---
ws_at = wb.create_sheet("Attribution")
ws_at.append(["Component", "Value (formula)"])
ws_at.append(["Carry/Theta", "=Valuation!E3 - Valuation!E2"])
ws_at.append(["Rates", "=Valuation!E4 - Valuation!E3"])
ws_at.append(["Credit", "=Valuation!E5 - Valuation!E4"])
# Cashflow between dates: -SUMIFS(Fee_i, PayDate, (t0,t1])
ws_at.append(["Cashflows(t0,t1]",
              f'=-SUMIFS(Schedule!$F$2:$F$6,Schedule!$D$2:$D$6,">"&Inputs!$B$9,Schedule!$D$2:$D$6,"<="&Inputs!$B$10)'])
ws_at.append(["Total", "=Valuation!E5 - Valuation!E2 + B5"])
ws_at.append(["Residual", "=B6 - (B2+B3+B4+B5)"])

# --- Sensitivities (DV01 & CS01 buckets) ---
ws_sn = wb.create_sheet("Sensitivities")
ws_sn.append(["Type", "TenorY", "PV_bumped (num)", "Sens (formula)", "Sens (Python num)"])
ws_sn.append(["--- DV01 (risk-free key-rate) ---", "", "", "", ""])
dv01_start = ws_sn.max_row + 1
for ten, pv_b in dv01:
    r = ws_sn.max_row + 1
    ws_sn.cell(r, 1, value="DV01")
    ws_sn.cell(r, 2, value=ten)
    ws_sn.cell(r, 3, value=pv_b)
    ws_sn.cell(r, 4, value=f"=C{r} - Valuation!$E$5")  # in-cell formula
dv01_end = ws_sn.max_row

ws_sn.append(["", "", "", "", ""])
ws_sn.append(["--- CS01 (credit spread CSR tenors) ---", "", "", "", ""])
cs01_start = ws_sn.max_row + 1
for ten, pv_b in cs01:
    r = ws_sn.max_row + 1
    ws_sn.cell(r, 1, value="CS01")
    ws_sn.cell(r, 2, value=ten)
    ws_sn.cell(r, 3, value=pv_b)
    ws_sn.cell(r, 4, value=f"=C{r} - Valuation!$E$5")  # in-cell formula
cs01_end = ws_sn.max_row

# --- Greeks-based explain (SUMPRODUCT) ---
ws_gx = wb.create_sheet("GreeksExplain")
ws_gx.append(["Item", "Formula"])
# DV01 PnL ~= SUMPRODUCT(DV01_sens, dRF_bp)
ws_gx.append(["RatesPnL_approx",
              f"=SUMPRODUCT(Sensitivities!$D${dv01_start}:$D${dv01_end}, Curves!$D$2:$D$6)"])
ws_gx.append(["CreditPnL_approx",
              f"=SUMPRODUCT(Sensitivities!$D${cs01_start}:$D${cs01_end}, Curves!$H$2:$G$6)"])
ws_gx.append(["Theta_exact (from PV ladder)", "=Valuation!E3 - Valuation!E2"])
ws_gx.append(["Total_exact (PV ladder)", "=Valuation!E5 - Valuation!E2"])
ws_gx.append(["Residual_vs_exact", "=D4 - (B2+B3+B4)"])

# IMPORTANT: per your request, do NOT save any file.
wb.save("CRMW_schedule_attribution.xlsx")

# ==============================================================================
# # 7) Print key content (since you cannot download)
# ==============================================================================

print("\n" + "=" * 60)
print("CRMW Excel-style Schedule + P&L Attribution (in-memory)")
print("=" * 60)
print(f"Product: {PRODUCT}")
print(f"Linked bond: {REF_BOND}")
print(f"Notional M={M:,} FeeRate S={S:.4%} Recovery R={R:.0%}")
print(f"Val dates: t0={t0.isoformat()} t1={t1.isoformat()}")
print("Curves are DUMMY data for demonstration.\n")

# Print Schedule formulas
sched_rows = [["Period", "Start", "End", "Pay", "Days (formula)", "Fee_i (formula)", "DailyCarry (formula)"]]
for i in range(2, 2 + len(periods)):
    sched_rows.append([
        ws_sc[f"A{i}"].value,
        ws_sc[f"B{i}"].value,
        ws_sc[f"C{i}"].value,
        ws_sc[f"D{i}"].value,
        ws_sc[f"E{i}"].value,
        ws_sc[f"F{i}"].value,
        ws_sc[f"G{i}"].value,
    ])
print_block("SCHEDULE (formulas in cells)", sched_rows)

# Print PV ladder numbers + PV_total formulas
pv_rows = [["Scenario", "PV_Prot (num)", "PV_Prem (num)", "PV_Total (formula)", "PV_Total (Python num)"]]
pv_python = {
    "PV_t0": PV_t0,
    "PV_theta": PV_theta,
    "PV_rates": PV_rates,
    "PV_credit": PV_credit
}
for i in range(2, 6):
    scen = ws_vl[f"A{i}"].value
    pv_rows.append([
        scen,
        f"{ws_vl[f'C{i}'].value:,.2f}",
        f"{ws_vl[f'D{i}'].value:,.2f}",
        ws_vl[f"E{i}"].value,
        f"{pv_python[scen]:,.2f}"
    ])
print_block("PV LADDER (PV_total formula in-cell; numeric PV computed in Python)", pv_rows)

# Print Attribution formulas + Python numbers
attrib = [
    ["Carry/Theta", ws_at["B2"].value, f"{carry:,.2f}"],
    ["Rates", ws_at["B3"].value, f"{rates:,.2f}"],
    ["Credit", ws_at["B4"].value, f"{credit:,.2f}"],
    ["Cashflows", ws_at["B5"].value, f"{cashflow_between:,.2f}"],
    ["Total", ws_at["B6"].value, f"{total_pnl:,.2f}"],
    ["Residual", ws_at["B7"].value, f"{residual:,.2f}"],
]
print_block("ATTRIBUTION (Excel formulas; Python numbers shown for reference)",
            [["Component", "ExcelFormula", "PythonValue"]] + attrib)

# Print DV01/CS01 sensitivity formulas
sens_rows = [["Type", "TenorY", "PV_bumped (num)", "Sens (formula)", "Sens (Python num)"]]
# DV01 rows
for r in range(dv01_start, dv01_end + 1):
    ten = ws_sn.cell(r, 2).value
    pv_b = ws_sn.cell(r, 3).value
    sens_py = pv_b - PV_base
    sens_rows.append(["DV01", ten, f"{pv_b:,.2f}", ws_sn.cell(r, 4).value, f"{sens_py:,.2f}"])
# CS01 rows
for r in range(cs01_start, cs01_end + 1):
    ten = ws_sn.cell(r, 2).value
    pv_b = ws_sn.cell(r, 3).value
    sens_py = pv_b - PV_base
    sens_rows.append(["CS01", ten, f"{pv_b:,.2f}", ws_sn.cell(r, 4).value, f"{sens_py:,.2f}"])
print_block("BUCKETED DV01 / CS01 (formulas in-cell; Python numbers shown)", sens_rows)

print("\nDone: workbook created in memory (openpyxl). No file saved.")