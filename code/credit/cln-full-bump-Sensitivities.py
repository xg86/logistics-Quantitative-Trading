from __future__ import annotations
from datetime import date
import math
from typing import List, Tuple, Dict, Any
from openpyxl import Workbook


# ==============================================================================
# 0) Utilities
# ==============================================================================

def d(iso: str) -> date:
    return date.fromisoformat(iso)


def act365(a: date, b: date) -> float:
    return (b - a).days / 365.0


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


def fmt_bp_amount(x: float) -> str:
    # x is a PV change amount (not a rate); print as amount
    return f"{x:,.2f}"


def fmt_rate_bp(x: float) -> str:
    return f"{x * 10000:.2f} bp"


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
    """
    Bump a single curve node at bump_tenor by +bump_bp (bp), leaving others unchanged.
    bump_bp=1 => +0.0001 in rate units.
    """
    bump = bump_bp / 10000.0
    out = ys[:]
    if bump_tenor not in xs:
        raise ValueError(f"Tenor {bump_tenor} not found in curve nodes {xs}")
    idx = xs.index(bump_tenor)
    out[idx] = out[idx] + bump
    return out


# ==============================================================================
# 1) Trade terms + schedule (dummy, consistent)
# ==============================================================================

N = 102_000_000
coupon = 0.1062
R = 0.40

t0 = d("2023-09-29")
t1 = d("2023-10-02")

last_coupon = d("2023-08-14")
maturity = d("2025-11-12")

pay_dates = list(map(d, [
    "2023-11-13",
    "2024-02-12",
    "2024-05-13",
    "2024-08-12",
    "2024-11-12",
    "2025-02-12",
    "2025-05-12",
    "2025-08-12",
    "2025-11-12",
]))

start_dates = [last_coupon] + pay_dates[:-1]

# cashflows per pay date (dirty CF schedule)
schedule = []
for sd, pd in zip(start_dates, pay_dates):
    yf = act365(sd, pd)  # ACT/365F
    coup_cf = N * coupon * yf
    prin_cf = N if pd == maturity else 0.0
    schedule.append({
        "StartDate": sd,
        "PayDate": pd,
        "YearFrac": yf,
        "TotalCF": coup_cf + prin_cf
    })

# ==============================================================================
# 2) Dummy curves (two tenor grids)
# - Key-rate tenors for swap (DV01)
# - CSR tenors for issuer/ref spreads (CS01)
# ==============================================================================

# Key-rate tenors (swap curve nodes) for DV01
kr_tenors = [0.25, 0.50, 1.00, 2.00, 3.00, 5.00, 10.00]

swap_t0 = [0.0830, 0.0840, 0.0850, 0.0860, 0.0870, 0.0880, 0.0890]
swap_t1 = [0.0836, 0.0846, 0.0856, 0.0866, 0.0876, 0.0886, 0.0896]  # +6bp

# CSR tenors for issuer/ref credit risk (CS01)
csr_tenors = [0.50, 1.00, 3.00, 5.00, 10.00]

# Issuer spread curve (Nedbank senior) over swap (dummy)
iss_t0 = [0.0145, 0.0150, 0.0160, 0.0165, 0.0170]
iss_t1 = [0.0147, 0.0152, 0.0162, 0.0167, 0.0172]  # +2bp

# Reference spread curve (proxy z-spread / hazard driver) (dummy)
ref_t0 = [0.0205, 0.0210, 0.0230, 0.0240, 0.0250]
ref_t1 = [0.0193, 0.0198, 0.0218, 0.0228, 0.0238]  # -12bp


# ==============================================================================
# 3) Valuation engine (same as earlier, but curves have separate tenor grids)
# ==============================================================================

def df_issuer(asof: date, pay: date,
              swap_curve: List[float],
              issuer_spread_curve: List[float]) -> float:
    """
    DF = exp(-(r_swap(tau) + s_issuer(tau)) * tau)
    where:
        tau = (pay-asof)/365
        r_swap interpolated on kr_tenors
        s_issuer interpolated on csr_tenors
    """
    tau = act365(asof, pay)
    if tau <= 0:
        return 0.0
    r = interp_linear(tau, kr_tenors, swap_curve)
    sI = interp_linear(tau, csr_tenors, issuer_spread_curve)
    return math.exp(-(r + sI) * tau)


def pv_note(asof: date,
            swap_curve: List[float],
            issuer_spread_curve: List[float]) -> float:
    """PV_note = sum CF_i * DF_issuer(asof, pay_i)"""
    pv = 0.0
    for row in schedule:
        pd = row["PayDate"]
        if pd <= asof:
            continue
        DF = df_issuer(asof, pd, swap_curve, issuer_spread_curve)
        pv += row["TotalCF"] * DF
    return pv


def cds_proxy_buy(asof: date,
                  swap_curve: List[float],
                  issuer_spread_curve: List[float],
                  ref_spread_curve: List[float],
                  S_fixed: float) -> Tuple[float, float, float]:
    """
    CDS-like proxy (buyer of protection):
        z(tau) interpolated on csr_tenors
        hazard h = z/(1-R) per period (piecewise)
        survival Q_end = Q_prev * exp(-h*delta)
        ProtPV = (1-R)*N*sum DF*(Q_prev - Q_end)
        RPV01 = N*sum DF*delta*Q_end
        PV_CDS_buy = ProtPV - S_fixed*RPV01
    Discounting uses issuer DF (swap+issuer spread) for consistency with CLN decomposition.
    """
    Q_prev = 1.0
    protPV = 0.0
    rpv01 = 0.0

    for sd, pd in zip(start_dates, pay_dates):
        if pd <= asof:
            continue

        start_eff = max(sd, asof)
        delta = act365(start_eff, pd)
        tau = act365(asof, pd)

        z = interp_linear(tau, csr_tenors, ref_spread_curve)
        h = z / (1.0 - R)

        Q_end = Q_prev * math.exp(-h * delta)
        DF = df_issuer(asof, pd, swap_curve, issuer_spread_curve)

        protPV += (1.0 - R) * N * DF * (Q_prev - Q_end)
        rpv01 += N * DF * delta * Q_end

        Q_prev = Q_end

    pv_cds_buy = protPV - S_fixed * rpv01
    return pv_cds_buy, protPV, rpv01


def calibrate_S_fixed_par_at_t0() -> float:
    """
    Par spread at t0:
        S = ProtPV(t0)/RPV01(t0)
    """
    pv0, prot0, rpv010 = cds_proxy_buy(t0, swap_t0, iss_t0, ref_t0, S_fixed=0.0)
    return prot0 / rpv010


S_fixed = calibrate_S_fixed_par_at_t0()


def pv_total_investor(asof: date,
                      swap_curve: List[float],
                      issuer_spread_curve: List[float],
                      ref_spread_curve: List[float]) -> Tuple[float, float, float]:
    """
    Investor CLN PV decomposition:
        PV_total = PV_note + PV_credit_investor
        PV_credit_investor = - PV_CDS_buy (investor sells protection)
    """
    note = pv_note(asof, swap_curve, issuer_spread_curve)
    pv_cds_buy, _, _ = cds_proxy_buy(asof, swap_curve, issuer_spread_curve, ref_spread_curve, S_fixed=S_fixed)
    credit = -pv_cds_buy
    total = note + credit
    return total, note, credit


# ==============================================================================
# 4) PV ladder + P&L attribution (no default, t0->t1)
# ==============================================================================

PV_t0, note_t0, cred_t0 = pv_total_investor(t0, swap_t0, iss_t0, ref_t0)
PV_theta, note_theta, cred_theta = pv_total_investor(t1, swap_t0, iss_t0, ref_t0)
PV_rates, note_rates, cred_rates = pv_total_investor(t1, swap_t1, iss_t0, ref_t0)
PV_issuer, note_issuer, cred_issuer = pv_total_investor(t1, swap_t1, iss_t1, ref_t0)
PV_ref, note_ref, cred_ref = pv_total_investor(t1, swap_t1, iss_t1, ref_t1)

carry = PV_theta - PV_t0
rates = PV_rates - PV_theta
issuer = PV_issuer - PV_rates
ref = PV_ref - PV_issuer
total_pnl = PV_ref - PV_t0
residual = total_pnl - (carry + rates + issuer + ref)

# Base PV for risk sensitivities: use PV_ref (asof t1, all curves t1)
PV_base_total = PV_ref
PV_base_note = note_ref
PV_base_credit = cred_ref

# ==============================================================================
# 5) Bucketed DV01 (key-rate tenors) and CS01 (CSR tenors)
# Definition: Sens(bucket) = PV(bumped by +1bp at that bucket) - PV(base)
# ==============================================================================

BUMP_BP = 1.0

dv01_rows = []
for k in kr_tenors:
    swap_b = bump_node(kr_tenors, swap_t1, k, bump_bp=BUMP_BP)
    pv_b_total, pv_b_note, pv_b_credit = pv_total_investor(t1, swap_b, iss_t1, ref_t1)
    dv01_rows.append({
        "Type": "DV01_SWAP",
        "Tenor": k,
        "PV_bumped_total": pv_b_total,
        "PV_bumped_note": pv_b_note,
        "PV_bumped_credit": pv_b_credit,
        "Sens_total": pv_b_total - PV_base_total,
        "Sens_note": pv_b_note - PV_base_note,
        "Sens_credit": pv_b_credit - PV_base_credit,
    })

cs01_iss_rows = []
for k in csr_tenors:
    iss_b = bump_node(csr_tenors, iss_t1, k, bump_bp=BUMP_BP)
    pv_b_total, pv_b_note, pv_b_credit = pv_total_investor(t1, swap_t1, iss_b, ref_t1)
    cs01_iss_rows.append({
        "Type": "CS01_ISSUER",
        "Tenor": k,
        "PV_bumped_total": pv_b_total,
        "PV_bumped_note": pv_b_note,
        "PV_bumped_credit": pv_b_credit,
        "Sens_total": pv_b_total - PV_base_total,
        "Sens_note": pv_b_note - PV_base_note,
        "Sens_credit": pv_b_credit - PV_base_credit,
    })

cs01_ref_rows = []
for k in csr_tenors:
    ref_b = bump_node(csr_tenors, ref_t1, k, bump_bp=BUMP_BP)
    pv_b_total, pv_b_note, pv_b_credit = pv_total_investor(t1, swap_t1, iss_t1, ref_b)
    cs01_ref_rows.append({
        "Type": "CS01_REF",
        "Tenor": k,
        "PV_bumped_total": pv_b_total,
        "PV_bumped_note": pv_b_note,
        "PV_bumped_credit": pv_b_credit,
        "Sens_total": pv_b_total - PV_base_total,
        "Sens_note": pv_b_note - PV_base_note,
        "Sens_credit": pv_b_credit - PV_base_credit,
    })

# ==============================================================================
# 6) Build an in-memory workbook with formulas IN cells (no saving)
# ==============================================================================

wb = Workbook()

# --- BasePV sheet ---
ws = wb.active
ws.title = "BasePV"
ws["A1"] = "Field"
ws["B1"] = "Value (numbers)"
ws["A2"] = "PV_base_note"
ws["B2"] = PV_base_note
ws["A3"] = "PV_base_credit"
ws["B3"] = PV_base_credit
ws["A4"] = "PV_base_total (=B2+B3)"
ws["B4"] = "=B2+B3"  # Excel function (not a separate formula column)

# --- PV Ladder sheet with formulas ---
ws = wb.create_sheet("PV_Ladder")
ws.append(["Scenario", "PV_note", "PV_credit", "PV_total (formula)"])
ws.append(["PV_t0", note_t0, cred_t0, "=B2+C2"])
ws.append(["PV_theta", note_theta, cred_theta, "=B3+C3"])
ws.append(["PV_rates", note_rates, cred_rates, "=B4+C4"])
ws.append(["PV_issuer", note_issuer, cred_issuer, "=B5+C5"])
ws.append(["PV_ref", note_ref, cred_ref, "=B6+C6"])

# --- Attribution sheet with formulas referencing PV totals ---
ws = wb.create_sheet("Attribution")
ws["A1"] = "Component"
ws["B1"] = "Value (formula)"
# cell references to PV_Ladder PV_total column (D)
ws["A2"] = "Carry/Theta"
ws["B2"] = "=PV_Ladder!D3 - PV_Ladder!D2"
ws["A3"] = "Rates"
ws["B3"] = "=PV_Ladder!D4 - PV_Ladder!D3"
ws["A4"] = "IssuerSpread"
ws["B4"] = "=PV_Ladder!D5 - PV_Ladder!D4"
ws["A5"] = "RefSpread"
ws["B5"] = "=PV_Ladder!D6 - PV_Ladder!D5"
ws["A6"] = "Total"
ws["B6"] = "=PV_Ladder!D6 - PV_Ladder!D2"
ws["A7"] = "Residual"
ws["B7"] = "=B6 - (B2+B3+B4+B5)"

# --- Sensitivities sheet (DV01 + CS01) ---
ws = wb.create_sheet("Sensitivities")
ws.append([
    "Type", "Tenor",
    "PV_bumped_total", "Sens_total (=PVb - BasePV!B4)",
    "PV_bumped_note", "Sens_note  (=PVb - BasePV!B2)",
    "PV_bumped_credit", "Sens_credit(=PVb - BasePV!B3)"
])


def add_sens_rows(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        row_idx = ws.max_row + 1
        ws.cell(row=row_idx, column=1, value=r["Type"])
        ws.cell(row=row_idx, column=2, value=r["Tenor"])

        # PV bumped values as numbers
        ws.cell(row=row_idx, column=3, value=r["PV_bumped_total"])
        ws.cell(row=row_idx, column=5, value=r["PV_bumped_note"])
        ws.cell(row=row_idx, column=7, value=r["PV_bumped_credit"])

        # Sensitivities as Excel formulas in the sensitivity cells
        # Sens_total in col 4: =C{row} - BasePV!$B$4
        ws.cell(row=row_idx, column=4, value=f"=C{row_idx} - BasePV!$B$4")
        # Sens_note in col 6: =E{row} - BasePV!$B$2
        ws.cell(row=row_idx, column=6, value=f"=E{row_idx} - BasePV!$B$2")
        # Sens_credit in col 8: =G{row} - BasePV!$B$3
        ws.cell(row=row_idx, column=8, value=f"=G{row_idx} - BasePV!$B$3")


# Add sections
ws.append(["", "", "", "", "", "", "", ""])
ws.append(["--- DV01 buckets (swap key-rate) ---", "", "", "", "", "", "", ""])
add_sens_rows(dv01_rows)

ws.append(["", "", "", "", "", "", "", ""])
ws.append(["--- CS01 buckets (issuer spread, CSR tenors) ---", "", "", "", "", "", "", ""])
add_sens_rows(cs01_iss_rows)

ws.append(["", "", "", "", "", "", "", ""])
ws.append(["--- CS01 buckets (reference spread, CSR tenors) ---", "", "", "", "", "", "", ""])
add_sens_rows(cs01_ref_rows)

# Do NOT save any file (per your request)
wb.save("CLN_with_Sensitivities.xlsx")