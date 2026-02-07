from datetime import date
import math
from openpyxl import Workbook


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def d(iso: str) -> date:
    return date.fromisoformat(iso)


def yearfrac_act365(a: date, b: date) -> float:
    return (b - a).days / 365.0


def lin_interp(x: float, xs: list[float], ys: list[float]) -> float:
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


def df_issuer(asof: date, pay: date, tenors: list[float],
              swap_curve: list[float], issuer_spread_curve: list[float]) -> float:
    tau = yearfrac_act365(asof, pay)
    if tau <= 0.0:
        return 0.0
    r = lin_interp(tau, tenors, swap_curve)
    s = lin_interp(tau, tenors, issuer_spread_curve)
    return math.exp(-(r + s) * tau)  # continuous comp


# ------------------------------------------------------------------
# Trade + dummy market data
# ------------------------------------------------------------------
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
    "2025-11-12"
]))

starts = [last_coupon] + pay_dates[:-1]

# Cashflows
cfs = []
for sd, pd in zip(starts, pay_dates):
    yf = yearfrac_act365(sd, pd)
    coup_cf = N * coupon * yf
    prin_cf = N if pd == maturity else 0.0
    cfs.append(coup_cf + prin_cf)

# Curves (dummy)
tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

swap_t0 = [0.0830, 0.0840, 0.0850, 0.0860, 0.0870, 0.0880]
swap_t1 = [0.0836, 0.0846, 0.0856, 0.0866, 0.0876, 0.0886]  # +6bp

iss_t0 = [0.0140, 0.0145, 0.0150, 0.0155, 0.0160, 0.0165]
iss_t1 = [0.0142, 0.0147, 0.0152, 0.0157, 0.0162, 0.0167]  # +2bp

ref_t0 = [0.0200, 0.0205, 0.0210, 0.0220, 0.0230, 0.0240]
ref_t1 = [0.0188, 0.0193, 0.0198, 0.0208, 0.0218, 0.0228]  # -12bp


# ------------------------------------------------------------------
# Valuation functions
# ------------------------------------------------------------------
def pv_note(asof: date, swap_curve, iss_curve) -> float:
    pv = 0.0
    for pd, cf in zip(pay_dates, cfs):
        if pd <= asof:
            continue
        pv += cf * df_issuer(asof, pd, tenors, swap_curve, iss_curve)
    return pv


def pv_cds_buy_protection(asof: date, swap_curve, iss_curve, ref_curve, S_fixed: float) -> tuple[float, float, float]:
    """
    CDS-like proxy:
    - hazard per interval from interpolated ref spread z(tau)/(1-R)
    - survival piecewise constant per coupon interval
    - discount using issuer curve (since embedded in issuer note)
    Returns: (PV_cds_buy, ProtPV, RPV01)
    """
    Q_prev = 1.0
    prot = 0.0
    rpv01 = 0.0

    for sd, pd in zip(starts, pay_dates):
        if pd <= asof:
            continue
        start = max(sd, asof)
        delta = yearfrac_act365(start, pd)
        tau_end = yearfrac_act365(asof, pd)
        z = lin_interp(tau_end, tenors, ref_curve)
        h = z / (1.0 - R)

        Q_end = Q_prev * math.exp(-h * delta)

        DF = df_issuer(asof, pd, tenors, swap_curve, iss_curve)

        prot += (1.0 - R) * N * DF * (Q_prev - Q_end)
        rpv01 += N * DF * delta * Q_end

        Q_prev = Q_end

    pv_cds_buy = prot - S_fixed * rpv01
    return pv_cds_buy, prot, rpv01


def calibrate_par_spread_at_t0() -> float:
    pv0, prot0, rpv010 = pv_cds_buy_protection(t0, swap_t0, iss_t0, ref_t0, S_fixed=0.0)
    return prot0 / rpv010


S_fixed = calibrate_par_spread_at_t0()


def pv_total_investor(asof: date, swap_curve, iss_curve, ref_curve) -> tuple[float, float, float]:
    """Investor view: long note + short protection => PV_total = PV_note - PV_CDS_buy"""
    note = pv_note(asof, swap_curve, iss_curve)
    pv_cds_buy, _, _ = pv_cds_buy_protection(asof, swap_curve, iss_curve, ref_curve, S_fixed=S_fixed)
    credit = -pv_cds_buy
    return note + credit, note, credit


# ------------------------------------------------------------------
# Stepwise PV ladder (t0 -> t1)
# ------------------------------------------------------------------
PV_t0, note_t0, cred_t0 = pv_total_investor(t0, swap_t0, iss_t0, ref_t0)
PV_theta, note_theta, cred_theta = pv_total_investor(t1, swap_t0, iss_t0, ref_t0)
PV_rates, note_rates, cred_rates = pv_total_investor(t1, swap_t1, iss_t0, ref_t0)
PV_issuer, note_issuer, cred_issuer = pv_total_investor(t1, swap_t1, iss_t1, ref_t0)
PV_ref, note_ref, cred_ref = pv_total_investor(t1, swap_t1, iss_t1, ref_t1)

carry = PV_theta - PV_t0
rates = PV_rates - PV_theta
issuer = PV_issuer - PV_rates
ref = PV_ref - PV_issuer
total = PV_ref - PV_t0
resid = total - (carry + rates + issuer + ref)

# ------------------------------------------------------------------
# Build an in-memory workbook (no saving)
# ------------------------------------------------------------------
wb = Workbook()

# Inputs sheet
ws = wb.active
ws.title = "Inputs"
ws.append(["Field", "Value"])
ws.append(["Notional N", N])
ws.append(["Coupon", coupon])
ws.append(["Recovery R", R])
ws.append(["t0", t0.isoformat()])
ws.append(["t1", t1.isoformat()])
ws.append(["Last coupon date", last_coupon.isoformat()])
ws.append(["Maturity", maturity.isoformat()])
ws.append(["Calibrated S_fixed (par @ t0)", S_fixed])

# Curves sheet
ws = wb.create_sheet("Curves")
ws.append(["Tenor", "Swap_t0", "Swap_t1", "IssuerSpr_t0", "IssuerSpr_t1", "RefSpr_t0", "RefSpr_t1"])
for i in range(len(tenors)):
    ws.append([tenors[i], swap_t0[i], swap_t1[i], iss_t0[i], iss_t1[i], ref_t0[i], ref_t1[i]])

# Schedule sheet
ws = wb.create_sheet("Schedule")
ws.append(["StartDate", "PayDate", "YearFrac", "Cashflow"])
for sd, pd, cf in zip(starts, pay_dates, cfs):
    ws.append([sd.isoformat(), pd.isoformat(), yearfrac_act365(sd, pd), cf])

# Valuation sheet (PV ladder)
ws = wb.create_sheet("Valuation")
ws.append(["Scenario", "PV_total", "PV_note", "PV_credit"])
ws.append(["PV_t0 (t0, all t0 curves)", PV_t0, note_t0, cred_t0])
ws.append(["PV_theta (t1, curves frozen @ t0)", PV_theta, note_theta, cred_theta])
ws.append(["PV_rates (t1, swap->t1)", PV_rates, note_rates, cred_rates])
ws.append(["PV_issuer (t1, issuer->t1)", PV_issuer, note_issuer, cred_issuer])
ws.append(["PV_ref (t1, ref->t1)", PV_ref, note_ref, cred_ref])

# Attribution sheet
ws = wb.create_sheet("Attribution")
ws.append(["Component", "Formula", "P&L"])
ws.append(["Carry/Theta", "PV_theta - PV_t0", carry])
ws.append(["Rates", "PV_rates - PV_theta", rates])
ws.append(["Issuer spread", "PV_issuer - PV_rates", issuer])
ws.append(["Reference spread", "PV_ref - PV_issuer", ref])
ws.append(["Total", "PV_ref - PV_t0", total])
ws.append(["Residual", "Total - sum(components)", resid])

# ------------------------------------------------------------------
# Print key outputs (since you cannot download files)
# ------------------------------------------------------------------
print("\n=== Calibrated S_fixed (par spread @ t0) ===")
print(f"S_fixed = {S_fixed * 100:.6f}%  ({S_fixed * 10000:.2f} bp)")

print("\n=== PV ladder (Investor view, dirty PV) ===")
for name, pv in [
    ("PV_t0", PV_t0),
    ("PV_theta", PV_theta),
    ("PV_rates", PV_rates),
    ("PV_issuer", PV_issuer),
    ("PV_ref", PV_ref),
]:
    print(f"{name:10s}: {pv:,.2f}")

print("\n=== Daily P&L Attribution (t0 -> t1) ===")
print(f"Carry/Theta     : {carry:,.2f}")
print(f"Rates           : {rates:,.2f}")
print(f"Issuer spread   : {issuer:,.2f}")
print(f"Reference spread: {ref:,.2f}")
print(f"Total           : {total:,.2f}")
print(f"Residual        : {resid:,.2f}")

# Optional: if you ever want to save locally, uncomment:
wb.save('ZAR_CLN_PnL_Attribution_20231002.xlsx')