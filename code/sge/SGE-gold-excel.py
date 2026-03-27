from openpyxl import Workbook

wb = Workbook()

# ----------- INPUTS -----------
ws_inputs = wb.active
ws_inputs.title = "INPUTS"

ws_inputs["A1"] = "Parameter"
ws_inputs["B1"] = "Value"
ws_inputs["C1"] = "Notes"

rows = [
    ("TradeDate", "", "e.g., 2025-12-27"),
    ("Unit_g_per_lot", 1000, "Au(T+D) = 1 kg/lot = 1000 g"),
    ("PD_settle_RMB_per_g", "", "Prior day settlement price (元/克)"),
    ("CD_settle_RMB_per_g", "", "Today settlement price (元/克)"),
    ("PD_long_lots", "", "Prior day long lots"),
    ("PD_short_lots", "", "Prior day short lots"),
    ("FeeRate", 0.0002, "Transaction fee rate (万分之二)"),
    ("DeferredRate_per_day", "", "Deferred compensation fee rate per calendar day"),
    ("DeferredDirection", "", "Enter: 多付空 or 空付多"),
    ("DayCount", 1, "Calendar days applied to deferred fee (1 normally; 3 over weekend)"),
    ("Statement_TotalPnL", "", "Optional: total P&L from broker statement"),
    ("", "", ""),
    ("DirectionSign", None, "Computed: 多付空=+1, 空付多=-1"),
]

for i, (p, v, n) in enumerate(rows, start=2):
    ws_inputs[f"A{i}"] = p
    ws_inputs[f"B{i}"] = v
    ws_inputs[f"C{i}"] = n

# DirectionSign in INPUTS!B14 (because the row index ends up at 14)
ws_inputs["B14"] = '=IF($B$10="多付空",1,IF($B$10="空付多",-1,0))'

# ----------- TRADES -----------
ws_trades = wb.create_sheet("TRADES")
trade_headers = [
    "TradeTime", "Side", "QtyLots", "TradePrice_RMB_per_g",
    "Unit_g", "CD_Settle", "TradeNotional_CNY", "Fee_CNY", "ExecPnL_vs_Settle"
]

for j, h in enumerate(trade_headers, start=1):
    ws_trades.cell(row=1, column=j, value=h)

# Rows 2..199 with formulas
for r in range(2, 200):
    ws_trades[f"E{r}"] = "=INPUTS!$B$3"
    ws_trades[f"F{r}"] = "=INPUTS!$B$5"
    ws_trades[f"G{r}"] = f"=$C{r}*$D{r}*$E{r}"
    ws_trades[f"H{r}"] = f"=$G{r}*INPUTS!$B$8"
    ws_trades[f"I{r}"] = (
        f'=IF(OR($B{r}="B",$B{r}="BUY",$B{r}="买"),'
        f'($F{r}-$D{r})*$C{r}*$E{r},'
        f'IF(OR($B{r}="S",$B{r}="SELL",$B{r}="卖"),'
        f'($D{r}-$F{r})*$C{r}*$E{r},0))'
    )

# Totals
ws_trades["G200"] = "Totals:"
ws_trades["H200"] = "=SUM($H$2:$H$199)"
ws_trades["I200"] = "=SUM($I$2:$I$199)"
ws_trades["G201"] = "BuyLots:"
ws_trades["H201"] = (
    '=SUMIF($B$2:$B$199,"B",$C$2:$C$199)'
    '+SUMIF($B$2:$B$199,"BUY",$C$2:$C$199)'
    '+SUMIF($B$2:$B$199,"买",$C$2:$C$199)'
)
ws_trades["G202"] = "SellLots:"
ws_trades["H202"] = (
    '=SUMIF($B$2:$B$199,"S",$C$2:$C$199)'
    '+SUMIF($B$2:$B$199,"SELL",$C$2:$C$199)'
    '+SUMIF($B$2:$B$199,"卖",$C$2:$C$199)'
)

# ----------- POSITIONS -----------
ws_pos = wb.create_sheet("POSITIONS")
ws_pos["A1"] = "Item"
ws_pos["B1"] = "Value"

pos_rows = [
    ("PD_NetLots", "=INPUTS!$B$6-INPUTS!$B$7"),
    ("BuyLots_Today",
     '=SUMIF(TRADES!$B$2:$B$199,"B",TRADES!$C$2:$C$199)'
     '+SUMIF(TRADES!$B$2:$B$199,"BUY",TRADES!$C$2:$C$199)'
     '+SUMIF(TRADES!$B$2:$B$199,"买",TRADES!$C$2:$C$199)'),
    ("SellLots_Today",
     '=SUMIF(TRADES!$B$2:$B$199,"S",TRADES!$C$2:$C$199)'
     '+SUMIF(TRADES!$B$2:$B$199,"SELL",TRADES!$C$2:$C$199)'
     '+SUMIF(TRADES!$B$2:$B$199,"卖",TRADES!$C$2:$C$199)'),
    ("Trade_NetLots", "=B3-B4"),
    ("EOD_NetLots", "=B2+B5"),
    ("", ""),
    ("Position_MTM_CNY", "=(INPUTS!$B$5-INPUTS!$B$4)*B2*INPUTS!$B$3"),
    ("Trade_ExecPnL_CNY", "=SUM(TRADES!$I$2:$I$199)"),
    ("Price_Total_CNY", "=B8+B9"),
    ("", ""),
    ("EOD_Notional_CNY", "=ABS(B6)*INPUTS!$B$5*INPUTS!$B$3"),
    ("PosSign", "=SIGN(B6)"),
    ("DirectionSign", "=INPUTS!$B$14"),
    ("Deferred_PnL_CNY", "=IF(B6=0,0,-B13*B14*B12*INPUTS!$B$9*INPUTS!$B$10)"),
    ("", ""),
    ("Fees_CNY", "=SUM(TRADES!$H$2:$H$199)"),
    ("Fees_PnL_CNY", "=-B17"),
]

for i, (item, formula) in enumerate(pos_rows, start=2):
    ws_pos[f"A{i}"] = item
    ws_pos[f"B{i}"] = formula

# ----------- PNL EXPLAIN -----------
ws_pnl = wb.create_sheet("PNL_EXPLAIN")
ws_pnl["A1"] = "Bucket"
ws_pnl["B1"] = "Amount_CNY"

pnl_rows = [
    ("Position_MTM", "=POSITIONS!B8"),
    ("Trade_execution_vs_settle", "=POSITIONS!B9"),
    ("Deferred_fee (延期补偿费)", "=POSITIONS!B15"),
    ("Transaction_fees", "=POSITIONS!B18"),
    ("Total_explained", "=SUM(B2:B5)"),
    ("Statement_total_P&L_(optional)", "=INPUTS!$B$12"),
    ("Residual_(Statement_-_Explained)", '=IF(B7="","",B7-B6)'),
]

for i, (item, formula) in enumerate(pnl_rows, start=2):
    ws_pnl[f"A{i}"] = item
    ws_pnl[f"B{i}"] = formula

wb.save("SGE_AuTD_PnL_Attribution_Template.xlsx")
print("Created: SGE_AuTD_PnL_Attribution_Template.xlsx")