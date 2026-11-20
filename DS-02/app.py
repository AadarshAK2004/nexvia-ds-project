import json
from pathlib import Path
import pandas as pd, plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output

BASE = Path(__file__).parent
pred = pd.read_csv(BASE / "outputs/test_predictions.csv", parse_dates=["date"])
orders = pd.read_csv(BASE / "outputs/orders.csv")
m = json.load(open(BASE / "outputs/metrics.json"))

app = Dash(__name__)
server = app.server
app.layout = html.Div(style={"maxWidth": "1000px", "margin": "auto", "fontFamily": "sans-serif"}, children=[
    html.H2("Demand Forecasting & Inventory Optimization"),
    html.B(f"WAPE LightGBM: {m['lgbm_wape']:.3f} | WAPE Naive: {m['naive_wape']:.3f} | "
           f"RMSSE LightGBM: {m['lgbm_rmsse']:.3f} | 80% interval coverage: {m['interval_coverage_80']:.2f}"),
    html.Br(), html.Br(),
    html.Div([
        html.Label("Store: "),
        dcc.Dropdown(sorted(pred.store.unique()), 1, id="store", clearable=False, style={"width": "120px", "display": "inline-block"}),
        html.Label("  Item: "),
        dcc.Dropdown(sorted(pred.item.unique()), 1, id="item", clearable=False, style={"width": "120px", "display": "inline-block"}),
    ]),
    dcc.Graph(id="chart"),
    html.H3("Order recommendations (Store 1, next 7 days)"),
    dash_table.DataTable(orders.to_dict("records"), page_size=10),
])

@app.callback(Output("chart", "figure"), Input("store", "value"), Input("item", "value"))
def update(store, item):
    d = pred[(pred.store == store) & (pred.item == item)]
    f = go.Figure()
    f.add_scatter(x=d.date, y=d.p90, line=dict(width=0), showlegend=False)
    f.add_scatter(x=d.date, y=d.p10, fill="tonexty", line=dict(width=0), name="80% interval")
    f.add_scatter(x=d.date, y=d.sales, name="Actual")
    f.add_scatter(x=d.date, y=d.pred, name="Forecast")
    f.update_layout(title=f"Store {store}, Item {item}", xaxis_title="Date", yaxis_title="Sales")
    return f

if __name__ == "__main__":
    app.run(debug=False)
