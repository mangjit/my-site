"""Render the research bulletin. Numbers come from the experiment record, not from prose."""

from __future__ import annotations

import html
import json
from pathlib import Path


def write_site(result: dict, index_path: Path, json_path: Path) -> None:
    public = {key: value for key, value in result.items() if key not in {"predictions", "trades"}}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    index_path.write_text(render(result))


def render(result: dict) -> str:
    verdict = result["verdict"]
    dataset = result["dataset"]
    expanding = result["oos"]["expanding"]
    rolling = result["oos"]["rolling"]
    audit = result["audit"]
    point = result.get("point_in_time") or {}
    synthetic = bool(dataset.get("synthetic") or dataset.get("not_market_data"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aegis research bulletin — not a trade recommendation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,550;0,6..72,650;1,6..72,450&display=swap" rel="stylesheet">
  <style>
    :root {{
      --paper: #f3efe6;
      --paper-2: #e7e0d4;
      --ink: #1c1915;
      --muted: #6d665c;
      --rule: #d4cbc0;
      --seal: #7c2f2a;
      --seal-ink: #f6f1e8;
      --good: #1e4d3a;
      --warn: #8a5a12;
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--paper); color: var(--ink); }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      font-size: 17px;
      line-height: 1.55;
    }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 28px 96px; }}
    .status {{
      display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 28px;
    }}
    .status span {{
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
      border: 1px solid var(--ink); padding: 5px 8px;
    }}
    .kicker {{
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 550; font-size: clamp(40px, 6vw, 68px);
      line-height: 0.96; letter-spacing: -0.03em; margin: 8px 0 12px;
    }}
    h2 {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 550; font-size: 32px; letter-spacing: -0.02em;
      margin: 56px 0 12px;
    }}
    h3 {{ font-size: 15px; letter-spacing: 0.08em; text-transform: uppercase; margin: 28px 0 8px; }}
    p {{ margin: 0 0 14px; }}
    .lede {{ font-size: 20px; max-width: 42rem; }}
    .banner {{
      background: #241c18; color: var(--seal-ink); padding: 18px 20px; margin: 22px 0;
    }}
    .banner strong {{ color: #f0c9a0; }}
    .verdict {{
      display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 28px;
      border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink);
      padding: 22px 0; margin: 12px 0 8px;
    }}
    .verdict b {{
      display: block; font-family: Newsreader, Georgia, serif;
      font-size: clamp(36px, 5vw, 54px); line-height: 0.95; font-weight: 550; letter-spacing: -0.03em;
    }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 14.5px; }}
    th, td {{ text-align: left; padding: 8px 8px 8px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }}
    th {{ font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 500; }}
    td.num, th.num {{ text-align: right; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; }}
    .ladder {{ display: grid; grid-template-columns: 180px 1fr; gap: 8px 18px; margin: 18px 0 8px; }}
    .ladder dt {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--seal); padding-top: 3px; }}
    .ladder dd {{ margin: 0 0 10px; }}
    .layers {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }}
    .layer {{ border: 1px solid var(--rule); padding: 12px 14px; background: rgba(255,255,255,0.35); }}
    .layer span {{ display: block; font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--muted); letter-spacing: 0.08em; }}
    .json {{
      background: #1c1915; color: #f3efe6; padding: 16px; overflow: auto; max-height: 460px;
      font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; line-height: 1.45;
    }}
    figure {{ margin: 18px 0; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    .pass {{ color: var(--good); }}
    .fail {{ color: var(--seal); font-weight: 600; }}
    ul {{ padding-left: 18px; }}
    li {{ margin: 4px 0; }}
    footer {{ margin-top: 64px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--rule); padding-top: 16px; }}
    a {{ color: var(--seal); }}
    @media (max-width: 720px) {{
      .verdict, .ladder, .layers {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 28px 16px 64px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <p class="kicker">Aegis · research only · Mangjit Singh</p>
    <h1>What was knowable then, and nothing after.</h1>
    <p class="lede">This bulletin does not recommend a trade. It records whether a hypothesis, frozen before the test years were scored, contained information that survived costs, purge rules, and a second training window.</p>
    <div class="status">
      <span>Research only</span>
      <span>Live execution disabled</span>
      <span>OANDA live forbidden</span>
      <span>Not investment advice</span>
    </div>
    {banner(synthetic, dataset)}
    <h2>Latest examination</h2>
    <p class="meta">{esc(result["experiment_id"])} · protocol {esc(result["protocol_version"])} · engine {esc(result["aegis_version"])}</p>
    <div class="verdict">
      <div>
        <p class="kicker">Market classification</p>
        <b>{esc(verdict["market_classification"].replace("_", " "))}</b>
        <p>A market call is refused when the dataset is synthetic, the audit fails, financing was not measured, or the point in time does not support one. Refusal is a result.</p>
      </div>
      <div>
        <p class="kicker">Laboratory process</p>
        <b>{esc(verdict["laboratory_process_verdict"].replace("_", " "))}</b>
        <p>This label describes the named dataset under the frozen rule. It is not permission to trade, and it is not a claim about a later year.</p>
      </div>
    </div>
    <p>{esc(reason_prose(verdict["reason_codes"]))}</p>
    <h3>Claim ladder</h3>
    <dl class="ladder">
      <dt>Fact</dt>
      <dd>The engine cannot place an order. The live OANDA host is rejected in code. In-sample performance was not computed. Holdout examined: {"yes" if result["holdout"]["examined"] else "no"}.</dd>
      <dt>Measurement</dt>
      <dd>Dataset {esc(dataset.get("dataset_id"))}, source {esc(dataset.get("source"))}, {esc(dataset.get("rows"))} bars. Bar opens run from {esc(dataset.get("start"))} to {esc(dataset.get("end"))}. Quality {esc(result["quality"]["verdict"])}. Synthetic: {"yes" if dataset.get("synthetic") else "no"}.</dd>
      <dt>Historical evidence</dt>
      <dd>Out-of-sample trades only, concatenated by procedure. Expanding trades: {esc(expanding.get("trades"))}. Rolling trades: {esc(rolling.get("trades"))}. These are counts, not a promise they will recur.</dd>
      <dt>Model estimate</dt>
      <dd>Logistic scores are one-versus-rest estimates normalized to sum to one. Nearest-neighbor shares are distance-weighted frequencies of past labels. Neither is a calibrated probability.</dd>
      <dt>Hypothesis</dt>
      <dd>{esc(result["hypothesis"]["hypothesis"])}</dd>
      <dt>Uncertainty</dt>
      <dd>{esc((point or {}).get("uncertainty") or "See limitations.")}</dd>
    </dl>
    <h2>Out-of-sample ledger</h2>
    <p>Decisions evaluated: {esc(result.get("decision_count"))}. A decision with no trade is the frozen rule refusing, not a missing calculation. Holdout bars withheld: {esc(result["holdout"]["bars_excluded"])}, years {esc(result["holdout"]["years"])}.</p>
    <p>The principal comparison is expanding versus rolling. Neither was chosen after seeing the other win. Flat days are inside the Sharpe-type ratio, which is descriptive and is not a significance test.</p>
    {estimate_table(result.get("model_estimates"))}
    {comparison_table(expanding, rolling)}
    <h3>Expanding window, by year</h3>
    {year_table(expanding)}
    <h3>Equity, expanding window</h3>
    {equity_figure(result.get("trades") or [], "expanding")}
    <h3>Equity, rolling window</h3>
    {equity_figure(result.get("trades") or [], "rolling")}
    <h3>Folds</h3>
    {fold_table(result["folds"])}
    <h2>Costs and stresses</h2>
    <p>Net pips already use the bid for a long exit and the ask for a short exit. Spread is not subtracted a second time. Extra slippage, adverse financing, and commission are explicit. Financing measured from a swap history: {"yes" if result["hypothesis"]["financing_is_measured"] else "no — the laboratory figure is an adverse declaration"}.</p>
    {robustness_block(result["robustness"])}
    <h2>Audit</h2>
    <p>Valid: <strong class="{ 'pass' if audit['valid'] else 'fail' }">{esc(audit["valid"])}</strong>. A failed audit invalidates the experiment. It is not explained away.</p>
    <ul>
      {''.join(finding_item(item) for item in audit["findings"])}
    </ul>
    <p class="meta">Feature prefix checks at bar indices {esc(audit.get("causality_indices_checked"))}.</p>
    <h2>Point in time</h2>
    <p>The object below is the last expanding-window decision in the research sample. If the dataset is synthetic, its research classification stays insufficient. Agreement between models, if any, is an estimate about the laboratory process.</p>
    <pre class="json">{esc(json.dumps(point, indent=2, sort_keys=True))}</pre>
    <h2>Protocol</h2>
    <p>The future does not get to train the past. At a simulated close T, features, scalers, neighbors, event records, and the decision rule may use only what a careful observer could have known. Neighbor outcomes must already have finished. Training labels that reach into the test year are embargoed. The final year in the file is not a tuning set.</p>
    <div class="layers">
      <div class="layer"><span>01</span>Market data. Bid and ask, UTC, immutable once hashed. Missing sides are not invented.</div>
      <div class="layer"><span>02</span>Features. Returns scaled by volatility, distances scaled by ATR, causal percentiles.</div>
      <div class="layer"><span>03</span>Market memory. Prior states only. Outcomes are labels, never inputs to similarity.</div>
      <div class="layer"><span>04</span>Event memory. Separate clock. Unverified times are unsafe and excluded.</div>
      <div class="layer"><span>05</span>Baselines. Logistic regression and nearest neighbors. A story is not a third model.</div>
      <div class="layer"><span>06</span>Walk-forward. Expanding and rolling. Principal curve is out-of-sample only.</div>
    </div>
    <h2>What this is not</h2>
    <ul>
      {''.join(f"<li>{esc(item)}</li>" for item in result["limitations"])}
      <li>Not an adviser, not a broker, not a portfolio manager, and not a live execution system.</li>
    </ul>
    <h2>Reproduce</h2>
    <p>From the repository root, with Python 3.11:</p>
    <pre class="json">python3 -m unittest discover -s tests
python3 -m aegis lab-run</pre>
    <p>An OANDA practice token, if you supply one in the environment, is used only for <span style="font-family: 'IBM Plex Mono', monospace">GET /v3/instruments/{{instrument}}/candles</span> on the practice host. The client has no order method. Do not commit the token.</p>
    <footer>
      Published as a research instrument on Mangjit Singh's site. Historical similarity is evidence, not destiny.
      A positive laboratory ledger would still not authorize a live order.
    </footer>
  </div>
</body>
</html>
"""


def reason_prose(codes: list[str]) -> str:
    text = {
        "NO_TRADES_UNDER_FROZEN_RULE": "The frozen agreement rule produced no trades.",
        "SYNTHETIC_OR_NON_MARKET_DATA": "The file is not market data, so the market classification stays insufficient.",
        "HOLDOUT_NOT_EXAMINED": "The locked year was not examined.",
        "LIVE_EXECUTION_NOT_AUTHORIZED": "Nothing in this record authorizes a live order.",
        "AUDIT_FAILED": "The audit failed. The numerical record is not evidence.",
        "TRADE_COUNT_BELOW_PRE_REGISTERED_MINIMUM": "The trade count is below the pre-registered minimum.",
        "OOS_RISK_ADJUSTED_RESULT_NOT_POSITIVE": "The out-of-sample risk-adjusted result was not positive.",
    }
    return " ".join(text.get(code, code.replace("_", " ").capitalize() + ".") for code in codes)


def banner(synthetic: bool, dataset: dict) -> str:
    if not synthetic:
        return (
            "<div class='banner'><strong>Market data is attached.</strong> "
            "That still does not make a walk-forward result a trade. Read the audit before the ledger.</div>"
        )
    return (
        "<div class='banner'><strong>Synthetic laboratory data.</strong> "
        f"{esc(dataset.get('notes') or 'This file is not a market.')} "
        "Do not read the ledger as evidence about EUR/USD, gold, or any other traded price.</div>"
    )


def estimate_table(estimates: dict | None) -> str:
    if not estimates:
        return ""
    rows = []
    for label, key in (
        ("Logistic, long", "logit_long"),
        ("Logistic, short", "logit_short"),
        ("Neighbors, long", "knn_long"),
        ("Neighbors, short", "knn_short"),
    ):
        span = estimates[key]
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td class='num'>{num(span['min'], 3)}</td>"
            f"<td class='num'>{num(span['mean'], 3)}</td>"
            f"<td class='num'>{num(span['max'], 3)}</td>"
            "</tr>"
        )
    return (
        "<h3>Model estimates, before the agreement rule</h3>"
        "<p class='meta'>Expanding-window decisions only. A high neighbor share that the logistic model does not confirm stays a no-trade.</p>"
        "<table><thead><tr><th>Estimate</th><th class='num'>Min</th><th class='num'>Mean</th><th class='num'>Max</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def comparison_table(expanding: dict, rolling: dict) -> str:
    rows = [
        ("Trades", "trades", 0),
        ("Net pips", "net_pips", 2),
        ("Expectancy, pips", "expectancy_pips", 3),
        ("Win rate", "win_rate", 3),
        ("Profit factor", "profit_factor", 3),
        ("Sharpe-type", "sharpe_type", 3),
        ("Sortino-type", "sortino_type", 3),
        ("Max drawdown, pips", "max_drawdown_pips", 2),
        ("Max drawdown, fraction", "max_drawdown_fraction", 3),
        ("Average holding, bars", "average_holding_bars", 2),
        ("Exposure", "exposure", 3),
        ("Cost drag, pips", "transaction_cost_pips", 2),
        ("Longest losing streak", "longest_losing_streak", 0),
    ]
    body = []
    for label, key, digits in rows:
        body.append(
            "<tr>"
            f"<td>{esc(label)}</td>"
            f"<td class='num'>{num(expanding.get(key), digits)}</td>"
            f"<td class='num'>{num(rolling.get(key), digits)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Measurement</th><th class='num'>Expanding</th>"
        "<th class='num'>Rolling</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def year_table(summary: dict) -> str:
    years = summary.get("by_year") or {}
    if not years:
        return "<p class='meta'>No out-of-sample trades to split by year.</p>"
    rows = []
    for year, bucket in years.items():
        rows.append(
            "<tr>"
            f"<td>{esc(year)}</td>"
            f"<td class='num'>{num(bucket.get('trades'), 0)}</td>"
            f"<td class='num'>{num(bucket.get('net_pips'), 2)}</td>"
            f"<td class='num'>{num(bucket.get('expectancy_pips'), 3)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Year</th><th class='num'>Trades</th>"
        "<th class='num'>Net pips</th><th class='num'>Expectancy</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def fold_table(folds: list[dict]) -> str:
    rows = []
    for fold in folds:
        train = fold["train_years"]
        span = f"{train[0]}–{train[-1]}" if train else "—"
        violations = fold["retrieval_violations"]["future_neighbor"] + fold["retrieval_violations"]["outcome_not_yet_known"]
        rows.append(
            "<tr>"
            f"<td>{esc(fold['procedure'])}</td>"
            f"<td>{esc(span)}</td>"
            f"<td class='num'>{esc(fold['test_year'])}</td>"
            f"<td class='num'>{esc(fold['train_rows'])}</td>"
            f"<td class='num'>{esc(fold['embargoed_rows'])}</td>"
            f"<td class='num'>{esc(fold['test_rows'])}</td>"
            f"<td class='num'>{esc(fold['trades'])}</td>"
            f"<td class='num'>{esc(violations)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Procedure</th><th>Train</th><th class='num'>Test</th>"
        "<th class='num'>Train rows</th><th class='num'>Embargoed</th><th class='num'>Test rows</th>"
        "<th class='num'>Trades</th><th class='num'>Retrieval violations</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def robustness_block(block: dict) -> str:
    return (
        "<table><tbody>"
        f"<tr><td>2× slippage, net pips</td><td class='num'>{num(block.get('slippage_2x_net_pips'), 2)}</td></tr>"
        f"<tr><td>2× slippage, expectancy</td><td class='num'>{num(block.get('slippage_2x_expectancy_pips'), 3)}</td></tr>"
        f"<tr><td>Net without best five trades</td><td class='num'>{num(block.get('net_without_best_five'), 2)}</td></tr>"
        f"<tr><td>Sign flips without best five</td><td class='num'>{esc(block.get('sign_flips_without_best_five'))}</td></tr>"
        f"<tr><td>Best year share of net</td><td class='num'>{num(block.get('best_year_share_of_net'), 3)}</td></tr>"
        "</tbody></table>"
        f"<p class='meta'>{esc(block.get('note'))}</p>"
    )


def finding_item(item: str) -> str:
    protocol = item in {"IN_SAMPLE_NOT_COMPUTED", "HOLDOUT_NOT_EXAMINED"}
    css = "meta" if protocol else "fail"
    return f"<li class='{css}'>{esc(item)}</li>"


def equity_figure(trades: list[dict], procedure: str) -> str:
    ordered = [trade for trade in trades if trade.get("procedure") == procedure]
    ordered.sort(key=lambda trade: trade["decision_time"])
    if not ordered:
        return "<p class='meta'>No trades. An empty ledger is preferable to a curve fitted to noise and then described as skill.</p>"
    points = [0.0]
    running = 0.0
    for trade in ordered:
        running += float(trade["net_pips"])
        points.append(running)
    width, height, pad = 860, 220, 18
    low = min(points)
    high = max(points)
    if abs(high - low) < 1e-9:
        high = low + 1.0
    span = high - low

    def xy(index: int, value: float) -> str:
        x = pad + (width - 2 * pad) * index / (len(points) - 1)
        y = pad + (height - 2 * pad) * (1 - (value - low) / span)
        return f"{x:.1f},{y:.1f}"

    polyline = " ".join(xy(i, value) for i, value in enumerate(points))
    zero_y = pad + (height - 2 * pad) * (1 - (0 - low) / span)
    return (
        "<figure>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='Cumulative out-of-sample net pips'>"
        f"<rect width='{width}' height='{height}' fill='#f7f4ee'></rect>"
        f"<line x1='{pad}' y1='{zero_y:.1f}' x2='{width - pad}' y2='{zero_y:.1f}' stroke='#b7ad9f' stroke-dasharray='3 4'></line>"
        f"<polyline fill='none' stroke='#1c1915' stroke-width='1.6' points='{polyline}'></polyline>"
        "</svg>"
        f"<figcaption>Cumulative net pips, {esc(procedure)} window, out-of-sample trades only, constant notional. "
        "Zero is a reference, not a target. This is not an account curve and not a market result when the dataset is synthetic.</figcaption>"
        "</figure>"
    )


def num(value, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return esc(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def esc(value) -> str:
    return html.escape("" if value is None else str(value))
