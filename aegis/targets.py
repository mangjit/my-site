"""Barrier labels and executable fills.

The training label is a property of the midpoint path after T. It is not a
feature, and it is not a fill. A taken trade is scored on bid/ask fills with
declared slippage, adverse financing, and commission. If both barriers print
inside one bar, the path is ambiguous and is not called a success.
"""

from __future__ import annotations

from aegis.costs import CostModel, financing_price
from aegis.table import CandleTable
from aegis.timeutil import granularity_delta


def bar_hours(granularity: str) -> float:
    return granularity_delta(granularity).total_seconds() / 3600.0


def label_end_index(decision_index: int, horizon: int, n_bars: int) -> int | None:
    """Last bar used by a label that enters at decision_index + 1."""
    end = decision_index + horizon
    if decision_index < 0 or end >= n_bars:
        return None
    return end


def forward_mid_return(table: CandleTable, decision_index: int, horizon: int) -> float | None:
    end = label_end_index(decision_index, horizon, len(table))
    if end is None:
        return None
    start_mid = _mid(table.bid_c[decision_index], table.ask_c[decision_index])
    end_mid = _mid(table.bid_c[end], table.ask_c[end])
    if start_mid <= 0:
        return None
    return end_mid / start_mid - 1.0


def path_label(
    table: CandleTable,
    decision_index: int,
    atr: float,
    barrier_atr: float,
    stop_atr: float,
    horizon: int,
) -> dict | None:
    """Midpoint barrier label. Known only after the horizon, so it cannot be a feature."""
    if atr <= 0:
        return None
    entry_index = decision_index + 1
    end = label_end_index(decision_index, horizon, len(table))
    if end is None:
        return None
    entry = _mid(table.bid_o[entry_index], table.ask_o[entry_index])
    up = entry + barrier_atr * atr
    down = entry - stop_atr * atr
    status = "TIME"
    exit_index = end
    for j in range(entry_index, end + 1):
        high = _mid(table.bid_h[j], table.ask_h[j])
        low = _mid(table.bid_l[j], table.ask_l[j])
        hit_up = high >= up
        hit_down = low <= down
        if hit_up and hit_down:
            status = "AMBIGUOUS_PATH"
            exit_index = j
            break
        if hit_up:
            status = "LONG_SUCCESS"
            exit_index = j
            break
        if hit_down:
            status = "SHORT_SUCCESS"
            exit_index = j
            break
    return {
        "label": status if status != "TIME" else "NO_RESOLUTION",
        "label_end_index": end,
        "path_exit_index": exit_index,
        "forward_mid_return": forward_mid_return(table, decision_index, horizon),
        "label_definition": "midpoint_barrier_adverse_same_bar_ambiguous",
    }


def simulate_side(
    table: CandleTable,
    decision_index: int,
    side: str,
    atr: float,
    barrier_atr: float,
    stop_atr: float,
    horizon: int,
    cost: CostModel,
    slippage_multiplier: float = 1.0,
) -> dict | None:
    """Score an actual side. This is an execution measurement, not the training label."""
    if side not in ("LONG", "SHORT"):
        raise ValueError(side)
    if atr <= 0:
        return None
    entry_index = decision_index + 1
    end = label_end_index(decision_index, horizon, len(table))
    if end is None:
        return None
    slip = cost.slippage_price_per_side * slippage_multiplier
    if side == "LONG":
        entry = table.ask_o[entry_index] + slip
        target = entry + barrier_atr * atr
        stop = entry - stop_atr * atr
        status, exit_index = _first_long(table, entry_index, end, target, stop)
        if status == "TARGET":
            exit_fill = target
            exit_slip = 0.0
        elif status in ("STOP", "AMBIGUOUS_PATH"):
            exit_fill = stop - slip
            exit_slip = slip
        else:
            exit_fill = table.bid_c[exit_index] - slip
            exit_slip = slip
        executable = exit_fill - entry
        favorable = max(table.bid_h[j] - entry for j in range(entry_index, exit_index + 1))
        adverse = max(entry - table.bid_l[j] for j in range(entry_index, exit_index + 1))
    else:
        entry = table.bid_o[entry_index] - slip
        target = entry - barrier_atr * atr
        stop = entry + stop_atr * atr
        status, exit_index = _first_short(table, entry_index, end, target, stop)
        if status == "TARGET":
            exit_fill = target
            exit_slip = 0.0
        elif status in ("STOP", "AMBIGUOUS_PATH"):
            exit_fill = stop + slip
            exit_slip = slip
        else:
            exit_fill = table.ask_c[exit_index] + slip
            exit_slip = slip
        executable = entry - exit_fill
        favorable = max(entry - table.ask_l[j] for j in range(entry_index, exit_index + 1))
        adverse = max(table.ask_h[j] - entry for j in range(entry_index, exit_index + 1))

    holding_bars = exit_index - entry_index + 1
    slippage_price = slip + exit_slip
    financing = financing_price(cost, holding_bars, bar_hours(table.granularity))
    commission = cost.commission_price_round_turn
    # executable already deducts slippage through the fills. gross adds it back
    # so the ledger can show gross - slippage - financing - commission = net.
    gross = executable + slippage_price
    net = executable - financing - commission
    pip = cost.pip_size
    spread_entry = table.ask_o[entry_index] - table.bid_o[entry_index]
    spread_exit = table.ask_c[exit_index] - table.bid_c[exit_index]
    return {
        "side": side,
        "execution_label": _execution_label(side, status),
        "path_status": status,
        "ambiguous": status == "AMBIGUOUS_PATH",
        "decision_index": decision_index,
        "entry_index": entry_index,
        "exit_index": exit_index,
        "holding_bars": holding_bars,
        "entry_fill": entry,
        "exit_fill": exit_fill,
        "gross_executable_pips": gross / pip,
        "slippage_pips": slippage_price / pip,
        "financing_pips": financing / pip,
        "commission_pips": commission / pip,
        "net_pips": net / pip,
        "net_return": net / entry if entry else None,
        "spread_at_entry_pips": spread_entry / pip,
        "spread_at_exit_pips": spread_exit / pip,
        "estimated_spread_cost_pips": ((spread_entry + spread_exit) / 2.0) / pip,
        "identity_residual_price": net - (gross - slippage_price - financing - commission),
        "mfe_pips": favorable / pip,
        "mae_pips": adverse / pip,
        "path_assumption": "ohlc_same_bar_ambiguous_not_success",
        "pnl_assumption_if_ambiguous": "adverse_stop_fill",
    }


def _first_long(table: CandleTable, start: int, end: int, target: float, stop: float) -> tuple[str, int]:
    for j in range(start, end + 1):
        hit_target = table.bid_h[j] >= target
        hit_stop = table.bid_l[j] <= stop
        if hit_target and hit_stop:
            return "AMBIGUOUS_PATH", j
        if hit_target:
            return "TARGET", j
        if hit_stop:
            return "STOP", j
    return "TIME", end


def _first_short(table: CandleTable, start: int, end: int, target: float, stop: float) -> tuple[str, int]:
    for j in range(start, end + 1):
        hit_target = table.ask_l[j] <= target
        hit_stop = table.ask_h[j] >= stop
        if hit_target and hit_stop:
            return "AMBIGUOUS_PATH", j
        if hit_target:
            return "TARGET", j
        if hit_stop:
            return "STOP", j
    return "TIME", end


def _execution_label(side: str, status: str) -> str:
    if status == "AMBIGUOUS_PATH":
        return "AMBIGUOUS_PATH"
    if status == "TIME":
        return "NO_RESOLUTION"
    if status == "TARGET":
        return "LONG_SUCCESS" if side == "LONG" else "SHORT_SUCCESS"
    if status == "STOP":
        return "SHORT_SUCCESS" if side == "LONG" else "LONG_SUCCESS"
    raise ValueError((side, status))


def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0
