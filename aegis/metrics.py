"""Performance measurements. These describe a sample. They do not prove an edge."""

from __future__ import annotations

import math


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = mean(values)
    assert center is not None
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def maximum_drawdown(levels: list[float]) -> dict:
    if not levels:
        return {"max_drawdown_fraction": None, "max_drawdown_pips": None}
    peak = levels[0]
    worst_frac = 0.0
    worst_abs = 0.0
    peak_pips = levels[0]
    for level in levels:
        if level > peak:
            peak = level
        if peak > 0:
            worst_frac = max(worst_frac, (peak - level) / peak)
        if level > peak_pips:
            peak_pips = level
        worst_abs = max(worst_abs, peak_pips - level)
    return {
        "max_drawdown_fraction": worst_frac,
        "max_drawdown_units": worst_abs,
    }


def longest_losing_streak(net_pips: list[float]) -> int:
    streak = 0
    worst = 0
    for value in net_pips:
        if value < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def summarize_trades(trades: list[dict], daily_returns: list[float], exposure: float) -> dict:
    nets = [trade["net_pips"] for trade in trades]
    returns = [trade["net_return"] for trade in trades if trade.get("net_return") is not None]
    winners = [value for value in nets if value > 0]
    losers = [value for value in nets if value < 0]
    gross_pos = sum(winners)
    gross_neg = sum(losers)
    wealth = [1.0]
    level = 1.0
    for value in daily_returns:
        level += value
        wealth.append(level)
    pip_curve = [0.0]
    running = 0.0
    for value in nets:
        running += value
        pip_curve.append(running)
    downside = [min(value, 0.0) for value in daily_returns]
    downside_dev = math.sqrt(sum(value * value for value in downside) / len(downside)) if downside else None
    daily_mean = mean(daily_returns)
    daily_std = sample_std(daily_returns)
    sharpe = None
    if daily_mean is not None and daily_std not in (None, 0):
        sharpe = (daily_mean / daily_std) * math.sqrt(252)
    sortino = None
    if daily_mean is not None and downside_dev not in (None, 0):
        sortino = (daily_mean / downside_dev) * math.sqrt(252)
    avg_win = mean(winners)
    avg_loss = mean(losers)
    by_year: dict[str, list[float]] = {}
    by_regime: dict[str, list[float]] = {}
    by_instrument: dict[str, list[float]] = {}
    for trade in trades:
        by_year.setdefault(str(trade["year"]), []).append(trade["net_pips"])
        by_regime.setdefault(str(trade["regime"]), []).append(trade["net_pips"])
        by_instrument.setdefault(str(trade["instrument"]), []).append(trade["net_pips"])
    total_net = sum(nets) if nets else 0.0
    year_shares = {
        year: (sum(values) / total_net if total_net else None) for year, values in by_year.items()
    }
    best_five = sorted(nets, reverse=True)[:5]
    without_best_five = nets[:]
    for value in best_five:
        if value in without_best_five:
            without_best_five.remove(value)
    return {
        "trades": len(trades),
        "net_pips": total_net,
        "expectancy_pips": mean(nets),
        "mean_net_return": mean(returns),
        "annualized_return_estimate": (mean(daily_returns) or 0.0) * 252 if daily_returns else None,
        "sharpe_type": sharpe,
        "sharpe_definition": "daily_simple_returns_including_flat_days_sqrt_252_not_a_significance_test",
        "sortino_type": sortino,
        "max_drawdown_fraction": maximum_drawdown(wealth)["max_drawdown_fraction"],
        "max_drawdown_pips": maximum_drawdown(pip_curve)["max_drawdown_units"],
        "profit_factor": (gross_pos / abs(gross_neg)) if gross_neg < 0 else None,
        "win_rate": (len(winners) / len(trades)) if trades else None,
        "average_winner_pips": avg_win,
        "average_loser_pips": avg_loss,
        "win_loss_ratio": (abs(avg_win / avg_loss) if avg_win is not None and avg_loss not in (None, 0) else None),
        "average_holding_bars": mean([trade["holding_bars"] for trade in trades]),
        "exposure": exposure,
        "turnover_trades": len(trades),
        "transaction_cost_pips": sum(
            trade["slippage_pips"] + trade["financing_pips"] + trade["commission_pips"] for trade in trades
        ),
        "longest_losing_streak": longest_losing_streak(nets),
        "by_year": {year: _bucket(values) for year, values in sorted(by_year.items())},
        "by_regime": {name: _bucket(values) for name, values in sorted(by_regime.items())},
        "by_instrument": {name: _bucket(values) for name, values in sorted(by_instrument.items())},
        "extreme_volatility": _bucket(
            [
                trade["net_pips"]
                for trade in trades
                if trade["regime"] in {"HIGH_VOLATILITY", "VOLATILITY_EXPANSION", "CRISIS"}
            ]
        ),
        "other_regimes": _bucket(
            [
                trade["net_pips"]
                for trade in trades
                if trade["regime"] not in {"HIGH_VOLATILITY", "VOLATILITY_EXPANSION", "CRISIS"}
            ]
        ),
        "best_year_share_of_net": None
        if total_net <= 0 or not year_shares
        else max((share for share in year_shares.values() if share is not None), default=None),
        "net_without_best_five_trades": sum(without_best_five),
        "best_five_pips": best_five,
        "wealth_index": "constant_notional_additive",
        "pnl_timing": "intraday_mark_to_exit_side_quote_costs_booked_at_exit",
    }


def _bucket(values: list[float]) -> dict:
    return {
        "trades": len(values),
        "net_pips": sum(values) if values else 0.0,
        "expectancy_pips": mean(values),
    }


def daily_returns_from_trades(trades: list[dict], oos_days: list[str], mark_paths: list[list[tuple[str, float]]]) -> list[float]:
    """Map exit-relative mark-to-market increments onto the OOS weekday calendar.

    `mark_paths` entries are (date, incremental_return) already computed by the caller.
    Days with no position contribute zero. That is intentional: idle capital is part of the ratio.
    """
    bucket = {day: 0.0 for day in oos_days}
    for path in mark_paths:
        for day, increment in path:
            if day in bucket:
                bucket[day] += increment
    return [bucket[day] for day in oos_days]
