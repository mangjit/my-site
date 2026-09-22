"""Declared execution costs. A missing declaration is a failed experiment, not a zero."""

from __future__ import annotations

from dataclasses import dataclass

from aegis.config import ResearchSettings


@dataclass(frozen=True)
class CostModel:
    slippage_price_per_side: float
    commission_price_round_turn: float
    financing_price_per_hour: float
    pip_size: float
    spread_measured_from_quotes: bool
    slippage_measured: bool
    financing_measured: bool
    commission_measured: bool
    financing_direction: str

    @property
    def complete_declaration(self) -> bool:
        return (
            self.slippage_price_per_side >= 0
            and self.commission_price_round_turn >= 0
            and self.financing_price_per_hour >= 0
        )


def cost_model_from_settings(settings: ResearchSettings, pip_size: float, bar_hours: float) -> CostModel:
    if not settings.financing_is_measured and settings.financing_pips_per_day < 0:
        raise ValueError("Undeclared or credit financing is not allowed. Declare an adverse cost or measure it.")
    return CostModel(
        slippage_price_per_side=settings.slippage_pips_per_side * pip_size,
        commission_price_round_turn=settings.commission_pips_per_side * 2.0 * pip_size,
        financing_price_per_hour=(settings.financing_pips_per_day * pip_size) / 24.0,
        pip_size=pip_size,
        spread_measured_from_quotes=True,
        slippage_measured=False,
        financing_measured=settings.financing_is_measured,
        commission_measured=False,
        financing_direction="adverse_only_assumption"
        if not settings.financing_is_measured
        else "measured",
    )


def financing_price(cost: CostModel, holding_bars: int, bar_hours: float) -> float:
    hours = max(holding_bars, 1) * bar_hours
    return cost.financing_price_per_hour * hours * 24.0 / 24.0
