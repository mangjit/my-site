"""Frozen laboratory hypothesis and research conventions.

These values are the pre-registration for the bundled laboratory experiment.
Changing them creates a different experiment. It does not revise an old one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


MAJORS = (
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "AUD_USD",
    "NZD_USD",
    "USD_CAD",
)
CROSSES = (
    "EUR_JPY",
    "EUR_GBP",
    "GBP_JPY",
    "AUD_JPY",
    "CAD_JPY",
)
METALS = (
    "XAU_USD",
    "XAG_USD",
)
INSTRUMENT_UNIVERSE = MAJORS + CROSSES + METALS

GRANULARITIES = ("M5", "M15", "M30", "H1", "H4", "D")

# Research convention for reporting. Not a claim about any broker's pip definition.
# Metals increments in particular vary by venue. Results are always stored in price
# units as well as in this increment.
PRICE_INCREMENT = {
    "EUR_USD": 1e-4,
    "GBP_USD": 1e-4,
    "USD_JPY": 1e-2,
    "USD_CHF": 1e-4,
    "AUD_USD": 1e-4,
    "NZD_USD": 1e-4,
    "USD_CAD": 1e-4,
    "EUR_JPY": 1e-2,
    "EUR_GBP": 1e-4,
    "GBP_JPY": 1e-2,
    "AUD_JPY": 1e-2,
    "CAD_JPY": 1e-2,
    "XAU_USD": 1e-2,
    "XAG_USD": 1e-4,
}

# Model features. All are dimensionless or cyclic. Raw price and raw ATR are excluded
# so that a state can be compared across periods without using the nominal level.
FEATURE_NAMES = (
    "ret_1_z",
    "ret_3_z",
    "ret_6_z",
    "ret_12_z",
    "ret_24_z",
    "rv_ratio",
    "vol_pct",
    "ma_dist_20_atr",
    "ma_dist_50_atr",
    "ma_slope_20_atr",
    "persist_12",
    "range_pos_24",
    "dist_high_24_atr",
    "compression_24",
    "spread_over_atr",
    "spread_pct",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "overlap_lon_ny",
    "rollover_bar",
)

LABELS = ("LONG_SUCCESS", "SHORT_SUCCESS", "NO_RESOLUTION", "AMBIGUOUS_PATH")


@dataclass(frozen=True)
class ResearchSettings:
    """Parameters that must be frozen before an out-of-sample year is scored."""

    hypothesis_id: str
    hypothesis: str
    instrument: str
    granularity: str
    horizon_bars: int
    barrier_atr: float
    stop_atr: float
    min_train_years: int
    holdout_years: int
    stride_bars: int
    knn_k: int
    min_neighbors: int
    retrieval_principal: str
    logit_l2: float
    logit_steps: int
    logit_lr: float
    min_probability: float
    margin_over_train_base: float
    min_margin_vs_opposite: float
    slippage_pips_per_side: float
    commission_pips_per_side: float
    financing_pips_per_day: float
    financing_is_measured: bool
    vol_percentile_window: int
    slow_vol_bars: int
    min_trades_for_pattern_claim: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def laboratory_settings() -> ResearchSettings:
    """Pre-registered null experiment on one instrument and one timeframe.

    H0: after the declared execution model, short-horizon barrier outcomes are not
    predicted well enough by causal state features — under logistic regression and
    prior-only nearest-neighbor retrieval — to clear the agreement rule on purged
    walk-forward folds.

    The bundled dataset is synthetic. A result on it is not evidence about EUR/USD.
    """
    return ResearchSettings(
        hypothesis_id="H0_EURUSD_H1_BARRIER_V1",
        hypothesis=(
            "On a completed H1 bar, the causal normalized state of price, volatility, "
            "spread, and session does not contain enough information to justify a "
            "long or short barrier trade after bid/ask execution, declared slippage, "
            "declared adverse financing, and commission, once training labels that "
            "overlap the test window have been purged."
        ),
        instrument="EUR_USD",
        granularity="H1",
        horizon_bars=24,
        barrier_atr=1.0,
        stop_atr=1.0,
        min_train_years=5,
        holdout_years=1,
        stride_bars=24,
        knn_k=25,
        min_neighbors=20,
        retrieval_principal="euclidean",
        logit_l2=1.0,
        logit_steps=250,
        logit_lr=0.15,
        min_probability=0.40,
        margin_over_train_base=0.05,
        min_margin_vs_opposite=0.05,
        slippage_pips_per_side=0.2,
        commission_pips_per_side=0.0,
        financing_pips_per_day=0.2,
        financing_is_measured=False,
        vol_percentile_window=24 * 63,
        slow_vol_bars=120,
        min_trades_for_pattern_claim=100,
        seed=20260922,
    )


def price_increment(instrument: str) -> float:
    try:
        return PRICE_INCREMENT[instrument]
    except KeyError as exc:
        raise KeyError(
            f"No research price-increment convention for {instrument}. "
            "Add one explicitly. Do not guess."
        ) from exc
