# Research protocol

This is the operating rule for the engine. A narrative is not allowed to overrule it.

## Claim ladder

Every conclusion has to say which of these it is.

1. Fact. A property of the code or of a stored file. Example: live execution is disabled.
2. Measurement. A number computed from a named dataset. Example: bid/ask spread on a bar.
3. Historical evidence. A distribution of outcomes after states that were already knowable.
4. Model estimate. A logistic score or a neighbor frequency. Not a calibrated probability unless a calibration test says so.
5. Hypothesis. A statement frozen before the test year is scored.
6. Uncertainty. What the bar, the sample, or the cost assumption cannot resolve.

A weak estimate is not promoted to a fact because the prose sounds sure.

## Time

Internal timestamps are UTC.

A candle timestamp is the bar open. The decision time is the bar close. Features of bar i may use bars `0..i` and nothing later. A label may use bars after i. A label is never a feature.

Another series may be joined only on its availability time: the close of its own bar, not its open. A daily bar is not known to an intraday decision that occurs before that daily bar closes.

## Walk-forward

Years are assigned by decision time, not by a convenient label.

- The last year in a file is a locked holdout. Ordinary research does not train on it, retrieve from it, or score it.
- Remaining years: the first five train, the next year is out of sample, then the window walks forward.
- Two procedures are both reported: expanding, and rolling five-year. Neither is selected because it won.
- A training decision whose label window touches the test year is embargoed.
- Neighbor outcomes must already have been knowable at the query. `retrieved_timestamp < T` is necessary and not sufficient.
- Scalers are fit on the training rows of that fold only.
- In-sample predictions are not computed.
- The principal ledger is the concatenation of out-of-sample trades only.

`python3 -m aegis holdout-once --confirm-once` opens the locked year once for a settings hash and then refuses a second call. That result is not mixed into the walk-forward curve.

## Costs

A result that exists only on midpoint prices is not a result.

The ledger is:

- executable bid/ask fill
- minus declared slippage
- minus declared adverse financing
- minus declared commission

Spread is inside the bid/ask fill. It is reported, and it is not subtracted a second time. If financing was not measured from swap history, the experiment says so. The laboratory run uses an adverse declaration, not a credit carry.

Same-bar barrier conflicts are ambiguous. They are not recorded as wins. The fill assumption, when a trade was on, is the adverse stop. That is an assumption, and it is labeled as one.

## Models

Two baselines, same folds, same costs.

- Model A: L2 logistic regression, one-versus-rest, training rows only.
- Model B: Euclidean nearest neighbors on the training window only. Cosine similarity is computed and not used to choose the reported rule.
- A language-model sentence may describe the evidence package. It may not invent a price, a cause, or a fill.

The frozen agreement rule is in `aegis/models.py`. It does not get retuned on the test year. Stresses that move a threshold or a slippage assumption are marked diagnostic and are not a selection step.

## Events

Event memory is separate from price memory. A record needs `event_time`, `public_information_available_time`, and `market_reaction_time` when those instants are actually known. If the clock time is not established, the record is `UNSAFE_FOR_BACKTEST` and is excluded from predictive tests. The bundled catalog is date-level on purpose. It does not encode "event X means instrument Y rises."

COVID-19 is one record in that catalog, not a special regime. Regimes come from measurable volatility, spread, and trend state.

## Data

A finalized dataset directory is immutable. The candle file is hashed. A mismatch refuses the load. Missing bid or ask is not filled in. Smoothed broker candles are refused.

`SYNTHETIC_*` files are laboratory processes. Their calendar labels exist so year-splits can be tested. They are not historical prices and must not be joined to the event catalog.

## Stop

Invalidate the experiment, do not narrate it, if:

- a feature at T changes when data after T changes
- a neighbor is from the future, or its outcome was not yet known
- a scaler or a label used the test year or the holdout
- an unsafe event was used as a predictor
- transaction costs were not declared
- data quality failed
- the result is a handful of trades, one year, or one event wearing a strategy's clothes

No edge is a valid output. Insufficient data is a valid output.

## Execution boundary

Research, practice, and live are not the same environment.

```text
historical research -> walk-forward -> locked holdout -> practice paper log -> still not live
```

This process cannot authorize the last step. A paper log may record an expected fill against a simulated fill. It cannot send an order.
