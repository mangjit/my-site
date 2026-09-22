# Data store

This repository saves research data. It is not a site to deploy, and it is not a trading system.

GitHub Pages may still be attached to `main`. Do not treat a push as a release of the research bulletin. The homepage stays the existing `index.html`. Reports, if generated, go under `research/` and are not the site.

## What is stored

| Path | What it is |
| --- | --- |
| `data/sets/<dataset_id>/` | Immutable candles plus a sha256 manifest. A finalized id is never overwritten. |
| `experiments/<id>/` | One examination: settings, audit, predictions, trades. Failed runs stay. |
| `experiments/registry.jsonl` | Append-only index of those examinations. |
| `experiments/holdout_locks/` | Created only if a locked year is opened once. |

`SYNTHETIC_LAB_EURUSD_H1_V1` is a laboratory process. It is not historical EUR/USD. Do not join it to a real event calendar.

An OANDA extract, if you add one, is a practice-host candle read only. Put the token in the environment, never in this repository. A missing bid or ask is not invented. Write the extract with a new dataset id.

## Do not commit

- `.env` or any API token
- a second copy of a dataset id that already exists
- a claim that a synthetic file is market data

The loader refuses a file whose bytes do not match its manifest.
