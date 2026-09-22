# Data

This directory is the archive.

`sets/<dataset_id>/` is written once. It contains `candles.csv` and `manifest.json`. If the hash does not match, do not use the file.

`SYNTHETIC_*` datasets are laboratory clocks, not market prices.

OANDA extracts belong here only after a read of historical candles. Do not store tokens, account ids, or order responses in this directory.
