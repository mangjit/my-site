# Experiment registry

Records in this directory are append-only evidence.

Do not delete a failed or unattractive run to make the remaining record look
cleaner. A new configuration is a new experiment id. It does not replace the old one.

`holdout_locks/` records a one-time locked-holdout examination. The engine
refuses a second scoring of the same frozen configuration.
