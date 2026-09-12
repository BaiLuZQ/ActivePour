# Historical generator source

`generation/` preserves the first-party generation packages from the source-locked final run. The code itself contains no machine-specific paths or credentials. This snapshot retains some historical helper functions to preserve the original file identities; the focused public entry points live in `activepour/`.

The original production trainer is `generation/activepour_pipeline/trainer.py`, with complete optimizer-boundary checkpoint handling. Its cache preparation expects the historical release/legacy-cache contracts, so this snapshot is supplied for **source audit**, not as a standalone one-command reproduction with the schema-example data. Private controllers, configs and caches are intentionally not copied.

Use the portable CLIs for new runs. They preserve the final architecture and objectives while replacing private orchestration and data bootstrap dependencies. Do not treat a portable run as bitwise identical to the archived experiment or overwrite archived checkpoints with it.
