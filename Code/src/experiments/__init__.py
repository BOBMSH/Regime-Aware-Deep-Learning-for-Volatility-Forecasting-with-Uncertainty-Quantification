"""Experiment drivers.

Each module here is a thin, reproducible entry point that wires the config,
data-assembly, models and evaluation engine together and writes its artifacts
(predictions, tables, figures, milestone note) into ``results/``. Drivers contain
no modelling logic of their own -- that lives in ``src/models`` and
``src/evaluation`` -- so the numbers cited in the dissertation always trace back
to unit-tested library code (roadmap §4/§9).
"""
