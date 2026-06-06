"""Smoke tests so the Phase 0 gate (`pytest -q` runs green) is always satisfied."""

import importlib


def test_package_imports():
    mod = importlib.import_module("src")
    assert hasattr(mod, "__version__")


def test_data_subpackage_imports():
    importlib.import_module("src.data.realized_vol")
    importlib.import_module("src.data.splits")
    importlib.import_module("src.data.features")


def test_utils_subpackage_imports():
    importlib.import_module("src.utils.seeding")
    importlib.import_module("src.utils.config")
    importlib.import_module("src.utils.io")


def test_set_seed_no_torch():
    from src.utils.seeding import set_seed

    set_seed(17)  # must not raise even when torch is absent or installed
