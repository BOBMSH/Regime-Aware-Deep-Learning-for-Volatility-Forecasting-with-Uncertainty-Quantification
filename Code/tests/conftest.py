"""Shared test fixtures, and the ``src`` import path.

Make the project importable as a top-level ``src`` package during tests.
"""

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def project_logs():
    """Factory capturing records from one of the project's *own* loggers.

    Use this instead of ``caplog`` for anything ``src.utils.logging.get_logger``
    produces.

    Why (2026-08-31)
    ----------------
    ``get_logger`` sets ``logger.propagate = False`` and attaches its own stdout
    handler -- deliberately, so runs are not double-logged. pytest's ``caplog``
    fixture attaches its handler to the **root** logger, so whether it sees a
    non-propagating logger's records is an implementation detail of the pytest
    version: it does under pytest 9.1.1 and does not under 9.0.3. Two tests that
    asserted on ``caplog.text`` therefore passed in one environment and failed in
    the other, on identical, correct code.

    A test should not depend on that. This fixture attaches a collector to the
    logger the code actually writes to, so the assertion holds on any pytest.

        def test_something(project_logs):
            records = project_logs("datasets")
            ...
            assert any("expected text" in r.getMessage() for r in records)

    Levels are forced to DEBUG for the duration and restored afterwards.
    """
    attached: list[tuple[logging.Logger, logging.Handler, int]] = []

    class _Collector(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    def _capture(name: str) -> list[logging.LogRecord]:
        logger = logging.getLogger(name)
        handler = _Collector()
        attached.append((logger, handler, logger.level))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return handler.records

    yield _capture

    for logger, handler, level in attached:
        logger.removeHandler(handler)
        logger.setLevel(level)


def log_text(records) -> str:
    """Join captured records into one searchable string (arguments applied)."""
    return "\n".join(r.getMessage() for r in records)
