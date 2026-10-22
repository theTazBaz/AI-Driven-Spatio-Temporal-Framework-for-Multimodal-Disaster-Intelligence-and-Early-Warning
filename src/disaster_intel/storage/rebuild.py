"""Increment 9 - cold-to-hot rebuild job.

The hot tier is a cache of the last ``retention_hours`` of the cold tier. After a Redis
wipe/crash/upgrade (or any suspicion of drift), this job repopulates it from ClickHouse:

    uv run python -m disaster_intel.storage.rebuild --clear

The rebuilt state is identical to the state a never-interrupted live hot tier would hold,
because both keep exactly the observations with ``bucket_end(ts) > horizon`` and apply the
same deduplicating write (pinned by tests/test_hot_unit.py and the integration tests).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from disaster_intel.models import Observation
from disaster_intel.storage.hot import HotStore

_ONE_MICROSECOND = timedelta(microseconds=1)


class ColdReader(Protocol):
    """The one ``ColdStore`` method the rebuild needs."""

    def read_cell(
        self,
        cell: str,
        variable: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Observation]: ...


@dataclass(frozen=True)
class RebuildReport:
    cells: int
    observations_read: int
    stored: int
    duplicates: int
    rejected_stale: int
    cleared_keys: int
    elapsed_s: float

    def summary(self) -> str:
        return (
            f"rebuilt {self.cells} cells from cold: read={self.observations_read:,} "
            f"stored={self.stored:,} duplicates={self.duplicates:,} "
            f"stale={self.rejected_stale:,} cleared_keys={self.cleared_keys:,} "
            f"in {self.elapsed_s:.1f}s"
        )


def rebuild_hot_from_cold(
    hot: HotStore,
    cold: ColdReader,
    cells: Iterable[str],
    *,
    wipe: bool = False,
    cells_per_write: int = 200,
    progress: Callable[[int, int], None] | None = None,
) -> RebuildReport:
    """Repopulate ``hot`` for ``cells`` from ``cold``.

    ``wipe=True`` first deletes EVERY key under the hot store's prefix (not just ``cells``),
    so a rebuild of a subset leaves only that subset. Re-running without ``wipe`` is safe:
    writes are idempotent.
    """
    if cells_per_write < 1:
        raise ValueError("cells_per_write must be >= 1")
    cell_list = list(cells)
    started = time.perf_counter()
    stale_before = hot.rejected_stale
    cleared = hot.clear() if wipe else 0

    # bucket_end(ts) > horizon  <=>  ts > horizon (the horizon is hour-aligned),
    # and read_cell's ``start`` is inclusive, hence +1 microsecond.
    start = hot.horizon() + _ONE_MICROSECOND

    read = stored = duplicates = 0
    pending: list[Observation] = []
    for index, cell in enumerate(cell_list, start=1):
        pending.extend(cold.read_cell(cell, start=start))
        if index % cells_per_write == 0 or index == len(cell_list):
            read += len(pending)
            if pending:
                s, d = hot.write(pending)
                stored += s
                duplicates += d
            pending = []
            if progress is not None:
                progress(index, len(cell_list))
    return RebuildReport(
        cells=len(cell_list),
        observations_read=read,
        stored=stored,
        duplicates=duplicates,
        rejected_stale=hot.rejected_stale - stale_before,
        cleared_keys=cleared,
        elapsed_s=time.perf_counter() - started,
    )


def main(argv: Sequence[str] | None = None) -> None:
    from disaster_intel.grid import CoastalGrid
    from disaster_intel.storage.cold import ColdStore

    parser = argparse.ArgumentParser(description="Rebuild the Redis hot tier from ClickHouse.")
    parser.add_argument("--clear", action="store_true", help="delete all hot-tier keys first")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cells")
    parser.add_argument("--database", default="disaster_intel")
    parser.add_argument("--clickhouse-host", default="localhost")
    parser.add_argument("--clickhouse-port", type=int, default=8123)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args(argv)

    cold = ColdStore.connect(
        host=args.clickhouse_host, port=args.clickhouse_port, database=args.database
    )
    hot = HotStore.connect(host=args.redis_host, port=args.redis_port)
    cells = CoastalGrid(use_cache=True).cells
    if args.limit is not None:
        cells = cells[: args.limit]

    def show(done: int, total: int) -> None:
        print(f"  {done}/{total} cells", end="\r", flush=True)

    report = rebuild_hot_from_cold(hot, cold, cells, wipe=args.clear, progress=show)
    print("\n" + report.summary())


if __name__ == "__main__":
    main()
