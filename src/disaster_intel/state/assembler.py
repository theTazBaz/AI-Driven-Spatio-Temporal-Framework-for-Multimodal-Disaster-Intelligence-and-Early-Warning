"""Increment 10 - the state vector assembler: ``get_state(cell, as_of)``.

Turns "rolling windows from a store" + "static layers" into one ``StateVector`` laid out by a
versioned ``FeatureSpec``. Both tiers answer through the same call, and must agree:

* **hot**  (Redis, ``HotStore``)  - "now" in about a millisecond. It only retains the last seven
  days, so it can answer only when the requested hour is the current hour (its windows would
  otherwise reach before its retained range and it raises ``HotRangeError``).
* **cold** (ClickHouse, ``ColdStore``) - the permanent record; answers ANY past time, and the
  answer for time T is the same whether it is computed live at T or years later (windows only
  ever include observations with ``ts <= as_of``; Increment 8 pins this).

``tier="auto"`` tries hot and falls back to cold on ``HotRangeError``. Because the hot tier is a
rebuildable cache, a wiped Redis would answer "no data"; ``compare_tiers`` is the check for
that, and ``rebuild_hot_from_cold`` is the repair (Increment 9).

A ``get_state`` for a time in the future is refused: it would silently equal "now".
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

import h3

from disaster_intel.clock import Clock, SystemClock
from disaster_intel.state.spec import NAN, FeatureSpec, StateVector
from disaster_intel.state.static_layers import StaticLookup
from disaster_intel.storage.hot import HotRangeError
from disaster_intel.windows import WINDOW_HOURS, WindowValue

Tier = Literal["auto", "hot", "cold"]
Windows = dict[str, dict[str, WindowValue]]


class WindowSource(Protocol):
    """What both ``HotStore`` and ``ColdStore`` provide."""

    def windows(
        self,
        cell: str,
        as_of: datetime,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> Windows: ...


@dataclass(frozen=True)
class TierComparison:
    """Result of ``compare_tiers``: do hot and cold agree for this cell and time?"""

    cell: str
    as_of: datetime
    differing: tuple[str, ...]  # feature names (``name[coverage]`` for coverage differences)
    max_abs_diff: float

    @property
    def agrees(self) -> bool:
        return not self.differing


class StateAssembler:
    def __init__(
        self,
        static: StaticLookup,
        hot: WindowSource | None = None,
        cold: WindowSource | None = None,
        spec: FeatureSpec | None = None,
        clock: Clock | None = None,
    ) -> None:
        if hot is None and cold is None:
            raise ValueError("StateAssembler needs at least one of hot / cold")
        self._static = static
        self._hot = hot
        self._cold = cold
        self._spec = spec or FeatureSpec.v1()
        self._clock: Clock = clock or SystemClock()
        self._window_hours = {w: WINDOW_HOURS[w] for w in self._spec.windows}

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    # ── public API ───────────────────────────────────────────────────

    def get_state(
        self, cell: str, as_of: datetime | None = None, tier: Tier = "auto"
    ) -> StateVector:
        """The state vector of ``cell`` at ``as_of`` (default: now)."""
        if not isinstance(cell, str) or not h3.is_valid_cell(cell):
            raise ValueError(f"Invalid H3 cell {cell!r}")
        now = self._clock.now()
        when = now if as_of is None else self._aware_utc(as_of)
        if when > now:
            raise ValueError(f"as_of {when.isoformat()} is in the future (now={now.isoformat()})")

        if tier == "hot":
            return self._from(self._require(self._hot, "hot"), "hot", cell, when)
        if tier == "cold":
            return self._from(self._require(self._cold, "cold"), "cold", cell, when)
        if self._hot is not None:
            try:
                return self._from(self._hot, "hot", cell, when)
            except HotRangeError:
                if self._cold is None:
                    raise
        return self._from(self._require(self._cold, "cold"), "cold", cell, when)

    def get_states(
        self, cells: Sequence[str], as_of: datetime | None = None, tier: Tier = "auto"
    ) -> list[StateVector]:
        """Vectors for many cells at ONE consistent instant (the matrix Objective 3 trains on)."""
        when = self._clock.now() if as_of is None else self._aware_utc(as_of)
        return [self.get_state(c, when, tier) for c in cells]

    def compare_tiers(
        self,
        cell: str,
        as_of: datetime | None = None,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-9,
    ) -> TierComparison:
        """Hot vs cold for the same cell and time (both tiers must be configured)."""
        hot = self.get_state(cell, as_of, tier="hot")
        cold = self.get_state(cell, as_of, tier="cold")
        differing: list[str] = []
        worst = 0.0
        for name, a, b, ca, cb in zip(
            self._spec.names, hot.values, cold.values, hot.coverage, cold.coverage, strict=True
        ):
            if math.isnan(a) != math.isnan(b):
                differing.append(name)
                worst = math.inf
            elif not math.isnan(a) and not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                differing.append(name)
                worst = max(worst, abs(a - b))
            if ca != cb:
                differing.append(f"{name}[coverage]")
        return TierComparison(cell, hot.as_of, tuple(differing), worst)

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _aware_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise ValueError("Naive datetime is not allowed")
        return ts.astimezone(UTC)

    @staticmethod
    def _require(source: WindowSource | None, name: str) -> WindowSource:
        if source is None:
            raise ValueError(f"The {name} tier is not configured on this assembler")
        return source

    def _from(self, source: WindowSource, tier: str, cell: str, when: datetime) -> StateVector:
        table = source.windows(cell, when, window_hours=self._window_hours)
        return self._assemble(cell, when, table, tier)

    def _assemble(self, cell: str, when: datetime, table: Windows, tier: str) -> StateVector:
        spec = self._spec
        values: list[float] = []
        coverage: list[float | None] = []
        for variable, window in spec.dynamic:
            wv = table[variable][window]
            values.append(NAN if wv.value is None else float(wv.value))  # no data -> NaN, not 0
            coverage.append(wv.coverage)
        static = self._static.get(cell)
        for name in spec.static:
            values.append(float(getattr(static, name)))
            coverage.append(None)
        known = dict(zip(spec.names, values, strict=False))
        for derived in spec.derive(known):
            values.append(derived)
            coverage.append(None)
        return StateVector(
            cell=cell,
            as_of=when,
            spec_version=spec.version,
            spec_fingerprint=spec.fingerprint,
            values=tuple(values),
            coverage=tuple(coverage),
            tier=tier,
        )


# ── CLI: python -m disaster_intel.state.assembler --cell <h3> ────────


def main(argv: Sequence[str] | None = None) -> None:
    from disaster_intel.state.static_layers import DEFAULT_STATIC_PATH, StaticLayers
    from disaster_intel.storage.cold import ColdStore
    from disaster_intel.storage.hot import HotStore

    parser = argparse.ArgumentParser(description="Print one cell's state vector.")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--as-of", help="ISO time with offset, e.g. 2024-07-10T12:00:00+00:00")
    parser.add_argument("--tier", choices=["auto", "hot", "cold"], default="auto")
    parser.add_argument("--static", default=str(DEFAULT_STATIC_PATH))
    parser.add_argument("--database", default="disaster_intel")
    args = parser.parse_args(argv)

    assembler = StateAssembler(
        static=StaticLayers.load(args.static),
        hot=HotStore.connect(),
        cold=ColdStore.connect(database=args.database),
    )
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else None
    vector = assembler.get_state(args.cell, as_of, args.tier)
    print(f"cell={vector.cell} as_of={vector.as_of.isoformat()} tier={vector.tier}")
    print(f"spec v{vector.spec_version} ({vector.spec_fingerprint}), {len(vector)} features")
    for name, value, cov in zip(assembler.spec.names, vector.values, vector.coverage, strict=True):
        shown = "NaN (missing)" if math.isnan(value) else f"{value:.4f}"
        suffix = "" if cov is None else f"  coverage {cov:.0%}"
        print(f"  {name:<38}{shown:>16}{suffix}")


if __name__ == "__main__":
    main()
