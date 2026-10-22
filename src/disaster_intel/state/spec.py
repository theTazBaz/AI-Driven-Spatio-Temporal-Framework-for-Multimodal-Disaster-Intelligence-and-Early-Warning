"""Increment 10 - the versioned ``FeatureSpec`` and the ``StateVector`` it describes.

A ``StateVector`` is a flat list of floats. What each position MEANS is defined by exactly one
object, the ``FeatureSpec``; the vector carries the spec's version and fingerprint so that:

* the order and length of a vector always match the spec that produced it;
* loading a vector that was saved under a different spec FAILS LOUDLY
  (``SpecMismatchError``) instead of silently shifting every column;
* a missing sensor is ``NaN``, never ``0`` (zero rain is a measurement, "no data" is not).

Spec v1 (what Objective 3 trains on):

* dynamic  : 5 variables x 5 rolling windows (1h, 6h, 24h, 72h, 7d) from ``windows.py``
             precipitation, soil_moisture_surface, soil_moisture_rootzone, discharge,
             seismic_magnitude   -> 25 features
* static   : elevation_m, slope_deg, population_density_km2                  ->  3 features
* derived  : compound_saturation = soil_moisture_rootzone__24h x precipitation__72h
             (a wet root zone AND a lot of recent rain: the compound-cascade term of the
             proposal, Section 3.2)                                           ->  1 feature
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from disaster_intel.models import VariableCatalog
from disaster_intel.windows import WINDOW_HOURS

NAN = float("nan")

V1_DYNAMIC_VARIABLES: tuple[str, ...] = (
    "precipitation",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "discharge",
    "seismic_magnitude",
)
V1_WINDOWS: tuple[str, ...] = ("1h", "6h", "24h", "72h", "7d")
V1_STATIC: tuple[str, ...] = ("elevation_m", "slope_deg", "population_density_km2")


class SpecMismatchError(ValueError):
    """A stored vector (or a request) does not match the FeatureSpec in use."""


def feature_name(variable: str, window: str) -> str:
    return f"{variable}__{window}"


def is_missing(value: float) -> bool:
    return math.isnan(value)


@dataclass(frozen=True)
class DerivedFeature:
    """``name = left x right`` over two earlier features; NaN if either input is missing."""

    name: str
    left: str
    right: str

    def compute(self, known: Mapping[str, float]) -> float:
        a, b = known[self.left], known[self.right]
        if math.isnan(a) or math.isnan(b):
            return NAN
        return a * b


@dataclass(frozen=True)
class FeatureSpec:
    """Ordered, versioned definition of a state vector."""

    version: int
    dynamic: tuple[tuple[str, str], ...]  # (variable, window), in vector order
    static: tuple[str, ...]
    derived: tuple[DerivedFeature, ...]
    definitions: Mapping[str, str]  # feature name -> human-readable definition (fingerprinted)

    def __post_init__(self) -> None:
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("FeatureSpec has duplicate feature names")
        seen: set[str] = {feature_name(v, w) for v, w in self.dynamic} | set(self.static)
        for d in self.derived:
            if d.left not in seen or d.right not in seen:
                raise ValueError(f"Derived feature {d.name!r} uses an unknown input")
            seen.add(d.name)
        missing = [n for n in names if n not in self.definitions]
        if missing:
            raise ValueError(f"Features without a definition: {missing}")

    @classmethod
    def v1(cls, catalog: VariableCatalog | None = None) -> FeatureSpec:
        cat = catalog or VariableCatalog.default()
        dynamic = tuple((v, w) for v in V1_DYNAMIC_VARIABLES for w in V1_WINDOWS)
        definitions: dict[str, str] = {}
        for variable, window in dynamic:
            var = cat[variable]
            definitions[feature_name(variable, window)] = (
                f"{var.agg_method.name} of {variable} [{var.canonical_unit}] "
                f"over the last {WINDOW_HOURS[window]} hour-ending buckets"
            )
        for name in V1_STATIC:
            definitions[name] = f"static layer {name}"
        derived = (
            DerivedFeature(
                name="compound_saturation",
                left="soil_moisture_rootzone__24h",
                right="precipitation__72h",
            ),
        )
        definitions["compound_saturation"] = (
            "soil_moisture_rootzone__24h * precipitation__72h; NaN if either is missing"
        )
        return cls(
            version=1,
            dynamic=dynamic,
            static=V1_STATIC,
            derived=derived,
            definitions=definitions,
        )

    # ── layout ───────────────────────────────────────────────────────

    @property
    def names(self) -> tuple[str, ...]:
        return (
            *(feature_name(v, w) for v, w in self.dynamic),
            *self.static,
            *(d.name for d in self.derived),
        )

    def __len__(self) -> int:
        return len(self.dynamic) + len(self.static) + len(self.derived)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise SpecMismatchError(f"Feature {name!r} is not in spec v{self.version}") from None

    @property
    def windows(self) -> tuple[str, ...]:
        """Distinct window names the dynamic features need, in first-use order."""
        return tuple(dict.fromkeys(w for _, w in self.dynamic))

    @property
    def fingerprint(self) -> str:
        """Hash of version + ordered (name, definition): any change to the layout changes it."""
        payload = json.dumps(
            {"version": self.version, "features": [[n, self.definitions[n]] for n in self.names]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def derive(self, known: Mapping[str, float]) -> list[float]:
        """Derived feature values, in spec order, from the dynamic + static values."""
        values = dict(known)
        out: list[float] = []
        for d in self.derived:
            value = d.compute(values)
            values[d.name] = value
            out.append(value)
        return out


@dataclass(frozen=True)
class StateVector:
    """One cell's state at one instant, laid out as ``spec`` says.

    ``coverage`` has one entry per feature: the observed fraction of the window for dynamic
    features, ``None`` for static and derived ones.
    """

    cell: str
    as_of: datetime
    spec_version: int
    spec_fingerprint: str
    values: tuple[float, ...]
    coverage: tuple[float | None, ...]
    tier: str = "unknown"  # which store answered: "hot" or "cold"

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("Naive datetime is not allowed")
        if len(self.values) != len(self.coverage):
            raise ValueError("values and coverage must have the same length")

    def __len__(self) -> int:
        return len(self.values)

    def as_mapping(self, spec: FeatureSpec) -> dict[str, float]:
        self.check_spec(spec)
        return dict(zip(spec.names, self.values, strict=True))

    def get(self, spec: FeatureSpec, name: str) -> float:
        return self.values[spec.index(name)]

    def check_spec(self, spec: FeatureSpec) -> None:
        if (self.spec_version, self.spec_fingerprint) != (spec.version, spec.fingerprint):
            raise SpecMismatchError(
                f"Vector was built under spec v{self.spec_version} "
                f"({self.spec_fingerprint}), but the active spec is v{spec.version} "
                f"({spec.fingerprint})"
            )
        if len(self.values) != len(spec):
            raise SpecMismatchError(
                f"Vector has {len(self.values)} values; spec v{spec.version} defines {len(spec)}"
            )

    def same_values(self, other: StateVector, rel_tol: float = 0.0, abs_tol: float = 0.0) -> bool:
        """Equality that treats NaN == NaN (dataclass ``==`` cannot) and allows a tolerance."""
        if (self.cell, self.spec_version, self.spec_fingerprint) != (
            other.cell,
            other.spec_version,
            other.spec_fingerprint,
        ):
            return False
        if self.coverage != other.coverage or len(self.values) != len(other.values):
            return False
        for a, b in zip(self.values, other.values, strict=True):
            if math.isnan(a) or math.isnan(b):
                if not (math.isnan(a) and math.isnan(b)):
                    return False
            elif not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        return True

    # ── serialisation (NaN <-> null: strict JSON has no NaN) ─────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "spec_version": self.spec_version,
            "spec_fingerprint": self.spec_fingerprint,
            "values": [None if math.isnan(v) else v for v in self.values],
            "coverage": list(self.coverage),
            "tier": self.tier,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], spec: FeatureSpec) -> StateVector:
        """Rebuild a vector, REFUSING anything saved under a different spec."""
        try:
            version = int(data["spec_version"])
            fingerprint = str(data["spec_fingerprint"])
            raw_values: Sequence[float | None] = data["values"]
            coverage = tuple(data["coverage"])
            cell = str(data["cell"])
            as_of = datetime.fromisoformat(str(data["as_of"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecMismatchError(f"Malformed state vector payload: {exc}") from exc
        vector = cls(
            cell=cell,
            as_of=as_of,
            spec_version=version,
            spec_fingerprint=fingerprint,
            values=tuple(NAN if v is None else float(v) for v in raw_values),
            coverage=coverage,
            tier=str(data.get("tier", "unknown")),
        )
        vector.check_spec(spec)
        return vector

    @classmethod
    def from_json(cls, text: str, spec: FeatureSpec) -> StateVector:
        return cls.from_dict(json.loads(text), spec)
