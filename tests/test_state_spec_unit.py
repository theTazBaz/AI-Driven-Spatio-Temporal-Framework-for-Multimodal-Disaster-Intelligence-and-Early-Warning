"""Increment 10 unit tests - FeatureSpec, StateVector, the compound saturation term.

Test list (plan.md, Increment 10), spec half:
 1. Vector length and order match the versioned FeatureSpec.
 2. compound saturation = soil_moisture x rolling_rainfall: zero if either is zero, rises with
    both, NaN (not 0) if either is missing.
 3. A missing sensor is NaN, never 0; NaN survives a JSON round trip.
 4. Loading a vector saved under another spec FAILS LOUDLY (version, fingerprint, length).
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from disaster_intel.models import VariableCatalog
from disaster_intel.state import FeatureSpec, SpecMismatchError, StateVector
from disaster_intel.state.spec import (
    V1_DYNAMIC_VARIABLES,
    V1_STATIC,
    V1_WINDOWS,
    feature_name,
)

pytestmark = pytest.mark.unit

NOW = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)
CELL = "872a1072bffffff"
SPEC = FeatureSpec.v1()
NAN = float("nan")
SOIL = "soil_moisture_rootzone__24h"
RAIN = "precipitation__72h"


def make_vector(spec: FeatureSpec = SPEC, values: tuple[float, ...] | None = None) -> StateVector:
    vals = values if values is not None else tuple(float(i) for i in range(len(spec)))
    return StateVector(
        cell=CELL,
        as_of=NOW,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        values=vals,
        coverage=tuple(None for _ in vals),
    )


# ── 1. Layout ────────────────────────────────────────────────────────


def test_v1_has_25_dynamic_3_static_1_derived_in_a_fixed_order() -> None:
    names = SPEC.names
    assert len(SPEC) == len(names) == 29
    expected_dynamic = [feature_name(v, w) for v in V1_DYNAMIC_VARIABLES for w in V1_WINDOWS]
    assert list(names[:25]) == expected_dynamic
    assert names[0] == "precipitation__1h"
    assert names[4] == "precipitation__7d"
    assert names[25:28] == V1_STATIC
    assert names[28] == "compound_saturation"
    assert len(set(names)) == len(names)


def test_index_finds_names_and_rejects_unknown_ones() -> None:
    assert SPEC.index("precipitation__1h") == 0
    assert SPEC.index("compound_saturation") == 28
    with pytest.raises(SpecMismatchError):
        SPEC.index("not_a_feature")


def test_fingerprint_is_stable_and_changes_with_version_or_definition() -> None:
    assert FeatureSpec.v1().fingerprint == FeatureSpec.v1().fingerprint
    assert dataclasses.replace(SPEC, version=2).fingerprint != SPEC.fingerprint
    changed = dict(SPEC.definitions)
    changed[RAIN] = "MAX of precipitation"  # e.g. someone swaps SUM for MAX
    assert dataclasses.replace(SPEC, definitions=changed).fingerprint != SPEC.fingerprint


def test_fingerprint_tracks_the_catalog_aggregation() -> None:
    catalog = VariableCatalog.default()
    assert "SUM of precipitation" in FeatureSpec.v1(catalog).definitions[RAIN]
    assert "AVG of soil_moisture_rootzone" in SPEC.definitions[SOIL]


def test_spec_rejects_duplicates_and_unknown_derived_inputs() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SPEC, static=("elevation_m", "elevation_m"))
    with pytest.raises(ValueError):
        dataclasses.replace(
            SPEC,
            derived=(dataclasses.replace(SPEC.derived[0], left="nope"),),
        )


# ── 2. Compound saturation ───────────────────────────────────────────


def test_compound_is_the_product_of_soil_moisture_and_rainfall() -> None:
    assert SPEC.derive({SOIL: 0.4, RAIN: 50.0}) == [pytest.approx(20.0)]


@pytest.mark.parametrize(("soil", "rain"), [(0.0, 50.0), (0.4, 0.0), (0.0, 0.0)])
def test_compound_is_zero_if_either_input_is_zero(soil: float, rain: float) -> None:
    assert SPEC.derive({SOIL: soil, RAIN: rain}) == [0.0]


@pytest.mark.parametrize(("soil", "rain"), [(NAN, 5.0), (0.4, NAN), (NAN, NAN)])
def test_compound_is_nan_not_zero_if_either_input_is_missing(soil: float, rain: float) -> None:
    (value,) = SPEC.derive({SOIL: soil, RAIN: rain})
    assert math.isnan(value)


@given(
    soil=st.floats(0.01, 0.6),
    rain=st.floats(0.1, 500.0),
    bump=st.floats(0.01, 100.0),
)
def test_compound_rises_with_both_inputs(soil: float, rain: float, bump: float) -> None:
    (base,) = SPEC.derive({SOIL: soil, RAIN: rain})
    (more_rain,) = SPEC.derive({SOIL: soil, RAIN: rain + bump})
    (wetter,) = SPEC.derive({SOIL: soil + bump / 1000.0, RAIN: rain})
    assert more_rain > base
    assert wetter > base


# ── 3. Serialisation keeps NaN as "missing" ──────────────────────────


def test_nan_survives_json_as_null_and_is_never_turned_into_zero() -> None:
    values = [float(i) for i in range(len(SPEC))]
    values[3] = NAN
    vector = make_vector(values=tuple(values))
    text = vector.to_json()
    assert "NaN" not in text  # strict JSON
    assert json.loads(text)["values"][3] is None
    restored = StateVector.from_json(text, SPEC)
    assert math.isnan(restored.values[3])
    assert restored.same_values(vector)


def test_same_values_treats_nan_as_equal_and_honours_tolerance() -> None:
    a = make_vector(values=tuple([NAN] + [1.0] * (len(SPEC) - 1)))
    b = make_vector(values=tuple([NAN] + [1.0 + 1e-12] * (len(SPEC) - 1)))
    assert not a.same_values(b)
    assert a.same_values(b, rel_tol=1e-9)
    c = make_vector(values=tuple([0.0] + [1.0] * (len(SPEC) - 1)))
    assert not a.same_values(c, rel_tol=1.0)  # NaN vs 0.0 is never "close"


def test_naive_as_of_is_rejected() -> None:
    with pytest.raises(ValueError):
        StateVector(CELL, datetime(2024, 7, 10), 1, "x", (1.0,), (None,))


# ── 4. Old vector under a new spec fails loudly ──────────────────────


def test_loading_a_vector_under_a_newer_spec_version_raises() -> None:
    saved = make_vector().to_json()
    newer = dataclasses.replace(SPEC, version=2)
    with pytest.raises(SpecMismatchError):
        StateVector.from_json(saved, newer)


def test_loading_under_a_spec_with_the_same_version_but_different_layout_raises() -> None:
    saved = make_vector().to_json()
    changed = dict(SPEC.definitions)
    changed[RAIN] = "something else"
    sneaky = dataclasses.replace(SPEC, definitions=changed)  # version unchanged!
    with pytest.raises(SpecMismatchError, match="active spec"):
        StateVector.from_json(saved, sneaky)


def test_loading_a_truncated_vector_raises() -> None:
    payload = make_vector().to_dict()
    payload["values"] = payload["values"][:-1]
    payload["coverage"] = payload["coverage"][:-1]
    with pytest.raises(SpecMismatchError):
        StateVector.from_dict(payload, SPEC)


def test_malformed_payload_raises_spec_mismatch_not_keyerror() -> None:
    with pytest.raises(SpecMismatchError):
        StateVector.from_dict({"cell": CELL}, SPEC)


def test_as_mapping_and_get_refuse_a_foreign_spec() -> None:
    vector = make_vector()
    assert vector.as_mapping(SPEC)["precipitation__1h"] == 0.0
    assert vector.get(SPEC, "compound_saturation") == 28.0
    with pytest.raises(SpecMismatchError):
        vector.as_mapping(dataclasses.replace(SPEC, version=2))
