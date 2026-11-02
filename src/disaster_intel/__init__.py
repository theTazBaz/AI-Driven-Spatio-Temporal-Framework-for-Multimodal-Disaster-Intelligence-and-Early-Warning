"""AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence."""

__version__ = "0.1.0"

from disaster_intel.adapters import (
    AdapterError,
    AdapterMetrics,
    OpenMeteoAdapter,
    ParseError,
    SourceAdapter,
    SourceError,
)
from disaster_intel.backfill import (
    BENCHMARK_WINDOWS,
    BackfillEngine,
    BackfillResult,
    HistoricalWindow,
    generate_historical_spell,
)
from disaster_intel.bus import (
    BufferFullError,
    Bus,
    BusError,
    InMemoryBus,
    Message,
)
from disaster_intel.clock import Clock, FakeClock, SystemClock
from disaster_intel.discretizers import (
    FanOutTable,
    GridNodeDiscretizer,
    PointDiscretizer,
    PolygonDiscretizer,
    SpatialDiscretizer,
    point_in_polygon,
)
from disaster_intel.evaluation import (
    Phase1EvaluationReport,
    Phase1Evaluator,
    run_evaluation_demo,
)
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import (
    AggMethod,
    CAPSeverity,
    Geometry,
    GridNodeGeometry,
    Observation,
    PointGeometry,
    PolygonGeometry,
    RawReading,
    Variable,
    VariableCatalog,
)
from disaster_intel.pipeline import (
    IngestionMetrics,
    IngestionPipeline,
    InMemorySink,
    ObservationSink,
    calculate_percentiles,
    run_replay_fixture,
)

__all__ = [
    "BENCHMARK_WINDOWS",
    "AdapterError",
    "AdapterMetrics",
    "AggMethod",
    "BackfillEngine",
    "BackfillResult",
    "BufferFullError",
    "Bus",
    "BusError",
    "CAPSeverity",
    "Clock",
    "CoastalGrid",
    "FakeClock",
    "FanOutTable",
    "Geometry",
    "GridNodeDiscretizer",
    "GridNodeGeometry",
    "HistoricalWindow",
    "InMemoryBus",
    "InMemorySink",
    "IngestionMetrics",
    "IngestionPipeline",
    "Message",
    "Observation",
    "ObservationSink",
    "OpenMeteoAdapter",
    "ParseError",
    "Phase1EvaluationReport",
    "Phase1Evaluator",
    "PointDiscretizer",
    "PointGeometry",
    "PolygonDiscretizer",
    "PolygonGeometry",
    "RawReading",
    "SourceAdapter",
    "SourceError",
    "SpatialDiscretizer",
    "SystemClock",
    "Variable",
    "VariableCatalog",
    "__version__",
    "calculate_percentiles",
    "generate_historical_spell",
    "point_in_polygon",
    "run_evaluation_demo",
    "run_replay_fixture",
]
