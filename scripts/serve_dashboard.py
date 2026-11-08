"""Launcher script for Disaster Intelligence Tactical Command-Center Server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src and tests are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Disaster Intel Tactical Command-Center Server")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")
    args = parser.parse_args()

    print("=" * 78)
    print("  AI-DRIVEN MULTIMODAL DISASTER INTELLIGENCE: COMMAND-CENTER SERVER")
    print("  National Institute of Technology Karnataka (NITK) Surathkal")
    print(f"  Tactical Dashboard listening at: http://localhost:{args.port}")
    print("=" * 78)

    uvicorn.run(
        "disaster_intel.dashboard.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
