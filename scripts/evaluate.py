"""Increment 13 — Phase 1 Evaluation Report and Evaluation Demo.

Runs historical replay over past disaster benchmark (e.g. July 2024 Shirur landslide),
demonstrates Western Ghats cell state vector progression across the storm, verifies
deterministic rollback, and outputs the certified metrics report meeting all
Objective 1 and Objective 2 criteria from plan.md Section 15.

Usage:
    uv run python scripts/evaluate.py
    # or
    make evaluate
"""

from __future__ import annotations

from disaster_intel.evaluation import main

if __name__ == "__main__":
    main()

