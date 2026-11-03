"""Baseline taxonomy shared by the runner and the report.

Three classes, never mixed into one claim table:

- ``MECHANISM_BASELINES``: lmpu.* instrumentation and the native control; these
  produce mechanism evidence only and are never end-to-end arms.
- ``END_TO_END_BASELINES``: host model in the loop; these compete for the
  viability claim.
- ``SIMULATION_BASELINES``: labelled simulations; quarantined from every
  claim table (V0.3).
"""

MECHANISM_BASELINES = (
    "lmpu.sync",
    "lmpu.sleep",
    "lmpu.slice",
    "lmpu.individual",
    "lmpu.batch",
    "native",
)
END_TO_END_BASELINES = (
    "direct_cot",
    "sync_tool_full_result",
    "async_poll_programmatic",
    "async_poll_model",
    "session_reopen",
    "app_agent",
    "engine_native",
)
SIMULATION_BASELINES = ("trace_token",)
ALL_BASELINES = (*MECHANISM_BASELINES, *END_TO_END_BASELINES, *SIMULATION_BASELINES)

TREATMENT_PREFIX = "lmpu.model_"


def baseline_kind(baseline: str) -> str:
    if baseline in MECHANISM_BASELINES:
        return "mechanism"
    if baseline in SIMULATION_BASELINES:
        return "simulation"
    return "end_to_end"


def record_is_simulated(record: dict) -> bool:
    return bool(record.get("metadata", {}).get("simulated"))
