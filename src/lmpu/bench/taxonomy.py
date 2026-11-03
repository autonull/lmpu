"""Baseline taxonomy shared by the runner and the report.

Three classes, never mixed into one claim table:

- ``MECHANISM_BASELINES``: lmpu.* instrumentation and the native control; these
  produce mechanism evidence only and are never end-to-end arms.
- ``END_TO_END_BASELINES``: host model in the loop; these compete for the
  viability claim. Every base arm has an ``<arm>_oracle`` twin in which the
  gold program is injected instead of a model-planned one (V1.8); the twins
  isolate architecture from model planning. The two-session arms
  (``lmpu.model_async_2session`` / ``app_agent_2session``) are oracle-planned
  by construction.
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

_END_TO_END_BASE = (
    "direct_cot",
    "sync_tool_full_result",
    "async_poll_programmatic",
    "async_poll_model",
    "session_reopen",
    "app_agent",
    "engine_native",
    "lmpu.model_async",
    "lmpu.model_sync",
    "lmpu.model_async_2session",
    "app_agent_2session",
)

ORACLE_SUFFIX = "_oracle"
TWO_SESSION_ARMS = frozenset({"lmpu.model_async_2session", "app_agent_2session"})
ORACLE_BASELINES = tuple(
    f"{name}{ORACLE_SUFFIX}" for name in _END_TO_END_BASE if name not in TWO_SESSION_ARMS
)
END_TO_END_BASELINES = (*_END_TO_END_BASE, *ORACLE_BASELINES)
SIMULATION_BASELINES = ("trace_token",)
ALL_BASELINES = (*MECHANISM_BASELINES, *END_TO_END_BASELINES, *SIMULATION_BASELINES)

TREATMENT_PREFIX = "lmpu.model_"


def split_oracle(baseline: str) -> tuple[str, bool]:
    """Split an ``<arm>_oracle`` baseline name into its stem and mode."""
    if baseline.endswith(ORACLE_SUFFIX):
        stem = baseline[: -len(ORACLE_SUFFIX)]
        if f"{stem}{ORACLE_SUFFIX}" in END_TO_END_BASELINES:
            return stem, True
    return baseline, False


def baseline_kind(baseline: str) -> str:
    stem, _ = split_oracle(baseline)
    if stem in MECHANISM_BASELINES:
        return "mechanism"
    if stem in SIMULATION_BASELINES:
        return "simulation"
    return "end_to_end"


def record_is_simulated(record: dict) -> bool:
    return bool(record.get("metadata", {}).get("simulated"))
