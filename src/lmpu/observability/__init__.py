"""LMPU Observability - Tracing, metrics, and dashboard."""

try:
    from .trace import TraceViewer, TraceReplayer, export_trace_archive, import_trace_archive
except ImportError:
    TraceViewer = None
    TraceReplayer = None
    export_trace_archive = None
    import_trace_archive = None

try:
    from .dashboard import app as dashboard_app, init_dashboard, start_dashboard_server
except ImportError:
    dashboard_app = None
    init_dashboard = None
    start_dashboard_server = None

__all__ = [
    "TraceViewer",
    "TraceReplayer",
    "export_trace_archive",
    "import_trace_archive",
    "dashboard_app",
    "init_dashboard",
    "start_dashboard_server",
]