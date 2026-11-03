"""LMPU CLI - Command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lmpu.doctor import run_doctor
from lmpu.kernel.runtime import Runtime

app = typer.Typer(name="lmpu", help="Language Model Processing Unit CLI")
console = Console()


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run environment diagnostics."""
    results = run_doctor(verbose=verbose)

    table = Table(title="LMPU Doctor Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    for check, result in results.items():
        status = "✓ PASS" if result.get("ok", False) else "✗ FAIL"
        style = "green" if result.get("ok", False) else "red"
        table.add_row(check, f"[{style}]{status}[/{style}]", result.get("details", ""))

    console.print(table)

    if not all(r.get("ok", False) for r in results.values()):
        raise typer.Exit(1)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Initial prompt for the model"),
    model: str = typer.Option("Qwen/Qwen3.5-0.8B", "--model", "-m", help="Model to use"),
    engine: str = typer.Option("hf", "--engine", "-e", help="Engine: hf, vllm, scripted"),
    socket: str = typer.Option("/tmp/lmpu/run.sock", "--socket", "-s", help="Runtime socket"),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Enable tracing"),
):
    """Run a session with the given prompt."""
    from lmpu.host.hf import HFDecoder, HFEngine
    from lmpu.host.scripted import ScriptedDecoder, free_segment
    from lmpu.host.session import SessionConfig, SessionController

    async def _run():
        # Start runtime
        runtime = Runtime(socket_path=socket)
        await runtime.start()

        controller = None
        try:
            # Create engine and decoder
            if engine == "hf":
                hf_engine = HFEngine(model_name=model)
                decoder = HFDecoder(hf_engine)
            elif engine == "scripted":
                from lmpu.host.engine import ScriptedEngine

                hf_engine = ScriptedEngine([])
                decoder = ScriptedDecoder([free_segment(prompt)])
            else:
                raise ValueError(f"Unknown engine: {engine}")

            # Create session controller
            config = SessionConfig()
            controller = SessionController(
                session_id="cli_session",
                runtime=runtime,
                engine=hf_engine,
                decoder=decoder,
                config=config,
            )

            # Initialize and run
            await controller.initialize(prompt)

            # Run turns until completion
            for _ in range(10):  # Max 10 turns for CLI
                should_continue = await controller.run_turn()
                if not should_continue:
                    break

            console.print("[green]Session completed[/green]")

        finally:
            if controller is not None:
                await controller.close()
            await runtime.stop()

    asyncio.run(_run())


@app.command()
def serve(
    socket: str = typer.Option("/tmp/lmpu/run.sock", "--socket", "-s", help="Runtime socket"),
    trace_dir: str = typer.Option("traces", "--trace-dir", help="Trace directory"),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Enable tracing"),
):
    """Start the runtime server and wait for clients."""
    runtime = Runtime(socket_path=socket, config={"trace_dir": trace_dir, "trace_enabled": trace})

    async def _serve() -> None:
        await runtime.start()
        try:
            await asyncio.Event().wait()
        finally:
            await runtime.stop()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        console.print("[yellow]Runtime stopped[/yellow]")


@app.command()
def demo(
    name: str = typer.Argument(..., help="Demo name: 01-12 or all"),
    socket: str = typer.Option("/tmp/lmpu/run.sock", "--socket", "-s", help="Runtime socket"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU"),
):
    """Run a demo. Canonical names run with assertions."""
    from lmpu.demos import run_demo

    raise typer.Exit(asyncio.run(run_demo(name, socket, gpu)))


@app.command()
def mission_control(
    socket: str = typer.Option("/tmp/lmpu/run.sock", "--socket", "-s", help="Runtime socket"),
    bind: str = typer.Option("127.0.0.1", "--bind", "-b", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port"),
):
    """Start Mission Control dashboard."""
    from lmpu.observability.dashboard import start_dashboard_server

    console.print(f"[cyan]Starting Mission Control on {bind}:{port}[/cyan]")

    async def _run():
        runtime = Runtime(socket_path=socket)
        await runtime.start()

        try:
            await start_dashboard_server(runtime, bind, port)
        finally:
            await runtime.stop()

    asyncio.run(_run())


@app.command()
def bench(
    workload: str = typer.Option("all", "--workload", "-w", help="Workload to run"),
    trials: int = typer.Option(20, "--trials", "-t", help="Number of trials"),
    output: str = typer.Option("benchmarks/results", "--output", "-o", help="Output directory"),
):
    """Run benchmarks."""
    from lmpu.bench.run_benchmark import run_benchmark

    asyncio.run(run_benchmark(workload, trials, output))


@app.command()
def trace(
    trace_id: str = typer.Argument(..., help="Trace ID or 'latest'"),
    trace_dir: str = typer.Option("traces", "--dir", "-d", help="Trace directory"),
    gaps: bool = typer.Option(False, "--gaps", help="Report per-session sequence gaps"),
    redact_payloads: bool = typer.Option(False, "--redact", help="Redact sensitive fields"),
    timeline: bool = typer.Option(False, "--timeline", help="Print a type timeline"),
    replay: bool = typer.Option(False, "--replay", help="Replay records in monotonic order"),
    speed: float = typer.Option(1000.0, "--speed", help="Replay speed; 0 disables sleeping"),
    json_output: bool = typer.Option(False, "--json", help="Emit records as JSON"),
):
    """Replay a trace, detect gaps, redact payloads, and summarize the timeline."""
    import json
    import time

    from lmpu.kernel.trace import redact, replay_trace
    from lmpu.observability.trace import TraceViewer

    directory = Path(trace_dir)
    if trace_id == "latest":
        traces = TraceViewer(trace_dir).list_traces()
        if not traces:
            console.print("[red]No traces found[/red]")
            raise typer.Exit(1)
        trace_path = traces[-1]
    else:
        trace_path = directory / f"trace-{trace_id}.jsonl"
        if not trace_path.exists():
            matches = sorted(directory.glob(f"*{trace_id}*.jsonl"))
            if not matches:
                console.print(f"[red]Trace not found: {trace_id}[/red]")
                raise typer.Exit(1)
            trace_path = matches[-1]

    replay_result = replay_trace(trace_path)
    records = (
        [redact(record) for record in replay_result.records]
        if redact_payloads
        else replay_result.records
    )
    stats = TraceViewer(trace_dir).compute_stats(records)
    ordered = sorted(records, key=lambda record: record.get("mono", 0))

    if json_output:
        typer.echo(json.dumps(ordered, default=str))
    elif replay:
        if not ordered:
            console.print("[yellow]Empty trace[/yellow]")
        origin = ordered[0].get("mono", 0.0) if ordered else 0.0
        previous = origin
        for record in ordered:
            mono = record.get("mono", previous)
            if speed > 0 and mono > previous:
                time.sleep((mono - previous) / speed)
            previous = mono
            console.print(
                f"{mono - origin:8.3f}s {record.get('type', 'record')} {record.get('job_id', '')}"
            )
    else:
        table = Table(title=f"Trace {trace_path.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Records", str(stats.get("total_records", 0)))
        table.add_row("Duration (mono)", f"{stats.get('duration_mono', 0):.3f}s")
        table.add_row("Host forwards", str(stats.get("host_forwards", 0)))
        table.add_row("Generated tokens", str(stats.get("total_tokens_generated", 0)))
        table.add_row("Events", str(stats.get("events", 0)))
        console.print(table)

    if timeline:
        counts: dict[str, int] = {}
        for record in ordered:
            record_type = str(record.get("type", "record"))
            counts[record_type] = counts.get(record_type, 0) + 1
        timeline_table = Table(title="Timeline")
        timeline_table.add_column("Type", style="cyan")
        timeline_table.add_column("Count", justify="right")
        for record_type, count in sorted(counts.items()):
            timeline_table.add_row(record_type, str(count))
        console.print(timeline_table)

    if gaps:
        if replay_result.is_contiguous():
            console.print("[green]No sequence gaps[/green]")
        else:
            for session_id, session_gaps in replay_result.gaps.items():
                for start, end in session_gaps:
                    console.print(f"[red]Gap in {session_id}: missing {start}..{end - 1}[/red]")
            raise typer.Exit(2)


@app.command()
def plugin(
    name: str = typer.Argument(..., help="Plugin name"),
    template: str = typer.Option(
        "basic", "--template", "-t", help="Template: basic, python, vector"
    ),
):
    """Create a new processor plugin."""
    from lmpu.plugin import create_plugin

    create_plugin(name, template)
    console.print(f"[green]Created plugin: {name}[/green]")


@app.command()
def version():
    """Show version."""
    from lmpu import __version__

    console.print(f"LMPU {__version__}")


def main():
    app()


if __name__ == "__main__":
    main()
