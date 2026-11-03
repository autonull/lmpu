"""LMPU CLI - Command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lmpu.doctor import run_doctor
from lmpu.kernel.runtime import Runtime
from lmpu.observability.dashboard import start_dashboard_server

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
def demo(
    name: str = typer.Argument(..., help="Demo name: 01-12 or all"),
    socket: str = typer.Option("/tmp/lmpu/run.sock", "--socket", "-s", help="Runtime socket"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU"),
):
    """Run a demo."""
    from lmpu.demos import run_demo

    asyncio.run(run_demo(name, socket, gpu))


@app.command()
def mission_control(
    socket: str = typer.Option("/tmp/lmpu/run.sock", "--socket", "-s", help="Runtime socket"),
    bind: str = typer.Option("127.0.0.1", "--bind", "-b", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port"),
):
    """Start Mission Control dashboard."""
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
):
    """View trace."""
    from lmpu.observability.trace import TraceViewer

    viewer = TraceViewer(trace_dir)

    if trace_id == "latest":
        traces = viewer.list_traces()
        if not traces:
            console.print("[red]No traces found[/red]")
            raise typer.Exit(1)
        trace_path = traces[-1]
    else:
        trace_path = Path(trace_dir) / f"trace-{trace_id}.jsonl"
        if not trace_path.exists():
            console.print(f"[red]Trace not found: {trace_path}[/red]")
            raise typer.Exit(1)

    records = viewer.load_trace(trace_path)
    stats = viewer.compute_stats(records)

    console.print(f"[cyan]Trace: {trace_path.name}[/cyan]")
    console.print(f"Records: {stats.get('total_records', 0)}")
    console.print(f"Duration: {stats.get('duration_mono', 0):.2f}s")
    console.print(f"Host forwards: {stats.get('host_forwards', 0)}")
    console.print(f"Tokens: {stats.get('total_tokens_generated', 0)}")
    console.print(f"Events: {stats.get('events', 0)}")


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
