"""LMPU Worker - Main entry point for worker processes."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from typing import Any

import orjson

from lmpu.processors import (
    JobContext,
    Processor,
    ProcessorContext,
)
from lmpu.processors.builtin import (
    HardwareStubProcessor,
    NeuralSummaryProcessor,
    PureFunctionProcessor,
    PythonVMProcessor,
    SymbolicPlannerProcessor,
    VectorEngineProcessor,
)

# Processor registry
BUILTIN_PROCESSORS = {
    "pure.func": PureFunctionProcessor,
    "python.vm": PythonVMProcessor,
    "vector.engine": VectorEngineProcessor,
    "symbolic.planner": SymbolicPlannerProcessor,
    "neural.summary": NeuralSummaryProcessor,
    "hardware.stub": HardwareStubProcessor,
}


class Worker:
    """Worker process that executes jobs for a specific processor."""

    def __init__(
        self,
        processor_name: str,
        worker_id: str,
        plugin_path: str | None = None,
    ):
        self.processor_name = processor_name
        self.worker_id = worker_id
        self.plugin_path = plugin_path
        self.processor: Processor | None = None
        self.current_job_id: str | None = None
        self.session_id: str | None = None
        self._running = True
        self._stdin_reader: asyncio.StreamReader | None = None
        self._stdout_writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        """Start the worker."""
        # Load processor
        if self.processor_name in BUILTIN_PROCESSORS:
            self.processor = BUILTIN_PROCESSORS[self.processor_name]()
        elif self.plugin_path:
            # Load from plugin
            self.processor = await self._load_plugin_processor()
        else:
            raise ValueError(f"Unknown processor: {self.processor_name}")

        # Setup stdin/stdout for communication (binary mode for orjson)
        loop = asyncio.get_event_loop()

        # Use a simpler approach: create readers/writers directly
        self._stdin_reader = asyncio.StreamReader()
        self._stdin_protocol = asyncio.StreamReaderProtocol(self._stdin_reader)

        # Connect stdin (binary)
        await loop.connect_read_pipe(
            lambda: self._stdin_protocol,
            sys.stdin.buffer,
        )

        # For stdout, use a simple approach with sys.stdout.buffer directly
        class BinaryStreamWriter:
            def __init__(self, stream):
                self._stream = stream
                self._loop = loop

            def write(self, data: bytes) -> None:
                self._stream.write(data)
                self._stream.flush()

            async def drain(self) -> None:
                pass

            def close(self) -> None:
                pass

            async def wait_closed(self) -> None:
                pass

        self._stdout_writer = BinaryStreamWriter(sys.stdout.buffer)

        # Initialize processor
        ctx = ProcessorContext(runtime_config={})
        await self.processor.setup(ctx)

        # Send ready message
        await self._send_message({
            "type": "ready",
            "worker_id": self.worker_id,
            "processor": self.processor_name,
            "manifest": self.processor.manifest.__dict__,
            "pid": os.getpid(),
        })

        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

        # Process commands
        await self._command_loop()

    async def _load_plugin_processor(self) -> Processor:
        """Load processor from plugin module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("plugin", self.plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Processor subclass
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Processor) and obj is not Processor:
                return obj()

        raise ValueError("No Processor subclass found in plugin")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            await asyncio.sleep(1.0)
            if self._running:
                await self._send_message({
                    "type": "heartbeat",
                    "worker_id": self.worker_id,
                    "session": self.session_id or "unknown",
                    "job_id": self.current_job_id,
                })

    async def _command_loop(self) -> None:
        """Process incoming commands."""
        while self._running:
            try:
                line = await self._stdin_reader.readline()
                if not line:
                    break

                msg = orjson.loads(line)
                await self._handle_command(msg)
            except orjson.JSONDecodeError:
                continue
            except Exception as e:
                await self._send_message({
                    "type": "error",
                    "worker_id": self.worker_id,
                    "job_id": self.current_job_id,
                    "error_code": "internal",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                })

    async def _handle_command(self, msg: dict[str, Any]) -> None:
        """Handle a command from parent."""
        cmd_type = msg.get("type")

        if cmd_type == "execute":
            await self._execute_job(msg)
        elif cmd_type == "cancel":
            await self._cancel_job(msg)
        elif cmd_type == "shutdown":
            await self._shutdown()

    async def _execute_job(self, msg: dict[str, Any]) -> None:
        """Execute a job."""
        job = msg.get("job", {})
        self.current_job_id = job.get("job_id")
        self.session_id = msg.get("session_id")

        # Prepare buffers (in real impl, these come from fd passing)
        buffers = {}

        # Create job context
        async def send_progress(fraction: float, message: str = None):
            await self._send_message({
                "type": "progress",
                "job_id": self.current_job_id,
                "fraction": fraction,
                "message": message,
            })

        async def send_needs_input(fields: list[str], prompt: str):
            await self._send_message({
                "type": "needs_input",
                "job_id": self.current_job_id,
                "fields": fields,
                "prompt": prompt,
            })

        async def submit_child(processor: str, op: str, inputs: dict, config: dict = None):
            # Would submit to parent runtime
            pass

        def log(level: str, message: str, **kwargs):
            pass

        def trace(event: str, data: dict):
            pass

        def cancelled() -> bool:
            return False  # Would check cancellation flag

        job_ctx = JobContext(
            job_id=self.current_job_id,
            session_id=self.session_id,
            buffers=buffers,
            send_progress=send_progress,
            send_needs_input=send_needs_input,
            submit_child=submit_child,
            log=log,
            trace=trace,
            cancelled=cancelled,
        )

        try:
            # Send started event
            await self._send_message({
                "type": "started",
                "job_id": self.current_job_id,
                "worker": self.worker_id,
            })

            # Execute
            result = await self.processor.execute(job, job_ctx)

            if result.success:
                await self._send_message({
                    "type": "result",
                    "job_id": self.current_job_id,
                    "result": result.data,
                    "buffer_refs": result.buffer_refs,
                    "checksum": result.checksum,
                    "cache_hit": result.cache_hit,
                })
            else:
                await self._send_message({
                    "type": "error",
                    "job_id": self.current_job_id,
                    "error_code": result.error_code or "execution_error",
                    "message": result.error_message or "Unknown error",
                    "retriable": result.retriable,
                })
        except Exception as e:
            await self._send_message({
                "type": "error",
                "job_id": self.current_job_id,
                "error_code": "execution_error",
                "message": str(e),
                "retriable": False,
            })
        finally:
            self.current_job_id = None

    async def _cancel_job(self, msg: dict[str, Any]) -> None:
        """Cancel current job."""
        job_id = msg.get("job_id")
        if self.current_job_id == job_id and self.processor:
            await self.processor.cancel(job_id)

    async def _shutdown(self) -> None:
        """Shutdown worker."""
        self._running = False
        if self.processor:
            await self.processor.shutdown()

    async def _send_message(self, msg: dict[str, Any]) -> None:
        """Send message to parent."""
        if self._stdout_writer:
            data = orjson.dumps(msg) + b"\n"
            self._stdout_writer.write(data)
            await self._stdout_writer.drain()


async def main():
    parser = argparse.ArgumentParser(description="LMPU Worker")
    parser.add_argument("--processor", required=True, help="Processor name")
    parser.add_argument("--worker-id", required=True, help="Worker ID")
    parser.add_argument("--plugin", help="Plugin path")
    parser.add_argument("--socket", help="Unix socket path (unused, uses stdio)")

    args = parser.parse_args()

    worker = Worker(
        processor_name=args.processor,
        worker_id=args.worker_id,
        plugin_path=args.plugin,
    )

    try:
        await worker.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
