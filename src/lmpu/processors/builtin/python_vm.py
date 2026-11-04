"""LMPU Processors - Python VM processor for running Python code."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import textwrap
from typing import Any

import numpy as np

from ...memory.checksum import compute_checksum
from ..base import JobContext, Manifest, Processor, ProcessorContext, Result


class PythonVMProcessor(Processor):
    """Python VM processor for executing Python programs."""

    manifest = Manifest(
        name="python.vm",
        version="0.1.0",
        protocol="lmpu/1",
        ops=[
            {"name": "run_program", "inputs": ["program", "state"], "outputs": ["result", "state"]},
            {"name": "run_file", "inputs": ["path", "state"], "outputs": ["result", "state"]},
        ],
        limits={
            "max_buffer_bytes": 100_000_000,
            "max_batch": 1,
        },
        determinism=["best_effort"],
        latency_class="s",
        streaming=False,
        progress=True,
        interactive=True,
        persistent_state=True,
        batching=False,
        zero_copy=False,
        sandbox=["subprocess", "nsjail", "docker"],
        executor="process",
    )

    def __init__(self):
        super().__init__()
        self._helper_code = self._get_helper_code()

    def _get_helper_code(self) -> str:
        """Get the helper module code injected into user programs."""
        return textwrap.dedent('''
        """LMPU Python VM Helper Module"""
        import numpy as np
        import json
        
        # These will be injected by the runtime
        _buffers = {}
        _progress_callback = None
        _needs_input_callback = None
        _log_callback = None
        
        def read_buffer(name: str):
            """Read a buffer by name. Returns numpy array."""
            if name not in _buffers:
                raise KeyError(f"Buffer '{name}' not found")
            return _buffers[name]
        
        def write_buffer(name: str, data):
            """Write data to a buffer by name."""
            if name not in _buffers:
                raise KeyError(f"Buffer '{name}' not found")
            buf = _buffers[name]
            if hasattr(buf, 'shape') and hasattr(data, 'shape'):
                if buf.shape != data.shape:
                    raise ValueError(f"Shape mismatch: {buf.shape} vs {data.shape}")
            _buffers[name][...] = data
        
        def progress(fraction: float, message: str = None):
            """Report progress (0.0 to 1.0)."""
            if _progress_callback:
                _progress_callback(fraction, message)
        
        def request_input(fields: list, prompt: str):
            """Request interactive input from host."""
            if _needs_input_callback:
                _needs_input_callback(fields, prompt)
        
        def finish(result: dict):
            """Finish with result dict."""
            return {"result": result}
        
        def log(level: str, message: str, **kwargs):
            """Log a message."""
            if _log_callback:
                _log_callback(level, message, **kwargs)
        
        # Exported API
        __all__ = ['read_buffer', 'write_buffer', 'progress', 'request_input', 'finish', 'log']
        ''')

    async def setup(self, ctx: ProcessorContext) -> None:
        pass

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        op = job.get("op")
        inputs = job.get("inputs", {})
        config = job.get("config", {})

        timeout = config.get("timeout_ms", 30000) / 1000

        try:
            if op == "run_program":
                return await self._run_program(inputs, ctx, timeout)
            elif op == "run_file":
                return await self._run_file(inputs, ctx, timeout)
            else:
                return Result(
                    success=False,
                    error_code="unknown_op",
                    error_message=f"Unknown operation: {op}",
                )
        except TimeoutError:
            return Result(
                success=False,
                error_code="timeout",
                error_message=f"Execution timed out after {timeout}s",
                retriable=True,
            )
        except Exception as e:
            return Result(
                success=False,
                error_code="execution_error",
                error_message=str(e),
                retriable=False,
            )

    async def _run_program(
        self,
        inputs: dict[str, Any],
        ctx: JobContext,
        timeout: float,
    ) -> Result:
        program = inputs.get("program", "")
        state_buffer = inputs.get("state")

        ctx.progress(0.05, "preparing sandbox")

        # Prepare execution environment
        with tempfile.TemporaryDirectory() as tmpdir:
            boot = self._write_buffer_files(ctx, tmpdir)

            # Write program to file
            program_path = os.path.join(tmpdir, "program.py")
            with open(program_path, "w") as f:
                f.write(self._helper_code + "\n\n" + boot + "\n\n" + program)

            # Execute in subprocess
            result = await self._execute_in_sandbox(
                program_path,
                {},
                ctx,
                timeout,
            )

        ctx.progress(1.0, "finished")
        return result

    def _write_buffer_files(self, ctx: JobContext, tmpdir: str) -> str:
        """Stage job buffers as .npy files and return boot code that maps them.

        The sandbox is a separate process, so buffers cannot be shared in memory
        without handing over descriptors. Staging keeps isolation and lets user
        programs call ``read_buffer(name)``, which returns a read-only memmap.
        """
        entries = []
        for idx, (name, arr) in enumerate(getattr(ctx, "_buffers", {}).items()):
            if not isinstance(arr, np.ndarray):
                continue
            path = os.path.join(tmpdir, f"buf_{idx}.npy")
            np.save(path, arr, allow_pickle=False)
            entries.append((str(name), path))

        if not entries:
            return ""
        items = ",\n    ".join(
            f"{name!r}: np.load({path!r}, mmap_mode='r')" for name, path in entries
        )
        return f"_buffers.update({{\n    {items},\n}})"

    async def _run_file(
        self,
        inputs: dict[str, Any],
        ctx: JobContext,
        timeout: float,
    ) -> Result:
        path = inputs.get("path", "")
        state_buffer = inputs.get("state")

        if not os.path.exists(path):
            return Result(
                success=False,
                error_code="file_not_found",
                error_message=f"File not found: {path}",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy file to temp dir
            import shutil

            dest = os.path.join(tmpdir, "program.py")
            shutil.copy2(path, dest)
            boot = self._write_buffer_files(ctx, tmpdir)

            # Prepend helper
            with open(dest) as f:
                content = f.read()
            with open(dest, "w") as f:
                f.write(self._helper_code + "\n\n" + boot + "\n\n" + content)

            result = await self._execute_in_sandbox(
                dest,
                {},
                ctx,
                timeout,
            )

        return result

    async def _execute_in_sandbox(
        self,
        program_path: str,
        buffer_data: dict[str, np.ndarray],
        ctx: JobContext,
        timeout: float,
    ) -> Result:
        """Execute program in sandboxed subprocess."""

        # Build command with resource limits
        cmd = [
            sys.executable,
            "-I",  # Isolated mode
            "-c",
            f"""
import sys
sys.path.insert(0, '{os.path.dirname(program_path)}')
exec(open('{program_path}').read())
""",
        ]

        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        env["LMPU_JOB_ID"] = ctx.job_id
        env["LMPU_SESSION_ID"] = ctx.session_id

        # Create process
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(program_path),
            start_new_session=True,
            # Resource limits would be set via preexec_fn
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            return Result(
                success=False,
                error_code="execution_failed",
                error_message=stderr.decode()[:1000],
                retriable=False,
            )

        # Parse result from stdout (last line should be JSON)
        output = stdout.decode().strip()
        try:
            result_data = json.loads(output.split("\n")[-1])
        except json.JSONDecodeError:
            result_data = {"output": output}

        return Result(
            success=True,
            data=result_data,
            checksum=compute_checksum(json.dumps(result_data, sort_keys=True).encode()),
        )

    async def cancel(self, job_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass
