"""LMPU Demos - Demo implementations."""

import asyncio
from typing import Any


async def run_demo(name: str, socket: str, gpu: bool = True) -> None:
    """Run a demo by name."""
    demos = {
        "01": demo_01_capability_query,
        "02": demo_02_sync_arithmetic,
        "03": demo_03_async_python_job,
        "04": demo_04_sleep_wake_matrix,
        "05": demo_05_large_buffer_read_slice,
        "06": demo_06_progress_and_cancel,
        "07": demo_07_needs_more_input,
        "08": demo_08_neural_summary,
        "09": demo_09_multi_processor_concurrent,
        "10": demo_10_batch_small_jobs,
        "11": demo_11_benchmark_vs_token_compute,
        "12": demo_12_mission_control,
    }
    
    if name == "all":
        for demo_name, demo_func in demos.items():
            print(f"\n=== Running Demo {demo_name} ===")
            await demo_func(socket, gpu)
    elif name in demos:
        await demos[name](socket, gpu)
    else:
        raise ValueError(f"Unknown demo: {name}")


async def demo_01_capability_query(socket: str, gpu: bool) -> None:
    """Demo 1: Capability query and routing."""
    from lmpu.kernel.runtime import Runtime
    from lmpu.protocol.codec import open_rpc_connection
    
    runtime = Runtime(socket_path=socket)
    await runtime.start()
    
    try:
        stream = await open_rpc_connection(socket)
        
        # Hello
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "lmpu.hello",
            "params": {"protocol": "lmpu/1"},
        })
        resp = await stream.recv()
        session_id = resp.result["session_id"]
        print(f"Session: {session_id}")
        
        # Capabilities
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "lmpu.capabilities",
            "params": {},
        })
        resp = await stream.recv()
        print("Processors:")
        for p in resp.result["processors"]:
            print(f"  - {p['name']} v{p['version']}: {[op['name'] for op in p['ops']]}")
        
    finally:
        await runtime.stop()


async def demo_02_sync_arithmetic(socket: str, gpu: bool) -> None:
    """Demo 2: Synchronous exact arithmetic."""
    from lmpu.kernel.runtime import Runtime
    from lmpu.protocol.codec import open_rpc_connection
    
    runtime = Runtime(socket_path=socket)
    await runtime.start()
    
    try:
        stream = await open_rpc_connection(socket)
        
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "lmpu.hello",
            "params": {"protocol": "lmpu/1"},
        })
        resp = await stream.recv()
        session_id = resp.result["session_id"]
        
        # Submit bigint multiplication
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "lmpu.submit",
            "params": {
                "processor": "pure.func",
                "op": "bigint_mul",
                "inputs": {"a": "12345678901234567890", "b": "98765432109876543210"},
                "config": {"mode": "sync"},
            },
        })
        resp = await stream.recv()
        print(f"Result: {resp.result}")
        
    finally:
        await runtime.stop()


async def demo_03_async_python_job(socket: str, gpu: bool) -> None:
    """Demo 3: Asynchronous Python job with poll/wait."""
    from lmpu.kernel.runtime import Runtime
    from lmpu.protocol.codec import open_rpc_connection
    
    runtime = Runtime(socket_path=socket)
    await runtime.start()
    
    try:
        stream = await open_rpc_connection(socket)
        
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "lmpu.hello",
            "params": {"protocol": "lmpu/1"},
        })
        resp = await stream.recv()
        session_id = resp.result["session_id"]
        
        # Submit Python job
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "lmpu.submit",
            "params": {
                "processor": "python.vm",
                "op": "run_program",
                "inputs": {
                    "program": "import time\ntime.sleep(2)\nresult = {'sum': sum(range(1000000))}\nfinish(result)"
                },
                "config": {"mode": "async", "timeout_ms": 10000},
            },
        })
        resp = await stream.recv()
        job_id = resp.result["job_id"]
        print(f"Job submitted: {job_id}")
        
        # Wait for completion
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "lmpu.wait",
            "params": {"job_id": job_id, "timeout_ms": 15000},
        })
        resp = await stream.recv()
        print(f"Result: {resp.result}")
        
    finally:
        await runtime.stop()


async def demo_04_sleep_wake_matrix(socket: str, gpu: bool) -> None:
    """Demo 4: Sleep/wake matrix multiplication."""
    print("Demo 4 requires GPU and HF engine - run via mission control")
    pass


async def demo_05_large_buffer_read_slice(socket: str, gpu: bool) -> None:
    """Demo 5: Large buffer partial read."""
    print("Demo 5 requires GPU - run via mission control")
    pass


async def demo_06_progress_and_cancel(socket: str, gpu: bool) -> None:
    """Demo 6: Progress, polling, cancellation."""
    print("Demo 6 requires GPU - run via mission control")
    pass


async def demo_07_needs_more_input(socket: str, gpu: bool) -> None:
    """Demo 7: Interactive NeedsMoreInput."""
    print("Demo 7 requires GPU - run via mission control")
    pass


async def demo_08_neural_summary(socket: str, gpu: bool) -> None:
    """Demo 8: Neural summarization."""
    print("Demo 8 requires GPU - run via mission control")
    pass


async def demo_09_multi_processor_concurrent(socket: str, gpu: bool) -> None:
    """Demo 9: Concurrent processors."""
    print("Demo 9 requires GPU - run via mission control")
    pass


async def demo_10_batch_small_jobs(socket: str, gpu: bool) -> None:
    """Demo 10: Batched small jobs."""
    from lmpu.kernel.runtime import Runtime
    from lmpu.protocol.codec import open_rpc_connection
    
    runtime = Runtime(socket_path=socket)
    await runtime.start()
    
    try:
        stream = await open_rpc_connection(socket)
        
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "lmpu.hello",
            "params": {"protocol": "lmpu/1"},
        })
        resp = await stream.recv()
        session_id = resp.result["session_id"]
        
        # Submit batch
        items = [{"inputs": {"a": i, "b": i+1}} for i in range(100)]
        await stream.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "lmpu.submit",
            "params": {
                "processor": "pure.func",
                "op": "bigint_add",
                "batch": {"items": items},
                "config": {"mode": "async"},
            },
        })
        resp = await stream.recv()
        print(f"Batch submitted: {resp.result}")
        
    finally:
        await runtime.stop()


async def demo_11_benchmark_vs_token_compute(socket: str, gpu: bool) -> None:
    """Demo 11: Benchmark vs token compute."""
    print("Demo 11 requires GPU - run via mission control")
    pass


async def demo_12_mission_control(socket: str, gpu: bool) -> None:
    """Demo 12: Mission Control dashboard."""
    from lmpu.observability.dashboard import start_dashboard_server
    from lmpu.kernel.runtime import Runtime
    
    runtime = Runtime(socket_path=socket)
    await runtime.start()
    
    try:
        await start_dashboard_server(runtime, "127.0.0.1", 8080)
    finally:
        await runtime.stop()