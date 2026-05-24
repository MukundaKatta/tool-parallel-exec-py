import asyncio
import time
import pytest
from tool_parallel_exec import ParallelExecutor, SideEffect, side_effect, get_side_effect, ToolError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@side_effect(SideEffect.READ)
def read_tool(key: str) -> str:
    return f"read:{key}"


@side_effect(SideEffect.IDEMPOTENT)
def idempotent_tool(value: int) -> int:
    return value * 2


@side_effect(SideEffect.WRITE)
def write_tool(data: str) -> str:
    return f"wrote:{data}"


@side_effect(SideEffect.DESTRUCTIVE)
def delete_tool(item: str) -> str:
    return f"deleted:{item}"


def plain_tool(x: int) -> int:
    return x + 1


TOOLS = {
    "read": read_tool,
    "idempotent": idempotent_tool,
    "write": write_tool,
    "delete": delete_tool,
    "plain": plain_tool,
}


# ---------------------------------------------------------------------------
# side_effect decorator
# ---------------------------------------------------------------------------

def test_side_effect_decorator_sets_attr():
    assert get_side_effect(read_tool) == SideEffect.READ


def test_side_effect_preserves_fn_behavior():
    assert read_tool(key="x") == "read:x"


def test_side_effect_preserves_name():
    assert read_tool.__name__ == "read_tool"


def test_plain_fn_defaults_to_write():
    assert get_side_effect(plain_tool) == SideEffect.WRITE


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

def test_single_read_call():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([{"name": "read", "args": {"key": "hello"}}])
    assert results[0]["result"] == "read:hello"
    assert results[0]["error"] is None


def test_single_write_call():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([{"name": "write", "args": {"data": "abc"}}])
    assert results[0]["result"] == "wrote:abc"


def test_empty_call_list():
    ex = ParallelExecutor(TOOLS)
    assert ex.run([]) == []


def test_result_order_matches_input():
    ex = ParallelExecutor(TOOLS)
    calls = [
        {"name": "read", "args": {"key": "a"}},
        {"name": "read", "args": {"key": "b"}},
        {"name": "read", "args": {"key": "c"}},
    ]
    results = ex.run(calls)
    assert results[0]["result"] == "read:a"
    assert results[1]["result"] == "read:b"
    assert results[2]["result"] == "read:c"


def test_result_name_field():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([{"name": "read", "args": {"key": "x"}}])
    assert results[0]["name"] == "read"


# ---------------------------------------------------------------------------
# Parallel execution (timing)
# ---------------------------------------------------------------------------

@side_effect(SideEffect.READ)
def slow_read(ms: int) -> str:
    time.sleep(ms / 1000)
    return f"done:{ms}"


def test_parallel_reads_faster_than_serial():
    tools = {"slow": slow_read}
    ex = ParallelExecutor(tools)
    calls = [{"name": "slow", "args": {"ms": 100}} for _ in range(4)]

    start = time.monotonic()
    ex.run(calls)
    elapsed = time.monotonic() - start

    # 4x 100ms in parallel should finish well under 400ms
    assert elapsed < 0.38


# ---------------------------------------------------------------------------
# WRITE / DESTRUCTIVE serialized
# ---------------------------------------------------------------------------

def test_write_calls_are_executed():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([
        {"name": "write", "args": {"data": "x"}},
        {"name": "write", "args": {"data": "y"}},
    ])
    assert results[0]["result"] == "wrote:x"
    assert results[1]["result"] == "wrote:y"


def test_destructive_call_executed():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([{"name": "delete", "args": {"item": "obj"}}])
    assert results[0]["result"] == "deleted:obj"


# ---------------------------------------------------------------------------
# Mixed batch
# ---------------------------------------------------------------------------

def test_mixed_read_and_write():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([
        {"name": "read", "args": {"key": "r1"}},
        {"name": "write", "args": {"data": "w1"}},
        {"name": "read", "args": {"key": "r2"}},
    ])
    assert results[0]["result"] == "read:r1"
    assert results[1]["result"] == "wrote:w1"
    assert results[2]["result"] == "read:r2"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_tool_raises_tool_error():
    ex = ParallelExecutor(TOOLS)
    with pytest.raises(ToolError) as exc_info:
        ex.run([{"name": "nonexistent", "args": {}}])
    assert "nonexistent" in str(exc_info.value)


@side_effect(SideEffect.WRITE)
def bad_tool() -> None:
    raise ValueError("boom")


def test_tool_exception_raises_tool_error():
    ex = ParallelExecutor({"bad": bad_tool})
    with pytest.raises(ToolError) as exc_info:
        ex.run([{"name": "bad", "args": {}}])
    assert exc_info.value.tool_name == "bad"
    assert isinstance(exc_info.value.original, ValueError)


# ---------------------------------------------------------------------------
# Idempotent treated as parallel-safe
# ---------------------------------------------------------------------------

def test_idempotent_treated_as_parallel():
    ex = ParallelExecutor(TOOLS)
    results = ex.run([
        {"name": "idempotent", "args": {"value": 3}},
        {"name": "idempotent", "args": {"value": 5}},
    ])
    assert results[0]["result"] == 6
    assert results[1]["result"] == 10


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------

def test_async_run_single_call():
    ex = ParallelExecutor(TOOLS)

    async def _run():
        return await ex.async_run([{"name": "read", "args": {"key": "async"}}])

    results = asyncio.run(_run())
    assert results[0]["result"] == "read:async"


def test_async_run_empty():
    ex = ParallelExecutor(TOOLS)

    async def _run():
        return await ex.async_run([])

    assert asyncio.run(_run()) == []


def test_async_run_multiple_reads():
    ex = ParallelExecutor(TOOLS)

    async def _run():
        return await ex.async_run([
            {"name": "read", "args": {"key": "a"}},
            {"name": "read", "args": {"key": "b"}},
        ])

    results = asyncio.run(_run())
    assert results[0]["result"] == "read:a"
    assert results[1]["result"] == "read:b"
