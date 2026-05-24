"""tool-parallel-exec-py — run non-conflicting agent tool calls concurrently."""

from __future__ import annotations

import asyncio
import enum
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


class SideEffect(enum.Enum):
    READ = "READ"
    IDEMPOTENT = "IDEMPOTENT"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"


# Attribute name stored on decorated functions
_SIDE_EFFECT_ATTR = "_tool_side_effect"

# Side effects that are safe to run in parallel with each other
_PARALLEL_SAFE = {SideEffect.READ, SideEffect.IDEMPOTENT}


def side_effect(effect: SideEffect) -> Callable:
    """Decorator that tags a tool function with its side-effect level."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)
        setattr(wrapper, _SIDE_EFFECT_ATTR, effect)
        return wrapper
    return decorator


def get_side_effect(fn: Callable) -> SideEffect:
    """Return the declared side effect, defaulting to WRITE if untagged."""
    return getattr(fn, _SIDE_EFFECT_ATTR, SideEffect.WRITE)


class ToolError(Exception):
    """Wraps an exception raised by a tool during execution."""
    def __init__(self, tool_name: str, original: Exception) -> None:
        self.tool_name = tool_name
        self.original = original
        super().__init__(f"Tool '{tool_name}' raised: {original!r}")


class ParallelExecutor:
    """
    Execute agent tool calls, running parallel-safe ones concurrently.

    Tool calls whose side effects are READ or IDEMPOTENT are grouped and
    executed with a ThreadPoolExecutor. WRITE and DESTRUCTIVE calls are
    serialized after the parallel batch.

    Example::

        @side_effect(SideEffect.READ)
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        executor = ParallelExecutor({"get_weather": get_weather})
        results = executor.run([{"name": "get_weather", "args": {"city": "Dallas"}}])
    """

    def __init__(
        self,
        tools: dict[str, Callable],
        max_workers: int = 8,
    ) -> None:
        self._tools = tools
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Sync execution
    # ------------------------------------------------------------------

    def run(self, tool_calls: list[dict]) -> list[dict]:
        """
        Execute tool_calls in order of safety (parallel-safe first, then serial).
        Returns results in the same order as tool_calls.
        """
        if not tool_calls:
            return []

        indexed = list(enumerate(tool_calls))
        parallel_batch = [(i, tc) for i, tc in indexed if self._is_parallel_safe(tc)]
        serial_batch = [(i, tc) for i, tc in indexed if not self._is_parallel_safe(tc)]

        results: dict[int, dict] = {}

        # Run parallel-safe batch concurrently
        if parallel_batch:
            with ThreadPoolExecutor(max_workers=min(self._max_workers, len(parallel_batch))) as pool:
                future_to_idx = {
                    pool.submit(self._call, tc): i
                    for i, tc in parallel_batch
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()  # propagates ToolError

        # Run serial batch in order
        for i, tc in serial_batch:
            results[i] = self._call(tc)

        return [results[i] for i in range(len(tool_calls))]

    # ------------------------------------------------------------------
    # Async execution
    # ------------------------------------------------------------------

    async def async_run(self, tool_calls: list[dict]) -> list[dict]:
        """Async version — runs parallel-safe tools in the default executor."""
        if not tool_calls:
            return []

        loop = asyncio.get_event_loop()
        indexed = list(enumerate(tool_calls))
        parallel_batch = [(i, tc) for i, tc in indexed if self._is_parallel_safe(tc)]
        serial_batch = [(i, tc) for i, tc in indexed if not self._is_parallel_safe(tc)]

        results: dict[int, dict] = {}

        if parallel_batch:
            coros = [
                loop.run_in_executor(None, self._call, tc)
                for _, tc in parallel_batch
            ]
            for (i, _), result in zip(parallel_batch, await asyncio.gather(*coros)):
                results[i] = result

        for i, tc in serial_batch:
            results[i] = await loop.run_in_executor(None, self._call, tc)

        return [results[i] for i in range(len(tool_calls))]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_parallel_safe(self, tool_call: dict) -> bool:
        name = tool_call.get("name", "")
        fn = self._tools.get(name)
        if fn is None:
            return False
        return get_side_effect(fn) in _PARALLEL_SAFE

    def _call(self, tool_call: dict) -> dict:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        fn = self._tools.get(name)
        if fn is None:
            raise ToolError(name, KeyError(f"Unknown tool: '{name}'"))
        try:
            result = fn(**args)
            return {"name": name, "result": result, "error": None}
        except Exception as exc:
            raise ToolError(name, exc) from exc


__all__ = [
    "SideEffect",
    "side_effect",
    "get_side_effect",
    "ParallelExecutor",
    "ToolError",
]
