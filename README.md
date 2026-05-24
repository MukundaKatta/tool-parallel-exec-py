# tool-parallel-exec-py

Execute non-conflicting agent tool calls concurrently based on side-effect tags.

```bash
pip install tool-parallel-exec-py
```

## Quick start

```python
from tool_parallel_exec import ParallelExecutor, SideEffect, side_effect

@side_effect(SideEffect.READ)
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

@side_effect(SideEffect.READ)
def get_news(topic: str) -> str:
    return f"Latest {topic} news"

executor = ParallelExecutor({"get_weather": get_weather, "get_news": get_news})

# READ calls run concurrently
results = executor.run([
    {"name": "get_weather", "args": {"city": "Dallas"}},
    {"name": "get_news", "args": {"topic": "AI"}},
])
# [{"name": "get_weather", "result": "Sunny in Dallas", "error": None}, ...]
```

## Side effect levels

| Tag | Behavior |
|-----|----------|
| `READ` | Parallel with other READ/IDEMPOTENT |
| `IDEMPOTENT` | Parallel with other READ/IDEMPOTENT |
| `WRITE` | Serialized (one at a time) |
| `DESTRUCTIVE` | Serialized (one at a time) |

Untagged functions default to `WRITE`.

## Async

```python
results = await executor.async_run(tool_calls)
```

## API

```python
@side_effect(SideEffect.READ | .IDEMPOTENT | .WRITE | .DESTRUCTIVE)

ParallelExecutor(tools: dict[str, callable], max_workers: int = 8)
executor.run(tool_calls: list[dict]) -> list[dict]
executor.async_run(tool_calls: list[dict]) -> list[dict]   # awaitable
```

Each tool call: `{"name": str, "args": dict}`. Each result: `{"name": str, "result": Any, "error": None}`.

## Zero dependencies
