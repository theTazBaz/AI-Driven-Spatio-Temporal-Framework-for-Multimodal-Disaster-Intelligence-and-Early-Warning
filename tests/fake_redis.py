"""A tiny in-memory stand-in for redis-py (bytes in, bytes out), for unit tests only.

Implements just the commands ``HotStore`` uses. ``round_trips`` counts pipeline executions
so tests can pin that a batch costs a bounded number of network round trips.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator
from typing import Any


def _b(x: str | bytes) -> bytes:
    return x if isinstance(x, bytes) else x.encode("utf-8")


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.ttls: dict[str, int] = {}
        self.round_trips = 0

    # direct commands
    def hgetall(self, name: str) -> dict[bytes, bytes]:
        return dict(self.hashes.get(name, {}))

    def hmget(self, name: str, keys: Any, *args: Any) -> list[bytes | None]:
        table = self.hashes.get(name, {})
        return [table.get(_b(k)) for k in keys]

    def hset(self, name: str, key: Any = None, value: Any = None, mapping: Any = None) -> int:
        table = self.hashes.setdefault(name, {})
        items = dict(mapping or {})
        if key is not None:
            items[key] = value
        for field, blob in items.items():
            if not isinstance(blob, bytes):
                raise TypeError("FakeRedis only stores bytes values here")
            table[_b(field)] = blob
        return len(items)

    def hdel(self, name: str, *keys: Any) -> int:
        table = self.hashes.get(name, {})
        removed = sum(1 for k in keys if table.pop(_b(k), None) is not None)
        if name in self.hashes and not self.hashes[name]:
            del self.hashes[name]  # Redis removes empty hashes
            self.ttls.pop(name, None)
        return removed

    def expire(self, name: str, time: Any) -> bool:
        if name not in self.hashes:
            return False
        self.ttls[name] = int(time)
        return True

    def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            if self.hashes.pop(name, None) is not None:
                removed += 1
            self.ttls.pop(name, None)
        return removed

    def scan_iter(self, match: Any = None, count: Any = None) -> Iterator[bytes]:
        for name in list(self.hashes):
            if match is None or fnmatch.fnmatchcase(name, match):
                yield name.encode("utf-8")

    def memory_usage(self, key: str) -> int | None:
        table = self.hashes.get(key)
        if table is None:
            return None
        return sum(len(f) + len(v) for f, v in table.items())

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, parent: FakeRedis) -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _queue(self, op: str, *args: Any, **kwargs: Any) -> FakePipeline:
        self._ops.append((op, args, kwargs))
        return self

    def hmget(self, name: str, keys: Any, *args: Any) -> FakePipeline:
        return self._queue("hmget", name, keys, *args)

    def hgetall(self, name: str) -> FakePipeline:
        return self._queue("hgetall", name)

    def hset(
        self, name: str, key: Any = None, value: Any = None, mapping: Any = None
    ) -> FakePipeline:
        return self._queue("hset", name, key=key, value=value, mapping=mapping)

    def hdel(self, name: str, *keys: Any) -> FakePipeline:
        return self._queue("hdel", name, *keys)

    def expire(self, name: str, time: Any) -> FakePipeline:
        return self._queue("expire", name, time)

    def execute(self) -> list[Any]:
        self._parent.round_trips += 1
        results = [getattr(self._parent, op)(*args, **kwargs) for op, args, kwargs in self._ops]
        self._ops = []
        return results
