"""Increment 12 - Redpanda / Kafka streaming bus implementation.

Features:
- Full implementation of the `Bus` interface backed by Redpanda (Kafka protocol).
- Message serialization/deserialization for Observations, dicts, strings, and bytes.
- Topic retention configuration enforcing the 24-72 hour storage budget policy.
- Bounded queue backpressure via BufferFullError.
- Pluggable transport architecture supporting RealKafkaTransport (real Redpanda cluster)
  and FakeKafkaTransport (isolated unit tests with pytest-socket).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from disaster_intel.bus.base import (
    BufferFullError,
    Bus,
    Message,
)
from disaster_intel.models import Observation

if TYPE_CHECKING:
    pass

MIN_RETENTION_HOURS = 24
MAX_RETENTION_HOURS = 72
DEFAULT_RETENTION_HOURS = 72
CONTENT_TYPE_HEADER = "content-type"
OBSERVATION_CONTENT_TYPE = "application/json; type=Observation"
JSON_CONTENT_TYPE = "application/json"
TEXT_CONTENT_TYPE = "text/plain"
OCTET_CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class KafkaRecord:
    """Internal low-level Kafka record representation."""

    topic: str
    key: bytes
    value: bytes
    offset: int
    timestamp_ms: int
    headers: list[tuple[str, bytes]] = field(default_factory=list)


def serialize_payload(
    payload: Any, headers: dict[str, str] | None = None
) -> tuple[bytes, dict[str, str]]:
    """Serialize a payload object into bytes and attach metadata headers."""
    final_headers = dict(headers or {})

    if isinstance(payload, Observation):
        raw_bytes = payload.to_json().encode("utf-8")
        final_headers[CONTENT_TYPE_HEADER] = OBSERVATION_CONTENT_TYPE
    elif isinstance(payload, (dict, list)):
        raw_bytes = json.dumps(payload).encode("utf-8")
        final_headers[CONTENT_TYPE_HEADER] = JSON_CONTENT_TYPE
    elif isinstance(payload, str):
        raw_bytes = payload.encode("utf-8")
        final_headers[CONTENT_TYPE_HEADER] = TEXT_CONTENT_TYPE
    elif isinstance(payload, (bytes, bytearray)):
        raw_bytes = bytes(payload)
        final_headers[CONTENT_TYPE_HEADER] = OCTET_CONTENT_TYPE
    else:
        # Default JSON dump for primitives or objects
        raw_bytes = json.dumps(payload).encode("utf-8")
        final_headers[CONTENT_TYPE_HEADER] = JSON_CONTENT_TYPE

    return raw_bytes, final_headers


def deserialize_payload(raw_bytes: bytes, headers: dict[str, str]) -> Any:
    """Deserialize raw bytes back into an object or Observation based on headers."""
    content_type = headers.get(CONTENT_TYPE_HEADER, "")

    if content_type == OBSERVATION_CONTENT_TYPE:
        data = json.loads(raw_bytes.decode("utf-8"))
        return Observation.from_dict(data)

    if content_type == JSON_CONTENT_TYPE:
        return json.loads(raw_bytes.decode("utf-8"))

    if content_type == TEXT_CONTENT_TYPE:
        return raw_bytes.decode("utf-8")

    if content_type == OCTET_CONTENT_TYPE:
        return raw_bytes

    # Fallback heuristic
    try:
        text = raw_bytes.decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw_bytes


class KafkaTransport(Protocol):
    """Protocol defining low-level transport operations for RedpandaBus."""

    def send(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        headers: list[tuple[str, bytes]],
        timestamp_ms: int,
    ) -> int:
        ...

    def poll(
        self,
        topic: str,
        group_id: str,
        max_messages: int,
        auto_commit: bool,
    ) -> list[KafkaRecord]:
        ...

    def commit(
        self,
        topic: str,
        group_id: str,
        offset: int | None = None,
    ) -> None:
        ...

    def seek(
        self,
        topic: str,
        group_id: str,
        offset: int,
    ) -> None:
        ...

    def get_offset(
        self,
        topic: str,
        group_id: str,
    ) -> int:
        ...

    def ensure_topic(
        self,
        topic: str,
        retention_ms: int,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class FakeKafkaTransport:
    """In-memory Kafka transport emulator for unit testing without sockets."""

    def __init__(self, max_queue_size: int | None = None) -> None:
        self.max_queue_size = max_queue_size
        self._topics: dict[str, list[KafkaRecord]] = defaultdict(list)
        self._topic_retention: dict[str, int] = {}
        # (topic, group_id) -> next offset to poll from
        self._group_positions: dict[tuple[str, str], int] = {}
        # (topic, group_id) -> last committed offset
        self._group_committed: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    def ensure_topic(self, topic: str, retention_ms: int) -> None:
        with self._lock:
            self._topic_retention[topic] = retention_ms
            if topic not in self._topics:
                self._topics[topic] = []

    def send(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        headers: list[tuple[str, bytes]],
        timestamp_ms: int,
    ) -> int:
        with self._lock:
            topic_records = self._topics[topic]
            if self.max_queue_size is not None and len(topic_records) >= self.max_queue_size:
                raise BufferFullError(
                    f"Topic '{topic}' buffer is full: capacity is {self.max_queue_size}"
                )

            offset = len(topic_records)
            rec = KafkaRecord(
                topic=topic,
                key=key,
                value=value,
                offset=offset,
                timestamp_ms=timestamp_ms,
                headers=headers,
            )
            topic_records.append(rec)
            return offset

    def poll(
        self,
        topic: str,
        group_id: str,
        max_messages: int,
        auto_commit: bool,
    ) -> list[KafkaRecord]:
        with self._lock:
            if topic not in self._topics:
                return []

            topic_records = self._topics[topic]
            group_key = (topic, group_id)
            pos = self._group_positions.get(group_key, 0)
            if pos >= len(topic_records):
                return []

            batch = topic_records[pos : pos + max_messages]
            self._group_positions[group_key] = pos + len(batch)

            if auto_commit and batch:
                self._group_committed[group_key] = batch[-1].offset

            return list(batch)

    def commit(
        self,
        topic: str,
        group_id: str,
        offset: int | None = None,
    ) -> None:
        with self._lock:
            group_key = (topic, group_id)
            if offset is not None:
                self._group_committed[group_key] = offset
                self._group_positions[group_key] = offset + 1
            else:
                current_pos = self._group_positions.get(group_key, 0)
                if current_pos > 0:
                    self._group_committed[group_key] = current_pos - 1

    def seek(
        self,
        topic: str,
        group_id: str,
        offset: int,
    ) -> None:
        with self._lock:
            if offset < 0:
                raise ValueError(f"Offset cannot be negative: {offset}")
            group_key = (topic, group_id)
            self._group_positions[group_key] = offset

    def get_offset(
        self,
        topic: str,
        group_id: str,
    ) -> int:
        with self._lock:
            return self._group_committed.get((topic, group_id), -1)

    def close(self) -> None:
        with self._lock:
            self._topics.clear()
            self._group_positions.clear()
            self._group_committed.clear()


class RealKafkaTransport:
    """Real network transport for Redpanda using kafka-python-ng."""

    def __init__(
        self,
        bootstrap_servers: str | list[str] = "localhost:9092",
        client_id: str = "disaster_intel_bus",
    ) -> None:
        from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
        from kafka.admin import NewTopic

        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._producer: KafkaProducer | None = None
        self._consumers: dict[tuple[str, str], KafkaConsumer] = {}
        self._lock = threading.RLock()
        self._KafkaAdminClient = KafkaAdminClient
        self._KafkaConsumer = KafkaConsumer
        self._KafkaProducer = KafkaProducer
        self._NewTopic = NewTopic

    def _get_producer(self) -> Any:
        if self._producer is None:
            self._producer = self._KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                acks="all",
                retries=3,
            )
        return self._producer

    def _get_consumer(self, topic: str, group_id: str, auto_commit: bool) -> Any:
        key = (topic, group_id)
        if key not in self._consumers:
            consumer = self._KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                client_id=f"{self.client_id}_{group_id}",
                enable_auto_commit=auto_commit,
                auto_offset_reset="earliest",
            )
            self._consumers[key] = consumer
        return self._consumers[key]

    def ensure_topic(self, topic: str, retention_ms: int) -> None:
        with self._lock:
            try:
                admin = self._KafkaAdminClient(
                    bootstrap_servers=self.bootstrap_servers,
                    client_id=f"{self.client_id}_admin",
                )
                try:
                    new_topic = self._NewTopic(
                        name=topic,
                        num_partitions=1,
                        replication_factor=1,
                        topic_configs={"retention.ms": str(retention_ms)},
                    )
                    admin.create_topics([new_topic])
                finally:
                    admin.close()
            except Exception:
                # Topic may already exist; ignore
                pass

    def send(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        headers: list[tuple[str, bytes]],
        timestamp_ms: int,
    ) -> int:
        producer = self._get_producer()
        future = producer.send(
            topic=topic,
            key=key,
            value=value,
            headers=headers,
            timestamp_ms=timestamp_ms,
        )
        record_metadata = future.get(timeout=10.0)
        return int(record_metadata.offset)

    def poll(
        self,
        topic: str,
        group_id: str,
        max_messages: int,
        auto_commit: bool,
    ) -> list[KafkaRecord]:
        with self._lock:
            consumer = self._get_consumer(topic, group_id, auto_commit)
            records_dict = consumer.poll(timeout_ms=500, max_records=max_messages)
            records: list[KafkaRecord] = []
            for tp, tp_records in records_dict.items():
                if tp.topic == topic:
                    for r in tp_records:
                        records.append(
                            KafkaRecord(
                                topic=r.topic,
                                key=r.key or b"",
                                value=r.value or b"",
                                offset=r.offset,
                                timestamp_ms=r.timestamp,
                                headers=[(k, v) for k, v in (r.headers or [])],
                            )
                        )
            return records

    def commit(
        self,
        topic: str,
        group_id: str,
        offset: int | None = None,
    ) -> None:
        with self._lock:
            key = (topic, group_id)
            if key in self._consumers:
                consumer = self._consumers[key]
                if offset is None:
                    consumer.commit()
                else:
                    from kafka import OffsetAndMetadata, TopicPartition

                    tp = TopicPartition(topic, 0)
                    consumer.commit({tp: OffsetAndMetadata(offset + 1, "")})

    def seek(
        self,
        topic: str,
        group_id: str,
        offset: int,
    ) -> None:
        with self._lock:
            consumer = self._get_consumer(topic, group_id, auto_commit=False)
            from kafka import TopicPartition

            tp = TopicPartition(topic, 0)
            consumer.seek(tp, offset)

    def get_offset(
        self,
        topic: str,
        group_id: str,
    ) -> int:
        with self._lock:
            key = (topic, group_id)
            if key in self._consumers:
                from kafka import TopicPartition

                tp = TopicPartition(topic, 0)
                comm = self._consumers[key].committed(tp)
                return (comm - 1) if comm is not None else -1
            return -1

    def close(self) -> None:
        with self._lock:
            if self._producer is not None:
                with contextlib.suppress(Exception):
                    self._producer.close()
                self._producer = None
            for consumer in self._consumers.values():
                with contextlib.suppress(Exception):
                    consumer.close()
            self._consumers.clear()


class RedpandaBus(Bus):
    """High-throughput Kafka-compatible bus implementation for Redpanda.

    Enforces:
    - Ordered partitions per key (cell)
    - Independent consumer-group offsets and crash recovery
    - 24-72 hour storage retention budget
    - Transparent Observation and JSON serialization
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] = "localhost:9092",
        client_id: str = "disaster_intel_bus",
        retention_hours: int = DEFAULT_RETENTION_HOURS,
        max_queue_size: int | None = None,
        test_mode: bool = False,
        transport: KafkaTransport | None = None,
    ) -> None:
        if not (MIN_RETENTION_HOURS <= retention_hours <= MAX_RETENTION_HOURS):
            raise ValueError(
                f"Topic retention must be between {MIN_RETENTION_HOURS} and "
                f"{MAX_RETENTION_HOURS} hours (storage budget enforcement), got {retention_hours}"
            )

        self.retention_hours = retention_hours
        self.retention_ms = retention_hours * 3600 * 1000
        self.max_queue_size = max_queue_size

        if transport is not None:
            self._transport = transport
        elif test_mode or os.environ.get("DISASTER_INTEL_TEST_BUS") == "1":
            self._transport = FakeKafkaTransport(max_queue_size=max_queue_size)
        else:
            self._transport = RealKafkaTransport(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
            )

    def publish(
        self,
        topic: str,
        key: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> int:
        value_bytes, final_headers = serialize_payload(payload, headers)
        key_bytes = key.encode("utf-8")
        kafka_headers = [(k, v.encode("utf-8")) for k, v in final_headers.items()]
        now_ms = int(datetime.now(UTC).timestamp() * 1000)

        self._transport.ensure_topic(topic, self.retention_ms)
        return self._transport.send(
            topic=topic,
            key=key_bytes,
            value=value_bytes,
            headers=kafka_headers,
            timestamp_ms=now_ms,
        )

    def publish_batch(
        self,
        topic: str,
        messages: Sequence[tuple[str, Any]],
        headers: dict[str, str] | None = None,
    ) -> list[int]:
        self._transport.ensure_topic(topic, self.retention_ms)
        offsets: list[int] = []
        for key, payload in messages:
            offset = self.publish(topic=topic, key=key, payload=payload, headers=headers)
            offsets.append(offset)
        return offsets

    def poll(
        self,
        topic: str,
        group_id: str,
        max_messages: int = 100,
        auto_commit: bool = True,
    ) -> list[Message]:
        records = self._transport.poll(
            topic=topic,
            group_id=group_id,
            max_messages=max_messages,
            auto_commit=auto_commit,
        )

        messages: list[Message] = []
        for r in records:
            headers_dict = {
                k: (v.decode("utf-8") if isinstance(v, bytes) else str(v))
                for k, v in r.headers
            }
            payload = deserialize_payload(r.value, headers_dict)
            ts = datetime.fromtimestamp(r.timestamp_ms / 1000.0, tz=UTC)
            msg = Message(
                topic=r.topic,
                key=r.key.decode("utf-8") if isinstance(r.key, bytes) else str(r.key),
                payload=payload,
                offset=r.offset,
                timestamp=ts,
                headers=headers_dict,
            )
            messages.append(msg)

        return messages

    def commit(
        self,
        topic: str,
        group_id: str,
        offset: int | None = None,
    ) -> None:
        self._transport.commit(topic=topic, group_id=group_id, offset=offset)

    def seek(
        self,
        topic: str,
        group_id: str,
        offset: int,
    ) -> None:
        self._transport.seek(topic=topic, group_id=group_id, offset=offset)

    def get_offset(
        self,
        topic: str,
        group_id: str,
    ) -> int:
        return self._transport.get_offset(topic=topic, group_id=group_id)

    def close(self) -> None:
        self._transport.close()
