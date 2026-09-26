"""telephony/latency_events.py — fire-and-forget event emission to
voice-latency-harness's /webhook/auvia (see AuviaAdapter in that repo for
the receiving side: payload shape, HMAC scheme, and the full canonical/raw
event vocabulary this module sends).

Fire-and-forget is a hard requirement, not a style choice: this runs on the
same event loop as the live Twilio media stream. A slow or dead harness
must never add latency to the call itself — that would corrupt the exact
numbers this instrumentation exists to measure. emit_event() schedules the
POST as a background asyncio.Task and returns immediately without awaiting
it; every failure (harness unreachable, timeout, non-2xx, missing config)
is caught inside the task and logged, never raised or awaited by the
caller.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 2.0

# Strong references to in-flight background tasks — required so they aren't
# garbage-collected mid-flight (asyncio only holds a weak reference once you
# stop holding the Task yourself). Discarded via the done-callback below.
_background_tasks: set[asyncio.Task] = set()


def emit_event(call_id: str, event_name: str, ts: datetime | None = None) -> None:
    """Schedules a harness webhook POST and returns immediately — never
    awaits the network call, never raises. `ts` lets the caller report an
    event's real historical instant (e.g. a TTS first-byte timestamp
    recovered from LatencyTracker.marks_wall_clock after the fact) rather
    than the moment emission happened to be scheduled; defaults to now."""
    event_ts = ts or datetime.now(timezone.utc)
    task = asyncio.create_task(_post_event(call_id, event_name, event_ts))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _post_event(call_id: str, event_name: str, ts: datetime) -> None:
    base_url = os.getenv("LATENCY_HARNESS_URL")
    secret = os.getenv("AUVIA_WEBHOOK_SECRET")
    if not base_url or not secret:
        logger.warning(
            "[LATENCY-HARNESS] not configured (LATENCY_HARNESS_URL/AUVIA_WEBHOOK_SECRET "
            "unset) — dropping event %r for call %s", event_name, call_id,
        )
        return

    body = json.dumps({"event": event_name, "call_id": call_id, "ts": ts.isoformat()}).encode()
    timestamp = str(int(time.time()))
    signature = "sha256=" + hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/webhook/auvia",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Auvia-Signature": signature,
                    "X-Auvia-Timestamp": timestamp,
                },
            )
            response.raise_for_status()
    except Exception as e:
        logger.warning("[LATENCY-HARNESS] failed to emit %r for call %s: %s", event_name, call_id, e)
