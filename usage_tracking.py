"""
usage_tracking.py — Auvia Engine
Lightweight anonymous usage gating for the free tier: a SQLite counter
keyed by a client-generated device ID (localStorage, sent as a header).

Known limitation: the database lives under TEMP_DIR, which on Cloud Run
is ephemeral local disk — it does NOT persist across cold starts, new
revisions, or scale-to-zero (min-instances=0 in this project's deploy
config). It's a real but soft gate, same spirit as an anonymous cookie,
not a hard enforcement mechanism. A durable store (Cloud SQL, Firestore)
is the correct fix once this needs to hold up against determined abuse
or real paying-customer guarantees.
"""

import datetime
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

FREE_TIER_LIMIT = 3

_DB_PATH = os.path.join(os.getenv("TEMP_DIR", "/tmp/auvia"), "usage.db")
_lock = threading.Lock()


def _init_db():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            device_id TEXT PRIMARY KEY,
            orchestrate_count INTEGER NOT NULL DEFAULT 0,
            is_pro INTEGER NOT NULL DEFAULT 0,
            plan TEXT,
            razorpay_subscription_id TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


_init_db()


@contextmanager
def _connect():
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def check_and_increment(device_id: str) -> dict:
    """
    Call before running a gated operation.
    Returns {"allowed": bool, "remaining": int | None, "is_pro": bool}.
    Pro users are never counted. Increments the counter only when allowed.
    """
    now = datetime.datetime.utcnow().isoformat()

    with _connect() as conn:
        row = conn.execute(
            "SELECT orchestrate_count, is_pro FROM usage WHERE device_id = ?", (device_id,)
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO usage (device_id, orchestrate_count, is_pro, updated_at) VALUES (?, 0, 0, ?)",
                (device_id, now),
            )
            count, is_pro = 0, 0
        else:
            count, is_pro = row

        if is_pro:
            return {"allowed": True, "remaining": None, "is_pro": True}

        if count >= FREE_TIER_LIMIT:
            return {"allowed": False, "remaining": 0, "is_pro": False}

        conn.execute(
            "UPDATE usage SET orchestrate_count = orchestrate_count + 1, updated_at = ? WHERE device_id = ?",
            (now, device_id),
        )
        return {"allowed": True, "remaining": FREE_TIER_LIMIT - count - 1, "is_pro": False}


def mark_pro(device_id: str, plan: str, subscription_id: str) -> None:
    """Called by the Razorpay webhook once a subscription is confirmed active."""
    now = datetime.datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO usage (device_id, orchestrate_count, is_pro, plan, razorpay_subscription_id, updated_at)
            VALUES (?, 0, 1, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                is_pro = 1,
                plan = excluded.plan,
                razorpay_subscription_id = excluded.razorpay_subscription_id,
                updated_at = excluded.updated_at
            """,
            (device_id, plan, subscription_id, now),
        )
    logger.info("[USAGE] device=%s upgraded to plan=%s (sub=%s)", device_id, plan, subscription_id)
