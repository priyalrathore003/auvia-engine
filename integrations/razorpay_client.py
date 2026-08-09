"""
integrations/razorpay_client.py
Thin wrapper around the Razorpay Subscriptions API — creates subscriptions
and verifies webhook signatures. No SDK, just httpx + hmac.
"""

import hashlib
import hmac
import logging
import os

import httpx

logger = logging.getLogger(__name__)

RAZORPAY_API_BASE = "https://api.razorpay.com/v1"

# billing="monthly" uses the plan IDs from Task 4. billing="annual" is an
# extension point: unset by default (no annual plans were created), and
# create_subscription() below falls back to monthly rather than silently
# mismatching what the pricing page displays vs what Razorpay actually bills.
PLAN_ENV_VARS = {
    ("pro", "monthly"): "RAZORPAY_PRO_PLAN_ID",
    ("pro", "annual"): "RAZORPAY_PRO_ANNUAL_PLAN_ID",
    ("studio", "monthly"): "RAZORPAY_STUDIO_PLAN_ID",
    ("studio", "annual"): "RAZORPAY_STUDIO_ANNUAL_PLAN_ID",
}


def _auth() -> tuple[str, str]:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise RuntimeError("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET is not set")
    return (key_id, key_secret)


def create_subscription(
    plan: str, device_id: str, billing: str = "monthly", total_count: int = 120
) -> dict:
    """
    Creates a Razorpay subscription for "pro" or "studio", tagging it with
    the anonymous device_id (via notes) so the webhook can identify which
    device to upgrade once payment is confirmed.

    billing="annual" falls back to the monthly plan (with a logged warning)
    if no annual plan ID has been configured yet, rather than failing the
    checkout outright — Razorpay's own hosted checkout page still shows the
    real amount before any card is charged either way.

    Returns {"subscription_id": str, "checkout_url": str, "billing": str}.
    """
    if plan not in ("pro", "studio"):
        raise ValueError(f"Unknown plan: {plan}")

    plan_id = os.getenv(PLAN_ENV_VARS.get((plan, billing), ""))
    actual_billing = billing

    if not plan_id and billing == "annual":
        logger.warning(
            "[RAZORPAY] no annual plan configured for %s — falling back to monthly", plan
        )
        plan_id = os.getenv(PLAN_ENV_VARS[(plan, "monthly")])
        actual_billing = "monthly"

    if not plan_id:
        raise RuntimeError(f"No Razorpay plan ID configured for plan={plan} billing={billing}")

    try:
        response = httpx.post(
            f"{RAZORPAY_API_BASE}/subscriptions",
            auth=_auth(),
            json={
                "plan_id": plan_id,
                "total_count": total_count,
                "customer_notify": 1,
                "notes": {"device_id": device_id, "plan": plan, "billing": actual_billing},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "subscription_id": data["id"],
            "checkout_url": data["short_url"],
            "billing": actual_billing,
        }

    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300]
        raise RuntimeError(f"Razorpay API error {e.response.status_code}: {detail}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Razorpay API request failed: {e}") from e


def verify_webhook_signature(raw_body: bytes, signature: str) -> bool:
    """HMAC-SHA256 over the raw request body, keyed by the webhook secret."""
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("[RAZORPAY] RAZORPAY_WEBHOOK_SECRET not set — rejecting webhook")
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
