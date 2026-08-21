#!/usr/bin/env python3
"""
scripts/compare_transports.py — phone (Twilio bridge) vs WebSocket
(browser) transport, per canonical stage, p50/p95, with an honest
explanation for every gap or difference.

Phone-path data comes from the real voice-latency-harness deployment
(GET /calls, GET /calls/{id}) — every number there was POSTed by
telephony/twilio_bridge.py's real webhook emission during an actual call
(see telephony/latency_events.py, AuviaAdapter). Stage math is
reimplemented here (not imported from voice_latency_harness) to keep
auvia_engine decoupled from that repo's internals — the two projects are
coupled only through the HTTP webhook contract, which is the right
boundary between them. The 7 stage definitions below are a direct,
unmodified copy of voice_latency_harness/core/stages.py's STAGES; if that
file changes, this one needs the same edit.

WS-path data comes from scripts/fake_ws_client.py's own JSONL output
(the existing latency_ms payload voice_agent_ws.py already returns per
turn — no new instrumentation needed there beyond the additive
endpointing_ms field added this session). The WS path was never wired to
the harness (out of scope for this session — see AuviaAdapter's docstring
in voice-latency-harness), so this script maps its native stage-name keys
(stt_ms, llm_ms, ...) onto the same canonical stage vocabulary by hand.

Every NOT MEASURABLE row is real: either the events genuinely were never
observed (phone path, from the harness's own coverage logic) or the
signal genuinely doesn't exist yet on this transport (WS path, stated
explicitly per stage) — never a fabricated or estimated number.

Usage:
    python scripts/compare_transports.py
    python scripts/compare_transports.py --harness-url http://localhost:8099 --ws-results /tmp/ws_path_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime

import httpx

# Verbatim copy of voice_latency_harness/core/stages.py's STAGES (name, from_event, to_event, description).
STAGES: list[tuple[str, str, str, str]] = [
    ("endpointing", "user_speech_end", "endpoint_decision",
     "User stops talking -> platform decides the turn has ended."),
    ("stt", "endpoint_decision", "transcript_final",
     "Endpoint decision -> final transcript is available."),
    ("llm_ttft", "transcript_final", "llm_first_token",
     "Transcript available -> first LLM token generated."),
    ("llm_complete", "llm_first_token", "llm_generation_complete",
     "First LLM token -> generation complete."),
    ("tts_ttfb", "llm_first_token", "tts_first_audio_byte",
     "First LLM token -> first TTS audio byte."),
    ("playback_start", "tts_first_audio_byte", "playback_start",
     "First TTS audio byte -> audio begins reaching the caller."),
    ("turn_total", "user_speech_end", "playback_start",
     "User stops talking -> audio begins reaching the caller. The end-to-end number the caller actually feels."),
]

# WS path: native LatencyTracker key -> canonical stage name it corresponds
# to, for the two stages the WS path can actually measure today.
WS_NATIVE_KEY_FOR_STAGE = {
    "endpointing": "endpointing_ms",
    "stt": "stt_ms",
}

# WS path: honest, stage-specific reasons the other five rows are NOT
# MEASURABLE — same root cause (no LLM_FIRST_TOKEN signal) as the phone
# path for three of them, plus one WS-specific gap (no playback beacon).
WS_NOT_MEASURABLE_REASON = {
    "llm_ttft": "blocking .invoke(), no streaming -> no first-token signal (same root cause as the phone path)",
    "llm_complete": "needs llm_first_token, which is never available (same root cause as the phone path)",
    "tts_ttfb": "needs llm_first_token, which is never available (same root cause as the phone path)",
    "playback_start": "no client-side playback beacon exists yet (~15-line addition, flagged as a TODO "
                       "since this transport was first instrumented) -- the phone path CAN measure this "
                       "because writing to the Twilio WS transport IS the playback-start signal",
    "turn_total": "needs playback_start, which is not measurable on this transport for the reason above",
}


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile — same convention as
    voice_latency_harness/core/report.py's percentile() (and numpy's default)."""
    if not values:
        raise ValueError("percentile() of an empty sequence is undefined")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


@dataclass
class StageStat:
    n: int = 0
    p50: float | None = None
    p95: float | None = None
    not_measurable_reason: str | None = None


def _measure_phone_stages(harness_url: str) -> dict[str, StageStat]:
    """Pulls every auvia-provider call from the harness, extracts each
    turn's events, and measures all 7 canonical stages by hand (see module
    docstring for why this isn't imported from voice_latency_harness)."""
    with httpx.Client(timeout=10.0) as client:
        calls = client.get(f"{harness_url}/calls").json()
        auvia_calls = [c for c in calls if c["provider"] == "auvia"]

        durations: dict[str, list[float]] = {name: [] for name, _, _, _ in STAGES}
        observed_events: set[str] = set()
        turn_count = 0

        for call in auvia_calls:
            detail = client.get(f"{harness_url}/calls/{call['call_id']}").json()
            for turn in detail["turns"]:
                turn_count += 1
                events_by_type: dict[str, datetime] = {}
                for e in turn["events"]:
                    observed_events.add(e["event_type"])
                    ts = e["provider_time"] or e["arrival_time"]
                    ts_parsed = datetime.fromisoformat(ts)
                    # first occurrence only, matching Turn.first() in the harness
                    events_by_type.setdefault(e["event_type"], ts_parsed)

                for name, from_ev, to_ev, _ in STAGES:
                    if from_ev in events_by_type and to_ev in events_by_type:
                        delta_ms = (events_by_type[to_ev] - events_by_type[from_ev]).total_seconds() * 1000
                        durations[name].append(delta_ms)

    result: dict[str, StageStat] = {}
    for name, from_ev, to_ev, _ in STAGES:
        vals = durations[name]
        if vals:
            result[name] = StageStat(n=len(vals), p50=percentile(vals, 50), p95=percentile(vals, 95))
        else:
            missing = [ev for ev in (from_ev, to_ev) if ev not in observed_events]
            reason = (f"needs event(s) never observed: {', '.join(missing)}" if missing
                      else "events observed but never both present in the same turn")
            result[name] = StageStat(n=0, not_measurable_reason=reason)

    print(f"[phone] {len(auvia_calls)} calls, {turn_count} turns pulled from {harness_url}")
    return result


def _measure_ws_stages(results_path: str) -> dict[str, StageStat]:
    with open(results_path) as f:
        turns = [json.loads(line) for line in f if line.strip()]

    result: dict[str, StageStat] = {}
    for name, *_ in STAGES:
        native_key = WS_NATIVE_KEY_FOR_STAGE.get(name)
        if native_key is None:
            result[name] = StageStat(n=0, not_measurable_reason=WS_NOT_MEASURABLE_REASON[name])
            continue
        vals = [t[native_key] for t in turns if native_key in t]
        if vals:
            result[name] = StageStat(n=len(vals), p50=percentile(vals, 50), p95=percentile(vals, 95))
        else:
            result[name] = StageStat(n=0, not_measurable_reason=f"no turn carried {native_key!r}")

    print(f"[ws] {len(turns)} turns loaded from {results_path}")
    return result


def render_ws_component_breakdown(results_path: str) -> str:
    """Informational only, not part of the canonical stage table: neither
    llm_ttft nor llm_complete is measurable on either transport (no
    llm_first_token signal), so the real cost of the LLM call — which
    dominates turn_total on the phone path — would otherwise be invisible.
    llm_ms here is LatencyTracker's own relative-ms measurement, the same
    number voice_agent_ws.py already logs per turn; not a canonical stage,
    just the honest reason turn_total is what it is."""
    with open(results_path) as f:
        turns = [json.loads(line) for line in f if line.strip()]

    lines = ["-" * 100, "ADDITIONAL CONTEXT — WS-path component breakdown (informational, not a canonical stage)", "-" * 100]
    llm_vals = [t["llm_ms"] for t in turns if "llm_ms" in t]
    total_vals = [t["total_ms"] for t in turns if "total_ms" in t]
    for key in ("stt_ms", "llm_ms", "tts_first_byte_ms", "tts_complete_ms", "total_ms"):
        vals = [t[key] for t in turns if key in t]
        if not vals:
            continue
        lines.append(f"  {key:<20} p50={percentile(vals, 50):.0f}ms  min={min(vals):.0f}ms  max={max(vals):.0f}ms  (n={len(vals)})")
    lines.append("")
    if llm_vals and total_vals:
        llm_p50, total_p50 = percentile(llm_vals, 50), percentile(total_vals, 50)
        lines.append(
            f"  llm_ms alone is p50={llm_p50:.0f}ms of the p50={total_p50:.0f}ms WS total_ms "
            f"({llm_p50 / total_p50 * 100:.0f}%) — but see the CAVEAT below before reading anything into "
            f"that number: llm_ms is highly variable turn to turn (min={min(llm_vals):.0f}ms, "
            f"max={max(llm_vals):.0f}ms) because most of these calls failed (504/429/timeout), not because "
            "successful generation time itself varies that much. It's the dominant contributor to the "
            "phone path's turn_total too (see table above), for the same reason."
        )
    return "\n".join(lines)


def _fmt(stat: StageStat) -> str:
    if stat.p50 is None:
        return "NOT MEASURABLE"
    return f"p50={stat.p50:.0f}ms p95={stat.p95:.0f}ms (n={stat.n})"


def render_table(phone: dict[str, StageStat], ws: dict[str, StageStat]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("PHONE (Twilio bridge) vs WEBSOCKET (browser) — per-stage latency")
    lines.append("=" * 100)
    lines.append("")
    header = f"{'stage':<16} {'phone':<32} {'ws':<32} {'delta (ws-phone p50)':<20}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, from_ev, to_ev, _desc in STAGES:
        p, w = phone[name], ws[name]
        if p.p50 is not None and w.p50 is not None:
            delta = f"{w.p50 - p.p50:+.0f}ms"
        else:
            delta = "-"
        lines.append(f"{name:<16} {_fmt(p):<32} {_fmt(w):<32} {delta:<20}")

    lines.append("")
    lines.append("-" * 100)
    lines.append("EXPLANATIONS")
    lines.append("-" * 100)
    for name, from_ev, to_ev, desc in STAGES:
        p, w = phone[name], ws[name]
        lines.append(f"  {name} ({desc})")
        if p.not_measurable_reason:
            lines.append(f"    phone: NOT MEASURABLE — {p.not_measurable_reason}")
        if w.not_measurable_reason:
            lines.append(f"    ws:    NOT MEASURABLE — {w.not_measurable_reason}")
        if p.p50 is not None and w.p50 is not None:
            if name == "endpointing":
                lines.append(
                    "    both measurable, real numbers — same TurnTaker class and 500ms silence "
                    "threshold on both transports, so near-equal p50s here are an instrumentation "
                    "sanity check, not an interesting transport difference."
                )
            elif name == "stt":
                lines.append(
                    "    both measurable, real numbers — same transcribe_vocal()/Sarvam call on "
                    "both paths; any gap reflects real audio-content or network variance between "
                    "the two runs, not a codec or architecture difference (see tests/asr_results.csv "
                    "for the codec-specific WER analysis, a separate concern from latency)."
                )
        lines.append("")

    return "\n".join(lines)


# Manually determined, not auto-derived: cross-referencing the bridge server
# log against both batches' call/session timestamps for the run these
# specific numbers came from (2026-08-21, ~18:56-19:14 local). A fast 429
# rejection and a genuine fast LLM success can both look like a short
# duration, so this isn't reconstructed from latency values alone — it's
# read directly from "[VOICE-AGENT] LLM failed" log lines (504
# DEADLINE_EXCEEDED / 429 RESOURCE_EXHAUSTED / 499 CANCELLED / read-timeout),
# which unambiguously mark _generate_reply_sync's exception-handler fallback
# path firing. If you re-run this comparison, re-check the log yourself
# rather than trusting this note to still be accurate for a new batch.
LLM_FAILURE_CAVEAT = """
################################################################################
CAVEAT — Gemini free-tier quota exhaustion contaminated this specific run
################################################################################
Both batches above were run back-to-back and burned through
gemini-3.6-flash's free-tier daily quota
(generate_content_free_tier_requests, limit: 20/day) partway through:

  phone path: 10/10 turns hit "504 DEADLINE_EXCEEDED" from Gemini — EVERY
              phone-path turn in this run used _generate_reply_sync's
              fallback reply ("Sorry, I couldn't process that."), not a
              real generated response.
  ws path:    9/10 turns failed (504 DEADLINE_EXCEEDED, a read timeout,
              499 CANCELLED, then 429 RESOURCE_EXHAUSTED once the daily
              quota was fully spent) — only 1/10 turns plausibly got a
              real LLM response.

What this means for the numbers above:
  - endpointing, stt, playback_start: UNAFFECTED. None of these stages
    depend on whether the LLM call succeeded.
  - turn_total (phone, p50=15977ms): measures "STT + a ~14-17s Gemini
    504 timeout + fallback-text TTS", not typical successful-turn latency.
    Real, measured, honestly labeled here — but not representative of a
    healthy LLM backend.
  - The WS component breakdown's llm_ms values above are a mix of ~9
    failed/timed-out calls and ~1 likely-real one blended into one set of
    percentiles — exactly the kind of contaminated aggregate this
    project's normalization/coverage-gap philosophy exists to avoid
    hiding. Do not treat that p50 as "typical LLM generation time."

This is itself a real, honestly-measured finding (the existing fallback
logic in voice_agent_ws.py / twilio_bridge.py caught every failure
gracefully — no crashed turns, no silent hangs, no fabricated numbers
substituted for a failure) — just not the number you'd want to quote as
"typical turn latency." Re-run scripts/compare_transports.py after the
quota resets (~24h) for a clean baseline; the instrumentation and stage
math are unaffected by this and don't need to change.
################################################################################
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harness-url", default="http://localhost:8099")
    parser.add_argument("--ws-results", default="/tmp/ws_path_results.jsonl")
    args = parser.parse_args()

    phone = _measure_phone_stages(args.harness_url)
    ws = _measure_ws_stages(args.ws_results)
    print()
    print(render_table(phone, ws))
    print()
    print(render_ws_component_breakdown(args.ws_results))
    print(LLM_FAILURE_CAVEAT)


if __name__ == "__main__":
    main()
