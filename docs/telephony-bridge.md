# Telephony Bridge

## What this is

A Twilio Media Streams bridge (`telephony/twilio_bridge.py`) that puts the existing real-time voice agent — VAD-driven turn-taking, Sarvam STT, Gemini, ElevenLabs TTS — behind a real PSTN phone call instead of (or alongside) the browser WebSocket path, transcoding between Twilio's fixed 8kHz μ-law and the pipeline's 16kHz PCM16 in both directions. This document covers how to run it, two pieces of verification work done alongside it (a codec bug caught by exhaustive testing, and an ASR-accuracy-under-codec experiment that didn't produce the result it went looking for), and the real, measured latency numbers comparing this transport against the WebSocket one.

## Running it

```bash
# 1. Install deps (adds jiwer + webrtcvad beyond the base project requirements)
pip install -r requirements.txt

# 2. .env — copy from .env.example and fill in:
#    PUBLIC_WS_URL=wss://<your-ngrok-domain>/twilio-stream
#    TWILIO_ACCOUNT_SID=        # reserved, not yet consumed — see Deferred items
#    TWILIO_AUTH_TOKEN=         # reserved, not yet consumed
#    LATENCY_HARNESS_URL=http://localhost:8099   # optional, see below
#    AUVIA_WEBHOOK_SECRET=<a generated secret>    # optional, see below

# 3. Start the bridge
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Tunnel it (a real Twilio webhook can't reach localhost)
ngrok http 8000
# note the https://<your-ngrok-domain> ngrok assigns you

# 5. Point a Twilio phone number's Voice webhook at:
#    https://<your-ngrok-domain>/voice        (HTTP POST, returns TwiML)
# The bridge itself only reads PUBLIC_WS_URL to build that TwiML's
# <Stream> tag — Twilio's number configuration is what actually routes a
# call there, and isn't something this repo can set up for you.
```

Optional — per-turn latency instrumentation, in a second terminal, from a sibling checkout of [voice-latency-harness](https://github.com/priyalrathore003/voice-latency-harness):

```bash
export AUVIA_WEBHOOK_SECRET=<the same secret as above>
uvicorn voice_latency_harness.server.app:app --port 8099
```

With that running, every call POSTs real timing events to it (fire-and-forget — see [Bugs caught](#bugs-caught-during-instrumentation) and the Latency instrumentation section below for what "fire-and-forget" is protecting). `GET http://localhost:8099/calls/<call_sid>/report` gives a per-call coverage + stage report; `python scripts/compare_transports.py` aggregates across every call and compares against the WebSocket path.

## Testing without a phone number

No Twilio number was available for this work — trial accounts can't purchase one. `scripts/fake_twilio_client.py` exists to make that a non-blocker: it's a local WebSocket client that speaks Twilio's actual Media Streams protocol from the client side (`connected`/`start`/`media`/`stop` events, base64 μ-law 8kHz 20ms frames, `streamSid` on every outbound-from-Twilio message), so the full bridge — TwiML-independent, since it connects straight to `/twilio-stream` — can be exercised end to end against real Sarvam/Gemini/ElevenLabs calls.

```bash
python scripts/fake_twilio_client.py --url ws://localhost:8000/twilio-stream --say "Hey, what's the status of my order?"
```

Stated plainly: this covers everything **except Twilio's own network leg** — the actual carrier call setup, Twilio's TwiML fetch from `/voice`, its real Media Streams WebSocket client behavior, and whatever Twilio's infrastructure does between a caller's handset and their servers. Everything downstream of "a Media Streams WebSocket connects and sends Twilio-shaped frames" is real and verified this way; everything upstream of that is untested, because it can't be, without a number.

One protocol constraint this client (and the bridge) has to respect, confirmed against Twilio's docs rather than assumed: **a bidirectional Media Stream can only be stopped by ending the call** — there is no graceful "unsubscribe" message. `fake_twilio_client.py` sends an explicit `stop` event at the end of its run to mirror this, but a real integration doesn't get an early-exit option; the stream and the call share a lifecycle.

## The mulaw codec bug

`telephony/mulaw.py` implements G.711 μ-law encode/decode as a numpy port of CPython's own `audioop.c`, not a from-memory reconstruction of the general G.711 spec — because the from-memory version was wrong, and the way it was wrong is the actual story here.

The first implementation used the textbook exponent/mantissa formula: work in a 16-bit domain, take `abs()` of the sample, walk a segment table, done. Decode was bit-exact against `audioop.ulaw2lin` across all 256 possible μ-law byte values. Encode was not: 18/5000 mismatches on a random-sample check. A first attempted fix (a "round half up" mantissa correction) made it *dramatically* worse — 32544/65536 mismatches, roughly 50%. Reverted, then ran an exhaustive sweep instead of another guess: **all 65,536 possible int16 values**, pure truncation, no correction. 381 mismatches — every one of them on a negative sample, every one exactly off-by-one, and every one landing precisely at a segment-exponent boundary.

That pattern was specific enough to chase down properly: pulling CPython's actual `Modules/audioop.c` source showed the real algorithm doesn't work in a 16-bit domain with `abs()` at all. It reduces the sample to a **14-bit domain via an arithmetic (sign-preserving) right shift by 2** — `(sample << 16) >> 18`, equivalently `sample >> 2` on a correctly sign-extended value — *before* taking the magnitude, with a correspondingly halved bias (`0x84 >> 2 = 0x21`) and segment table. For a positive sample, arithmetic right-shift and "shift after abs()" agree. For a negative one, they don't — an arithmetic shift floors toward negative infinity, so e.g. `-1 >> 2 == -1`, not `-(1 >> 2) == 0`. That's the entire bug: one sign-handling assumption, wrong only in the negative half of the range, and only visible at the exact sample values where it changes which segment a value falls into. Rewritten to match the real algorithm, then reverified: bit-exact against `audioop` across all 65,536 values, both directions, plus round-trip.

**Why exhaustive testing mattered here, specifically:** 381/65536 is 0.58% of all possible samples — and every wrong value is off by exactly one least-significant unit, at a codec that's already lossy by design. That's not a crash, not a garbled call, not anything a listener would report as a bug. It's a slightly wrong sample once every ~172 in the worst case, which at 8kHz is roughly one wrong sample every 21ms — indistinguishable from ordinary line noise, and exactly what you'd blame on the network or the caller's connection rather than your own code. The random 5000-sample spot-check that ran first *did* catch 18 mismatches, but "18 wrong out of 5000, in a random sample" doesn't tell you the pattern is systematic and boundary-located rather than incidental noise in the test itself — only checking every single value, and noticing they were *all* negative and *all* at segment boundaries, made the real cause findable. A spot check would have shipped this.

`audioop` was used here purely as a **development-time oracle** — never a runtime dependency of `telephony/mulaw.py`, which imports only `numpy`. That's deliberate, not incidental: `audioop` was deprecated in Python 3.11 and removed from the standard library entirely in 3.13 (PEP 594). Verifying against it at dev time and then never importing it again is the only way to use it as a correctness check without also depending on a module that's already gone in the next Python major version this project could reasonably run on.

## Codec effect on ASR: a null result

**The hypothesis — that the 8kHz μ-law codec measurably degrades Sarvam STT accuracy versus 16kHz wideband audio — is not supported by this experiment.** `scripts/asr_codec_experiment.py` transcribed 20 real recordings (the user's own voice) both directly and after a round-trip through the exact `telephony/mulaw.py` functions the bridge calls, scored both against a fixed reference set via WER, and produced this:

| Group | Description | Wideband WER | Narrowband WER | n |
|---|---|---|---|---|
| A | Control (no numbers/spelling) | 0.000 | 0.250 | 4 |
| B | Fricative-heavy | 0.071 | 0.071 | 4 |
| C | Digit-heavy | 0.503 | 0.343 | 5 |
| D | Alphanumeric | 0.244 | 0.244 | 4 |
| E | Code-switched Hindi-English | 1.000 | 1.000 | 3 |
| **All** | | **0.339** | **0.349** | **20** |

Read naively, that table says "codec makes things worse" (A), "codec makes things better" (C), and "no measurable difference" (B, D) — three different, mutually confusing stories from one experiment. None of them is trustworthy, for five separate reasons:

**(a) Absolute WER here mostly measures a words-vs-numerals normalization mismatch, not transcription accuracy.** The reference text was written as spoken words ("one thousand four hundred and sixteen rupees"); Sarvam, like any real STT system, renders numbers as numerals. Row 12's reference is that exact phrase; **both** the wideband and narrowband transcript is `"The amount was ₹1416."` — a transcription that is, if anything, *better* than a literal word-for-word rendering — and it scores 0.4 WER, because the normalization policy doesn't expand `₹` to "rupees" or otherwise reconcile currency-symbol formatting. That's a real, documented limitation of the normalization policy (see the policy notes in `tests/asr_results.csv`'s header row and `scripts/asr_codec_experiment.py`'s docstring), not a transcription failure. Given that, **absolute WER per row isn't the number to trust — only the wideband-vs-narrowband *delta*, per row, is**, since the same normalization mismatch applies identically to both arms. That delta is ≈0 for 18 of 20 rows (identical WER on both arms); the two exceptions are discussed below.

**(b) Group C's apparent improvement is a Sarvam output-formatting artifact, not a codec effect.** The one row responsible for most of Group C's wideband-vs-narrowband gap (row 9, delta −0.8) has Sarvam transcribing the *same spoken digit sequence* as one merged numeral on the wideband path (`4127330966148205`) and as space-separated digit words on the narrowband path (`four one two seven three three zero nine six six one four eight two zero five`) — both are accurate hearings of the same digits, just formatted differently, and only one of those formats happens to match the reference's own space-separated-digit-word convention after normalization. That's Sarvam choosing a different output convention on different inputs, not the codec changing what it heard.

**(c) Group E is not measurable with this normalization policy at all.** Sarvam transcribed all three code-switched Hindi-English lines in Devanagari script on *both* the wideband and narrowband path, against a reference written in Roman script. WER 1.0 on both arms isn't a finding about codec degradation — it's a script mismatch the current normalization policy has no transliteration step for. This needs to be built before Group E says anything.

**(d) Sample size is small.** Four samples in Group A, four in Group B, five in Group C, four in Group D. That's enough to notice a pattern worth investigating, not enough to call any of these group-level WER numbers stable.

**(e) The source recordings themselves have known capture artifacts.** The 20 WAV files were recorded via macOS's AVFoundation audio capture, which had occasional dropouts during recording — a source of noise present identically in *both* the wideband and narrowband arms (since narrowband is derived from the same recording), which inflates both WER numbers roughly equally rather than favoring either arm, but is still a confound worth naming rather than leaving implicit.

**One thing worth following up, stated as what it is — an observation, not a finding:** row 2 (group A, plain English, no numbers) transcribed correctly on the wideband path and flipped to full Devanagari script on the narrowband path — the only case in this dataset of a purely English sentence having its detected *language* change as a function of the codec alone. n=1. That is not evidence the codec causes language misdetection; it's a single occurrence that would be worth deliberately trying to reproduce (e.g., a dedicated test set of plain-English utterances run narrowband multiple times) before concluding anything from it.

Full data: `tests/asr_results.csv` (includes the normalization policy in its header row, so every number in it is auditable against how it was produced). Script: `scripts/asr_codec_experiment.py`.

## Per-stage latency: phone vs WebSocket

Real numbers, from `scripts/compare_transports.py` against 10 real phone-path turns (`fake_twilio_client.py`) and 10 real WebSocket-path turns (`fake_ws_client.py`), instrumented via `telephony/latency_events.py` → [voice-latency-harness](https://github.com/priyalrathore003/voice-latency-harness)'s `AuviaAdapter`.

| Stage | Phone | WebSocket |
|---|---|---|
| endpointing | p50=535ms, p95=539ms (n=11) | p50=533ms, p95=538ms (n=10) |
| stt | p50=474ms, p95=736ms (n=10) | p50=449ms, p95=1067ms (n=10) |
| llm_ttft | NOT MEASURABLE | NOT MEASURABLE |
| llm_complete | NOT MEASURABLE | NOT MEASURABLE |
| tts_ttfb | NOT MEASURABLE | NOT MEASURABLE |
| playback_start | p50=92ms, p95=102ms (n=10) | NOT MEASURABLE |
| turn_total | p50=15977ms, p95=16427ms (n=10) | NOT MEASURABLE |

**The three `NOT MEASURABLE` rows in the middle are one root cause, not three.** `llm_ttft`, `llm_complete`, and `tts_ttfb` are all defined relative to an `llm_first_token` event, and that event is never emitted on either transport — the LLM call (`get_llm().invoke()`, both `voice_agent_ws.py` and `telephony/twilio_bridge.py`) is a single blocking, non-streaming call. There is no first-token instant to report; time-to-first-token and time-to-completion are the same instant by construction. Reporting a fabricated ~0ms `llm_ttft` against that timestamp would technically be non-zero-effort but actively misleading, so instead the event simply isn't emitted, and the harness's own coverage-gap mechanism reports all three stages as missing `llm_first_token` — one gap, three symptoms, not three separate problems.

`playback_start` and `turn_total` are `NOT MEASURABLE` on the WebSocket path specifically because there's no client-side playback beacon yet — the phone path can measure "audio begins reaching the caller" because writing a frame to the Twilio WebSocket transport *is* that signal; the browser path has no equivalent signal without an explicit ~15-line addition to the client (a known, still-open gap — see Deferred items).

`endpointing` and `stt` being nearly identical across both transports (535ms vs 533ms; 474ms vs 449ms) is worth reading as an instrumentation sanity check rather than an interesting transport difference: both paths run the exact same `TurnTaker` class with the same 500ms silence threshold, and the exact same `transcribe_vocal()` Sarvam call — near-equal numbers here are what "correctly wired" looks like, not a discovery.

### LLM_FAILURE_CAVEAT

**This specific run's `turn_total` and LLM-related numbers should not be read as typical latency**, and this is stated as loudly here as it is in `compare_transports.py`'s own output for the same reason: the batches that produced this table ran back-to-back and burned through `gemini-3.6-flash`'s free-tier **daily** quota (`generate_content_free_tier_requests`, limit 20/day) partway through. Cross-referenced directly against the bridge's server log:

- **Phone path: 10/10 turns** hit `504 DEADLINE_EXCEEDED` from Gemini. Every single phone-path turn in this run used `_generate_reply_sync`'s exception-handler fallback reply ("Sorry, I couldn't process that."), not a real generated response.
- **WebSocket path: 9/10 turns failed** — `504 DEADLINE_EXCEEDED`, a read timeout, `499 CANCELLED`, then `429 RESOURCE_EXHAUSTED` once the daily quota was fully spent. Only 1/10 plausibly got a genuine LLM response.

`turn_total`'s p50 of 15977ms is therefore mostly measuring "STT + a ~14-17s Gemini timeout + fallback-text TTS," not the latency of a healthy successful turn. `endpointing`, `stt`, and `playback_start` are unaffected — none of them depend on whether the LLM call succeeded. This is itself a real, honestly-measured, if unplanned, finding: the existing fallback logic in both `voice_agent_ws.py` and `telephony/twilio_bridge.py` caught every one of these failures gracefully — no crashed turns, no hangs, no fabricated numbers substituted for the gap. It's just not the number to quote as "how fast is a normal turn." A clean re-run needs the daily quota to reset or a different key; the instrumentation and stage math themselves don't need to change.

## Bugs caught during instrumentation

- **`session.call_id` vs `session.call_sid`** — every `emit_event()` call in the first pass of instrumentation referenced a field that doesn't exist on `TwilioCallSession` (the actual field is `call_sid`). This would have raised `AttributeError` on the very first inbound media frame of any real call, crashing `_handle_inbound_media` synchronously — notably *before* `emit_event`'s own fire-and-forget try/except ever got a chance to run, since the bug was in the caller, not inside the background task. Caught by an actual smoke-test call against the rebuilt bridge, not by review; fixed by a straightforward rename across all 12 call sites, then re-verified live.
- **The harness doesn't auto-load `.env`** — matching its own existing `VAANI_WEBHOOK_SECRET` convention (documented in its README as `export`, not dotenv), `AUVIA_WEBHOOK_SECRET` has to be exported into the harness process's environment, not just written to its `.env` file. Missing this produced a `500` on every webhook POST (the endpoint's own "no secret configured" branch, not a signature failure) — an operational miss, not a code bug, but one that looked enough like a real failure to be worth writing down here so it isn't rediscovered the same way twice.

## Deferred items

- **Twilio webhook signature validation** (`X-Twilio-Signature`) on `/voice` — Twilio's own docs warn that hand-rolling the exact-URL HMAC reconstruction is easy to get subtly wrong, especially behind a tunnel, and recommend their SDK instead. Adding that SDK is a new dependency that was never explicitly cleared for this work. The `/voice` endpoint is currently unauthenticated.
- **True mid-stream TTS cancellation.** Barge-in today is "naive": a flag checked between outbound frames that stops forwarding audio and sends Twilio a `clear` message, not a request that actually stops ElevenLabs generation mid-flight. `stream_speech()` (`integrations/elevenlabs_client.py`) is a sync generator consumed via `asyncio.to_thread`; true cancellation needs an async ElevenLabs client, which wasn't in scope.
- **LLM token streaming / `llm_first_token`.** Both transports use a single blocking `.invoke()` call. Streaming would resolve the `llm_ttft`/`llm_complete`/`tts_ttfb` coverage gap directly (see the latency section above) but is separate, not-yet-started work.
- **WebSocket-path playback beacon.** No client-side signal exists yet for "audio actually started playing" on the browser path, which is why `playback_start`/`turn_total` are `NOT MEASURABLE` there. Estimated at roughly a 15-line addition to the WS client.
- **Devanagari transliteration normalization** for the ASR experiment's Group E (code-switched Hindi-English) — without it, that group's WER numbers aren't interpretable at all (see the codec-effect section above).
- **Transcoding and barge-in state-machine tests.** Not yet written. `telephony/mulaw.py` was verified exhaustively by hand against `audioop` during development (see above) but that verification isn't captured as an automated, repeatable test yet; the barge-in flag/clear logic in `telephony/twilio_bridge.py` has no automated coverage either.
