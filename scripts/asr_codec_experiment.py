#!/usr/bin/env python3
"""
scripts/asr_codec_experiment.py — measures what the Twilio bridge's 8kHz
mu-law codec costs Sarvam STT accuracy, using the EXACT transcoding
functions the bridge uses (telephony/mulaw.py), not a reimplementation.

For each of the 20 recordings in tests/recordings/{01..20}.wav:
  A) transcribe the original 16kHz WAV directly via Sarvam   -> wideband
  B) round-trip the SAME PCM through telephony.mulaw's
     pipeline_pcm16_to_twilio_ulaw() / twilio_ulaw_to_pipeline_pcm16()
     (16k PCM16 -> 8k mu-law -> 16k PCM16) and transcribe that
                                                              -> narrowband
  C) WER for both against the reference line (tests/asr_test_set.txt),
     via jiwer.

Output:
  tests/asr_results.csv  (id, group, reference, wideband transcript,
                           narrowband transcript, wideband WER,
                           narrowband WER, delta)
  stdout: per-group (A-E) and overall mean WER, wideband vs narrowband.

Run:
    python scripts/asr_codec_experiment.py

Costs 40 real Sarvam API calls (2 per recording). No fabricated numbers —
every WER in the output comes from an actual transcription of an actual
recording; failures are recorded as FAILED, never dropped or substituted.

---------------------------------------------------------------------------
NORMALIZATION POLICY (decided once here, applied identically to the
reference and to BOTH hypotheses, before WER is computed)
---------------------------------------------------------------------------
1. lowercase
2. strip punctuation (keep only letters, digits, whitespace)
3. convert spoken number-words to digit form.

   Why: every reference line was dictated as WORDS ("four one two seven",
   "one thousand four hundred and sixteen"), but ASR systems conventionally
   render numbers as numerals. Leaving the reference in word form would
   inflate WER on every numeric line as a pure formatting mismatch, not a
   real transcription difference. So a small, hand-verified word->digit
   converter (scoped exactly to the vocabulary that appears in
   tests/asr_test_set.txt — see _ONES/_TEENS/_TENS/_ORDINALS below) is run
   on the reference AND on both hypotheses (a no-op on any side that's
   already in digit form, which is the expected case for Sarvam's output).

   Two sub-rules, disambiguated by whether "hundred"/"thousand"/a tens-word
   appears anywhere in the contiguous run of number-words:
     - a bare run of ones/teens words only (no hundred/thousand/tens) is
       read as a DIGIT SEQUENCE — the style used for card/reference/phone
       numbers spoken digit-by-digit. Each word becomes its own digit
       token, independently, never merged:
           "four one two seven" -> "4 1 2 7"
     - a run containing hundred/thousand/a tens-word is read as a single
       COMPOUND QUANTITY and folded arithmetically, left to right,
       dropping "and":
           "one thousand four hundred and sixteen" -> "1416"
           "two thousand and three"                -> "2003"
           "fifty five"                             -> "55"
   Standalone ordinals (first/third/sixth/...) convert to "1st"/"3rd"/"6th".

4. collapse whitespace.

KNOWN, DELIBERATELY UNHANDLED risks (flagged, not silently patched over):
  - Spelled-out letters (group D, e.g. "P R I Y A L") are left as
    individual lowercase letters. If Sarvam collapses them into a word
    ("priyal"), that's a real tokenization difference and will show up as
    WER — arguably correct, since letter-survival is exactly what group D
    is testing, not a formatting artifact to erase.
  - "at" / "dot" in the email line (15) are left as literal words. If
    Sarvam renders "@"/"." instead, punctuation-stripping will merge
    tokens differently on each side. Not specially handled — outside the
    digit policy the user asked for.
  - Group E (code-switched Hindi-English, 18-20) is scored as Roman-script
    text. If Sarvam transcribes the Hindi portions in Devanagari script,
    WER for those lines will be inflated by a script mismatch, not a real
    transcription failure. No transliteration is attempted.

FAILURE DETECTION: integrations.sarvam_client.transcribe_vocal() is a
best-effort wrapper that swallows its own exceptions and returns
{"transcript": "", "language_code": ""} on ANY failure (network error, bad
API key, HTTP error). It's reused here unmodified rather than forked, to
avoid two divergent implementations of "call Sarvam" drifting apart. That
means a row is classified FAILED via the heuristic
(transcript == "" and language_code == "") — a real API failure always
produces this exact pair, whereas a genuine successful-but-empty
transcription would still carry a real language_code. Given these are
~9s clear-speech recordings, a true empty-but-successful transcript is
very unlikely; this heuristic's limitation is stated here explicitly
rather than hidden.
"""
from __future__ import annotations

import re
import string
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import jiwer  # noqa: E402

from integrations.sarvam_client import transcribe_vocal  # noqa: E402
from telephony.mulaw import (  # noqa: E402
    PIPELINE_SAMPLE_RATE,
    pipeline_pcm16_to_twilio_ulaw,
    twilio_ulaw_to_pipeline_pcm16,
)
from voice_agent_pipeline import pcm16_to_wav_bytes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "tests" / "recordings"
TEST_SET_PATH = REPO_ROOT / "tests" / "asr_test_set.txt"
GROUPS_PATH = REPO_ROOT / "tests" / "asr_groups.txt"
RESULTS_CSV_PATH = REPO_ROOT / "tests" / "asr_results.csv"

FAILED = "FAILED"

# --- number-word vocabulary, scoped exactly to tests/asr_test_set.txt -----

_ONES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_TEENS = {
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19",
}
_TENS = {
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}
_ORDINALS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th",
}
_NUMBER_WORDS = set(_ONES) | set(_TEENS) | set(_TENS) | {"hundred", "thousand", "and"}


def _parse_number_run(run: list[str]) -> str:
    """Converts one contiguous run of number-words to digit-token(s).

    Two disjoint readings, disambiguated by whether the run contains a
    tens-word or hundred/thousand (a COMPOUND QUANTITY, folded
    arithmetically) or is a bare run of ones/teens words (a DIGIT
    SEQUENCE, each word kept as its own separate digit token).
    """
    has_structure = any(t in _TENS or t in ("hundred", "thousand") for t in run)

    if not has_structure:
        return " ".join(_ONES.get(t) or _TEENS[t] for t in run)

    total = 0
    current = 0
    for t in run:
        if t == "and":
            continue
        elif t in _ONES:
            current += int(_ONES[t])
        elif t in _TEENS:
            current += int(_TEENS[t])
        elif t in _TENS:
            current += int(_TENS[t])
        elif t == "hundred":
            current *= 100
        elif t == "thousand":
            total += current * 1000
            current = 0
    total += current
    return str(total)


def _convert_number_words(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _ORDINALS:
            out.append(_ORDINALS[tok])
            i += 1
            continue
        if tok in _NUMBER_WORDS and tok != "and":
            run = []
            j = i
            while j < n and tokens[j] in _NUMBER_WORDS:
                run.append(tokens[j])
                j += 1
            while run and run[-1] == "and":
                run.pop()
                j -= 1
            out.append(_parse_number_run(run))
            i = j
            continue
        out.append(tok)
        i += 1
    return out


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Applies the full policy documented in the module docstring above:
    lowercase -> strip punctuation -> number-word -> digit conversion ->
    collapse whitespace. Identical function, run on the reference and on
    both hypotheses."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    tokens = _convert_number_words(tokens)
    return _WS_RE.sub(" ", " ".join(tokens)).strip()


# --- input loading ----------------------------------------------------------

def load_reference_lines(path: Path) -> dict[int, str]:
    lines: dict[int, str] = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        num_str, _, sentence = raw.partition(".")
        lines[int(num_str.strip())] = sentence.strip()
    return lines


def load_groups(path: Path) -> dict[int, str]:
    groups: dict[int, str] = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        id_str, group = raw.split(",")
        groups[int(id_str.strip())] = group.strip()
    return groups


# --- transcription ------------------------------------------------------------

def _is_failure(result: dict) -> bool:
    return not result["transcript"] and not result["language_code"]


def transcribe_wideband(wav_path: Path) -> str | None:
    """Returns the transcript, or None on failure."""
    audio_bytes = wav_path.read_bytes()
    result = transcribe_vocal(audio_bytes, "wav")
    if _is_failure(result):
        return None
    return result["transcript"]


def transcribe_narrowband(wav_path: Path) -> str | None:
    """Round-trips the recording through the EXACT mulaw functions the
    Twilio bridge calls, then transcribes the recovered 16k PCM16 audio.
    Returns the transcript, or None on failure."""
    pcm16, sr = sf.read(wav_path, dtype="int16")
    if sr != PIPELINE_SAMPLE_RATE:
        raise ValueError(f"{wav_path.name}: expected {PIPELINE_SAMPLE_RATE}Hz, got {sr}Hz")

    ulaw_bytes = pipeline_pcm16_to_twilio_ulaw(pcm16, sr)
    recovered_pcm16 = twilio_ulaw_to_pipeline_pcm16(ulaw_bytes)

    wav_bytes = pcm16_to_wav_bytes(recovered_pcm16.tobytes(), PIPELINE_SAMPLE_RATE)
    result = transcribe_vocal(wav_bytes, "wav")
    if _is_failure(result):
        return None
    return result["transcript"]


# --- main experiment ----------------------------------------------------------

def main() -> None:
    references = load_reference_lines(TEST_SET_PATH)
    groups = load_groups(GROUPS_PATH)

    print("=" * 78)
    print(__doc__.split("NORMALIZATION POLICY")[1].split("FAILURE DETECTION")[0]
          .replace("---------------------------------------------------------------------------\n", "")
          .strip())
    print("=" * 78)
    print()

    rows = []
    for i in range(1, 21):
        wav_path = RECORDINGS_DIR / f"{i:02d}.wav"
        reference = references[i]
        group = groups[i]

        print(f"[{i:2d}/20] group={group}  {reference[:60]}...")

        wideband_raw = transcribe_wideband(wav_path)
        narrowband_raw = transcribe_narrowband(wav_path)

        ref_norm = normalize(reference)

        if wideband_raw is None:
            wideband_transcript, wideband_wer = FAILED, FAILED
            print("         wideband:   FAILED")
        else:
            wideband_transcript = wideband_raw
            wideband_wer = jiwer.wer(ref_norm, normalize(wideband_raw))
            print(f"         wideband:   WER={wideband_wer:.3f}  {wideband_raw!r}")

        if narrowband_raw is None:
            narrowband_transcript, narrowband_wer = FAILED, FAILED
            print("         narrowband: FAILED")
        else:
            narrowband_transcript = narrowband_raw
            narrowband_wer = jiwer.wer(ref_norm, normalize(narrowband_raw))
            print(f"         narrowband: WER={narrowband_wer:.3f}  {narrowband_raw!r}")

        if wideband_wer != FAILED and narrowband_wer != FAILED:
            delta = narrowband_wer - wideband_wer
        else:
            delta = FAILED

        rows.append({
            "id": i,
            "group": group,
            "reference": reference,
            "wideband_transcript": wideband_transcript,
            "narrowband_transcript": narrowband_transcript,
            "wideband_wer": wideband_wer,
            "narrowband_wer": narrowband_wer,
            "delta": delta,
        })

    _write_csv(rows)
    _print_summary(rows)


def _write_csv(rows: list[dict]) -> None:
    import csv

    with RESULTS_CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "# Normalization policy: lowercase, strip punctuation, collapse "
            "whitespace, spoken number-words -> digits (digit-by-digit runs "
            "kept as separate tokens, hundred/thousand/tens runs folded "
            "arithmetically, ordinals -> 1st/3rd/6th form). See "
            "scripts/asr_codec_experiment.py module docstring for full policy "
            "and known unhandled edge cases (spelled letters, email at/dot, "
            "Hindi script)."
        ])
        writer.writerow([
            "id", "group", "reference", "wideband_transcript",
            "narrowband_transcript", "wideband_wer", "narrowband_wer", "delta",
        ])
        for row in rows:
            writer.writerow([
                row["id"], row["group"], row["reference"],
                row["wideband_transcript"], row["narrowband_transcript"],
                row["wideband_wer"], row["narrowband_wer"], row["delta"],
            ])

    print(f"\nWrote {RESULTS_CSV_PATH.relative_to(REPO_ROOT)}")


def _print_summary(rows: list[dict]) -> None:
    print()
    print("=" * 78)
    print("SUMMARY — mean WER by group, wideband vs narrowband")
    print("=" * 78)

    excluded = [r["id"] for r in rows if r["wideband_wer"] == FAILED or r["narrowband_wer"] == FAILED]

    def mean_wer(subset: list[dict], key: str) -> str:
        vals = [r[key] for r in subset if r[key] != FAILED]
        if not vals:
            return "N/A (all failed)"
        return f"{sum(vals) / len(vals):.3f}  (n={len(vals)})"

    print(f"{'group':<8}{'wideband':<22}{'narrowband':<22}")
    for group in ["A", "B", "C", "D", "E"]:
        subset = [r for r in rows if r["group"] == group]
        print(f"{group:<8}{mean_wer(subset, 'wideband_wer'):<22}{mean_wer(subset, 'narrowband_wer'):<22}")

    print("-" * 52)
    print(f"{'ALL':<8}{mean_wer(rows, 'wideband_wer'):<22}{mean_wer(rows, 'narrowband_wer'):<22}")

    if excluded:
        print(f"\nNote: {len(excluded)} row(s) had a FAILED transcription and were "
              f"excluded from the aggregates above (ids: {excluded}). "
              f"They remain in {RESULTS_CSV_PATH.name} as FAILED, not dropped.")


if __name__ == "__main__":
    main()
