# Borealis-etterbehandling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Legge til et etterbehandlingssteg som kjører Borealis lokalt via ollama på et ferdig transkript: språkvasket VTT (tidskoder bevart), oppsummering og nøkkelpunkter.

**Architecture:** Ny modul `borealis_postprocess.py` som nås via ollama sitt HTTP-API (`urllib`, ingen ny avhengighet). VTT-språkvask gjøres med batch-by-index (strategi A): segmentene sendes nummerert i bolker, modellen returnerer JSON, renset tekst mappes tilbake på indeks så tidskoder aldri forskyves. Kalles automatisk på slutten av `transcribe-diarize.py`, kan skrus av med `--no-postprocess`, og kan kjøres frittstående på en eksisterende `.diarized.vtt`.

**Tech Stack:** Python 3.14 (stdlib: `urllib`, `json`, `re`, `argparse`), ollama HTTP-API, Borealis 27B GGUF. Tester med stdlib `unittest` + `unittest.mock`.

---

## Filstruktur

- Create: `borealis_postprocess.py` — hele etterbehandlingsmodulen (parser, OllamaClient, clean_vtt, summarize, extract_keypoints, postprocess, CLI).
- Create: `tests/test_borealis_postprocess.py` — unittest-tester med fake ollama-klient.
- Modify: `transcribe-diarize.py` — nye CLI-flagg og kall til `postprocess` i `main()`.

Alt av etterbehandling samles i én modul fordi delene henger tett sammen (samme datamodell for segmenter, samme klient). Modulen er liten nok til å holdes i hodet.

---

## Task 1: VTT parse + render (roundtrip)

**Files:**
- Create: `borealis_postprocess.py`
- Test: `tests/test_borealis_postprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_borealis_postprocess.py
import unittest

import borealis_postprocess as bp

SAMPLE_VTT = """WEBVTT

1
00:00:00.060 --> 00:00:05.060
<v SPEAKER_00>Hva jeg ville ha sagt til en som gir blod?

2
00:00:05.060 --> 00:00:11.940
<v UNKNOWN>Jeg ville ha sagt at det er livsnodvendig.

3
00:00:11.940 --> 00:00:17.140
<v SPEAKER_00>Ja, sa tusen takk.
"""


class TestVttParsing(unittest.TestCase):
    def test_parse_extracts_fields(self):
        segs = bp.parse_diarized_vtt(SAMPLE_VTT)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0]["index"], 0)
        self.assertEqual(segs[0]["start"], "00:00:00.060")
        self.assertEqual(segs[0]["end"], "00:00:05.060")
        self.assertEqual(segs[0]["speaker"], "SPEAKER_00")
        self.assertEqual(segs[0]["text"], "Hva jeg ville ha sagt til en som gir blod?")
        self.assertEqual(segs[1]["speaker"], "UNKNOWN")

    def test_roundtrip_preserves_timing_and_speaker(self):
        segs = bp.parse_diarized_vtt(SAMPLE_VTT)
        rendered = bp.render_vtt(segs)
        reparsed = bp.parse_diarized_vtt(rendered)
        self.assertEqual(len(reparsed), len(segs))
        for a, b in zip(segs, reparsed):
            self.assertEqual((a["start"], a["end"], a["speaker"], a["text"]),
                             (b["start"], b["end"], b["speaker"], b["text"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'borealis_postprocess'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Etterbehandling av transkript med Borealis via ollama.

Sprakvask av VTT (tidskoder bevart), oppsummering og nokkelpunkter.
Borealis transkriberer ikke lyd; dette steget jobber kun pa ferdig tekst.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "hf.co/NbAiLab/borealis-27b-gguf"
DEFAULT_URL = "http://localhost:11434"
BATCH_SIZE = 40

_TS = r"(\d{2}:\d{2}:\d{2}\.\d{3})"
_VOICE = re.compile(r"<v\s+([^>]+)>(.*)", re.DOTALL)


def parse_diarized_vtt(content):
    """Parse en .diarized.vtt til segmenter med indeks, tid, taler og tekst."""
    segments = []
    lines = content.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(rf"{_TS}\s*-->\s*{_TS}", line)
        if match:
            start, end = match.group(1), match.group(2)
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            speaker = None
            voice = _VOICE.match(text)
            if voice:
                speaker = voice.group(1).strip()
                text = voice.group(2)
            text = text.strip()
            if text:
                segments.append({
                    "index": len(segments),
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                })
            continue
        i += 1
    return segments


def render_vtt(segments):
    """Skriv segmenter tilbake til VTT med taler-tagger."""
    out = ["WEBVTT", ""]
    for n, seg in enumerate(segments, 1):
        out.append(str(n))
        out.append(f"{seg['start']} --> {seg['end']}")
        speaker = seg.get("speaker")
        if speaker:
            out.append(f"<v {speaker}>{seg['text']}")
        else:
            out.append(seg["text"])
        out.append("")
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/dev/transkriber
git add borealis_postprocess.py tests/test_borealis_postprocess.py
git commit -m "Add VTT parser and renderer for Borealis postprocessing"
```

---

## Task 2: clean_vtt — batch-by-index med fallback

**Files:**
- Modify: `borealis_postprocess.py`
- Test: `tests/test_borealis_postprocess.py`

- [ ] **Step 1: Write the failing test**

Legg til i `tests/test_borealis_postprocess.py` (over `if __name__`):

```python
class FakeClient:
    """Erstatter OllamaClient i tester. Returnerer forhandsdefinerte svar."""

    def __init__(self, responses=None, available=True):
        self.responses = list(responses or [])
        self.calls = []
        self._available = available

    def is_available(self):
        return self._available

    def generate(self, prompt, json_format=False):
        self.calls.append((prompt, json_format))
        return self.responses.pop(0)


def _segs(*texts):
    return [
        {"index": i, "start": "00:00:00.000", "end": "00:00:01.000",
         "speaker": "SPEAKER_00", "text": t}
        for i, t in enumerate(texts)
    ]


class TestCleanVtt(unittest.TestCase):
    def test_maps_cleaned_text_by_index(self):
        segs = _segs(" raatekst en", "raatekst to")
        client = FakeClient(responses=[
            json.dumps({"segments": [
                {"i": 0, "text": "Raatekst en."},
                {"i": 1, "text": "Raatekst to."},
            ]})
        ])
        cleaned = bp.clean_vtt(segs, client)
        self.assertEqual(cleaned[0]["text"], "Raatekst en.")
        self.assertEqual(cleaned[1]["text"], "Raatekst to.")
        # tidskoder uendret
        self.assertEqual(cleaned[0]["start"], "00:00:00.000")

    def test_missing_index_keeps_original(self):
        segs = _segs("forste", "andre")
        client = FakeClient(responses=[
            json.dumps({"segments": [{"i": 0, "text": "Forste."}]})
        ])
        cleaned = bp.clean_vtt(segs, client)
        self.assertEqual(cleaned[0]["text"], "Forste.")
        self.assertEqual(cleaned[1]["text"], "andre")

    def test_invalid_json_keeps_batch(self):
        segs = _segs("en", "to")
        client = FakeClient(responses=["ikke gyldig json"])
        cleaned = bp.clean_vtt(segs, client)
        self.assertEqual(cleaned[0]["text"], "en")
        self.assertEqual(cleaned[1]["text"], "to")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: FAIL with `AttributeError: module 'borealis_postprocess' has no attribute 'clean_vtt'`

- [ ] **Step 3: Write minimal implementation**

Legg til i `borealis_postprocess.py`:

```python
def _clean_prompt(batch):
    lines = "\n".join(f'{s["index"]}: {s["text"]}' for s in batch)
    return (
        "Du far nummererte tekstsegmenter fra tale-til-tekst pa norsk. "
        "Rens hvert segment: riktig tegnsetting, fjern fyllord som «eh» og "
        "gjentakelser. Ikke endre meningen, ikke oversett, ikke sla sammen eller "
        "del opp segmenter. Behold noyaktig samme antall segmenter med samme nummer. "
        "Svar med JSON: {\"segments\":[{\"i\":<nummer>,\"text\":\"<renset tekst>\"}]}.\n\n"
        f"{lines}"
    )


def clean_vtt(segments, client):
    """Sprakvask hvert segment via Borealis. Tidskoder og talere bevares (strategi A)."""
    cleaned = [dict(s) for s in segments]
    by_index = {s["index"]: s for s in cleaned}
    for start in range(0, len(segments), BATCH_SIZE):
        batch = segments[start:start + BATCH_SIZE]
        try:
            response = client.generate(_clean_prompt(batch), json_format=True)
            items = json.loads(response).get("segments", [])
        except (json.JSONDecodeError, urllib.error.URLError, OSError) as exc:
            print(f"  Advarsel: bolk fra segment {start} feilet ({exc}), "
                  "beholder original tekst", file=sys.stderr)
            continue
        for item in items:
            idx = item.get("i")
            text = item.get("text")
            if idx in by_index and isinstance(text, str) and text.strip():
                by_index[idx]["text"] = text.strip()
    return cleaned
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: PASS (5 tests totalt)

- [ ] **Step 5: Commit**

```bash
cd ~/dev/transkriber
git add borealis_postprocess.py tests/test_borealis_postprocess.py
git commit -m "Add batch-by-index VTT cleanup with fallback"
```

---

## Task 3: grouped_text + summarize + extract_keypoints

**Files:**
- Modify: `borealis_postprocess.py`
- Test: `tests/test_borealis_postprocess.py`

- [ ] **Step 1: Write the failing test**

Legg til i `tests/test_borealis_postprocess.py`:

```python
class TestTextHelpers(unittest.TestCase):
    def test_grouped_text_groups_by_speaker(self):
        segs = [
            {"index": 0, "start": "0", "end": "1", "speaker": "SPEAKER_00", "text": "Hei."},
            {"index": 1, "start": "1", "end": "2", "speaker": "SPEAKER_00", "text": "Hvordan gar det?"},
            {"index": 2, "start": "2", "end": "3", "speaker": "SPEAKER_01", "text": "Bra."},
        ]
        text = bp.grouped_text(segs)
        self.assertEqual(text, "SPEAKER_00:\nHei. Hvordan gar det?\n\nSPEAKER_01:\nBra.")

    def test_grouped_text_handles_missing_speaker(self):
        segs = [{"index": 0, "start": "0", "end": "1", "speaker": None, "text": "Noe."}]
        self.assertEqual(bp.grouped_text(segs), "UKJENT:\nNoe.")

    def test_summarize_passes_text_to_client(self):
        client = FakeClient(responses=["Et sammendrag."])
        result = bp.summarize("SPEAKER_00:\nHei.", client)
        self.assertEqual(result, "Et sammendrag.")
        prompt, json_format = client.calls[0]
        self.assertIn("SPEAKER_00:\nHei.", prompt)
        self.assertFalse(json_format)

    def test_extract_keypoints_passes_text_to_client(self):
        client = FakeClient(responses=["- Punkt en"])
        result = bp.extract_keypoints("SPEAKER_00:\nHei.", client)
        self.assertEqual(result, "- Punkt en")
        self.assertIn("SPEAKER_00:\nHei.", client.calls[0][0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: FAIL with `AttributeError: module 'borealis_postprocess' has no attribute 'grouped_text'`

- [ ] **Step 3: Write minimal implementation**

Legg til i `borealis_postprocess.py`:

```python
def grouped_text(segments):
    """Sla sammen segmenter til lopende tekst gruppert per taler."""
    blocks = []
    current = object()
    buffer = []
    for seg in segments:
        speaker = seg.get("speaker") or "UKJENT"
        if speaker != current:
            if buffer:
                blocks.append(f"{current}:\n" + " ".join(buffer))
                buffer = []
            current = speaker
        buffer.append(seg["text"])
    if buffer:
        blocks.append(f"{current}:\n" + " ".join(buffer))
    return "\n\n".join(blocks)


def summarize(text, client):
    """Kort oppsummering pa bokmal."""
    prompt = (
        "Oppsummer dette intervju-transkriptet kort pa bokmal, 3-5 setninger. "
        "Skriv kun sammendraget.\n\n" + text
    )
    return client.generate(prompt).strip()


def extract_keypoints(text, client):
    """Nokkelpunkter og sitater som Markdown-punktliste pa bokmal."""
    prompt = (
        "Trekk ut de viktigste punktene og sitatene fra dette transkriptet. "
        "Svar pa bokmal som en punktliste i Markdown.\n\n" + text
    )
    return client.generate(prompt).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: PASS (9 tests totalt)

- [ ] **Step 5: Commit**

```bash
cd ~/dev/transkriber
git add borealis_postprocess.py tests/test_borealis_postprocess.py
git commit -m "Add grouped text, summary and keypoints helpers"
```

---

## Task 4: OllamaClient + postprocess-orkestrering

**Files:**
- Modify: `borealis_postprocess.py`
- Test: `tests/test_borealis_postprocess.py`

- [ ] **Step 1: Write the failing test**

Legg til i `tests/test_borealis_postprocess.py`:

```python
import tempfile
from pathlib import Path


class TestModelMatch(unittest.TestCase):
    def test_matches_with_and_without_latest(self):
        names = ["hf.co/NbAiLab/borealis-27b-gguf:latest", "gemma4:26b"]
        self.assertTrue(bp._model_matches("hf.co/NbAiLab/borealis-27b-gguf", names))
        self.assertFalse(bp._model_matches("hf.co/NbAiLab/borealis-12b-gguf", names))


class TestPostprocess(unittest.TestCase):
    def test_skips_when_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "x.diarized.vtt").write_text(SAMPLE_VTT, encoding="utf-8")
            bp.postprocess(out, "x", FakeClient(available=False))
            self.assertFalse((out / "x.cleaned.vtt").exists())
            self.assertFalse((out / "x.summary.md").exists())

    def test_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "x.diarized.vtt").write_text(SAMPLE_VTT, encoding="utf-8")
            client = FakeClient(responses=[
                json.dumps({"segments": [{"i": 0, "text": "A."}, {"i": 1, "text": "B."},
                                         {"i": 2, "text": "C."}]}),
                "Et sammendrag.",
                "- Punkt",
            ])
            bp.postprocess(out, "x", client)
            self.assertTrue((out / "x.cleaned.vtt").exists())
            self.assertEqual((out / "x.summary.md").read_text(encoding="utf-8").strip(),
                             "Et sammendrag.")
            self.assertEqual((out / "x.keypoints.md").read_text(encoding="utf-8").strip(),
                             "- Punkt")
            self.assertIn("A.", (out / "x.cleaned.vtt").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: FAIL with `AttributeError: module 'borealis_postprocess' has no attribute '_model_matches'`

- [ ] **Step 3: Write minimal implementation**

Legg til i `borealis_postprocess.py`:

```python
def _model_matches(model, names):
    """Match modellnavn med eller uten :latest-suffiks."""
    base = model.split(":")[0]
    return any(n.split(":")[0] == base for n in names)


class OllamaClient:
    """Tynn wrapper over ollama sitt HTTP-API."""

    def __init__(self, model=DEFAULT_MODEL, url=DEFAULT_URL, timeout=600):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _get(self, path):
        req = urllib.request.Request(f"{self.url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_available(self):
        try:
            self._get("/api/version")
            tags = self._get("/api/tags")
        except (urllib.error.URLError, OSError):
            return False
        names = [m.get("name", "") for m in tags.get("models", [])]
        return _model_matches(self.model, names)

    def generate(self, prompt, json_format=False):
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if json_format:
            payload["format"] = "json"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/generate", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "")


def postprocess(output_dir, basename, client=None):
    """Kjor sprakvask, oppsummering og nokkelpunkter pa en .diarized.vtt."""
    client = client or OllamaClient()
    output_dir = Path(output_dir)
    vtt_path = output_dir / f"{basename}.diarized.vtt"
    if not vtt_path.exists():
        print(f"  Advarsel: fant ikke {vtt_path}, hopper over etterbehandling",
              file=sys.stderr)
        return
    if not client.is_available():
        print("  Advarsel: ollama/Borealis ikke tilgjengelig, hopper over "
              "etterbehandling", file=sys.stderr)
        return
    segments = parse_diarized_vtt(vtt_path.read_text(encoding="utf-8"))
    print(f"  Sprakvask av {len(segments)} segmenter med Borealis...")
    cleaned = clean_vtt(segments, client)
    (output_dir / f"{basename}.cleaned.vtt").write_text(
        render_vtt(cleaned), encoding="utf-8")
    text = grouped_text(cleaned)
    print("  Lager oppsummering...")
    (output_dir / f"{basename}.summary.md").write_text(
        summarize(text, client) + "\n", encoding="utf-8")
    print("  Trekker ut nokkelpunkter...")
    (output_dir / f"{basename}.keypoints.md").write_text(
        extract_keypoints(text, client) + "\n", encoding="utf-8")
    print(f"  Etterbehandling ferdig: {basename}.cleaned.vtt, "
          f"{basename}.summary.md, {basename}.keypoints.md")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/dev/transkriber && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: PASS (13 tests totalt)

- [ ] **Step 5: Commit**

```bash
cd ~/dev/transkriber
git add borealis_postprocess.py tests/test_borealis_postprocess.py
git commit -m "Add OllamaClient and postprocess orchestration"
```

---

## Task 5: CLI + integrasjon i transcribe-diarize.py

**Files:**
- Modify: `borealis_postprocess.py` (legg til `main()`)
- Modify: `transcribe-diarize.py:206-242`

- [ ] **Step 1: Legg til CLI i borealis_postprocess.py**

Legg til nederst i `borealis_postprocess.py`:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Borealis-etterbehandling av et diarized transkript")
    parser.add_argument("vtt", help="Sti til .diarized.vtt")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_URL)
    args = parser.parse_args()

    vtt_path = Path(args.vtt)
    if not vtt_path.exists():
        print(f"Error: File not found: {vtt_path}")
        sys.exit(1)

    basename = vtt_path.name
    for suffix in (".diarized.vtt", ".vtt"):
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break

    client = OllamaClient(model=args.ollama_model, url=args.ollama_url)
    postprocess(vtt_path.parent, basename, client)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verifiser at modulen fortsatt importerer og testene passerer**

Run: `cd ~/dev/transkriber && ./venv/bin/python -c "import borealis_postprocess" && ./venv/bin/python -m unittest discover -s tests -t . -v`
Expected: import OK, PASS (13 tests)

- [ ] **Step 3: Legg til CLI-flagg i transcribe-diarize.py**

I `transcribe-diarize.py`, etter linjen `parser.add_argument("--hf-token", ...)` (linje 211), legg til:

```python
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Hopp over Borealis-etterbehandling")
    parser.add_argument("--ollama-model", default="hf.co/NbAiLab/borealis-27b-gguf",
                        help="Ollama-modell for etterbehandling")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama-server-URL")
```

- [ ] **Step 4: Kall postprocess i main()**

I `transcribe-diarize.py`, erstatt kallet til `transcribe_with_diarization(...)` på slutten av `main()` (linje 236-242) slik at det følges av etterbehandling:

```python
    transcribe_with_diarization(
        audio_path=audio_path,
        output_dir=output_dir,
        hf_token=hf_token,
        whisper_model=args.model,
        language=args.language
    )

    if not args.no_postprocess:
        from borealis_postprocess import OllamaClient, postprocess
        print("Etterbehandling med Borealis...")
        client = OllamaClient(model=args.ollama_model, url=args.ollama_url)
        postprocess(output_dir, audio_path.stem, client)
```

- [ ] **Step 5: Verifiser frittstående CLI mot eksempel (ollama må kjøre)**

Run: `cd ~/dev/transkriber && ./venv/bin/python borealis_postprocess.py ut/eksempel/eksempel.diarized.vtt`
Expected: skriver `ut/eksempel/eksempel.cleaned.vtt`, `eksempel.summary.md`, `eksempel.keypoints.md`. Sjekk at `.cleaned.vtt` har samme tidskoder som `.diarized.vtt`:

Run: `diff <(grep -- '-->' ut/eksempel/eksempel.diarized.vtt) <(grep -- '-->' ut/eksempel/eksempel.cleaned.vtt)`
Expected: ingen forskjell (tidskodene er bevart).

- [ ] **Step 6: Verifiser --no-postprocess-flagget**

Run: `cd ~/dev/transkriber && ./venv/bin/python transcribe-diarize.py --help`
Expected: `--no-postprocess`, `--ollama-model` og `--ollama-url` vises i hjelpeteksten.

- [ ] **Step 7: Commit**

```bash
cd ~/dev/transkriber
git add borealis_postprocess.py transcribe-diarize.py
git commit -m "Wire Borealis postprocessing into transcribe-diarize CLI"
```

---

## Task 6: Oppdater dokumentasjon

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Oppdater AGENTS.md**

I `AGENTS.md`, under Filstruktur, legg til linje:

```
- `borealis_postprocess.py` -- Etterbehandling via Borealis (sprakvask, oppsummering, nokkelpunkter)
```

Under Output, legg til at `transcribe-diarize.py` na ogsa produserer (med mindre `--no-postprocess`):

```
- `.cleaned.vtt` -- Sprakvasket transkript (tidskoder bevart)
- `.summary.md` -- Oppsummering
- `.keypoints.md` -- Nokkelpunkter og sitater
```

Under Bruk, legg til:

```
python borealis_postprocess.py ut/fil/fil.diarized.vtt   # Etterbehandle eksisterende transkript
```

- [ ] **Step 2: Commit**

```bash
cd ~/dev/transkriber
git add AGENTS.md
git commit -m "Document Borealis postprocessing step"
```

---

## Self-Review

**Spec coverage:**
- OllamaClient (urllib, generate, is_available) → Task 4 ✓
- clean_vtt batch-by-index + fallback → Task 2 ✓
- summarize / extract_keypoints → Task 3 ✓
- postprocess orkestrering + skip-when-unavailable → Task 4 ✓
- VTT-parsing av voice-tagger → Task 1 ✓
- Output `.cleaned.vtt`, `.summary.md`, `.keypoints.md` → Task 4 ✓
- Integrasjon + flagg `--no-postprocess`, `--ollama-model`, `--ollama-url` → Task 5 ✓
- Feilhåndtering (connection refused, modell mangler, ugyldig JSON, manglende indeks) → Task 2 + Task 4 ✓
- Tester (parser roundtrip, indeks-mapping, fallback, skip) → Task 1-4 ✓

**Type consistency:** Segment-dicten (`index`, `start`, `end`, `speaker`, `text`) brukes likt i alle tasks. `client.generate(prompt, json_format=...)` og `client.is_available()` har samme signatur i FakeClient og OllamaClient. `_model_matches(model, names)` brukes i is_available.

**Placeholder scan:** Ingen TBD/TODO. All kode er komplett.
