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
    print(f"  Skrev {basename}.cleaned.vtt")
    text = grouped_text(cleaned)
    try:
        print("  Lager oppsummering...")
        (output_dir / f"{basename}.summary.md").write_text(
            summarize(text, client) + "\n", encoding="utf-8")
        print("  Trekker ut nokkelpunkter...")
        (output_dir / f"{basename}.keypoints.md").write_text(
            extract_keypoints(text, client) + "\n", encoding="utf-8")
        print(f"  Etterbehandling ferdig: {basename}.cleaned.vtt, "
              f"{basename}.summary.md, {basename}.keypoints.md")
    except (urllib.error.URLError, OSError) as exc:
        print(f"  Advarsel: oppsummering/nokkelpunkter feilet ({exc}), hopper over. "
              f"{basename}.cleaned.vtt er skrevet.", file=sys.stderr)


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
