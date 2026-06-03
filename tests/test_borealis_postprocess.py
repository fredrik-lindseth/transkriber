import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

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

    def test_summary_network_error_degrades_gracefully(self):
        """Nettverksfeil under oppsummering skal ikke velte steget.

        .cleaned.vtt er alt skrevet, og vi hopper over resten med en advarsel.
        """
        class DropsAfterCleanup:
            def is_available(self):
                return True

            def generate(self, prompt, json_format=False):
                if json_format:
                    return json.dumps({"segments": [
                        {"i": 0, "text": "A."}, {"i": 1, "text": "B."},
                        {"i": 2, "text": "C."}]})
                raise urllib.error.URLError("connection lost")

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "x.diarized.vtt").write_text(SAMPLE_VTT, encoding="utf-8")
            bp.postprocess(out, "x", DropsAfterCleanup())  # skal ikke kaste
            self.assertTrue((out / "x.cleaned.vtt").exists())
            self.assertFalse((out / "x.summary.md").exists())
            self.assertFalse((out / "x.keypoints.md").exists())


if __name__ == "__main__":
    unittest.main()
