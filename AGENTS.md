# AGENTS.md -- Transkriber

Transkriberer norske video- og lydfiler til tekst med automatisk taler-identifikasjon (pyannote + nb-whisper).

## Bruk

```bash
source venv/bin/activate
python transcribe-diarize.py inn/fil.mp4    # Transkriber med taler-ID
python vtt-to-dovetail.py ut/fil/fil.vtt    # Konverter VTT til Dovetail-format
python borealis_postprocess.py ut/fil/fil.diarized.vtt   # Etterbehandle eksisterende transkript (sprakvask, oppsummering, nokkelpunkter)
bash transcribe.sh                          # Rask modus via whisper.cpp (uten taler-ID), prosesserer alt i inn/
```

## Oppsett

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch openai-whisper pyannote.audio
```

Krever HuggingFace-token for pyannote-modeller: `cp .env.example .env` og fyll inn `HF_TOKEN`.
Alternativt: `export HF_TOKEN="..."` eller `--hf-token`-flagg.

## Filstruktur

- `transcribe-diarize.py` -- Hovedskript (pyannote diarisering + whisper transkripsjon)
- `transcribe.sh` -- Batch-prosessering via whisper.cpp (rask, uten taler-ID)
- `vtt-to-dovetail.py` -- VTT -> Dovetail-konverterer (legger til talerbytte basert pa pauser)
- `borealis_postprocess.py` -- Etterbehandling via Borealis (sprakvask, oppsummering, nokkelpunkter)
- `setup-nb-whisper.sh` -- Oppsett av nb-whisper-large for whisper.cpp
- `inn/` -> input, `ut/` -> output

## Output

`transcribe-diarize.py` produserer per fil i `ut/<filnavn>/`:
- `.diarized.vtt` -- Undertekster med taler-markering
- `.diarized.txt` -- Ren tekst gruppert etter taler
- `.diarized.log` -- Metadata (modell, enhet, antall talere)
- `.cleaned.vtt` -- Sprakvasket transkript (tidskoder bevart)
- `.summary.md` -- Oppsummering
- `.keypoints.md` -- Nokkelpunkter og sitater

De tre siste krever ollama med Borealis-modellen (hf.co/NbAiLab/borealis-27b-gguf) og hoppes over hvis ollama ikke kjorer. Kan deaktiveres med `--no-postprocess`.
