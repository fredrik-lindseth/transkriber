# Borealis-etterbehandling av transkript

**Dato:** 2026-06-03
**Status:** Godkjent design

## Mål

Legge til et etterbehandlingssteg som kjører Borealis (NB AiLab tekst-LLM) lokalt via
ollama på et ferdig transkript fra `transcribe-diarize.py`. Borealis transkriberer ikke
lyd, så whisper + pyannote beholdes uendret for selve transkripsjonen. Etterbehandlingen
produserer tre ting: språkvasket VTT, oppsummering og nøkkelpunkter.

## Avgrensning

- Borealis erstatter ikke whisper. Den jobber kun på tekst som allerede er transkribert.
- Ingen nye Python-avhengigheter. Ollama nås via HTTP-API med `urllib` fra standardbiblioteket.
- Etterbehandling skal aldri velte en ferdig transkripsjon. Feiler den, er de tre
  `.diarized.*`-filene allerede lagret, og vi avslutter pent med en advarsel.

## Arkitektur

Ny modul `borealis_postprocess.py`, kjørbar både som import (fra `transcribe-diarize.py`)
og som frittstående CLI på en eksisterende `.diarized.vtt`.

### Komponenter

**`OllamaClient`**
Tynn wrapper over `urllib.request` mot `http://localhost:11434/api/generate`, `stream=false`.
- Konstanter for default modell (`hf.co/NbAiLab/borealis-27b-gguf`) og URL, begge overstyrbare.
- `generate(prompt, json_format=False) -> str` som setter `format=json` når vi vil ha
  strukturert svar.
- `is_available() -> bool` som sjekker `/api/version` og at modellen finnes i `/api/tags`.

**`clean_vtt(segments, client) -> list[segment]`** (strategi A: batch + indeks)
1. Segmentene kommer parset til `(i, start, end, speaker, text)`.
2. Send i bolker à 40 som nummerert liste. Be om JSON `{"segments":[{"i":N,"text":"..."}]}`.
3. Map renset tekst tilbake på indeks `i`. Tidskoder og taler-tagger settes på i Python,
   modellen ser dem aldri og kan derfor ikke forskyve dem.
4. Mangler en indeks i svaret, beholdes originalteksten for det segmentet.

**`summarize(text, client) -> str`** og **`extract_keypoints(text, client) -> str`**
Bygger full tekst gruppert per taler fra de rensede segmentene, ett kall hver.

**`postprocess(output_dir, basename, client)`**
Orkestrerer: leser `.diarized.vtt`, kjører `clean_vtt`, skriver `.cleaned.vtt`, bygger
gruppert tekst, kjører `summarize` og `extract_keypoints`, skriver `.summary.md` og
`.keypoints.md`. Sjekker `client.is_available()` først og hopper pent over hvis ikke.

### VTT-parsing

`.diarized.vtt` har formatet `<v SPEAKER_00>tekst` per segment. Modulen får en egen liten
parser som bevarer indeks, start, slutt, taler og tekst. (Parseren i `vtt-to-dovetail.py`
håndterer ikke voice-tagger og gjenbrukes ikke.)

## Dataflyt

```
lyd ──transcribe-diarize.py──> ut/<navn>/<navn>.diarized.{vtt,txt,log}
                                         │
                                         ▼ (med mindre --no-postprocess)
                              borealis_postprocess.postprocess()
                                         │
              ┌──────────────────────────┼───────────────────────────┐
              ▼                          ▼                            ▼
     <navn>.cleaned.vtt          <navn>.summary.md          <navn>.keypoints.md
```

## Integrasjon i `transcribe-diarize.py`

Etter at de tre diarized-filene er skrevet i `transcribe_with_diarization`, kall
`postprocess(...)`. Nye CLI-flagg:
- `--no-postprocess` — hopp over etterbehandlingen.
- `--ollama-model` — default `hf.co/NbAiLab/borealis-27b-gguf`.
- `--ollama-url` — default `http://localhost:11434`.

## Prompts

Bokmål. Rens tegnsetting og fjern fyllord, men ikke endre mening, ikke slå sammen eller del
opp segmenter, ikke oversett. For oppsummering og nøkkelpunkter: konsist, på bokmål, basert
kun på transkriptet.

## Feilhåndtering

| Situasjon | Oppførsel |
|-----------|-----------|
| ollama svarer ikke / connection refused | Advarsel, hopp over etterbehandling, avslutt pent |
| modell mangler i `/api/tags` | Advarsel, hopp over |
| JSON-parsing feiler på en bolk | Behold originaltekst for bolken, advar, fortsett |
| segment mangler i svaret | Behold original for det segmentet |

## Output

I `ut/<navn>/`:
- `<navn>.cleaned.vtt` — språkvasket, tidskoder og talere bevart
- `<navn>.summary.md` — oppsummering
- `<navn>.keypoints.md` — nøkkelpunkter/sitater

## Testing (TDD)

`OllamaClient` mockes så testene ikke krever en kjørende modell.
- VTT-parser roundtrip (parse → skriv → samme tidskoder/talere).
- `clean_vtt` indeks-mapping med mocket klient (renset tekst havner på riktig segment).
- Fallback når en indeks mangler i svaret (original beholdes).
- Fallback når en bolk gir ugyldig JSON.
- `postprocess` hopper pent over når `is_available()` er `False`.
