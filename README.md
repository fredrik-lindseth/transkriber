# Transkriber

Transkriber norske video- og lydfiler til tekst med **automatisk taler-identifikasjon**.

## Hva du får ut

- **VTT-fil** med undertekster og taler-markering (kan importeres i Dovetail, YouTube, etc.)
- **Tekstfil** gruppert etter hvem som snakker

### Eksempel

Input: `inn/eksempel.mp4` (17 sekunder video)

Output `ut/eksempel/eksempel.diarized.vtt`:
```
WEBVTT

1
00:00:00.000 --> 00:00:03.000
<v SPEAKER_00>Hva jeg ville ha sagt til en som gir blod?

2
00:00:05.000 --> 00:00:10.000
<v SPEAKER_00>Jeg ville ha sagt at det er livsnødvendig for at jeg skal kunne leve et normalt liv.

3
00:00:11.000 --> 00:00:16.000
<v SPEAKER_00>Ja, så tusen, tusen takk for at du gir blod.
```

Output `ut/eksempel/eksempel.diarized.txt`:
```
SPEAKER_00:
Hva jeg ville ha sagt til en som gir blod?

Jeg ville ha sagt at det er livsnødvendig for at jeg skal kunne leve et normalt liv.

Ja, så tusen, tusen takk for at du gir blod.
```

## Oppsett

### Forutsetninger

- macOS (testet) eller Linux
- Python 3.9+
- ffmpeg
- Ca. 10 GB ledig diskplass for modeller

### 1. Installer avhengigheter

```bash
# macOS
brew install cmake ffmpeg python@3.11

# Eller Linux (Ubuntu/Debian)
sudo apt install cmake ffmpeg python3-pip python3-venv
```

### 2. Klon og sett opp prosjektet

```bash
git clone git@github.com:fredrik-lindseth/transkriber.git
cd transkriber

# Lag virtuelt miljø
python3 -m venv venv
source venv/bin/activate

# Installer Python-pakker
pip install --upgrade pip
pip install torch openai-whisper pyannote.audio
```

### 3. Sett opp HuggingFace-tilgang (for taler-identifikasjon)

Pyannote-modellene krever at du godtar bruksvilkår:

1. Lag konto på [huggingface.co](https://huggingface.co/join)

2. Gå til [Token-innstillinger](https://huggingface.co/settings/tokens) og lag en token med "Read" tilgang

3. Godta vilkårene for disse modellene (klikk "Agree and access repository"):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

4. Sett token som miljøvariabel:
   ```bash
   export HF_TOKEN="hf_din_token_her"
   ```
   
   Eller legg til i `~/.zshrc` / `~/.bashrc` for permanent bruk.

## Bruk

### Transkribere med taler-identifikasjon

```bash
source venv/bin/activate
python3 transcribe-diarize.py ./din-video.mp4
```

Output havner i `./ut/din-video/`:
- `din-video.diarized.vtt` - Undertekster med taler-markering
- `din-video.diarized.txt` - Ren tekst gruppert etter taler
- `din-video.diarized.log` - Loggfil

### Valg

```bash
# Angi output-mappe
python3 transcribe-diarize.py video.mp4 -o ./min-mappe

# Bruk mindre modell (raskere, men mindre nøyaktig)
python3 transcribe-diarize.py video.mp4 -m medium

# Transkriber engelsk lyd
python3 transcribe-diarize.py video.mp4 -l en
```

### Konverter VTT til Dovetail-format

Hvis du har en VTT-fil uten taler-markering og vil legge til taler-bytte basert på pauser:

```bash
python3 vtt-to-dovetail.py ./ut/video/video.vtt

# Juster pause-terskel (standard 2 sekunder)
python3 vtt-to-dovetail.py ./ut/video/video.vtt 3.0
```

## Rask modus (uten taler-identifikasjon)

For raskere transkribering uten å identifisere hvem som snakker, bruk whisper.cpp:

### Oppsett (kun første gang)

```bash
# Klon whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp.git

# Kjør setup-scriptet
./setup-nb-whisper.sh
```

### Transkribere

```bash
# Legg filer i inn-mappen
cp din-video.mp4 ./inn/

# Kjør transkribering
./transcribe.sh
```

Output havner i `./ut/din-video/din-video.vtt`

## Modeller

Prosjektet bruker:

- **[nb-whisper-large](https://huggingface.co/NbAiLab/nb-whisper-large)** - Norsk Whisper-modell fra Nasjonalbiblioteket/NB-AiLab
- **[pyannote-audio](https://github.com/pyannote/pyannote-audio)** - State-of-the-art taler-identifikasjon
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Tale-til-tekst modell

## Tidsbruk

Omtrentlig tid per minutt med lyd (på M1 Mac):

| Modell | Tid | Nøyaktighet |
|--------|-----|-------------|
| `large` | ~2-3 min | Best |
| `medium` | ~1 min | God |
| `small` | ~30 sek | OK |

Taler-identifikasjon legger til ca. 30 sek per minutt lyd.

## Feilsøking

### "HuggingFace token required"
Sett `HF_TOKEN` miljøvariabel, se oppsett steg 3.

### "Access to model denied"
Du må godta vilkårene for pyannote-modellene på HuggingFace. Se lenker i oppsett steg 3.

### "No module named 'torch'"
Aktiver virtuelt miljø: `source venv/bin/activate`

### Transkribering stopper/henger
Prøv en mindre modell: `-m medium` eller `-m small`
