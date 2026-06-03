#!/usr/bin/env python3
"""
Transcribe audio with speaker diarization using:
- pyannote-audio for speaker identification (who spoke when)
- whisper for transcription (what was said)

Requires HuggingFace token with access to pyannote models.
"""

import sys
import os
import argparse
from pathlib import Path

def load_dotenv():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        missing.append("pyannote.audio")
    
    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

def transcribe_with_diarization(audio_path, output_dir, hf_token, whisper_model="large", language="no"):
    """
    Transcribe audio with speaker diarization.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory for output files
        hf_token: HuggingFace token for pyannote models
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., 'no' for Norwegian)
    """
    import torch
    # Fix for PyTorch 2.6+ security change - monkey patch torch.load
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    
    import whisper
    from pyannote.audio import Pipeline
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basename = audio_path.stem
    
    # Step 1: Speaker diarization with pyannote
    print("Running speaker diarization with pyannote...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    if device != "cpu":
        diarization_pipeline.to(torch.device(device))
    
    diarization_result = diarization_pipeline(str(audio_path))
    
    # Extract speaker segments (handle both old and new pyannote API)
    speaker_segments = []
    # New API (pyannote 4.x) returns DiarizeOutput dataclass
    if hasattr(diarization_result, 'speaker_diarization'):
        diarization = diarization_result.speaker_diarization
    else:
        diarization = diarization_result
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    print(f"Found {len(set(s['speaker'] for s in speaker_segments))} speakers")
    
    # Step 2: Transcribe with Whisper
    print(f"Transcribing with Whisper ({whisper_model})...")
    print("  Loading model...")
    whisper_model_obj = whisper.load_model(whisper_model, device=device)
    print("  Model loaded. Transcribing (this may take several minutes)...")
    
    # Get audio duration for progress estimation
    import subprocess
    try:
        duration_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True
        )
        audio_duration = float(duration_result.stdout.strip())
        print(f"  Audio duration: {audio_duration/60:.1f} minutes")
    except:
        audio_duration = None
    
    result = whisper_model_obj.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=False,  # Disabled - MPS doesn't support float64 for DTW
        verbose=True  # Show progress during transcription
    )
    
    # Step 3: Align transcription with speaker diarization
    print("Aligning transcription with speakers...")
    
    def get_speaker_at_time(time):
        """Find which speaker is talking at a given time."""
        for seg in speaker_segments:
            if seg["start"] <= time <= seg["end"]:
                return seg["speaker"]
        return "UNKNOWN"
    
    # Assign speakers to each segment
    aligned_segments = []
    for segment in result["segments"]:
        mid_time = (segment["start"] + segment["end"]) / 2
        speaker = get_speaker_at_time(mid_time)
        aligned_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "speaker": speaker
        })
    
    # Step 4: Write outputs
    
    # VTT with speaker labels
    vtt_path = output_dir / f"{basename}.diarized.vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(aligned_segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"<v {seg['speaker']}>{seg['text']}\n\n")
    
    print(f"Created: {vtt_path}")
    
    # Plain text grouped by speaker
    txt_path = output_dir / f"{basename}.diarized.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in aligned_segments:
            if seg["speaker"] != current_speaker:
                if current_speaker is not None:
                    f.write("\n\n")
                f.write(f"{seg['speaker']}:\n")
                current_speaker = seg["speaker"]
            f.write(f"{seg['text']} ")
    
    print(f"Created: {txt_path}")
    
    # Log file
    log_path = output_dir / f"{basename}.diarized.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_path}\n")
        f.write(f"Whisper model: {whisper_model}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Speakers detected: {len(set(s['speaker'] for s in speaker_segments))}\n")
        f.write(f"Segments: {len(aligned_segments)}\n")
    
    print(f"Created: {log_path}")
    
    return aligned_segments

def format_timestamp(seconds):
    """Convert seconds to VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output directory", default="./ut")
    parser.add_argument("-m", "--model", help="Whisper model (tiny/base/small/medium/large)", default="large")
    parser.add_argument("-l", "--language", help="Language code", default="no")
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Hopp over Borealis-etterbehandling")
    parser.add_argument("--ollama-model", default="hf.co/NbAiLab/borealis-27b-gguf",
                        help="Ollama-modell for etterbehandling")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama-server-URL")

    args = parser.parse_args()
    
    # Get HuggingFace token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HuggingFace token required for pyannote models.")
        print("Either:")
        print("  1. Set HF_TOKEN environment variable")
        print("  2. Pass --hf-token argument")
        print("")
        print("Get your token at: https://huggingface.co/settings/tokens")
        print("Then accept the model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    
    check_dependencies()
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    output_dir = Path(args.output) / audio_path.stem
    
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

if __name__ == "__main__":
    main()
