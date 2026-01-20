#!/usr/bin/env python3
"""
Convert VTT files to Dovetail-friendly format with speaker labels.
Detects speaker turns based on pauses between segments.
"""

import sys
import re
from pathlib import Path

def parse_vtt(vtt_content):
    """Parse VTT file and return list of segments."""
    segments = []
    lines = vtt_content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for timestamp line
        if '-->' in line:
            match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                
                # Collect text lines until empty line or next timestamp
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1
                
                text = ' '.join(text_lines)
                if text:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'start_seconds': timestamp_to_seconds(start_time),
                        'end_seconds': timestamp_to_seconds(end_time)
                    })
                continue
        i += 1
    
    return segments

def timestamp_to_seconds(ts):
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts.replace(',', '.').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def detect_speaker_turns(segments, pause_threshold=2.0):
    """
    Detect speaker turns based on pauses between segments.
    Returns segments with speaker labels added.
    """
    if not segments:
        return segments
    
    current_speaker = 1
    segments[0]['speaker'] = current_speaker
    
    for i in range(1, len(segments)):
        prev_end = segments[i-1]['end_seconds']
        curr_start = segments[i]['start_seconds']
        pause = curr_start - prev_end
        
        # If there's a significant pause, assume speaker change
        if pause >= pause_threshold:
            current_speaker = 2 if current_speaker == 1 else 1
        
        segments[i]['speaker'] = current_speaker
    
    return segments

def write_dovetail_vtt(segments, output_path):
    """Write VTT with speaker labels for Dovetail."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('WEBVTT\n\n')
        
        for i, seg in enumerate(segments, 1):
            speaker = seg.get('speaker', 1)
            f.write(f'{i}\n')
            f.write(f"{seg['start']} --> {seg['end']}\n")
            f.write(f"<v Speaker {speaker}>{seg['text']}\n\n")

def write_dovetail_txt(segments, output_path):
    """Write plain text transcript with speaker labels for Dovetail."""
    with open(output_path, 'w', encoding='utf-8') as f:
        current_speaker = None
        
        for seg in segments:
            speaker = seg.get('speaker', 1)
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write('\n\n')
                f.write(f"Speaker {speaker}:\n")
                current_speaker = speaker
            
            f.write(f"{seg['text']} ")

def main():
    if len(sys.argv) < 2:
        print("Usage: python vtt-to-dovetail.py <input.vtt> [pause_threshold]")
        print("  pause_threshold: seconds of pause to detect speaker change (default: 2.0)")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    pause_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Read and parse VTT
    vtt_content = input_path.read_text(encoding='utf-8')
    segments = parse_vtt(vtt_content)
    
    print(f"Parsed {len(segments)} segments")
    
    # Detect speaker turns
    segments = detect_speaker_turns(segments, pause_threshold)
    
    # Write outputs
    output_vtt = input_path.with_suffix('.dovetail.vtt')
    output_txt = input_path.with_suffix('.dovetail.txt')
    
    write_dovetail_vtt(segments, output_vtt)
    write_dovetail_txt(segments, output_txt)
    
    print(f"Created: {output_vtt}")
    print(f"Created: {output_txt}")

if __name__ == '__main__':
    main()
