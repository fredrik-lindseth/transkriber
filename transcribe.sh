#!/bin/bash

# Define input and output directories
input_dir="./inn"
output_dir="./ut"

# Process each file in the input directory
for file in "$input_dir"/*; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file" | cut -d'.' -f 1)
        
        # Create a folder for this file's output
        file_output_dir="$output_dir/$filename"
        mkdir -p "$file_output_dir"
        
        wav_output="$file_output_dir/${filename}.wav"
        vtt_output="$file_output_dir/${filename}.vtt"
        log_output="$file_output_dir/${filename}.log"
        
        # Skip if already transcribed
        if [[ -f "$vtt_output" ]]; then
            echo "Skipping: $filename (already transcribed)"
            continue
        fi
        
        echo "Processing: $filename"
        echo "=== Transcription Log for $filename ===" > "$log_output"
        echo "Source file: $file" >> "$log_output"
        echo "Started: $(date)" >> "$log_output"
        echo "" >> "$log_output"
        
        # Convert to WAV if needed
        if [[ ! -f "$wav_output" ]]; then
            echo "Converting to WAV..." | tee -a "$log_output"
            ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 16000 "$wav_output" 2>> "$log_output"
            echo "" >> "$log_output"
        fi
        
        # Transcribe
        echo "Transcribing with nb-whisper-large..." | tee -a "$log_output"
        ./whisper.cpp/build/bin/whisper-cli \
            -m ./whisper.cpp/models/nb-large-ggml-model.bin \
            -l no \
            --output-vtt \
            -of "$file_output_dir/${filename}" \
            -f "$wav_output" 2>&1 | tee -a "$log_output"
        
        echo "" >> "$log_output"
        echo "Completed: $(date)" >> "$log_output"
        echo "Done: $filename"
    fi
done

echo "All files processed."
