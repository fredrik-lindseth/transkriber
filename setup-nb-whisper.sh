#!/bin/bash

# Setup script for nb-whisper-large model with whisper.cpp
# This downloads and converts the Norwegian nb-whisper-large model from NbAiLab

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up nb-whisper-large for whisper.cpp ==="

# Build whisper.cpp if not already built
if [ ! -f "./whisper.cpp/build/bin/whisper-cli" ]; then
    echo "Building whisper.cpp..."
    cd whisper.cpp
    cmake -B build
    cmake --build build -j --config Release
    cd ..
    echo "whisper.cpp built successfully!"
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install torch transformers huggingface_hub

# Clone OpenAI whisper repo if not present (needed for mel_filters.npz)
if [ ! -d "whisper" ]; then
    echo "Cloning OpenAI whisper repository (needed for mel filters)..."
    git clone https://github.com/openai/whisper.git
fi

# Download nb-whisper-large model from HuggingFace
MODEL_DIR="./nb-whisper-large"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading nb-whisper-large from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='NbAiLab/nb-whisper-large',
    local_dir='./nb-whisper-large',
    local_dir_use_symlinks=False
)
"
fi

# Convert the model to ggml format
OUTPUT_DIR="./whisper.cpp/models"
OUTPUT_FILE="$OUTPUT_DIR/nb-large-ggml-model.bin"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Converting nb-whisper-large to ggml format..."
    python3 ./whisper.cpp/models/convert-h5-to-ggml.py \
        "$MODEL_DIR" \
        ./whisper \
        "$OUTPUT_DIR"
    
    # Rename the output file to match what transcribe.sh expects
    mv "$OUTPUT_DIR/ggml-model.bin" "$OUTPUT_FILE"
    echo "Model converted and saved to: $OUTPUT_FILE"
else
    echo "Model already exists at: $OUTPUT_FILE"
fi

# Deactivate virtual environment
deactivate

echo ""
echo "=== Setup complete! ==="
echo "The nb-whisper-large model is ready at: $OUTPUT_FILE"
echo ""
echo "You can now run transcription with:"
echo "  ./transcribe.sh"
