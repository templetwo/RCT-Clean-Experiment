#!/bin/bash
# Aelara Launcher - Activates venv and starts the interface

cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q transformers peft torch textual rich
fi

# Launch Aelara
cd interface
python3 aelara.py "$@"
