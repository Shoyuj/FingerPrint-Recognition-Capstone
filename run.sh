#!/usr/bin/env bash
set -e

# Fingerprint Recognition Runner Script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

PYTHON=".venv/bin/python3"

CMD="${1:-app}"

case "$CMD" in
    app|web|serve)
        PORT="${2:-8080}"
        echo "============================================================"
        echo " Starting Biometric Fingerprint Recognition Web App"
        echo " Open in browser: http://localhost:$PORT"
        echo "============================================================"
        $PYTHON app.py "$PORT"
        ;;
    verify)
        echo "Running 1:1 Verification Demo..."
        $PYTHON recognize.py --mode verify --num_tests "${2:-3}"
        ;;
    search)
        echo "Running 1:N Database Identification Search..."
        $PYTHON recognize.py --mode search --num_tests "${2:-3}"
        ;;
    eval)
        echo "Running Test Set Metrics Evaluation..."
        $PYTHON recognize.py --mode eval
        ;;
    train)
        EPOCHS="${2:-8}"
        BATCH="${3:-64}"
        echo "Running Siamese CNN Training ($EPOCHS epochs, batch $BATCH)..."
        $PYTHON train.py --epochs "$EPOCHS" --batch_size "$BATCH"
        ;;
    quick-train)
        echo "Running Quick Training (3 epochs on 10,000 samples)..."
        $PYTHON train.py --epochs 3 --batch_size 64 --quick --subset_size 10000
        ;;
    *)
        echo "Usage: ./run.sh [app|verify|search|eval|train|quick-train] [optional args]"
        exit 1
        ;;
esac
