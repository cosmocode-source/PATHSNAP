#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
fi

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
