#!/usr/bin/env bash
# Launch the EMI Virtual Lab on macOS / Linux:  ./run_app.sh
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
python3 -m streamlit run app.py
