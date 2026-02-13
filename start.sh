#!/bin/bash
set -e

echo "Starting FastAPI backend on :8000..."
nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

echo "Starting Streamlit UI on :7860..."
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 7860