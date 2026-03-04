# F1 Probabilities Viewer (2023)

A Streamlit app that visualizes Formula 1 race outcome probabilities (Win / Podium / Top 10) and includes model diagnostics.

## What it does
- Pick a race and target market (Win, Podium, Top 10)
- View predicted probabilities per driver
- Simulate “book” odds and explore EV/Kelly sizing
- Season bankroll simulation
- Diagnostics tab (calibration, AUC, Brier, etc.)

## Run locally

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
