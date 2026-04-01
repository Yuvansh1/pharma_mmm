# Pharma Marketing Mix Model (MMM)

[![CI](https://github.com/Yuvansh1/pharma_mmm/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuvansh1/pharma_mmm/actions)

An end-to-end Marketing Mix Modeling pipeline for pharmaceutical brands. Combines OLS regression with an LLM-powered agent (Google Gemini) to measure channel effectiveness, compute ROI, and generate actionable business insights — served via a FastAPI REST API and a Streamlit dashboard.

## What This Solves

Pharma marketing teams struggle to quantify which channels — sales rep visits, speaker programs, free samples, emails, or physician ad clicks — actually drive Rx claims written by HCPs. This project attributes every Rx claim to a channel, computes elasticities and ROI, simulates budget scenarios, and explains results in plain English using Gemini.

## Architecture

```
pharma_mmm/
├── main.py                        # FastAPI app
├── streamlit_app.py               # Streamlit dashboard UI
├── src/
│   ├── utils/
│   │   ├── data_generator.py      # Synthetic weekly pharma data (3 years)
│   │   └── preprocessing.py       # Log transforms, lag features, train/test split
│   ├── models/
│   │   └── mmm_model.py           # OLS regression, elasticities, ROI, scenario sim
│   └── agents/
│       └── llm_agent.py           # Gemini-powered insights and recommendations
├── tests/
│   └── test_mmm.py                # 18 unit + integration tests
├── Dockerfile                     # FastAPI container
├── Dockerfile.streamlit           # Streamlit container
├── docker-compose.yml             # Run both services together
├── .github/workflows/ci.yml       # Lint + test on every push
└── requirements.txt
```

## Channels Modeled

| Channel | Description |
|---|---|
| Sales rep visits | Frequency of rep visits to HCPs |
| Speaker programs | Medical education events |
| Free samples | Samples distributed by reps (2-week lagged effect) |
| Emails | Email sends to HCPs |
| Ad clicks | Physician ad clicks on medical websites |

## Quickstart

**1. Clone and install:**

```bash
git clone https://github.com/Yuvansh1/pharma_mmm.git
cd pharma_mmm
pip install -r requirements.txt
```

**2. Set up environment:**

```bash
cp .env.example .env
# Add GEMINI_API_KEY to .env (optional — API works without it)
```

**3. Run FastAPI:**

```bash
uvicorn main:app --reload --port 8000
```

**4. Run Streamlit UI (separate terminal):**

```bash
streamlit run streamlit_app.py
```

**5. Train the model:**

```bash
curl -X POST http://localhost:8000/train
```

Open http://localhost:8501 for the dashboard, http://localhost:8000/docs for the API.

## Run with Docker

```bash
# Both API + UI together
docker-compose up --build

# API only (no env file needed)
docker build -t pharma-mmm .
docker run -p 8000:8000 pharma-mmm
```

URLs:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/train` | Train MMM on data |
| GET | `/elasticities` | Channel elasticities |
| GET | `/roi` | ROI per channel |
| POST | `/simulate` | Budget scenario simulation |
| GET | `/insights` | LLM channel insights |
| GET | `/recommend` | LLM budget recommendation |

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Tech Stack

Python, FastAPI, Streamlit, statsmodels, scikit-learn, pandas, Google Gemini, Docker, GitHub Actions
