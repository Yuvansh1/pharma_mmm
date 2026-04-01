# Pharma Marketing Mix Model (MMM) with LLM Insights

[![CI](https://github.com/Yuvansh1/pharma-mmm/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuvansh1/pharma-mmm/actions)

An end-to-end Marketing Mix Modeling pipeline for pharmaceutical brands, combining classical statistical regression with an LLM-powered agent (Google Gemini) to measure channel effectiveness, compute ROI, and generate actionable business insights — all served via a FastAPI REST API.

## What This Solves

Pharma marketing teams struggle to quantify which channels — sales rep visits, speaker programs, free samples, emails, or physician ad clicks — actually drive Rx claims written by HCPs. This project answers that with a log-log regression model that attributes every Rx claim to a channel, computes elasticities and ROI, simulates budget scenarios, and explains the results in plain English using an LLM agent.

## Architecture

```
pharma-mmm/
├── main.py                        # FastAPI app (train, elasticities, ROI, simulate, insights)
├── src/
│   ├── utils/
│   │   ├── data_generator.py      # Synthetic weekly pharma data (2 years)
│   │   └── preprocessing.py       # Lag features, log transforms, interaction terms
│   ├── models/
│   │   └── mmm_model.py           # OLS regression, elasticities, contributions, ROI, scenario sim
│   └── agents/
│       └── llm_agent.py           # Gemini-powered insights, budget recommendations, anomaly detection
├── tests/
│   └── test_mmm.py                # Unit + integration tests (data, preprocessing, model, API)
├── data/                          # Generated CSV data (git-ignored)
├── Dockerfile
├── .github/workflows/ci.yml
└── requirements.txt
```

## Channels Modeled

| Channel | Description |
|---|---|
| Sales rep visits | Frequency of rep visits to HCPs |
| Speaker programs | Medical education events with HCP attendance |
| Free samples | Samples distributed by reps (correlated with visit logs) |
| Emails | Email sends, opens, and click-through rates |
| Ad clicks | Physician ad clicks on medical websites |

## Features

- **Log-log OLS regression** — coefficients are direct channel elasticities
- **Lag feature engineering** — captures delayed Rx response (samples peak 2 weeks after distribution)
- **Interaction terms** — models correlated channel effects (reps + samples)
- **Seasonality dummies** — flu season, conference season, holidays
- **ROI per channel** — revenue contribution vs. channel cost
- **Scenario simulation** — predict Rx impact of changing any channel budget
- **LLM agent (Gemini)** — plain-English insights, budget recommendations, anomaly explanations
- **REST API** — train, query, and simulate via FastAPI endpoints
- **Full test suite** — pytest with coverage

## Quickstart

**1. Clone and install:**

```bash
git clone https://github.com/yourusername/pharma-mmm.git
cd pharma-mmm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Set up environment variables:**

```bash
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

**3. Run the API:**

```bash
uvicorn main:app --reload --port 8000
```

**4. Train the model:**

```bash
curl -X POST http://localhost:8000/train
```

**5. Get channel elasticities:**

```bash
curl http://localhost:8000/elasticities
```

**6. Get LLM-powered insights:**

```bash
curl http://localhost:8000/insights
```

**7. Simulate a budget scenario (20% more rep visits):**

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"adjustments": {"log_rep_visits": 1.2}, "description": "20% increase in rep visits"}'
```

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

## Running with Docker

```bash
docker build -t pharma-mmm .
docker run -p 8000:8000 --env-file .env pharma-mmm
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Example Results

**Channel Elasticities (hypothetical):**

| Channel | Elasticity | Interpretation |
|---|---|---|
| Sales rep visits | +0.31 | 1% more visits = +0.31% Rx claims |
| Free samples (lagged) | +0.26 | Strongest delayed effect |
| Speaker programs | +0.14 | Moderate impact |
| Ad clicks | +0.11 | Steady background lift |
| Emails | +0.09 | Lowest elasticity |

**LLM Insight Example:**
> "Rep visits and free samples are your highest-leverage channels. The 2-week lag on samples suggests HCPs need time to trial before writing. Speaker programs punch above their cost. Email ROI is low — consider reducing frequency and improving targeting."

## Tech Stack

Python, FastAPI, statsmodels, scikit-learn, pandas, numpy, Google Gemini, pytest, Docker, GitHub Actions

## Background

This project is based on the Marketing Mix Modeling methodology used in pharma commercial analytics — measuring the incremental Rx lift from each promotional channel using time-series regression. The LLM layer adds interpretability and business accessibility to what is otherwise a highly technical output.
