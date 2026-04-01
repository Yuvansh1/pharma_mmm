"""
Pharma MMM FastAPI Application

Endpoints:
  GET  /health        Health check
  POST /train         Train MMM on generated or existing data
  GET  /elasticities  Channel elasticities from trained model
  GET  /roi           ROI per channel
  POST /simulate      Budget scenario simulation + LLM explanation
  GET  /insights      LLM-generated channel insights
  GET  /recommend     LLM budget recommendation
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.llm_agent import MMMLLMAgent
from src.models.mmm_model import MarketingMixModel
from src.utils.data_generator import generate_mmm_data
from src.utils.preprocessing import build_feature_matrix, train_test_split_temporal

load_dotenv()

app = FastAPI(
    title="Pharma Marketing Mix Model API",
    description="AI-powered MMM for pharma brands — channel attribution, ROI, and LLM insights.",
    version="1.0.0",
)

_model: MarketingMixModel = None
_X_test: pd.DataFrame = None
_y_test: pd.Series = None
_agent = MMMLLMAgent()

DATA_PATH = Path("data/pharma_mmm_data.csv")

CHANNEL_COSTS = {
    "log_rep_visits": 250_000,
    "log_speaker_programs": 80_000,
    "log_samples_distributed": 120_000,
    "log_emails_sent": 15_000,
    "log_ad_clicks": 40_000,
}


class ScenarioRequest(BaseModel):
    adjustments: dict
    description: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": _model is not None}


@app.post("/train")
def train():
    global _model, _X_test, _y_test

    if not DATA_PATH.exists():
        df = generate_mmm_data(n_weeks=156)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["week"])

    X, y = build_feature_matrix(df)
    train_df, test_df = train_test_split_temporal(
        pd.concat([X, y], axis=1), test_weeks=20
    )

    y_col = y.name
    X_train = train_df.drop(columns=[y_col])
    y_train = train_df[y_col]
    _X_test = test_df.drop(columns=[y_col])
    _y_test = test_df[y_col]

    _model = MarketingMixModel()
    _model.fit(X_train, y_train)

    train_metrics = _model.evaluate(X_train, y_train)
    test_metrics = _model.evaluate(_X_test, _y_test)
    elasticities = _model.get_elasticities()

    top_channels = sorted(
        [
            (k, v)
            for k, v in elasticities.items()
            if "log_" in k and "seasonality" not in k
        ],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "status": "trained",
        "train_weeks": len(X_train),
        "test_weeks": len(_X_test),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "top_channels_by_elasticity": top_channels[:5],
    }


@app.get("/elasticities")
def elasticities():
    if _model is None or _model.model is None:
        raise HTTPException(400, "Model not trained. Call POST /train first.")
    return _model.get_elasticities()


@app.get("/roi")
def roi():
    if _model is None or _model.model is None:
        raise HTTPException(400, "Model not trained. Call POST /train first.")
    contributions = _model.get_contributions(_X_test, _y_test)
    return _model.compute_roi(contributions, CHANNEL_COSTS)


@app.post("/simulate")
def simulate(request: ScenarioRequest):
    if _model is None or _model.model is None:
        raise HTTPException(400, "Model not trained. Call POST /train first.")
    result = _model.simulate_scenario(_X_test, request.adjustments)
    result["description"] = request.description
    result["llm_explanation"] = _agent.explain_scenario(result)
    return result


@app.get("/insights")
def insights():
    if _model is None or _model.model is None:
        raise HTTPException(400, "Model not trained. Call POST /train first.")
    elast = _model.get_elasticities()
    metrics = _model.evaluate(_X_test, _y_test)
    return {
        "elasticities": elast,
        "metrics": metrics,
        "llm_insights": _agent.interpret_elasticities(elast, metrics),
    }


@app.get("/recommend")
def recommend():
    if _model is None or _model.model is None:
        raise HTTPException(400, "Model not trained. Call POST /train first.")
    contributions = _model.get_contributions(_X_test, _y_test)
    roi_scores = _model.compute_roi(contributions, CHANNEL_COSTS)
    return {
        "roi": roi_scores,
        "llm_recommendation": _agent.recommend_budget(roi_scores, CHANNEL_COSTS),
    }
