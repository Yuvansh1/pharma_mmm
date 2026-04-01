"""
LLM Agent for MMM Insights

Uses Google Gemini to generate human-readable interpretations of:
- Channel elasticities and ROI
- Budget optimization recommendations
- Scenario simulation results
- Anomaly explanations
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class MMMLLMAgent:
    """
    LLM-powered agent that wraps MMM results and provides:
    1. Plain-English channel insights
    2. Budget optimization recommendations
    3. Scenario simulation narratives
    4. Anomaly detection explanations
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.model = None

        if GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)

    def _call(self, prompt: str) -> str:
        if self.model is None:
            return "[LLM unavailable — set GEMINI_API_KEY in .env]"
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def interpret_elasticities(self, elasticities: dict, metrics: dict) -> str:
        prompt = f"""
You are a pharma marketing analytics expert. Interpret the following Marketing Mix Model results
for a pharmaceutical brand. The dependent variable is weekly Rx claims written by HCPs.

Channel Elasticities (log-log model — coefficient = % change in Rx per 1% change in channel):
{json.dumps(elasticities, indent=2)}

Model Performance Metrics:
{json.dumps(metrics, indent=2)}

Please provide:
1. A plain-English summary of which channels drive the most Rx claims
2. Which channels are underperforming relative to their likely cost
3. Any surprising or counterintuitive findings
4. 2-3 actionable recommendations for the marketing team

Keep the response concise, business-friendly, and avoid technical jargon.
"""
        return self._call(prompt)

    def recommend_budget(self, roi: dict, current_budget: dict) -> str:
        prompt = f"""
You are a pharma marketing strategist. Based on the ROI analysis below, recommend
how to reallocate the marketing budget to maximize Rx claims.

Current Channel ROI (higher = better return per dollar spent):
{json.dumps(roi, indent=2)}

Current Budget Allocation (USD):
{json.dumps(current_budget, indent=2)}

Please provide:
1. Which channels to increase investment in and why
2. Which channels to reduce or maintain
3. A suggested % reallocation (must sum to 100%)
4. Expected impact on Rx claims

Be specific and quantitative where possible.
"""
        return self._call(prompt)

    def explain_scenario(self, scenario_result: dict) -> str:
        prompt = f"""
A pharma brand ran a scenario simulation on their Marketing Mix Model.

Scenario Details:
{json.dumps(scenario_result, indent=2)}

Please explain:
1. What this scenario means in plain English
2. Whether the projected change in Rx claims is significant
3. What risks or caveats the marketing team should consider
4. Whether this scenario is worth pursuing based on the numbers

Keep it under 200 words and use business language.
"""
        return self._call(prompt)

    def detect_anomalies(self, weekly_data: list[dict]) -> str:
        prompt = f"""
You are a pharma data analyst. Review the following weekly Rx claims data and identify
any anomalies, unexpected drops, or unusual spikes that warrant investigation.

Weekly Data (last 12 weeks):
{json.dumps(weekly_data, indent=2)}

Please:
1. Identify any anomalous weeks and explain what might have caused them
2. Flag any channel activity that seems inconsistent with Rx outcomes
3. Suggest what additional data to collect to explain these anomalies

Be concise and practical.
"""
        return self._call(prompt)
