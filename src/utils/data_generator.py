"""
Synthetic Pharma MMM Data Generator

Generates realistic weekly pharma marketing and Rx claims data
for model training and demonstration. Outputs 3 years (156 weeks) by default.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_mmm_data(n_weeks: int = 156) -> pd.DataFrame:
    """
    Generate synthetic weekly pharma marketing data.

    Channels: sales rep visits, speaker programs, free samples,
              emails sent, physician ad clicks.

    Seasonality: flu season (Oct-Feb boost), conference season (May, Sep).
    Lagged effect: samples drive Rx claims 2 weeks after distribution.

    Returns:
        pd.DataFrame with weekly time-series MMM data.
    """
    dates = pd.date_range(start="2022-01-03", periods=n_weeks, freq="W-MON")
    week_of_year = np.array([d.isocalendar()[1] for d in dates])

    flu_season = np.where(
        (week_of_year >= 40) | (week_of_year <= 8),
        1.2,
        1.0,
    )
    conference_season = np.where(
        ((week_of_year >= 18) & (week_of_year <= 22))
        | ((week_of_year >= 36) & (week_of_year <= 39)),
        1.1,
        1.0,
    )
    seasonality = flu_season * conference_season

    rep_visits = np.clip(
        (np.random.poisson(lam=55, size=n_weeks) * seasonality).astype(int), 10, 120
    )
    speaker_programs = np.clip(np.random.poisson(lam=3, size=n_weeks), 0, 10)
    samples = np.clip(
        (rep_visits * np.random.uniform(3.5, 5.5, n_weeks)).astype(int), 50, 600
    )
    emails_sent = np.random.randint(600, 1400, size=n_weeks)
    ad_clicks = np.random.randint(200, 700, size=n_weeks)

    samples_lagged = np.roll(samples, 2)
    samples_lagged[:2] = samples[:2]

    trend = np.linspace(0, 150, n_weeks)
    rx_claims = (
        900
        + trend
        + 0.30 * rep_visits * 10
        + 0.15 * speaker_programs * 50
        + 0.25 * samples_lagged * 0.8
        + 0.10 * emails_sent * 0.05
        + 0.12 * ad_clicks * 0.15
        + np.random.normal(0, 40, n_weeks)
    ) * seasonality
    rx_claims = np.clip(rx_claims, 500, 3500).astype(int)

    events = []
    for d in dates:
        woy = d.isocalendar()[1]
        if woy in [18, 19]:
            events.append("Medical_Conference")
        elif woy in [40, 41, 42]:
            events.append("Flu_Season_Start")
        elif woy in [1, 2]:
            events.append("New_Year")
        else:
            events.append("None")

    return pd.DataFrame(
        {
            "week": dates,
            "rx_claims": rx_claims,
            "rep_visits": rep_visits,
            "speaker_programs": speaker_programs,
            "samples_distributed": samples,
            "emails_sent": emails_sent,
            "ad_clicks": ad_clicks,
            "seasonality_index": seasonality.round(3),
            "event": events,
        }
    )


if __name__ == "__main__":
    df = generate_mmm_data()
    out = Path(__file__).parent.parent.parent / "data" / "pharma_mmm_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} weeks -> {out}")
    print(df.head(5).to_string())
