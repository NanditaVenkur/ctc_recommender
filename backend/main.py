from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import ROOT, city_tier, init_db, read_offers
from .model import acceptance_curve, predict_acceptance, train_acceptance_model
from .recommender import flexible_percentiles
from .agent import chat_with_agent

FRONTEND_DIR = ROOT / "frontend"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class CandidateRequest(BaseModel):
    current_ctc: float = Field(gt=0)
    expected_ctc: float = Field(gt=0)
    offered_ctc: float = Field(gt=0)
    relevant_experience_years: float = Field(ge=0)
    notice_period_days: float = Field(ge=0)
    offered_band: str
    candidate_source: str
    lob: str
    primary_skill: str
    previous_company_type: str
    location: str
    joining_bonus: int = 0
    relocation: int = 0
    flexibility: str = "balanced"


app = FastAPI(title="CTC Offer Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict = {}


@app.on_event("startup")
def startup() -> None:
    init_db()
    df = read_offers()
    state["df"] = df
    state["model"] = train_acceptance_model(df)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict:
    df = state["df"]
    fields = ["offered_band", "candidate_source", "lob", "primary_skill", "previous_company_type", "location", "city_tier"]
    return {
        field: sorted([value for value in df[field].dropna().unique().tolist() if str(value) != "nan"])
        for field in fields
    }


@app.get("/api/summary")
def summary() -> dict:
    df = state["df"]
    total = len(df)
    accepted = int(df["accepted"].sum())
    declined = int(total - accepted)
    no_show = int((df["status"] == "No Show").sum())
    avg_hike = float(df["offered_hike_pct"].mean())

    status_counts = df["status"].value_counts().to_dict()
    trend = (
        df.groupby(["offer_year", "offer_month"], dropna=False)["accepted"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values(["offer_year", "offer_month"])
    )
    trend["period"] = trend["offer_year"].astype(int).astype(str) + "-" + trend["offer_month"].astype(int).astype(str).str.zfill(2)

    by_band = (
        df.groupby("offered_band")
        .agg(offers=("accepted", "count"), acceptance_rate=("accepted", "mean"), median_ctc=("offered_ctc", "median"))
        .reset_index()
        .sort_values("offered_band")
    )
    by_source = (
        df.groupby("candidate_source")
        .agg(offers=("accepted", "count"), acceptance_rate=("accepted", "mean"))
        .reset_index()
        .sort_values("offers", ascending=False)
    )

    accepted_df = df[df["accepted"] == 1]
    percentiles = accepted_df["offered_ctc"].quantile([0.2, 0.5, 0.8]).to_dict()

    trend_rows = []
    for row in trend.to_dict(orient="records"):
        trend_rows.append({
            "period": row["period"],
            "offers": int(row["count"]),
            "acceptance_rate": round(float(row["mean"]), 3),
        })

    band_rows = []
    for row in by_band.to_dict(orient="records"):
        band_rows.append({
            "band": row["offered_band"],
            "offers": int(row["offers"]),
            "acceptance_rate": round(float(row["acceptance_rate"]), 3),
            "median_ctc": round(float(row["median_ctc"]), 2),
        })

    source_rows = []
    for row in by_source.to_dict(orient="records"):
        source_rows.append({
            "source": row["candidate_source"],
            "offers": int(row["offers"]),
            "acceptance_rate": round(float(row["acceptance_rate"]), 3),
        })

    by_skill = (
        df.groupby("primary_skill")
        .agg(offers=("accepted", "count"), acceptance_rate=("accepted", "mean"), median_ctc=("offered_ctc", "median"))
        .reset_index()
        .sort_values("acceptance_rate", ascending=False)
    )

    by_location = (
        df.groupby("location")
        .agg(offers=("accepted", "count"), acceptance_rate=("accepted", "mean"), median_ctc=("offered_ctc", "median"))
        .reset_index()
        .sort_values("offers", ascending=False)
    )

    skill_rows = []
    for row in by_skill.to_dict(orient="records"):
        skill_rows.append({
            "skill": row["primary_skill"],
            "offers": int(row["offers"]),
            "acceptance_rate": round(float(row["acceptance_rate"]), 3),
            "median_ctc": round(float(row["median_ctc"]), 2),
        })

    location_rows = []
    for row in by_location.to_dict(orient="records"):
        location_rows.append({
            "location": row["location"],
            "offers": int(row["offers"]),
            "acceptance_rate": round(float(row["acceptance_rate"]), 3),
            "median_ctc": round(float(row["median_ctc"]), 2),
        })

    # Auto-generate a smart insight
    best_skill = by_skill.iloc[0] if len(by_skill) > 0 else None
    worst_skill = by_skill.iloc[-1] if len(by_skill) > 1 else None
    insight = ""
    if best_skill is not None and worst_skill is not None:
        insight = (
            f"{best_skill['primary_skill']} has the highest acceptance rate at "
            f"{round(float(best_skill['acceptance_rate']) * 100)}%, while "
            f"{worst_skill['primary_skill']} is the lowest at "
            f"{round(float(worst_skill['acceptance_rate']) * 100)}%. "
            f"Consider adjusting offer strategies for underperforming skill segments."
        )

    return {
        "kpis": {
            "total_offers": total,
            "accepted_or_joined": accepted,
            "declined_or_no_show": declined,
            "acceptance_rate": round(accepted / total, 3),
            "no_show": no_show,
            "avg_offered_hike_pct": round(avg_hike, 1),
            "median_offered_ctc": round(float(df["offered_ctc"].median()), 2),
        },
        "status_counts": status_counts,
        "trend": trend_rows,
        "by_band": band_rows,
        "by_source": source_rows,
        "by_skill": skill_rows,
        "by_location": location_rows,
        "accepted_ctc_percentiles": {
            "p20": round(float(percentiles[0.2]), 2),
            "p50": round(float(percentiles[0.5]), 2),
            "p80": round(float(percentiles[0.8]), 2),
        },
        "model_metrics": state["model"].metrics,
        "insight": insight,
    }


@app.get("/api/candidates")
def candidates(limit: int = 30) -> list[dict]:
    df = state["df"].sort_values("offer_date", ascending=False).head(limit)
    columns = [
        "candidate_ref",
        "offer_date",
        "primary_skill",
        "lob",
        "location",
        "offered_band",
        "current_ctc",
        "expected_ctc",
        "offered_ctc",
        "offered_hike_pct",
        "status",
    ]
    return df[columns].round(2).to_dict(orient="records")


@app.post("/api/recommend")
def recommend(payload: CandidateRequest) -> dict:
    candidate = payload.model_dump()
    candidate["city_tier"] = city_tier(candidate["location"])
    candidate["offer_year"] = 2026
    candidate["offer_month"] = 7
    candidate["offer_quarter"] = 3

    model = state["model"]
    df = state["df"]
    probability = predict_acceptance(model, candidate, candidate["offered_ctc"])
    curve = acceptance_curve(model, candidate)
    percentiles = flexible_percentiles(df, candidate, flexibility=payload.flexibility)

    target_probability = 0.70
    suggested = None
    for point in curve:
        if point["acceptance_probability"] >= target_probability:
            suggested = point["offered_ctc"]
            break
    if suggested is None and percentiles.get("p50_offered_ctc") is not None:
        suggested = percentiles["p50_offered_ctc"]

    if suggested is None:
        raise HTTPException(status_code=422, detail="Not enough data to suggest CTC")

    return {
        "candidate": candidate,
        "acceptance_probability": round(probability, 3),
        "suggested_ctc": round(float(suggested), 2),
        "target_probability": target_probability,
        "percentile_recommendation": percentiles,
        "acceptance_curve": curve,
        "warnings": _recommendation_warnings(percentiles),
    }


def _recommendation_warnings(percentiles: dict) -> list[str]:
    warnings = []
    if percentiles.get("specificity") == "Broad benchmark":
        warnings.append("The CTC range is based on a broad benchmark, not a close profile match.")
    if percentiles.get("warning"):
        warnings.append(percentiles["warning"])
    return warnings

@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    messages = [msg.model_dump() for msg in payload.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    response_text, ui_actions = chat_with_agent(messages, state)
    return {"response": response_text, "ui_actions": ui_actions}


class BenchmarkRequest(BaseModel):
    filters: dict


@app.post("/api/benchmark-records")
def benchmark_records(payload: BenchmarkRequest) -> list[dict]:
    """Return the actual offer records matching the benchmark filters."""
    from .recommender import _filter, _clean_value
    df = state["df"]
    subset, _ = _filter(df, payload.filters)
    columns = [
        "candidate_ref", "offer_date", "primary_skill", "lob", "location",
        "offered_band", "current_ctc", "expected_ctc", "offered_ctc",
        "offered_hike_pct", "status",
    ]
    available = [c for c in columns if c in subset.columns]
    return subset[available].round(2).to_dict(orient="records")
