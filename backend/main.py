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


FRONTEND_DIR = ROOT / "frontend"


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
        trend_rows.append(
            {
                "period": row["period"],
                "offers": int(row["count"]),
                "acceptance_rate": round(float(row["mean"]), 3),
            }
        )

    band_rows = []
    for row in by_band.to_dict(orient="records"):
        band_rows.append(
            {
                "band": row["offered_band"],
                "offers": int(row["offers"]),
                "acceptance_rate": round(float(row["acceptance_rate"]), 3),
                "median_ctc": round(float(row["median_ctc"]), 2),
            }
        )

    source_rows = []
    for row in by_source.to_dict(orient="records"):
        source_rows.append(
            {
                "source": row["candidate_source"],
                "offers": int(row["offers"]),
                "acceptance_rate": round(float(row["acceptance_rate"]), 3),
            }
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
        "accepted_ctc_percentiles": {
            "p20": round(float(percentiles[0.2]), 2),
            "p50": round(float(percentiles[0.5]), 2),
            "p80": round(float(percentiles[0.8]), 2),
        },
        "model_metrics": state["model"].metrics,
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
    percentiles = flexible_percentiles(df, candidate, flexibility=payload.flexibility, min_records=8)

    target_probability = 0.70
    target_offer_ctc = None
    for point in curve:
        if point["acceptance_probability"] >= target_probability:
            target_offer_ctc = point["offered_ctc"]
            break

    benchmark_p80 = percentiles.get("p80_offered_ctc")
    benchmark_p50 = percentiles.get("p50_offered_ctc")
    suggested = _choose_suggested_ctc(
        offered_ctc=candidate["offered_ctc"],
        target_offer_ctc=target_offer_ctc,
        benchmark_p50=benchmark_p50,
        benchmark_p80=benchmark_p80,
    )

    if suggested is None:
        raise HTTPException(status_code=422, detail="Not enough data to suggest CTC")

    probability_at_suggested = predict_acceptance(model, candidate, suggested)
    profile_match = _profile_match_counts(df, candidate)
    benchmark_records = _accepted_benchmark_records(df, percentiles.get("filters_used", {}))

    return {
        "candidate": candidate,
        "acceptance_probability": round(probability, 3),
        "suggested_ctc": round(float(suggested), 2),
        "probability_at_suggested_ctc": round(float(probability_at_suggested), 3),
        "target_offer_ctc": None if target_offer_ctc is None else round(float(target_offer_ctc), 2),
        "target_probability": target_probability,
        "profile_match": profile_match,
        "percentile_recommendation": percentiles,
        "accepted_benchmark_records": benchmark_records,
        "acceptance_curve": curve,
        "warnings": _recommendation_warnings(
            percentiles,
            target_offer_ctc,
            benchmark_p80,
            profile_match,
            candidate["offered_ctc"],
        ),
    }


def _choose_suggested_ctc(
    offered_ctc: float,
    target_offer_ctc: float | None,
    benchmark_p50: float | None,
    benchmark_p80: float | None,
) -> float | None:
    if target_offer_ctc is not None and benchmark_p80 is not None:
        return max(offered_ctc, min(target_offer_ctc, benchmark_p80))
    if target_offer_ctc is not None:
        return max(offered_ctc, target_offer_ctc)
    if benchmark_p80 is not None and offered_ctc > benchmark_p80:
        return offered_ctc
    if benchmark_p50 is not None:
        return max(offered_ctc, benchmark_p50)
    return offered_ctc


def _profile_match_counts(df: pd.DataFrame, candidate: dict) -> dict:
    skill_lob = df[
        (df["primary_skill"] == candidate["primary_skill"])
        & (df["lob"] == candidate["lob"])
    ]
    exact = skill_lob[
        (skill_lob["location"] == candidate["location"])
        & (skill_lob["offered_band"] == candidate["offered_band"])
    ]
    return {
        "skill_lob_records": int(len(skill_lob)),
        "exact_profile_records": int(len(exact)),
    }


def _accepted_benchmark_records(df: pd.DataFrame, filters: dict) -> list[dict]:
    subset = df[df["accepted"] == 1].copy()
    for column, value in filters.items():
        if value is None or column not in subset.columns:
            continue
        subset = subset[subset[column] == value]

    columns = [
        "candidate_ref",
        "offer_date",
        "primary_skill",
        "lob",
        "location",
        "city_tier",
        "offered_band",
        "current_ctc",
        "expected_ctc",
        "offered_ctc",
        "offered_hike_pct",
        "offer_gap_pct",
        "candidate_source",
        "previous_company_type",
        "status",
    ]
    return (
        subset[columns]
        .sort_values("offered_ctc", ascending=False)
        .round(2)
        .to_dict(orient="records")
    )


def _recommendation_warnings(
    percentiles: dict,
    target_offer_ctc: float | None,
    benchmark_p80: float | None,
    profile_match: dict,
    offered_ctc: float,
) -> list[str]:
    warnings = []
    if percentiles.get("specificity") == "Broad benchmark":
        warnings.append("The CTC range is based on a broad benchmark, not a close profile match.")
    if profile_match["skill_lob_records"] == 0:
        warnings.append("No historical records match this skill and LOB combination. Check whether the selected skill belongs to this business unit.")
    elif profile_match["exact_profile_records"] == 0:
        warnings.append("No exact historical records match this skill, LOB, location, and band combination.")
    if target_offer_ctc is not None and benchmark_p80 is not None and target_offer_ctc > benchmark_p80:
        warnings.append(
            f"The model reaches the 70% probability target near {target_offer_ctc:.2f} LPA, "
            f"which is above the benchmark P80 of {benchmark_p80:.2f} LPA. Treat this as an escalation case, not an automatic offer."
        )
    if benchmark_p80 is not None and offered_ctc > benchmark_p80:
        warnings.append(
            f"The current offer of {offered_ctc:.2f} LPA is already above the benchmark P80 of {benchmark_p80:.2f} LPA."
        )
    if percentiles.get("warning"):
        warnings.append(percentiles["warning"])
    return warnings
