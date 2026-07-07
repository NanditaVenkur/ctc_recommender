from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import ROOT, city_tier, init_db, read_offers
from .model import acceptance_curve, category_support, predict_acceptance, train_acceptance_model
from .recommender import flexible_percentiles


FRONTEND_DIR = ROOT / "frontend"
TARGET_PROBABILITY = 0.70
MIN_BENCHMARK_RECORDS = 8


class RecommendationStatus(str, Enum):
    OK = "ok"
    REVIEW_LOW_SUPPORT = "review_low_support"
    ESCALATE_LOW_PROB = "escalate_low_probability"
    ESCALATE_ABOVE_MARKET = "escalate_above_market"
    NO_TARGET_IN_RANGE = "no_target_in_range"
    INSUFFICIENT_DATA = "insufficient_data"


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
    no_show = int((df["status"] == "No Show").sum())

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

    return {
        "kpis": {
            "total_offers": total,
            "accepted_or_joined": accepted,
            "declined_or_no_show": int(total - accepted),
            "acceptance_rate": round(accepted / total, 3),
            "no_show": no_show,
            "avg_offered_hike_pct": round(float(df["offered_hike_pct"].mean()), 1),
            "median_offered_ctc": round(float(df["offered_ctc"].median()), 2),
        },
        "status_counts": status_counts,
        "trend": [
            {"period": row["period"], "offers": int(row["count"]), "acceptance_rate": round(float(row["mean"]), 3)}
            for row in trend.to_dict(orient="records")
        ],
        "by_band": [
            {
                "band": row["offered_band"],
                "offers": int(row["offers"]),
                "acceptance_rate": round(float(row["acceptance_rate"]), 3),
                "median_ctc": round(float(row["median_ctc"]), 2),
            }
            for row in by_band.to_dict(orient="records")
        ],
        "by_source": [
            {"source": row["candidate_source"], "offers": int(row["offers"]), "acceptance_rate": round(float(row["acceptance_rate"]), 3)}
            for row in by_source.to_dict(orient="records")
        ],
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

    df = state["df"]
    model = state["model"]
    _apply_latest_offer_period(candidate, df)

    probability = predict_acceptance(model, candidate, candidate["offered_ctc"])
    curve = acceptance_curve(model, candidate)
    percentiles = flexible_percentiles(df, candidate, flexibility=payload.flexibility, min_records=MIN_BENCHMARK_RECORDS)

    target_offer_ctc = None
    curve_max_offer = curve[-1]["offered_ctc"]
    probability_at_curve_max = curve[-1]["acceptance_probability"]
    for point in curve:
        if point["acceptance_probability"] >= TARGET_PROBABILITY:
            target_offer_ctc = point["offered_ctc"]
            break

    benchmark_p80 = percentiles.get("p80_offered_ctc")
    benchmark_p50 = percentiles.get("p50_offered_ctc")
    suggested, rec_status = _choose_suggested_ctc(
        offered_ctc=candidate["offered_ctc"],
        probability_at_offer=probability,
        target_offer_ctc=target_offer_ctc,
        benchmark_p50=benchmark_p50,
        benchmark_p80=benchmark_p80,
    )

    probability_at_suggested = predict_acceptance(model, candidate, suggested) if suggested is not None else None
    profile_match = _profile_match_counts(df, candidate)
    support = category_support(df, candidate, ["primary_skill", "lob", "offered_band", "location"])
    if rec_status == RecommendationStatus.OK and profile_match["skill_lob_records"] == 0:
        rec_status = RecommendationStatus.REVIEW_LOW_SUPPORT
    benchmark_records = _accepted_benchmark_records(df, percentiles.get("filters_used", {}))

    return {
        "candidate": candidate,
        "recommendation_status": rec_status.value,
        "recommendation_message": _recommendation_status_message(rec_status),
        "acceptance_probability": round(probability, 3),
        "suggested_ctc": None if suggested is None else round(float(suggested), 2),
        "probability_at_suggested_ctc": None if probability_at_suggested is None else round(float(probability_at_suggested), 3),
        "target_offer_ctc": None if target_offer_ctc is None else round(float(target_offer_ctc), 2),
        "target_probability": TARGET_PROBABILITY,
        "curve_max_offer_ctc": round(float(curve_max_offer), 2),
        "probability_at_curve_max": round(float(probability_at_curve_max), 3),
        "profile_match": profile_match,
        "category_support": support,
        "percentile_recommendation": percentiles,
        "accepted_benchmark_records": benchmark_records,
        "acceptance_curve": curve,
        "warnings": _recommendation_warnings(
            percentiles=percentiles,
            recommendation_status=rec_status,
            target_offer_ctc=target_offer_ctc,
            benchmark_p80=benchmark_p80,
            profile_match=profile_match,
            offered_ctc=candidate["offered_ctc"],
            support=support,
            probability_at_offer=probability,
            curve_max_offer=curve_max_offer,
            probability_at_curve_max=probability_at_curve_max,
        ),
    }


def _apply_latest_offer_period(candidate: dict, df: pd.DataFrame) -> None:
    latest_year = int(df["offer_year"].max())
    latest_month = int(df.loc[df["offer_year"] == latest_year, "offer_month"].max())
    candidate["offer_year"] = latest_year
    candidate["offer_month"] = latest_month
    candidate["offer_quarter"] = (latest_month - 1) // 3 + 1


def _choose_suggested_ctc(
    offered_ctc: float,
    probability_at_offer: float,
    target_offer_ctc: float | None,
    benchmark_p50: float | None,
    benchmark_p80: float | None,
    min_probability_ok: float = 0.5,
) -> tuple[float | None, RecommendationStatus]:
    if benchmark_p50 is None and benchmark_p80 is None:
        return None, RecommendationStatus.INSUFFICIENT_DATA

    if target_offer_ctc is None:
        if probability_at_offer >= min_probability_ok:
            return offered_ctc, RecommendationStatus.OK
        return None, RecommendationStatus.NO_TARGET_IN_RANGE

    if benchmark_p80 is not None and target_offer_ctc > benchmark_p80 * 1.05:
        return None, RecommendationStatus.ESCALATE_ABOVE_MARKET

    if benchmark_p80 is not None and offered_ctc > benchmark_p80 and probability_at_offer < min_probability_ok:
        return None, RecommendationStatus.ESCALATE_LOW_PROB

    suggested = max(offered_ctc, min(target_offer_ctc, benchmark_p80 or target_offer_ctc))
    return suggested, RecommendationStatus.OK


def _recommendation_status_message(status: RecommendationStatus) -> str:
    messages = {
        RecommendationStatus.OK: "Confident recommendation available",
        RecommendationStatus.REVIEW_LOW_SUPPORT: "Review carefully: recommendation is directional because this skill + LOB combination has weak or no history",
        RecommendationStatus.ESCALATE_LOW_PROB: "Escalate: offer is high versus benchmark, but acceptance probability is still low",
        RecommendationStatus.ESCALATE_ABOVE_MARKET: "Escalate: reaching the target probability would exceed the benchmark range",
        RecommendationStatus.NO_TARGET_IN_RANGE: "No confident CTC suggestion: target probability is not reached in the searched range",
        RecommendationStatus.INSUFFICIENT_DATA: "No confident CTC suggestion: insufficient benchmark data",
    }
    return messages[status]


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
    recommendation_status: RecommendationStatus,
    target_offer_ctc: float | None,
    benchmark_p80: float | None,
    profile_match: dict,
    offered_ctc: float,
    support: dict,
    probability_at_offer: float,
    curve_max_offer: float,
    probability_at_curve_max: float,
) -> list[str]:
    warnings = []

    conflict = _reconcile_signals(probability_at_offer, offered_ctc, benchmark_p80)
    if conflict:
        warnings.append(conflict)
    if recommendation_status != RecommendationStatus.OK:
        warnings.append(_recommendation_status_message(recommendation_status))
    if percentiles.get("specificity") == "Broad benchmark":
        warnings.append("The CTC range is based on a broad benchmark, not a close profile match.")

    low_support_cols = [column for column, count in support.items() if count < 15]
    if low_support_cols:
        warnings.append(
            "Low training data coverage for: "
            + ", ".join(low_support_cols)
            + ". Acceptance probability for this candidate may be unreliable."
        )

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
    if target_offer_ctc is None:
        warnings.append(
            f"Even at {curve_max_offer:.2f} LPA, predicted acceptance is {probability_at_curve_max:.0%}; "
            "the 70% target was not reached within the searched range."
        )
    if percentiles.get("warning"):
        warnings.append(percentiles["warning"])
    return warnings


def _reconcile_signals(probability_at_offer: float, offered_ctc: float, benchmark_p80: float | None) -> str | None:
    if benchmark_p80 is None:
        return None

    above_market = offered_ctc > benchmark_p80
    low_probability = probability_at_offer < 0.35

    if above_market and low_probability:
        return (
            "Conflicting signal: offer is above the benchmark P80 for accepted candidates, "
            "yet the model predicts low acceptance. Check band fit, notice period, competing offers, "
            "or candidate-specific concerns before raising pay further."
        )
    if not above_market and low_probability:
        return (
            "Offer is within the normal benchmark range, but predicted acceptance is low. "
            "Raising CTC alone may not fix this; investigate candidate-specific factors."
        )
    return None
