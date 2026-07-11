from __future__ import annotations

from enum import Enum

import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import chat_with_agent
from .data import (
    ROOT,
    city_tier,
    init_db,
    read_offers,
    map_dataframe_columns,
    clean_and_prepare_uploaded_data,
    update_db_with_df,
)
from .enrichment import forge_offer_letter, github_scan, market_wire
from .model import acceptance_curve, category_support, predict_acceptance, train_acceptance_model
from .negotiation import run_negotiation
from .recommender import flexible_percentiles
from .risk_agent import scan_at_risk_offers


FRONTEND_DIR = ROOT / "frontend"
TARGET_PROBABILITY = 0.70
MIN_BENCHMARK_RECORDS = 5


class RecommendationStatus(str, Enum):
    OK = "ok"
    REVIEW_LOW_SUPPORT = "review_low_support"
    REVIEW_BELOW_BENCHMARK = "review_below_benchmark"
    ESCALATE_LOW_PROB = "escalate_low_probability"
    ESCALATE_ABOVE_MARKET = "escalate_above_market"
    NO_TARGET_IN_RANGE = "no_target_in_range"
    INSUFFICIENT_DATA = "insufficient_data"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: dict | None = None


class BenchmarkRequest(BaseModel):
    filters: dict


class OfferLetterRequest(BaseModel):
    candidate: dict
    quote: dict


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


class NegotiationRequest(BaseModel):
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
    target_probability: float = Field(default=0.75, gt=0, lt=1)
    max_rounds: int = Field(default=6, ge=1, le=12)
    budget_cap: float | None = Field(default=None, gt=0)


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
    init_db(force=True)
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

    accepted_df = df[df["accepted"] == 1]
    percentiles = accepted_df["offered_ctc"].quantile([0.2, 0.5, 0.8]).to_dict()

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
        "by_skill": [
            {
                "skill": row["primary_skill"],
                "offers": int(row["offers"]),
                "acceptance_rate": round(float(row["acceptance_rate"]), 3),
                "median_ctc": round(float(row["median_ctc"]), 2),
            }
            for row in by_skill.to_dict(orient="records")
        ],
        "by_location": [
            {
                "location": row["location"],
                "offers": int(row["offers"]),
                "acceptance_rate": round(float(row["acceptance_rate"]), 3),
                "median_ctc": round(float(row["median_ctc"]), 2),
            }
            for row in by_location.to_dict(orient="records")
        ],
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

    benchmark_p20 = percentiles.get("p20_offered_ctc")
    benchmark_p80 = percentiles.get("p80_offered_ctc")
    benchmark_p50 = percentiles.get("p50_offered_ctc")
    suggested, rec_status = _choose_suggested_ctc(
        offered_ctc=candidate["offered_ctc"],
        probability_at_offer=probability,
        target_offer_ctc=target_offer_ctc,
        benchmark_p20=benchmark_p20,
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
            benchmark_p20=benchmark_p20,
            benchmark_p80=benchmark_p80,
            profile_match=profile_match,
            offered_ctc=candidate["offered_ctc"],
            support=support,
            probability_at_offer=probability,
            curve_max_offer=curve_max_offer,
            probability_at_curve_max=probability_at_curve_max,
        ),
    }


@app.post("/api/negotiate")
def negotiate(payload: NegotiationRequest) -> dict:
    candidate = payload.model_dump(exclude={"target_probability", "max_rounds", "budget_cap"})
    candidate["city_tier"] = city_tier(candidate["location"])

    df = state["df"]
    model = state["model"]
    _apply_latest_offer_period(candidate, df)

    result = run_negotiation(
        df,
        model,
        candidate,
        target_probability=payload.target_probability,
        max_rounds=payload.max_rounds,
        budget_cap=payload.budget_cap,
    )
    result["candidate"] = candidate
    return result


@app.get("/api/risk-scan")
def risk_scan(queue_size: int = 40, risk_threshold: float = 0.55, top_n: int = 10) -> dict:
    df = state["df"]
    model = state["model"]
    return scan_at_risk_offers(df, model, queue_size=queue_size, risk_threshold=risk_threshold, top_n=top_n)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "uploaded_file"
    content = await file.read()
    
    # Save the file to the datasets directory on disk
    datasets_dir = ROOT / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    save_path = datasets_dir / filename
    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file to datasets folder: {str(e)}")
        
    try:
        if filename.endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Only CSV or Excel (.xlsx, .xls) files are supported."
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
        
    try:
        df_mapped = map_dataframe_columns(df_raw)
        
        mandatory = ["offered_band", "current_ctc", "expected_ctc", "offered_ctc", "status", "relevant_experience_years"]
        missing = [col for col in mandatory if col not in df_mapped.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing mandatory columns (or unrecognized synonyms): {', '.join(missing)}"
            )
            
        df_cleaned = clean_and_prepare_uploaded_data(df_mapped)
        update_db_with_df(df_cleaned)
        
        state["df"] = df_cleaned
        state["model"] = train_acceptance_model(df_cleaned)
        
        return {
            "success": True,
            "message": f"Successfully loaded {len(df_cleaned)} records and re-trained model.",
            "metrics": state["model"].metrics,
            "record_count": len(df_cleaned),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    messages = [msg.model_dump() for msg in payload.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    response_text, ui_actions, cards = chat_with_agent(messages, state, payload.context)
    return {"response": response_text, "ui_actions": ui_actions, "cards": cards}


@app.get("/api/github-scan")
def github_scan_endpoint(username: str, skill: str = "") -> dict:
    return github_scan(username, skill)


@app.get("/api/market-wire")
def market_wire_endpoint(lpa: float | None = None, skill: str = "", location: str = "") -> dict:
    return market_wire(lpa=lpa, skill=skill, location=location)


@app.post("/api/offer-letter")
def offer_letter(payload: OfferLetterRequest) -> dict:
    result = forge_offer_letter(payload.candidate, payload.quote)
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("reason", "Unable to draft the offer letter"))
    return result


@app.post("/api/benchmark-records")
def benchmark_records(payload: BenchmarkRequest) -> list[dict]:
    df = state["df"]
    return _accepted_benchmark_records(df, payload.filters)


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
    benchmark_p20: float | None,
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
    if benchmark_p20 is not None and suggested < benchmark_p20 * 0.9:
        return float(benchmark_p20), RecommendationStatus.REVIEW_BELOW_BENCHMARK
    return suggested, RecommendationStatus.OK


def _recommendation_status_message(status: RecommendationStatus) -> str:
    messages = {
        RecommendationStatus.OK: "Confident recommendation available",
        RecommendationStatus.REVIEW_LOW_SUPPORT: "Review carefully: recommendation is directional because this skill + LOB combination has weak or no history",
        RecommendationStatus.REVIEW_BELOW_BENCHMARK: "Review carefully: the model target is below the successful benchmark floor for this band/profile",
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
    exact_profile = df[
        (df["primary_skill"] == candidate["primary_skill"])
        & (df["lob"] == candidate["lob"])
        & (df["location"] == candidate["location"])
        & (df["offered_band"] == candidate["offered_band"])
    ]
    exact_experience = exact_profile[
        exact_profile["relevant_experience_years"] == candidate["relevant_experience_years"]
    ]
    experience_low = candidate["relevant_experience_years"] - 2
    experience_high = candidate["relevant_experience_years"] + 2
    experience_band = exact_profile[
        exact_profile["relevant_experience_years"].between(experience_low, experience_high, inclusive="both")
    ]
    return {
        "skill_lob_records": int(len(skill_lob)),
        "exact_profile_records": int(len(exact_profile)),
        "exact_experience_records": int(len(exact_experience)),
        "experience_band_records": int(len(experience_band)),
        "experience_band": f"{experience_low:.1f}-{experience_high:.1f} years",
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
        "relevant_experience_years",
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
    benchmark_p20: float | None,
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
    if recommendation_status == RecommendationStatus.REVIEW_BELOW_BENCHMARK and benchmark_p20 is not None:
        warnings.append(
            f"The acceptance model reaches the target near {target_offer_ctc:.2f} LPA, "
            f"but accepted/joined offers for this benchmark start around P20 {benchmark_p20:.2f} LPA. "
            "Suggested CTC was raised to the benchmark floor; verify band fit and internal parity."
        )
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
            "Conflicting signal: offer is above the benchmark P80 for accepted/joined candidates, "
            "yet the model predicts low acceptance. Check band fit, notice period, competing offers, "
            "or candidate-specific concerns before raising pay further."
        )
    if not above_market and low_probability:
        return (
            "Offer is within the normal benchmark range, but predicted acceptance is low. "
            "Raising CTC alone may not fix this; investigate candidate-specific factors."
        )
    return None


