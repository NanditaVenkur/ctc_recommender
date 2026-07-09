from __future__ import annotations

import pandas as pd

from .model import TrainedAcceptanceModel, acceptance_curve, predict_acceptance
from .recommender import flexible_percentiles


RISK_PROBABILITY_THRESHOLD = 0.55
TARGET_PROBABILITY = 0.70
MIN_BENCHMARK_RECORDS = 8


def scan_at_risk_offers(
    df: pd.DataFrame,
    model: TrainedAcceptanceModel,
    queue_size: int = 40,
    risk_threshold: float = RISK_PROBABILITY_THRESHOLD,
    top_n: int = 10,
) -> dict:
    """Proactively scan the most recent offers and flag the ones the acceptance model
    would treat as at-risk today, with a drafted escalation alert for each.

    The dataset is historical (every offer already has a final outcome), so the "queue"
    here is the most recent N offers scored as if a decision were still pending - this
    is what a live version of this agent would run continuously against open offers.
    """
    queue_size = max(1, int(queue_size))
    top_n = max(1, int(top_n))
    queue = df.sort_values("offer_date", ascending=False).head(queue_size).copy()

    flagged = []
    for _, row in queue.iterrows():
        candidate = _candidate_from_row(row)
        probability = predict_acceptance(model, candidate, candidate["offered_ctc"])
        if probability >= risk_threshold:
            continue

        percentiles = flexible_percentiles(df, candidate, flexibility="balanced", min_records=MIN_BENCHMARK_RECORDS)
        curve = acceptance_curve(model, candidate)
        target_offer = next(
            (point["offered_ctc"] for point in curve if point["acceptance_probability"] >= TARGET_PROBABILITY),
            None,
        )
        benchmark_p80 = percentiles.get("p80_offered_ctc")

        if target_offer is not None and (benchmark_p80 is None or target_offer <= benchmark_p80 * 1.05):
            suggested_ctc = target_offer
            action = "raise_offer"
        elif benchmark_p80 is not None:
            suggested_ctc = benchmark_p80
            action = "escalate_above_market"
        else:
            suggested_ctc = None
            action = "escalate_insufficient_data"

        flagged.append(
            {
                "candidate_ref": row["candidate_ref"],
                "offer_date": row["offer_date"],
                "primary_skill": candidate["primary_skill"],
                "lob": candidate["lob"],
                "location": candidate["location"],
                "offered_band": candidate["offered_band"],
                "offered_ctc": round(candidate["offered_ctc"], 2),
                "acceptance_probability": round(probability, 3),
                "suggested_ctc": None if suggested_ctc is None else round(float(suggested_ctc), 2),
                "action": action,
                "urgency": "High" if probability < 0.35 else "Medium",
                "alert_message": _draft_alert(row, candidate, probability, suggested_ctc, action),
                "actual_outcome": row["status"],
            }
        )

    flagged.sort(key=lambda item: item["acceptance_probability"])
    flagged = flagged[:top_n]

    return {
        "queue_scanned": int(len(queue)),
        "flagged_count": len(flagged),
        "risk_threshold": risk_threshold,
        "flagged_offers": flagged,
    }


def _candidate_from_row(row: pd.Series) -> dict:
    return {
        "current_ctc": float(row["current_ctc"]),
        "expected_ctc": float(row["expected_ctc"]),
        "offered_ctc": float(row["offered_ctc"]),
        "relevant_experience_years": float(row["relevant_experience_years"]),
        "notice_period_days": float(row["notice_period_days"]),
        "offered_band": row["offered_band"],
        "candidate_source": row["candidate_source"],
        "lob": row["lob"],
        "primary_skill": row["primary_skill"],
        "previous_company_type": row["previous_company_type"],
        "location": row["location"],
        "city_tier": row["city_tier"],
        "joining_bonus": row.get("joining_bonus", 0) or 0,
        "relocation": row.get("relocation", 0) or 0,
        "offer_year": int(row["offer_year"]),
        "offer_month": int(row["offer_month"]),
        "offer_quarter": int(row["offer_quarter"]),
    }


def _draft_alert(row: pd.Series, candidate: dict, probability: float, suggested_ctc: float | None, action: str) -> str:
    ref = row["candidate_ref"]
    skill = candidate["primary_skill"]
    band = candidate["offered_band"]
    offered = candidate["offered_ctc"]

    if action == "raise_offer" and suggested_ctc is not None:
        return (
            f"Risk alert: offer {ref} ({skill}, {band}) is at {probability:.0%} predicted acceptance "
            f"at {offered:.2f} LPA. Raising to {suggested_ctc:.2f} LPA is projected to reach the 70% target "
            "without exceeding the historical benchmark. Recommend adjusting before the candidate goes cold."
        )
    if action == "escalate_above_market" and suggested_ctc is not None:
        return (
            f"Risk alert: offer {ref} ({skill}, {band}) is at {probability:.0%} predicted acceptance. "
            f"Reaching a safe acceptance likelihood would require exceeding the benchmark P80 of {suggested_ctc:.2f} LPA. "
            "Escalate to the hiring manager for a compensation exception or explore non-cash levers (joining bonus, relocation support)."
        )
    return (
        f"Risk alert: offer {ref} ({skill}, {band}) is at {probability:.0%} predicted acceptance and there is "
        "insufficient historical data to size a confident correction. Recommend manual review."
    )
