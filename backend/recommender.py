from __future__ import annotations

import numpy as np
import pandas as pd


FALLBACK_LEVELS = {
    "strict": [
        ["primary_skill", "lob", "location", "offered_band"],
        ["primary_skill", "lob", "city_tier", "offered_band"],
        ["primary_skill", "lob", "offered_band"],
        ["lob", "location", "offered_band"],
        ["lob", "city_tier", "offered_band"],
        ["lob", "offered_band"],
        ["offered_band"],
    ],
    "balanced": [
        ["primary_skill", "lob", "location", "offered_band"],
        ["primary_skill", "lob", "offered_band"],
        ["lob", "city_tier", "offered_band"],
        ["lob", "offered_band"],
        ["primary_skill", "lob"],
        ["offered_band"],
    ],
    "broad": [
        ["lob", "city_tier", "offered_band"],
        ["lob", "offered_band"],
        ["primary_skill", "lob"],
        ["lob"],
        ["offered_band"],
        [],
    ],
}


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _filter(data: pd.DataFrame, filters: dict) -> tuple[pd.DataFrame, dict]:
    subset = data.copy()
    applied = {}
    for column, value in filters.items():
        value = _clean_value(value)
        if value is None or column not in subset.columns:
            continue
        subset = subset[subset[column] == value]
        applied[column] = value
    return subset, applied


def _summarize(subset: pd.DataFrame, filters: dict, min_records: int) -> dict:
    accepted = subset[subset["accepted"] == 1]
    percentiles = accepted["offered_ctc"].quantile([0.2, 0.5, 0.8]).to_dict()
    accepted_count = int(len(accepted))
    warning = None
    if accepted_count < min_records:
        warning = f"Only {accepted_count} accepted records found. Broaden filters or treat recommendation cautiously."
    return {
        "filters_used": filters,
        "similar_records": int(len(subset)),
        "accepted_similar_records": accepted_count,
        "acceptance_rate": None if len(subset) == 0 else round(float(subset["accepted"].mean()), 3),
        "p20_offered_ctc": None if np.isnan(percentiles.get(0.2, np.nan)) else round(float(percentiles[0.2]), 2),
        "p50_offered_ctc": None if np.isnan(percentiles.get(0.5, np.nan)) else round(float(percentiles[0.5]), 2),
        "p80_offered_ctc": None if np.isnan(percentiles.get(0.8, np.nan)) else round(float(percentiles[0.8]), 2),
        "warning": warning,
    }


def flexible_percentiles(data: pd.DataFrame, candidate: dict, flexibility: str = "balanced", min_records: int = 20) -> dict:
    if flexibility not in FALLBACK_LEVELS:
        flexibility = "balanced"

    exact = {
        "primary_skill": candidate.get("primary_skill"),
        "lob": candidate.get("lob"),
        "location": candidate.get("location"),
        "city_tier": candidate.get("city_tier"),
        "offered_band": candidate.get("offered_band"),
    }

    attempts = []
    best = None
    for level in FALLBACK_LEVELS[flexibility]:
        filters = {field: exact[field] for field in level if exact.get(field) is not None}
        subset, applied = _filter(data, filters)
        summary = _summarize(subset, applied, min_records)
        attempts.append(summary)

        if summary["accepted_similar_records"] >= min_records:
            summary = dict(summary)
            summary["fallback_attempts"] = _serializable_attempts(attempts)
            summary["confidence"] = "High" if summary["accepted_similar_records"] >= min_records * 2 else "Medium"
            summary["specificity"] = _specificity_label(summary["filters_used"])
            return summary

        if best is None or summary["accepted_similar_records"] > best["accepted_similar_records"]:
            best = summary

    best = dict(best)
    best["fallback_attempts"] = _serializable_attempts(attempts)
    best["confidence"] = "Low"
    best["specificity"] = _specificity_label(best["filters_used"])
    return best


def _serializable_attempts(attempts: list[dict]) -> list[dict]:
    cleaned = []
    for attempt in attempts:
        item = dict(attempt)
        item.pop("fallback_attempts", None)
        cleaned.append(item)
    return cleaned


def _specificity_label(filters: dict) -> str:
    count = len(filters)
    if count >= 4:
        return "Exact profile"
    if count >= 2:
        return "Related segment"
    return "Broad benchmark"
