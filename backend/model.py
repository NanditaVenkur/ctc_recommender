from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FEATURES = [
    "current_ctc",
    "expected_ctc",
    "offered_ctc",
    "relevant_experience_years",
    "notice_period_days",
    "joining_bonus",
    "relocation",
    "offer_year",
    "offer_month",
    "offer_quarter",
    "expected_hike_pct",
    "offered_hike_pct",
    "offer_gap_pct",
    "offer_gap_amount",
    "offered_band",
    "candidate_source",
    "lob",
    "primary_skill",
    "previous_company_type",
    "location",
    "city_tier",
]


@dataclass
class TrainedAcceptanceModel:
    pipeline: Pipeline
    metrics: dict


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[MODEL_FEATURES].copy()


def train_acceptance_model(df: pd.DataFrame) -> TrainedAcceptanceModel:
    X = _feature_frame(df)
    y = df["accepted"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "brier_score": round(float(brier_score_loss(y_test, proba)), 3),
        "test_records": int(len(y_test)),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    return TrainedAcceptanceModel(pipeline=pipeline, metrics=metrics)


def candidate_to_feature_row(candidate: dict, offer: float) -> pd.DataFrame:
    current_ctc = float(candidate["current_ctc"])
    expected_ctc = float(candidate["expected_ctc"])
    offered_ctc = float(offer)

    row = {
        "current_ctc": current_ctc,
        "expected_ctc": expected_ctc,
        "offered_ctc": offered_ctc,
        "relevant_experience_years": float(candidate["relevant_experience_years"]),
        "notice_period_days": float(candidate["notice_period_days"]),
        "joining_bonus": float(candidate.get("joining_bonus", 0)),
        "relocation": float(candidate.get("relocation", 0)),
        "offer_year": int(candidate.get("offer_year", 2026)),
        "offer_month": int(candidate.get("offer_month", 7)),
        "offer_quarter": int(candidate.get("offer_quarter", 3)),
        "expected_hike_pct": (expected_ctc - current_ctc) / current_ctc * 100,
        "offered_hike_pct": (offered_ctc - current_ctc) / current_ctc * 100,
        "offer_gap_pct": (offered_ctc - expected_ctc) / expected_ctc * 100,
        "offer_gap_amount": offered_ctc - expected_ctc,
        "offered_band": candidate["offered_band"],
        "candidate_source": candidate["candidate_source"],
        "lob": candidate["lob"],
        "primary_skill": candidate["primary_skill"],
        "previous_company_type": candidate["previous_company_type"],
        "location": candidate["location"],
        "city_tier": candidate["city_tier"],
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def acceptance_curve(
    model: TrainedAcceptanceModel, candidate: dict, points: int = 40
) -> list[dict]:
    low = max(
        float(candidate["current_ctc"]) * 1.02, float(candidate["offered_ctc"]) * 0.75
    )
    high = max(
        float(candidate["expected_ctc"]) * 1.6, float(candidate["offered_ctc"]) * 2.0
    )
    offers = np.linspace(low, high, points)
    rows = []
    for offer in offers:
        features = candidate_to_feature_row(candidate, float(offer))
        probability = model.pipeline.predict_proba(features)[0, 1]
        rows.append(
            {
                "offered_ctc": round(float(offer), 2),
                "acceptance_probability": round(float(probability), 3),
            }
        )
    return rows


def predict_acceptance(
    model: TrainedAcceptanceModel, candidate: dict, offer: float
) -> float:
    features = candidate_to_feature_row(candidate, offer)
    return float(model.pipeline.predict_proba(features)[0, 1])


def category_support(
    df: pd.DataFrame, candidate: dict, columns: list[str]
) -> dict[str, int]:
    support = {}
    for column in columns:
        value = candidate.get(column)
        support[column] = (
            int((df[column] == value).sum())
            if value is not None and column in df.columns
            else 0
        )
    return support
