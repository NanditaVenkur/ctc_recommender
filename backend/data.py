from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "synthetic_hr_offer_acceptance_dataset.csv"
DB_PATH = ROOT / "ctc_recommender.sqlite3"


COLUMN_MAP = {
    "SLNO": "slno",
    "Candidate Ref": "candidate_ref",
    "Offer Date": "offer_date",
    "DOJ Extended": "doj_extended",
    "Duration to accept offer": "duration_to_accept_days",
    "Notice period": "notice_period_days",
    "Offered band": "offered_band",
    "Current CTC": "current_ctc",
    "Expected CTC": "expected_ctc",
    "Offered CTC": "offered_ctc",
    "Negotiated CTC": "negotiated_ctc",
    "Final CTC": "final_ctc",
    "Pecent hike expected in CTC": "expected_hike_pct_original",
    "Percent hike offered in CTC": "offered_hike_pct_original",
    "Percent difference CTC": "offer_gap_pct_original",
    "Joining Bonus": "joining_bonus",
    "Candidate relocate actual": "relocation",
    "Gender": "gender",
    "Candidate Source": "candidate_source",
    "Rex in Yrs": "relevant_experience_years",
    "LOB": "lob",
    "Primary Skill": "primary_skill",
    "Previous Company Type": "previous_company_type",
    "Location": "location",
    "Age": "age",
    "Status": "status",
}

NUMERIC_COLUMNS = [
    "duration_to_accept_days",
    "notice_period_days",
    "current_ctc",
    "expected_ctc",
    "offered_ctc",
    "negotiated_ctc",
    "final_ctc",
    "expected_hike_pct_original",
    "offered_hike_pct_original",
    "offer_gap_pct_original",
    "relevant_experience_years",
    "age",
]

TIER_1_CITIES = {
    "Bangalore",
    "Mumbai",
    "Delhi",
    "NCR",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
}
TIER_2_CITIES = {
    "Noida",
    "Gurgaon",
    "Gurugram",
    "Ahmedabad",
    "Kochi",
    "Coimbatore",
    "Indore",
    "Jaipur",
    "Chandigarh",
    "Mysore",
}


def city_tier(location: object) -> str:
    if pd.isna(location):
        return "Unknown"
    value = str(location).strip()
    if value in TIER_1_CITIES:
        return "Tier 1"
    if value in TIER_2_CITIES:
        return "Tier 2"
    return "Tier 3/Other"


def yes_no_to_int(value: object) -> float:
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    return np.nan


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_MAP)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["offer_date"] = pd.to_datetime(df["offer_date"], errors="coerce")
    df["offer_year"] = df["offer_date"].dt.year
    df["offer_month"] = df["offer_date"].dt.month
    df["offer_quarter"] = df["offer_date"].dt.quarter

    for column in ["joining_bonus", "relocation", "doj_extended"]:
        df[column] = df[column].apply(yes_no_to_int)

    categorical_columns = [
        "offered_band",
        "candidate_source",
        "lob",
        "primary_skill",
        "previous_company_type",
        "location",
        "status",
    ]
    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip()

    df["city_tier"] = df["location"].apply(city_tier)
    df["accepted"] = df["status"].isin({"Joined", "Accepted"}).astype(int)
    df["expected_hike_pct"] = (
        (df["expected_ctc"] - df["current_ctc"]) / df["current_ctc"] * 100
    )
    df["offered_hike_pct"] = (
        (df["offered_ctc"] - df["current_ctc"]) / df["current_ctc"] * 100
    )
    df["offer_gap_pct"] = (
        (df["offered_ctc"] - df["expected_ctc"]) / df["expected_ctc"] * 100
    )
    df["offer_gap_amount"] = df["offered_ctc"] - df["expected_ctc"]
    df["offer_date"] = df["offer_date"].dt.strftime("%Y-%m-%d")

    return df


def init_db() -> None:
    df = load_clean_data()
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("offers", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_status ON offers(status)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_offers_profile ON offers(primary_skill, lob, location, offered_band)"
        )


def read_offers() -> pd.DataFrame:
    if not DB_PATH.exists():
        init_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM offers", conn)
