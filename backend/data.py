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

TIER_1_CITIES = {"Bangalore", "Mumbai", "Delhi", "NCR", "Hyderabad", "Chennai", "Pune", "Kolkata"}
TIER_2_CITIES = {"Noida", "Gurgaon", "Gurugram", "Ahmedabad", "Kochi", "Coimbatore", "Indore", "Jaipur", "Chandigarh", "Mysore"}


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
    df["expected_hike_pct"] = (df["expected_ctc"] - df["current_ctc"]) / df["current_ctc"] * 100
    df["offered_hike_pct"] = (df["offered_ctc"] - df["current_ctc"]) / df["current_ctc"] * 100
    df["offer_gap_pct"] = (df["offered_ctc"] - df["expected_ctc"]) / df["expected_ctc"] * 100
    df["offer_gap_amount"] = df["offered_ctc"] - df["expected_ctc"]
    df["offer_date"] = df["offer_date"].dt.strftime("%Y-%m-%d")

    return df


def init_db(force: bool = False) -> None:
    if not force and DB_PATH.exists():
        return
    df = load_clean_data()
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("offers", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_status ON offers(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_profile ON offers(primary_skill, lob, location, offered_band)")


def read_offers() -> pd.DataFrame:
    if not DB_PATH.exists():
        init_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM offers", conn)


SYNONYMS = {
    "slno": ["slno", "sl no", "serial number", "sl_no"],
    "candidate_ref": ["candidate_ref", "candidate ref", "ref", "candidate id", "id", "candidate_id"],
    "offer_date": ["offer_date", "offer date", "date", "offer_dt", "offer dt"],
    "doj_extended": ["doj_extended", "doj extended", "doj", "date of joining extended"],
    "duration_to_accept_days": ["duration to accept offer", "duration_to_accept_days", "duration", "accept_duration"],
    "notice_period_days": ["notice period", "notice_period", "notice_period_days", "notice", "np"],
    "offered_band": ["offered band", "offered_band", "band", "off_band"],
    "current_ctc": ["current ctc", "current_ctc", "current", "current ctc (lpa)", "current_ctc_lpa"],
    "expected_ctc": ["expected ctc", "expected_ctc", "expected", "expected ctc (lpa)", "expected_ctc_lpa"],
    "offered_ctc": ["offered ctc", "offered_ctc", "offered", "offered ctc (lpa)", "offered_ctc_lpa"],
    "negotiated_ctc": ["negotiated ctc", "negotiated_ctc"],
    "final_ctc": ["final ctc", "final_ctc"],
    "joining_bonus": ["joining bonus", "joining_bonus", "join bonus", "jb"],
    "relocation": ["candidate relocate actual", "candidate relocate", "relocation", "relocate"],
    "gender": ["gender", "sex"],
    "candidate_source": ["candidate source", "candidate_source", "source"],
    "relevant_experience_years": ["rex in yrs", "rex_in_yrs", "rex", "relevant experience years", "relevant_experience_years", "experience", "exp", "relevant_exp_years"],
    "lob": ["lob", "line of business", "business unit", "bu"],
    "primary_skill": ["primary skill", "primary_skill", "skill", "skills"],
    "previous_company_type": ["previous company type", "previous_company_type", "previous company work type", "company type", "prev_company_type"],
    "location": ["location", "city", "job_location"],
    "age": ["age"],
    "status": ["status", "outcome"],
}


def llm_map_columns(columns: list[str]) -> dict[str, str]:
    try:
        from .agent import get_llm
        llm, err = get_llm()
        if err or not llm:
            print(f"LLM Column mapping disabled: {err}")
            return {}
            
        system_prompt = (
            "You are a precise data mapper assistant.\n"
            "Your task is to map a list of raw column headers from an uploaded spreadsheet to a standard set of target columns for an HR offer dataset.\n\n"
            "Target columns:\n"
            "- candidate_ref (Candidate reference identifier, ID)\n"
            "- offer_date (Date of the offer)\n"
            "- doj_extended (DOJ extended yes/no)\n"
            "- duration_to_accept_days (Days to accept)\n"
            "- notice_period_days (Notice period in days)\n"
            "- offered_band (Offered band or grade or level)\n"
            "- current_ctc (Current CTC or salary)\n"
            "- expected_ctc (Expected CTC or salary)\n"
            "- offered_ctc (Offered CTC or salary)\n"
            "- negotiated_ctc (Negotiated CTC)\n"
            "- final_ctc (Final CTC)\n"
            "- joining_bonus (Joining bonus yes/no or amount)\n"
            "- relocation (Relocation yes/no)\n"
            "- gender (Gender or sex)\n"
            "- candidate_source (Candidate source or channel)\n"
            "- relevant_experience_years (Relevant experience in years or Rex)\n"
            "- lob (Line of business or business unit or department)\n"
            "- primary_skill (Primary skill or technology)\n"
            "- previous_company_type (Previous company type or previous company work type)\n"
            "- location (Location or city)\n"
            "- age (Age)\n"
            "- status (Offer status or outcome or joining status)\n\n"
            "Return a JSON object where the keys are the raw column headers and the values are the mapped target columns.\n"
            "Only map raw columns that have a clear semantic match. If a raw column does not map to any target column, do not include it or set it to null.\n"
            "Respond ONLY with the raw JSON object, no explanation, no markdown blocks."
        )
        
        user_message = f"Please map these raw column headers: {columns}"
        
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        text = str(response.content).strip()
        
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        import json
        mapping = json.loads(text)
        
        valid_targets = {
            "candidate_ref", "offer_date", "doj_extended", "duration_to_accept_days",
            "notice_period_days", "offered_band", "current_ctc", "expected_ctc",
            "offered_ctc", "negotiated_ctc", "final_ctc", "joining_bonus",
            "relocation", "gender", "candidate_source", "relevant_experience_years",
            "lob", "primary_skill", "previous_company_type", "location", "age", "status"
        }
        
        cleaned_mapping = {}
        for k, v in mapping.items():
            if v in valid_targets and k in columns:
                cleaned_mapping[k] = v
        return cleaned_mapping
    except Exception as exc:
        print(f"LLM mapping error: {exc}")
        return {}


def map_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_mapping = {}
    
    # 1. Try LLM semantic mapping first
    llm_mapped = llm_map_columns(df.columns.tolist())
    if llm_mapped:
        for k, v in llm_mapped.items():
            if v not in col_mapping.values():
                col_mapping[k] = v
                
    # 2. Fallback to rule-based synonyms mapping
    for col in df.columns:
        if col in col_mapping:
            continue
        normalized = str(col).lower().replace("_", " ").replace("-", " ").strip()
        for std_key, syn_list in SYNONYMS.items():
            if normalized in syn_list or any(str(syn).lower().replace("_", " ").replace("-", " ").strip() == normalized for syn in syn_list):
                if std_key not in col_mapping.values():
                    col_mapping[col] = std_key
                    break
    return df.rename(columns=col_mapping)


def clean_and_prepare_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Standardize numeric types
    for numeric_col in ["current_ctc", "expected_ctc", "offered_ctc", "relevant_experience_years"]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
            
    # Check notice_period_days
    if "notice_period_days" in df.columns:
        df["notice_period_days"] = pd.to_numeric(df["notice_period_days"], errors="coerce")
    else:
        df["notice_period_days"] = 30.0
        
    # Check joining_bonus
    if "joining_bonus" in df.columns:
        df["joining_bonus"] = df["joining_bonus"].apply(yes_no_to_int)
    else:
        df["joining_bonus"] = 0
        
    # Check relocation
    if "relocation" in df.columns:
        df["relocation"] = df["relocation"].apply(yes_no_to_int)
    else:
        df["relocation"] = 0
        
    # Check doj_extended
    if "doj_extended" in df.columns:
        df["doj_extended"] = df["doj_extended"].apply(yes_no_to_int)
    else:
        df["doj_extended"] = 0
        
    # Check age
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = 30.0
        
    # Check duration_to_accept_days
    if "duration_to_accept_days" in df.columns:
        df["duration_to_accept_days"] = pd.to_numeric(df["duration_to_accept_days"], errors="coerce")
    else:
        df["duration_to_accept_days"] = 5.0
        
    # Check candidate_ref
    if "candidate_ref" not in df.columns or df["candidate_ref"].isna().all():
        df["candidate_ref"] = [f"CAN_{i+1:05d}" for i in range(len(df))]
        
    # Check offer_date
    if "offer_date" in df.columns:
        df["offer_date"] = pd.to_datetime(df["offer_date"], errors="coerce")
    else:
        df["offer_date"] = pd.to_datetime("2026-07-09")
        
    df["offer_year"] = df["offer_date"].dt.year.fillna(2026).astype(int)
    df["offer_month"] = df["offer_date"].dt.month.fillna(7).astype(int)
    df["offer_quarter"] = df["offer_date"].dt.quarter.fillna(3).astype(int)
    
    # Categoricals defaults
    categorical_defaults = {
        "offered_band": "E2",
        "candidate_source": "Direct",
        "lob": "Digital",
        "primary_skill": "Unknown",
        "previous_company_type": "Service",
        "location": "Bangalore",
        "gender": "Male",
        "status": "Joined",
    }
    
    for col, default in categorical_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str).str.strip()
        
    # Calculated columns
    df["city_tier"] = df["location"].apply(city_tier)
    df["accepted"] = df["status"].isin({"Joined", "Accepted", "Joined", "accepted", "joined"}).astype(int)
    
    # Safely compute percentages
    df["expected_hike_pct"] = np.where(df["current_ctc"] > 0, (df["expected_ctc"] - df["current_ctc"]) / df["current_ctc"] * 100, 0.0)
    df["offered_hike_pct"] = np.where(df["current_ctc"] > 0, (df["offered_ctc"] - df["current_ctc"]) / df["current_ctc"] * 100, 0.0)
    df["offer_gap_pct"] = np.where(df["expected_ctc"] > 0, (df["offered_ctc"] - df["expected_ctc"]) / df["expected_ctc"] * 100, 0.0)
    df["offer_gap_amount"] = df["offered_ctc"] - df["expected_ctc"]
    
    # Re-format date to string
    df["offer_date"] = df["offer_date"].dt.strftime("%Y-%m-%d")
    
    final_cols = [
        "candidate_ref", "offer_date", "doj_extended", "duration_to_accept_days",
        "notice_period_days", "offered_band", "current_ctc", "expected_ctc",
        "offered_ctc", "joining_bonus", "relocation", "gender", "candidate_source",
        "relevant_experience_years", "lob", "primary_skill", "previous_company_type",
        "location", "age", "status", "offer_year", "offer_month", "offer_quarter",
        "city_tier", "accepted", "expected_hike_pct", "offered_hike_pct",
        "offer_gap_pct", "offer_gap_amount"
    ]
    
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    return df[final_cols]


def update_db_with_df(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("offers", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_status ON offers(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_profile ON offers(primary_skill, lob, location, offered_band)")


