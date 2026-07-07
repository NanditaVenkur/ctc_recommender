from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets" / "synthetic_hr_offer_acceptance_dataset.csv"

LOCATIONS = ["Bangalore", "Chennai", "Hyderabad", "Pune", "Noida", "Mumbai", "Kochi"]
LOBS = ["ERS", "INFRA", "Digital", "Cloud", "Data", "Cybersecurity", "SAP", "QA"]
SKILLS_BY_LOB = {
    "ERS": ["Embedded C", "Autosar", "VLSI", "IoT"],
    "INFRA": ["Linux Admin", "Network", "VMware", "Windows Server"],
    "Digital": ["Java Spring", "React", "Node.js", "Python"],
    "Cloud": ["AWS", "Azure", "DevOps", "Kubernetes"],
    "Data": ["SQL", "Power BI", "Data Engineering", "ML"],
    "Cybersecurity": ["SOC", "IAM", "Penetration Testing", "GRC"],
    "SAP": ["SAP ABAP", "SAP MM", "SAP FICO", "SAP Basis"],
    "QA": ["Manual Testing", "Selenium", "API Testing", "Performance Testing"],
}
SOURCES = ["Agency", "Employee Referral", "LinkedIn", "Naukri", "Campus", "Direct"]
COMPANY_TYPES = ["Service", "Product", "Startup", "Captive", "Consulting"]
GENDERS = ["Female", "Male", "Other"]

LOB_BASE = {
    "ERS": 6.0,
    "INFRA": 5.5,
    "Digital": 7.0,
    "Cloud": 8.0,
    "Data": 7.5,
    "Cybersecurity": 8.5,
    "SAP": 7.8,
    "QA": 5.2,
}
LOCATION_MULTIPLIER = {
    "Bangalore": 1.15,
    "Hyderabad": 1.08,
    "Pune": 1.02,
    "Chennai": 0.96,
    "Noida": 0.94,
    "Mumbai": 1.10,
    "Kochi": 0.88,
}
BAND_MULTIPLIER = {"E1": 0.85, "E2": 1.0, "E3": 1.25, "M1": 1.55, "M2": 1.95}
COMPANY_MULTIPLIER = {"Service": 0.95, "Product": 1.20, "Startup": 1.10, "Captive": 1.08, "Consulting": 1.05}


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def choose_band(exp: float) -> str:
    if exp < 3:
        return random.choices(["E1", "E2"], [0.8, 0.2])[0]
    if exp < 6:
        return random.choices(["E1", "E2", "E3"], [0.15, 0.7, 0.15])[0]
    if exp < 10:
        return random.choices(["E2", "E3", "M1"], [0.35, 0.5, 0.15])[0]
    return random.choices(["E3", "M1", "M2"], [0.25, 0.5, 0.25])[0]


def yes_no(probability_yes: float) -> str:
    return "Yes" if random.random() < probability_yes else "No"


def build_row(idx: int, start_date: date) -> dict:
    candidate_ref = 2110000 + idx
    exp = round(max(0.8, random.gauss(6.0, 3.0)), 1)
    age = int(round(22 + exp + random.gauss(2.5, 2.0)))
    lob = random.choice(LOBS)
    skill = random.choice(SKILLS_BY_LOB[lob])
    location = random.choice(LOCATIONS)
    band = choose_band(exp)
    source = random.choices(SOURCES, [0.18, 0.24, 0.20, 0.20, 0.05, 0.13])[0]
    company_type = random.choices(COMPANY_TYPES, [0.45, 0.18, 0.12, 0.15, 0.10])[0]
    gender = random.choices(GENDERS, [0.36, 0.62, 0.02])[0]
    notice_period = random.choice([30, 45, 60, 75, 90])
    relocate = yes_no(0.42 if location in ["Bangalore", "Hyderabad", "Pune"] else 0.30)
    joining_bonus = yes_no(0.18)

    market_ctc = (
        LOB_BASE[lob]
        * (1 + exp * 0.18)
        * LOCATION_MULTIPLIER[location]
        * BAND_MULTIPLIER[band]
        * COMPANY_MULTIPLIER[company_type]
    )
    current_ctc = round(max(2.2, random.gauss(market_ctc * 0.82, market_ctc * 0.16)), 2)
    expected_hike = round(random.gauss(38, 18), 2)
    expected_ctc = round(current_ctc * (1 + expected_hike / 100), 2)

    offer_strategy = random.gauss(32, 16)
    if source == "Employee Referral":
        offer_strategy += 3
    if joining_bonus == "Yes":
        offer_strategy += 4
    if company_type == "Product":
        offer_strategy += 5
    offered_ctc = round(max(current_ctc * 1.02, current_ctc * (1 + offer_strategy / 100)), 2)

    negotiated_ctc = ""
    if random.random() < 0.45:
        negotiated_ctc = round(offered_ctc * random.uniform(1.02, 1.12), 2)
        final_ctc = negotiated_ctc
    else:
        final_ctc = offered_ctc

    offered_hike = round((offered_ctc - current_ctc) / current_ctc * 100, 2)
    expected_hike_actual = round((expected_ctc - current_ctc) / current_ctc * 100, 2)
    percent_difference = round((offered_ctc - expected_ctc) / expected_ctc * 100, 2)

    competitiveness = (offered_ctc - expected_ctc) / max(expected_ctc, 1)
    accept_score = (
        -0.25
        + 4.2 * competitiveness
        + 0.012 * offered_hike
        - 0.008 * max(notice_period - 45, 0)
        + (0.25 if source == "Employee Referral" else 0)
        + (0.18 if joining_bonus == "Yes" else 0)
        + (0.12 if relocate == "Yes" else -0.08)
        + random.gauss(0, 0.45)
    )
    joined = random.random() < sigmoid(accept_score)

    if joined:
        status = random.choices(["Joined", "Accepted"], [0.86, 0.14])[0]
        duration = max(1, int(random.gauss(11, 5) + notice_period * 0.05))
        doj_extended = yes_no(0.20)
    else:
        status = random.choices(["Declined", "No Show"], [0.82, 0.18])[0]
        duration = max(1, int(random.gauss(8, 5)))
        doj_extended = yes_no(0.08)

    offer_date = start_date + timedelta(days=random.randint(0, 730))
    return {
        "SLNO": idx,
        "Candidate Ref": candidate_ref,
        "Offer Date": offer_date.isoformat(),
        "DOJ Extended": doj_extended,
        "Duration to accept offer": duration,
        "Notice period": notice_period,
        "Offered band": band,
        "Current CTC": current_ctc,
        "Expected CTC": expected_ctc,
        "Offered CTC": offered_ctc,
        "Negotiated CTC": negotiated_ctc,
        "Final CTC": final_ctc,
        "Pecent hike expected in CTC": expected_hike_actual,
        "Percent hike offered in CTC": offered_hike,
        "Percent difference CTC": percent_difference,
        "Joining Bonus": joining_bonus,
        "Candidate relocate actual": relocate,
        "Gender": gender,
        "Candidate Source": source,
        "Rex in Yrs": exp,
        "LOB": lob,
        "Primary Skill": skill,
        "Previous Company Type": company_type,
        "Location": location,
        "Age": age,
        "Status": status,
    }


def generate(rows: int, seed: int, output: Path) -> None:
    random.seed(seed)
    start_date = date(2024, 1, 1)
    generated = [build_row(idx, start_date) for idx in range(1, rows + 1)]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(generated[0].keys()))
        writer.writeheader()
        writer.writerows(generated)

    print(f"Wrote {len(generated)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.rows, args.seed, args.output)


if __name__ == "__main__":
    main()
