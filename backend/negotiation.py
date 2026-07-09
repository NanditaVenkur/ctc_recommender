from __future__ import annotations

import pandas as pd

from .model import TrainedAcceptanceModel, predict_acceptance
from .recommender import flexible_percentiles


MIN_BENCHMARK_RECORDS = 8
CANDIDATE_CONCESSION_RATE = 0.25
RECRUITER_CONCESSION_RATE = 0.5


def run_negotiation(
    df: pd.DataFrame,
    model: TrainedAcceptanceModel,
    candidate: dict,
    target_probability: float = 0.75,
    max_rounds: int = 6,
    budget_cap: float | None = None,
) -> dict:
    """Simulate a bilateral negotiation between a Recruiter Agent and a Candidate Agent.

    Both agents only ever see numbers produced by the trained acceptance model and the
    historical benchmark engine - the negotiation logic decides who moves and by how much,
    it never invents an acceptance probability or a CTC out of thin air.
    """
    current_ctc = float(candidate["current_ctc"])
    expected_ctc = float(candidate["expected_ctc"])
    starting_offer = float(candidate["offered_ctc"])
    max_rounds = max(1, int(max_rounds))
    target_probability = max(0.05, min(float(target_probability), 0.99))

    percentiles = flexible_percentiles(df, candidate, flexibility="balanced", min_records=MIN_BENCHMARK_RECORDS)
    benchmark_p80 = percentiles.get("p80_offered_ctc")

    if budget_cap is None:
        candidates_for_cap = [starting_offer * 1.15, expected_ctc * 1.25]
        if benchmark_p80:
            candidates_for_cap.append(benchmark_p80 * 1.1)
        budget_cap = max(candidates_for_cap)
    budget_cap = float(budget_cap)

    recruiter_offer = max(starting_offer, current_ctc * 1.02)
    candidate_ask = max(expected_ctc, recruiter_offer)

    rounds: list[dict] = []
    status = "in_progress"
    final_offer = recruiter_offer

    for round_number in range(1, max_rounds + 1):
        round_candidate = dict(candidate)
        round_candidate["offered_ctc"] = recruiter_offer
        probability = predict_acceptance(model, round_candidate, recruiter_offer)
        final_offer = recruiter_offer

        rounds.append(
            {
                "round": round_number,
                "recruiter_offer": round(recruiter_offer, 2),
                "candidate_ask": round(candidate_ask, 2),
                "acceptance_probability": round(probability, 3),
                "recruiter_message": _recruiter_message(round_number, recruiter_offer, probability, budget_cap),
                "candidate_message": _candidate_message(candidate_ask, expected_ctc, probability),
            }
        )

        if probability >= target_probability:
            status = "agreed"
            break
        if recruiter_offer >= budget_cap - 0.01:
            status = "impasse"
            break
        if round_number == max_rounds:
            status = "max_rounds_reached"
            break

        gap = candidate_ask - recruiter_offer
        candidate_ask = max(recruiter_offer, candidate_ask - gap * CANDIDATE_CONCESSION_RATE)
        recruiter_offer = min(budget_cap, recruiter_offer + gap * RECRUITER_CONCESSION_RATE)

    final_candidate = dict(candidate)
    final_candidate["offered_ctc"] = final_offer
    final_probability = predict_acceptance(model, final_candidate, final_offer)

    return {
        "status": status,
        "rounds": rounds,
        "final_offer": round(float(final_offer), 2),
        "final_probability": round(float(final_probability), 3),
        "target_probability": target_probability,
        "budget_cap": round(budget_cap, 2),
        "benchmark_p50": percentiles.get("p50_offered_ctc"),
        "benchmark_p80": percentiles.get("p80_offered_ctc"),
        "summary": _summary_message(status, final_offer, final_probability, budget_cap, target_probability, max_rounds),
    }


def _recruiter_message(round_number: int, offer: float, probability: float, budget_cap: float) -> str:
    if round_number == 1:
        return f"Recruiter Agent: Opening at {offer:.2f} LPA. Model estimates {probability:.0%} acceptance likelihood."
    headroom = budget_cap - offer
    if headroom <= 0.01:
        return f"Recruiter Agent: This is our ceiling at {offer:.2f} LPA ({probability:.0%} predicted acceptance). No further budget available."
    return f"Recruiter Agent: Moving to {offer:.2f} LPA ({probability:.0%} predicted acceptance), {headroom:.2f} LPA below the authorized ceiling."


def _candidate_message(ask: float, expected_ctc: float, probability: float) -> str:
    if abs(ask - expected_ctc) < 0.05:
        return f"Candidate Agent: My expectation is {expected_ctc:.2f} LPA based on current market standing."
    if probability >= 0.7:
        return f"Candidate Agent: This is close to acceptable; I could move to around {ask:.2f} LPA."
    return f"Candidate Agent: I would need closer to {ask:.2f} LPA to seriously consider this offer."


def _summary_message(
    status: str,
    final_offer: float,
    final_probability: float,
    budget_cap: float,
    target_probability: float,
    max_rounds: int,
) -> str:
    if status == "agreed":
        return f"Agreement reached at {final_offer:.2f} LPA with {final_probability:.0%} predicted acceptance (target was {target_probability:.0%})."
    if status == "impasse":
        return (
            f"Impasse: reached the authorized budget ceiling of {budget_cap:.2f} LPA with only "
            f"{final_probability:.0%} predicted acceptance. Recommend escalating for a compensation exception "
            "or offering non-cash levers (joining bonus, relocation support, faster onboarding)."
        )
    return (
        f"No agreement within {max_rounds} rounds; best reached offer is {final_offer:.2f} LPA at "
        f"{final_probability:.0%} predicted acceptance. Consider extending rounds or adjusting the target probability."
    )
