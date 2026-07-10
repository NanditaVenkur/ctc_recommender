from __future__ import annotations

import numpy as np
import pandas as pd

from .model import TrainedAcceptanceModel, predict_acceptance
from .recommender import flexible_percentiles


MIN_BENCHMARK_RECORDS = 8
CANDIDATE_CONCESSION_RATE = 0.30
RECRUITER_CONCESSION_RATE = 0.55


def run_negotiation(
    df: pd.DataFrame,
    model: TrainedAcceptanceModel,
    candidate: dict,
    target_probability: float = 0.75,
    max_rounds: int = 6,
    budget_cap: float | None = None,
) -> dict:
    """Simulate a practical offer negotiation between a recruiter and candidate.

    The model probability is treated as risk evidence, not as the same thing as a
    candidate saying yes. A negotiation can therefore end as a clean agreement,
    an agreement with residual joining risk, an impasse, or no agreement.
    """
    current_ctc = float(candidate["current_ctc"])
    expected_ctc = float(candidate["expected_ctc"])
    starting_offer = float(candidate["offered_ctc"])
    max_rounds = max(1, int(max_rounds))
    target_probability = max(0.05, min(float(target_probability), 0.99))

    percentiles = flexible_percentiles(df, candidate, flexibility="balanced", min_records=MIN_BENCHMARK_RECORDS)
    benchmark_p50 = percentiles.get("p50_offered_ctc")
    benchmark_p80 = percentiles.get("p80_offered_ctc")
    support_warning = _support_warning(df, candidate)

    if budget_cap is None:
        budget_cap = _default_budget_cap(current_ctc, expected_ctc, starting_offer, benchmark_p80)
    budget_cap = float(max(starting_offer, budget_cap))

    recruiter_offer = max(starting_offer, current_ctc * 1.02)
    candidate_ask = max(expected_ctc, recruiter_offer)
    candidate_floor = _candidate_floor(current_ctc, expected_ctc, starting_offer)
    lever_state = {"joining_bonus": bool(int(candidate.get("joining_bonus", 0))), "notice_buyout": False}
    minimum_offer_for_target = _minimum_offer_for_target(model, candidate, recruiter_offer, budget_cap, target_probability, lever_state)

    rounds: list[dict] = []
    status = "in_progress"
    final_offer = recruiter_offer
    final_candidate_ask = candidate_ask
    final_probability = 0.0

    for round_number in range(1, max_rounds + 1):
        if round_number >= 3:
            _activate_realistic_levers(candidate, lever_state)

        probability = _predict_with_levers(model, candidate, recruiter_offer, lever_state)
        commercial_agreement = recruiter_offer + 0.01 >= candidate_ask
        target_probability_met = probability >= target_probability
        final_offer = recruiter_offer
        final_candidate_ask = candidate_ask
        final_probability = probability
        levers = _active_levers(lever_state)

        rounds.append(
            {
                "round": round_number,
                "recruiter_offer": round(recruiter_offer, 2),
                "candidate_ask": round(candidate_ask, 2),
                "acceptance_probability": round(probability, 3),
                "commercial_agreement": commercial_agreement,
                "target_probability_met": target_probability_met,
                "levers": levers,
                "recruiter_message": _recruiter_message(
                    round_number,
                    recruiter_offer,
                    probability,
                    budget_cap,
                    candidate_ask,
                    target_probability,
                    levers,
                    minimum_offer_for_target,
                ),
                "candidate_message": _candidate_message(
                    candidate_ask,
                    expected_ctc,
                    current_ctc,
                    recruiter_offer,
                    probability,
                    levers,
                ),
                "recruiter_reason": _recruiter_reason(
                    recruiter_offer,
                    candidate_ask,
                    probability,
                    target_probability,
                    budget_cap,
                    benchmark_p50,
                    benchmark_p80,
                ),
                "candidate_reason": _candidate_reason(candidate_ask, expected_ctc, current_ctc, recruiter_offer, levers),
            }
        )

        if commercial_agreement and target_probability_met:
            status = "agreed"
            break
        if commercial_agreement:
            status = "agreement_with_risk"
            break
        if recruiter_offer >= budget_cap - 0.01:
            status = "impasse"
            break
        if round_number == max_rounds:
            status = "max_rounds_reached"
            break

        gap = max(candidate_ask - recruiter_offer, 0)
        next_candidate_ask = max(candidate_floor, candidate_ask - gap * CANDIDATE_CONCESSION_RATE)
        next_recruiter_offer = recruiter_offer + gap * RECRUITER_CONCESSION_RATE

        if minimum_offer_for_target is not None and round_number >= max_rounds - 1:
            next_recruiter_offer = max(next_recruiter_offer, minimum_offer_for_target)
        if benchmark_p50 is not None and round_number >= 2:
            benchmark_anchor = min(float(benchmark_p50), candidate_ask, budget_cap)
            next_recruiter_offer = max(next_recruiter_offer, benchmark_anchor)

        recruiter_offer = min(budget_cap, max(recruiter_offer, next_recruiter_offer))
        candidate_ask = max(candidate_floor, min(candidate_ask, next_candidate_ask))

    final_probability = _predict_with_levers(model, candidate, final_offer, lever_state)
    commercial_agreement = final_offer + 0.01 >= final_candidate_ask
    target_probability_met = final_probability >= target_probability

    return {
        "status": status,
        "rounds": rounds,
        "final_offer": round(float(final_offer), 2),
        "final_candidate_ask": round(float(final_candidate_ask), 2),
        "final_probability": round(float(final_probability), 3),
        "target_probability": target_probability,
        "target_probability_met": target_probability_met,
        "commercial_agreement": commercial_agreement,
        "budget_cap": round(budget_cap, 2),
        "minimum_offer_for_target": None if minimum_offer_for_target is None else round(float(minimum_offer_for_target), 2),
        "active_levers": _active_levers(lever_state),
        "benchmark_p50": benchmark_p50,
        "benchmark_p80": benchmark_p80,
        "summary": _summary_message(
            status,
            final_offer,
            final_candidate_ask,
            final_probability,
            budget_cap,
            target_probability,
            max_rounds,
            minimum_offer_for_target,
            _active_levers(lever_state),
        ),
        "support_warning": support_warning,
        "next_action": _next_action(status, final_offer, final_candidate_ask, final_probability, target_probability, budget_cap, minimum_offer_for_target, support_warning),
        "method_notes": [
            "Commercial agreement means recruiter offer met or exceeded the candidate ask.",
            "Target acceptance means the ML model reached the selected probability threshold.",
            "Joining bonus/notice-buyout levers can reduce risk, but the final CTC remains the fixed offer shown.",
            *( [support_warning] if support_warning else [] ),
        ],
    }


def _support_warning(df: pd.DataFrame, candidate: dict) -> str | None:
    skill_lob = df[
        (df["primary_skill"] == candidate.get("primary_skill"))
        & (df["lob"] == candidate.get("lob"))
    ]
    exact = skill_lob[
        (skill_lob["location"] == candidate.get("location"))
        & (skill_lob["offered_band"] == candidate.get("offered_band"))
    ]
    if len(skill_lob) == 0:
        return "Weak evidence: no historical records match this skill and LOB combination, so treat the agreement as recruiter review rather than auto-approval."
    if len(exact) < 3:
        return "Limited evidence: fewer than 3 exact skill, LOB, location, and band records support this negotiation."
    return None


def _with_support_note(summary: str, support_warning: str | None) -> str:
    return f"{summary} {support_warning}" if support_warning else summary


def _default_budget_cap(current_ctc: float, expected_ctc: float, starting_offer: float, benchmark_p80: float | None) -> float:
    policy_caps = [expected_ctc * 1.15, current_ctc * 2.2]
    if benchmark_p80:
        policy_caps.append(float(benchmark_p80) * 1.05)
    return max(starting_offer, min(policy_caps))


def _candidate_floor(current_ctc: float, expected_ctc: float, starting_offer: float) -> float:
    return max(starting_offer, current_ctc * 1.20, expected_ctc * 0.92)


def _activate_realistic_levers(candidate: dict, lever_state: dict) -> None:
    if not lever_state["joining_bonus"]:
        lever_state["joining_bonus"] = True
    if float(candidate.get("notice_period_days", 0)) >= 60:
        lever_state["notice_buyout"] = True


def _active_levers(lever_state: dict) -> list[str]:
    levers = []
    if lever_state.get("joining_bonus"):
        levers.append("one-time joining bonus")
    if lever_state.get("notice_buyout"):
        levers.append("notice-period buyout support")
    return levers


def _predict_with_levers(model: TrainedAcceptanceModel, candidate: dict, offer: float, lever_state: dict) -> float:
    model_candidate = dict(candidate)
    if lever_state.get("joining_bonus"):
        model_candidate["joining_bonus"] = 1
    model_candidate["offered_ctc"] = offer
    probability = predict_acceptance(model, model_candidate, offer)
    if lever_state.get("notice_buyout"):
        probability = min(0.99, probability + 0.035)
    return float(probability)


def _minimum_offer_for_target(
    model: TrainedAcceptanceModel,
    candidate: dict,
    low: float,
    high: float,
    target_probability: float,
    lever_state: dict,
) -> float | None:
    if high < low:
        return None
    for offer in np.linspace(low, high, 48):
        if _predict_with_levers(model, candidate, float(offer), lever_state) >= target_probability:
            return float(offer)
    return None


def _recruiter_message(
    round_number: int,
    offer: float,
    probability: float,
    budget_cap: float,
    candidate_ask: float,
    target_probability: float,
    levers: list[str],
    minimum_offer_for_target: float | None,
) -> str:
    gap = max(candidate_ask - offer, 0)
    lever_text = f" I can also include {', '.join(levers)}." if levers else ""
    if round_number == 1:
        return (
            f"Recruiter Agent: I am opening at {offer:.2f} LPA. That is below your ask by {gap:.2f} LPA, "
            f"and the model estimates {probability:.0%} acceptance, so I know we may need to improve the package."
        )
    if gap <= 0.01:
        risk_text = "meets our target" if probability >= target_probability else "still carries joining risk"
        position_text = "exceed your current ask" if offer > candidate_ask + 0.25 else "match your ask"
        return f"Recruiter Agent: I can {position_text} at {offer:.2f} LPA. The model says {probability:.0%}, so this {risk_text}.{lever_text}"
    target_text = ""
    if minimum_offer_for_target is not None:
        target_text = f" Our model target is around {minimum_offer_for_target:.2f} LPA."
    return (
        f"Recruiter Agent: I can move to {offer:.2f} LPA, leaving a {gap:.2f} LPA gap to your ask. "
        f"Predicted acceptance is {probability:.0%}; my approval ceiling is {budget_cap:.2f} LPA.{target_text}{lever_text}"
    )


def _candidate_message(ask: float, expected_ctc: float, current_ctc: float, offer: float, probability: float, levers: list[str]) -> str:
    hike_at_offer = (offer - current_ctc) / current_ctc * 100
    gap = max(ask - offer, 0)
    if gap <= 0.01:
        if probability >= 0.7:
            return f"Candidate Agent: At {offer:.2f} LPA, I can accept. The hike is {hike_at_offer:.0f}% and the package is close enough to my expectation."
        return f"Candidate Agent: Commercially this matches my ask, but I would still want clarity on role, project, and joining support before I commit."
    lever_text = " The joining support helps, but fixed CTC is still my main concern." if levers else ""
    if abs(ask - expected_ctc) < 0.05:
        return f"Candidate Agent: My stated expectation is {expected_ctc:.2f} LPA because I am looking for a meaningful move from {current_ctc:.2f} LPA.{lever_text}"
    return f"Candidate Agent: I can come down to {ask:.2f} LPA, but I am still seeing a {gap:.2f} LPA gap versus your offer.{lever_text}"


def _recruiter_reason(
    offer: float,
    ask: float,
    probability: float,
    target_probability: float,
    budget_cap: float,
    benchmark_p50: float | None,
    benchmark_p80: float | None,
) -> str:
    reasons = []
    if offer < ask:
        reasons.append("offer remains below candidate ask")
    if probability < target_probability:
        reasons.append("model risk is below target")
    if benchmark_p50 is not None and offer < float(benchmark_p50):
        reasons.append("offer is below benchmark median")
    if benchmark_p80 is not None and offer > float(benchmark_p80):
        reasons.append("offer is above benchmark P80")
    if budget_cap - offer < 0.5:
        reasons.append("limited budget headroom")
    return "; ".join(reasons) if reasons else "offer is aligned with ask and model target"


def _candidate_reason(ask: float, expected_ctc: float, current_ctc: float, offer: float, levers: list[str]) -> str:
    hike_at_offer = (offer - current_ctc) / current_ctc * 100
    reasons = [f"offer implies {hike_at_offer:.0f}% hike"]
    if offer < expected_ctc:
        reasons.append("below stated expectation")
    if levers:
        reasons.append("non-cash support included")
    if ask <= offer + 0.01:
        reasons.append("commercial gap closed")
    return "; ".join(reasons)


def _summary_message(
    status: str,
    final_offer: float,
    final_candidate_ask: float,
    final_probability: float,
    budget_cap: float,
    target_probability: float,
    max_rounds: int,
    minimum_offer_for_target: float | None,
    levers: list[str],
) -> str:
    lever_text = f" Package includes {', '.join(levers)}." if levers else ""
    if status == "agreed":
        return f"Agreement reached at {final_offer:.2f} LPA with {final_probability:.0%} predicted acceptance against a {target_probability:.0%} target.{lever_text}"
    if status == "agreement_with_risk":
        return (
            f"Commercial agreement reached at {final_offer:.2f} LPA, but predicted acceptance is {final_probability:.0%}, "
            f"below the {target_probability:.0%} target. Treat as accepted in principle with joining risk.{lever_text}"
        )
    if status == "impasse":
        target_text = ""
        if minimum_offer_for_target is not None:
            target_text = f" Model target is near {minimum_offer_for_target:.2f} LPA."
        return (
            f"Impasse: recruiter reached the {budget_cap:.2f} LPA ceiling while the candidate ask was {final_candidate_ask:.2f} LPA. "
            f"Predicted acceptance is {final_probability:.0%}.{target_text}"
        )
    return (
        f"No agreement within {max_rounds} rounds. Best offer was {final_offer:.2f} LPA against candidate ask "
        f"{final_candidate_ask:.2f} LPA, with {final_probability:.0%} predicted acceptance."
    )


def _next_action(
    status: str,
    final_offer: float,
    final_candidate_ask: float,
    final_probability: float,
    target_probability: float,
    budget_cap: float,
    minimum_offer_for_target: float | None,
    support_warning: str | None = None,
) -> str:
    if support_warning:
        return "Review manually before offer release: the negotiation reached terms, but historical support for this profile is weak."
    if status == "agreed":
        return "Proceed to verbal confirmation and document the final package details."
    if status == "agreement_with_risk":
        return "Confirm role/project fit and joining constraints before releasing the offer letter. Consider adding a joining bonus or notice buyout if not already included."
    if minimum_offer_for_target is not None and minimum_offer_for_target <= budget_cap:
        return f"Ask approval to move near {minimum_offer_for_target:.2f} LPA, or keep CTC fixed and add non-cash levers."
    if final_offer < final_candidate_ask:
        return "Escalate only if the business can justify a compensation exception; otherwise reset expectations with the candidate."
    if final_probability < target_probability:
        return "Money may not be the only blocker. Validate competing offers, notice period, role fit, and relocation constraints."
    return "Review with the hiring manager before proceeding."
