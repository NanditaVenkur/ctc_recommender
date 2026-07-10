from __future__ import annotations

import json
import os
import re
from dotenv import load_dotenv

from .data import city_tier
from .model import acceptance_curve, predict_acceptance
from .recommender import flexible_percentiles


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MIN_BENCHMARK_RECORDS = 5

SYSTEM_INSTRUCTION = """
You are an HR recruiter assistant inside the CTC Offer Intelligence platform.
You help recruiters understand historical offers, benchmark CTC ranges, and predicted acceptance probability.

Use tools for calculations instead of guessing. Treat model outputs as decision support, not as an automatic compensation decision.
Explain uncertainty clearly when benchmark support is broad or sample counts are low.

Critical compensation logic:
- Raw model probability is NOT the final recommendation when it conflicts with P20/P50/P80 benchmarks.
- If the model target is below benchmark P20, say the raw model probability is overconfident / below-benchmark and recommend the benchmark-adjusted suggested CTC.
- Use the phrase "raw model probability" for the ML score, and "benchmark-adjusted suggestion" for the final suggested CTC.
- If the user did not provide current CTC, expected CTC, offered CTC, experience, band, LOB, skill, and location, state any assumptions clearly or ask for the missing inputs. Do not imply a precise personalized probability without those fields.
- For band-sensitive questions, emphasize whether the offer is below P20, between P20-P50, between P50-P80, or above P80.

Useful behavior:
- Use simulator when asked about a candidate profile, acceptance probability, P20/P50/P80, or whether a given offer is competitive.
- Use optimize_offer when asked what CTC is needed for a target acceptance probability.
- Use kpis when asked about overall offer funnel, acceptance rate, or dashboard-level metrics.
- Use filter_ui when asked to open or show dashboard, simulator, negotiation twin, risk radar, or recent offers.

Keep responses concise, practical, and recruiter-friendly.
"""


def get_llm():
    try:
        from langchain_openai import ChatOpenAI

        key = os.environ.get("GROQ_API_KEY")
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not key:
            return None, "GROQ_API_KEY is not configured in backend/.env"

        return ChatOpenAI(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
            temperature=0.1,
        ), None
    except Exception as exc:
        return None, f"Groq LLM Error: {exc}"


def _latest_offer_period(df) -> dict:
    latest_year = int(df["offer_year"].max())
    latest_month = int(df.loc[df["offer_year"] == latest_year, "offer_month"].max())
    return {
        "offer_year": latest_year,
        "offer_month": latest_month,
        "offer_quarter": (latest_month - 1) // 3 + 1,
    }


def _candidate_payload(
    state: dict,
    offered_ctc: float,
    current_ctc: float,
    expected_ctc: float,
    relevant_experience_years: float,
    notice_period_days: float,
    offered_band: str,
    candidate_source: str,
    lob: str,
    primary_skill: str,
    previous_company_type: str,
    location: str,
    joining_bonus: int,
    relocation: int,
) -> dict:
    candidate = {
        "current_ctc": current_ctc,
        "expected_ctc": expected_ctc,
        "offered_ctc": offered_ctc,
        "relevant_experience_years": relevant_experience_years,
        "notice_period_days": notice_period_days,
        "offered_band": offered_band,
        "candidate_source": candidate_source,
        "lob": lob,
        "primary_skill": primary_skill,
        "previous_company_type": previous_company_type,
        "location": location,
        "city_tier": city_tier(location),
        "joining_bonus": joining_bonus,
        "relocation": relocation,
    }
    candidate.update(_latest_offer_period(state["df"]))
    return candidate


def _first_offer_at_probability(model, candidate: dict, probability: float, points: int = 60) -> float | None:
    curve = acceptance_curve(model, candidate, points=points)
    match = next((point for point in curve if point["acceptance_probability"] >= probability), None)
    return None if match is None else match["offered_ctc"]


def _benchmark_adjusted_offer(
    model,
    candidate: dict,
    percentiles: dict,
    offered_ctc: float,
    target_probability: float = 0.70,
) -> dict:
    probability = predict_acceptance(model, candidate, offered_ctc)
    target_offer = _first_offer_at_probability(model, candidate, target_probability)
    p20 = percentiles.get("p20_offered_ctc")
    p50 = percentiles.get("p50_offered_ctc")
    p80 = percentiles.get("p80_offered_ctc")
    model_target = target_offer if target_offer is not None else offered_ctc
    suggested = max(offered_ctc, model_target)
    status = "ok"
    warnings = []

    if p20 is not None and suggested < float(p20) * 0.9:
        suggested = float(p20)
        status = "review_below_benchmark"
        warnings.append(
            f"Raw model target is below benchmark P20 ({p20:.2f} LPA). Use benchmark-adjusted suggestion and verify band fit."
        )
    if p20 is not None and offered_ctc < float(p20):
        warnings.append(f"Current offer is below benchmark P20 by {float(p20) - offered_ctc:.2f} LPA.")
    if percentiles.get("accepted_similar_records", 0) < MIN_BENCHMARK_RECORDS:
        warnings.append("Benchmark has low accepted/joined support; treat as directional.")

    suggested_candidate = dict(candidate)
    suggested_candidate["offered_ctc"] = suggested
    suggested_probability = predict_acceptance(model, suggested_candidate, suggested)

    return {
        "raw_model_probability": round(float(probability), 3),
        "model_target_offer": target_offer,
        "benchmark_adjusted_suggestion": round(float(suggested), 2),
        "probability_at_suggestion": round(float(suggested_probability), 3),
        "recommendation_status": status,
        "benchmark_percentiles": {"p20": p20, "p50": p50, "p80": p80},
        "benchmark_position": _benchmark_position(offered_ctc, p20, p50, p80),
        "warnings": warnings,
    }


def _benchmark_position(offer: float, p20: float | None, p50: float | None, p80: float | None) -> str:
    if p20 is not None and offer < float(p20):
        return "below_p20"
    if p50 is not None and offer < float(p50):
        return "p20_to_p50"
    if p80 is not None and offer <= float(p80):
        return "p50_to_p80"
    if p80 is not None and offer > float(p80):
        return "above_p80"
    return "unknown"


def _fmt_lpa(value) -> str:
    if value is None:
        return "not available"
    return f"{float(value):.2f} LPA"


def _known_value_from_text(text: str, values) -> str | None:
    normalized = text.lower()
    candidates = [str(value).strip() for value in values if str(value).strip() and str(value).strip().lower() != "nan"]
    for value in sorted(set(candidates), key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            return value
    return None


def _profile_from_text(text: str, df) -> dict | None:
    profile = {
        "primary_skill": _known_value_from_text(text, df["primary_skill"].dropna().unique()),
        "lob": _known_value_from_text(text, df["lob"].dropna().unique()),
        "location": _known_value_from_text(text, df["location"].dropna().unique()),
        "offered_band": _known_value_from_text(text, df["offered_band"].dropna().unique()),
    }
    if not all(profile.values()):
        return None
    profile["city_tier"] = city_tier(profile["location"])
    return profile


def _first_lpa_amount(text: str) -> float | None:
    amounts = _lpa_amounts(text)
    return amounts[0] if amounts else None


def _lpa_amounts(text: str) -> list[float]:
    matches = re.findall(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?)\b", text, flags=re.IGNORECASE)
    return [float(match) for match in matches]


def _benchmark_text(percentiles: dict) -> str:
    return (
        f"P20 {_fmt_lpa(percentiles.get('p20_offered_ctc'))}, "
        f"P50 {_fmt_lpa(percentiles.get('p50_offered_ctc'))}, "
        f"P80 {_fmt_lpa(percentiles.get('p80_offered_ctc'))}"
    )


def _profile_text(profile: dict) -> str:
    return (
        f"{profile.get('offered_band')} {profile.get('primary_skill')} candidate "
        f"in {profile.get('lob')}, {profile.get('location')}"
    )

def _filters_text(filters: dict | None, rule: str | None = None) -> str:
    if not filters:
        return "Benchmark match: broad historical fallback."
    labels = {
        "primary_skill": "skill",
        "lob": "LOB",
        "location": "location",
        "city_tier": "city tier",
        "offered_band": "band",
    }
    parts = [f"{labels.get(key, key)}={value}" for key, value in filters.items()]
    text = "Benchmark match: " + ", ".join(parts) + "."
    if rule:
        text += f" Rule: {rule}."
    return text


def _support_text(count: int) -> str:
    if count <= 0:
        return "I do not see accepted/joined matches for this exact benchmark, so treat this as directional."
    if count < MIN_BENCHMARK_RECORDS:
        return f"This is based on only {count} accepted/joined records, so I would treat it as directional."
    return f"This is based on {count} accepted/joined records."


def _offer_position_text(offer: float, percentiles: dict) -> str:
    p20 = percentiles.get("p20_offered_ctc")
    p50 = percentiles.get("p50_offered_ctc")
    p80 = percentiles.get("p80_offered_ctc")
    if p20 is not None and offer < float(p20):
        gap = float(p20) - offer
        return (
            f"I would flag {_fmt_lpa(offer)} as low. It is {_fmt_lpa(gap)} below the P20 floor, "
            "so it is weaker than the offers that usually worked for this segment."
        )
    if p50 is not None and offer < float(p50):
        gap = float(p50) - offer
        return (
            f"{_fmt_lpa(offer)} is usable, but conservative. It clears P20 and is {_fmt_lpa(gap)} below the median successful offer."
        )
    if p80 is not None and offer <= float(p80):
        return f"{_fmt_lpa(offer)} is in a strong zone: above the median and still within the P80 benchmark."
    if p80 is not None and offer > float(p80):
        extra = offer - float(p80)
        return (
            f"{_fmt_lpa(offer)} is above P80 by {_fmt_lpa(extra)}. It may be valid for a must-win candidate, "
            "but I would check budget and internal parity before proceeding."
        )
    return f"{_fmt_lpa(offer)} can be compared with the benchmark range above."


def _missing_probability_note() -> str:
    return (
        "I am not showing a personalized acceptance probability here because I would need the full candidate inputs: "
        "current CTC, expected CTC, experience, notice period, source, company type, joining bonus, and relocation."
    )


def _local_compensation_answer(messages: list[dict], state: dict) -> tuple[str, list[dict]] | None:
    if not messages:
        return None
    text = str(messages[-1].get("content", ""))
    lowered = text.lower()

    if "p20" in lowered and "p50" in lowered and "p80" in lowered and any(word in lowered for word in ["mean", "meaning", "what does"]):
        return (
            "P20, P50, and P80 are cut points from historical accepted/joined offers for the matched profile. "
            "P20 is a practical floor, P50 is the median successful offer, and P80 is the stronger upper benchmark. "
            "So below P20 needs review, P20-P50 is conservative, P50-P80 is strong, and above P80 needs a budget/parity check.",
            [],
        )

    if "why" in lowered and "suggest" in lowered and "model" in lowered and "target" in lowered:
        return (
            "The model target and the benchmark suggestion are answering different questions. The raw model target asks, "
            "'where does the ML score cross the probability threshold?' The benchmark asks, 'what actually worked historically for similar candidates?' "
            "If the raw model target falls below P20, the app raises the recommendation to the benchmark floor so recruiters do not under-offer against the historical evidence.",
            [],
        )

    if "probability" in lowered and any(phrase in lowered for phrase in ["lower", "go down", "went down", "less"]):
        return (
            "That is a model caveat, not a compensation rule. For the same candidate, a higher CTC should normally not reduce acceptance likelihood. "
            "For partial-profile chatbot questions, the safer answer is the P20/P50/P80 benchmark position. Use simulator probability only when the full candidate inputs are provided.",
            [],
        )

    if "probability" in lowered and any(phrase in lowered for phrase in ["why", "not showing", "not show", "missing"]):
        return (
            "I only show acceptance probability when the candidate profile is complete. The model needs current CTC, expected CTC, offered CTC, experience, notice period, source, company type, joining bonus, relocation, band, LOB, skill, and location. "
            "If only the role segment is provided, I use P20/P50/P80 benchmarks instead because they are clearer and less misleading.",
            [],
        )

    profile = _profile_from_text(text, state["df"])
    if profile is None:
        return None

    is_comp_question = any(
        phrase in lowered
        for phrase in [
            "what ctc",
            "should i offer",
            "how much",
            "safe",
            "starting offer",
            "minimum",
            "floor",
            "avoid going below",
            "compare",
            "versus",
            " vs ",
            "too low",
            "too high",
            "too much",
            "okay",
            " ok ",
            "good",
            "fine",
            "reasonable",
            "acceptable",
            "competitive",
            "p20",
            "p50",
            "p80",
            "range",
            "benchmark",
        ]
    )
    if not is_comp_question:
        return None

    percentiles = flexible_percentiles(state["df"], profile, flexibility="balanced", min_records=MIN_BENCHMARK_RECORDS)
    range_text = _benchmark_text(percentiles)
    filters = percentiles.get("filters_used") or profile
    support = percentiles.get("accepted_similar_records", 0)
    amounts = _lpa_amounts(text)
    amount = amounts[0] if amounts else None
    profile_label = _profile_text(profile)
    support_note = _support_text(support)
    filter_note = _filters_text(filters, percentiles.get("similarity_rule"))

    if any(phrase in lowered for phrase in ["minimum", "floor", "avoid going below"]):
        return (
            f"For this {profile_label}, I would avoid going below {_fmt_lpa(percentiles.get('p20_offered_ctc'))}. "
            f"A balanced starting point is closer to the median, {_fmt_lpa(percentiles.get('p50_offered_ctc'))}. "
            f"The accepted/joined benchmark range is {range_text}. {filter_note} {support_note}"
            + (f" {_missing_probability_note()}" if "probability" in lowered else ""),
            [],
        )

    if len(amounts) >= 2 and any(phrase in lowered for phrase in ["compare", "versus", " vs "]):
        comparisons = " ".join(_offer_position_text(value, percentiles) for value in amounts[:3])
        return (
            f"For this {profile_label}, here is the benchmark comparison: {comparisons} "
            f"The accepted/joined range is {range_text}. {filter_note} {support_note} {_missing_probability_note()}",
            [],
        )

    if "70" in lowered and any(word in lowered for word in ["probability", "acceptance", "target"]):
        return (
            f"For a personalized 70% acceptance target, I need the full candidate details, not just the segment. "
            f"For this {profile_label}, the historical benchmark is {range_text}. {filter_note} {support_note} "
            f"Until the full inputs are available, I would use {_fmt_lpa(percentiles.get('p20_offered_ctc'))} as the floor and {_fmt_lpa(percentiles.get('p50_offered_ctc'))} as a balanced starting point.",
            [],
        )

    if amount is not None:
        verdict = _offer_position_text(amount, percentiles)
        return (
            f"Short answer: {verdict} For this {profile_label}, the accepted/joined benchmark range is {range_text}. "
            f"{filter_note} {support_note} {_missing_probability_note()}",
            [],
        )

    return (
        f"For this {profile_label}, I would use {_fmt_lpa(percentiles.get('p50_offered_ctc'))} as the balanced starting point and avoid going below {_fmt_lpa(percentiles.get('p20_offered_ctc'))}. "
        f"The accepted/joined benchmark range is {range_text}. {filter_note} {support_note} {_missing_probability_note()}",
        [],
    )

def _local_ui_action(messages: list[dict]) -> tuple[str, list[dict]] | None:
    if not messages:
        return None
    text = str(messages[-1].get("content", "")).lower()
    tab = None
    if any(phrase in text for phrase in ["open simulator", "open the simulator", "show simulator", "show the simulator", "go to simulator", "go to the simulator", "offer simulator"]):
        tab = "simulator"
    elif any(phrase in text for phrase in ["open dashboard", "open the dashboard", "show dashboard", "show the dashboard", "go to dashboard", "go to the dashboard"]):
        tab = "dashboard"
    elif any(phrase in text for phrase in ["open recent", "open the recent", "show recent", "show the recent", "recent offers", "offers table"]):
        tab = "table"
    elif any(phrase in text for phrase in ["negotiat", "open negotiation", "negotiation twin"]):
        tab = "negotiation"
    elif any(phrase in text for phrase in ["risk radar", "at-risk offers", "at risk offers", "open risk", "show risk"]):
        tab = "risk"

    if tab is None:
        return None

    return (
        f"Opening the {tab.replace('_', ' ')} view.",
        [{"type": "FILTER_UI", "tab": tab, "column": None, "value": None}],
    )


def chat_with_agent(messages: list[dict], state: dict) -> tuple[str, list[dict]]:
    local_action = _local_ui_action(messages)
    if local_action is not None:
        return local_action

    local_compensation = _local_compensation_answer(messages, state)
    if local_compensation is not None:
        return local_compensation
    llm, err = get_llm()
    if err:
        return f"Error: {err}", []

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    ui_actions_captured = []

    @tool
    def kpis() -> str:
        """Return overall offer KPIs and status counts."""
        df = state["df"]
        total = len(df)
        accepted = int(df["accepted"].sum())
        status_counts = {str(key): int(value) for key, value in df["status"].value_counts().to_dict().items()}
        return json.dumps({
            "total_offers": total,
            "accepted_or_joined": accepted,
            "declined_or_no_show": int(total - accepted),
            "acceptance_rate": round(accepted / total, 3),
            "status_counts": status_counts,
        })
    @tool
    def simulator(
        offered_ctc: float,
        current_ctc: float = 15.0,
        expected_ctc: float = 20.0,
        relevant_experience_years: float = 5.0,
        notice_period_days: float = 60.0,
        offered_band: str = "E2",
        candidate_source: str = "Direct",
        lob: str = "Digital",
        primary_skill: str = "Java Spring",
        previous_company_type: str = "Service",
        location: str = "Bangalore",
        joining_bonus: int = 0,
        relocation: int = 0,
        flexibility: str = "balanced",
    ) -> str:
        """Simulate an offer and return raw model probability plus benchmark-adjusted recommendation."""
        candidate = _candidate_payload(
            state,
            offered_ctc,
            current_ctc,
            expected_ctc,
            relevant_experience_years,
            notice_period_days,
            offered_band,
            candidate_source,
            lob,
            primary_skill,
            previous_company_type,
            location,
            joining_bonus,
            relocation,
        )
        percentiles = flexible_percentiles(
            state["df"],
            candidate,
            flexibility=flexibility,
            min_records=MIN_BENCHMARK_RECORDS,
        )
        adjusted = _benchmark_adjusted_offer(state["model"], candidate, percentiles, offered_ctc, target_probability=0.70)

        return json.dumps({
            **adjusted,
            "offered_ctc": offered_ctc,
            "target_probability": 0.70,
            "benchmark_filters": percentiles.get("filters_used"),
            "benchmark_rule": percentiles.get("similarity_rule"),
            "similar_records": percentiles.get("similar_records"),
            "accepted_joined_records": percentiles.get("accepted_similar_records"),
            "benchmark_confidence": percentiles.get("confidence"),
            "benchmark_warning": percentiles.get("warning"),
            "important_instruction": (
                "Do not present raw_model_probability as the final recommendation if recommendation_status is review_below_benchmark. "
                "Explain the conflict and use benchmark_adjusted_suggestion."
            ),
        })


    @tool
    def optimize_offer(
        target_probability: float,
        current_ctc: float = 15.0,
        expected_ctc: float = 20.0,
        relevant_experience_years: float = 5.0,
        notice_period_days: float = 60.0,
        offered_band: str = "E2",
        candidate_source: str = "Direct",
        lob: str = "Digital",
        primary_skill: str = "Java Spring",
        previous_company_type: str = "Service",
        location: str = "Bangalore",
        joining_bonus: int = 0,
        relocation: int = 0,
        flexibility: str = "balanced",
    ) -> str:
        """Find model target CTC and benchmark-adjusted CTC for a target acceptance probability."""
        target_probability = max(0.01, min(float(target_probability), 0.99))
        starting_offer = max(float(current_ctc) * 1.02, float(expected_ctc) * 0.75)
        candidate = _candidate_payload(
            state,
            starting_offer,
            current_ctc,
            expected_ctc,
            relevant_experience_years,
            notice_period_days,
            offered_band,
            candidate_source,
            lob,
            primary_skill,
            previous_company_type,
            location,
            joining_bonus,
            relocation,
        )
        percentiles = flexible_percentiles(
            state["df"],
            candidate,
            flexibility=flexibility,
            min_records=MIN_BENCHMARK_RECORDS,
        )
        model_target = _first_offer_at_probability(state["model"], candidate, target_probability, points=60)
        if model_target is None:
            curve = acceptance_curve(state["model"], candidate, points=60)
            last = curve[-1]
            return json.dumps({
                "success": False,
                "target_probability": target_probability,
                "message": f"Could not reach {target_probability:.0%} within the searched range.",
                "max_searched_offer": last["offered_ctc"],
                "probability_at_max": last["acceptance_probability"],
                "benchmark_percentiles": {
                    "p20": percentiles.get("p20_offered_ctc"),
                    "p50": percentiles.get("p50_offered_ctc"),
                    "p80": percentiles.get("p80_offered_ctc"),
                },
            })
        target_candidate = dict(candidate)
        target_candidate["offered_ctc"] = model_target
        adjusted = _benchmark_adjusted_offer(state["model"], target_candidate, percentiles, model_target, target_probability=target_probability)
        return json.dumps({
            "success": True,
            "target_probability": target_probability,
            "raw_model_target_offer": model_target,
            "raw_model_probability_at_target": predict_acceptance(state["model"], target_candidate, model_target),
            **adjusted,
            "benchmark_filters": percentiles.get("filters_used"),
            "benchmark_rule": percentiles.get("similarity_rule"),
            "accepted_joined_records": percentiles.get("accepted_similar_records"),
            "important_instruction": "If benchmark_adjusted_suggestion is higher than raw_model_target_offer, explain that the benchmark floor overrode the raw model target.",
        })


    @tool
    def filter_ui(tab: str, filter_column: str | None = None, filter_value: str | None = None) -> str:
        """Switch UI tab. tab must be dashboard, simulator, negotiation, risk, or table."""
        action = {
            "type": "FILTER_UI",
            "tab": tab,
            "column": filter_column,
            "value": filter_value,
        }
        ui_actions_captured.append(action)
        return json.dumps({"status": "success", "ui_updated": True, "action": action})

    tools = [simulator, kpis, optimize_offer, filter_ui]
    llm_with_tools = llm.bind_tools(tools)

    lc_messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]
    for message in messages[-12:]:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role in {"model", "assistant"}:
            lc_messages.append(AIMessage(content=content))

    try:
        ai_msg = llm_with_tools.invoke(lc_messages)
        lc_messages.append(ai_msg)

        if ai_msg.tool_calls:
            tool_outputs = []
            for tool_call in ai_msg.tool_calls:
                selected_tool = next((item for item in tools if item.name == tool_call["name"]), None)
                if selected_tool:
                    tool_output = selected_tool.invoke(tool_call["args"])
                    tool_outputs.append(str(tool_output))
                    lc_messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
            if tool_outputs:
                return _tool_output_fallback(tool_outputs[-1]), ui_actions_captured

        return str(ai_msg.content or "").strip(), ui_actions_captured
    except Exception as exc:
        return f"Sorry, an error occurred while generating the response: {exc}", []


def _tool_output_fallback(raw_output: str) -> str:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output

    def fmt_lpa(value):
        if value is None:
            return "not available"
        return f"{float(value):.2f} LPA"

    if {"total_offers", "accepted_or_joined", "acceptance_rate"}.issubset(data):
        return (
            f"There are {data['total_offers']} historical offers. "
            f"{data['accepted_or_joined']} were accepted or joined, "
            f"for an overall success rate of {data['acceptance_rate']:.0%}."
        )

    if "raw_model_probability" in data:
        percentiles = data.get("benchmark_percentiles") or {}
        p20 = percentiles.get("p20")
        p50 = percentiles.get("p50")
        p80 = percentiles.get("p80")
        suggestion = data.get("benchmark_adjusted_suggestion")
        offered = data.get("offered_ctc")
        model_target = data.get("model_target_offer") or data.get("raw_model_target_offer")
        probability_at_suggestion = data.get("probability_at_suggestion")
        status = data.get("recommendation_status")
        position = data.get("benchmark_position")
        filter_note = _filters_text(data.get("benchmark_filters"), data.get("benchmark_rule"))

        range_text = f"P20 {fmt_lpa(p20)}, P50 {fmt_lpa(p50)}, P80 {fmt_lpa(p80)}"
        offer_text = fmt_lpa(offered) if offered is not None else "this offer"
        parts = [f"Model-only acceptance probability at {offer_text}: {data['raw_model_probability']:.0%}."]

        if p20 is not None or p50 is not None or p80 is not None:
            parts.append(f"Historical successful-offer range: {range_text}. {filter_note}")

        if status == "review_below_benchmark" and suggestion is not None:
            parts.append(
                f"Recruiter action: treat this as a review case, because the model target "
                f"({fmt_lpa(model_target)}) is below the benchmark floor. I would not recommend going below {fmt_lpa(p20)} for this segment."
            )
            parts.append(
                f"Benchmark-adjusted suggestion: {fmt_lpa(suggestion)}"
                + (f"; model probability there is {probability_at_suggestion:.0%}." if probability_at_suggestion is not None else ".")
            )
        elif suggestion is not None:
            if offered is not None and abs(float(suggestion) - float(offered)) < 0.01:
                parts.append("Recruiter action: this evaluated offer is within the benchmark guardrails.")
            else:
                parts.append(f"Recruiter action: benchmark-adjusted suggestion is {fmt_lpa(suggestion)}.")

        if position == "below_p20":
            parts.append("Benchmark position: below P20, so it is weak versus similar accepted/joined offers.")
        elif position == "p20_to_p50":
            parts.append("Benchmark position: between P20 and P50, so it is conservative but defensible.")
        elif position == "p50_to_p80":
            parts.append("Benchmark position: between P50 and P80, a stronger historical zone.")
        elif position == "above_p80":
            parts.append("Benchmark position: above P80, so check budget and internal parity.")

        return " ".join(parts)
    if data.get("success") is True and "benchmark_adjusted_suggestion" in data:
        raw_target = data.get("raw_model_target_offer")
        suggestion = data.get("benchmark_adjusted_suggestion")
        parts = [
            f"For a {data['target_probability']:.0%} target, the raw model target is {fmt_lpa(raw_target)}."
        ]
        if raw_target is not None and suggestion is not None and float(suggestion) > float(raw_target) + 0.01:
            parts.append(f"Because the raw target is below the benchmark floor, use the benchmark-adjusted suggestion of {fmt_lpa(suggestion)}.")
        elif suggestion is not None:
            parts.append(f"Suggested CTC is {fmt_lpa(suggestion)}.")
        if data.get("probability_at_suggestion") is not None:
            parts.append(f"Predicted acceptance at the suggestion is {data['probability_at_suggestion']:.0%}.")
        for warning in data.get("warnings") or []:
            parts.append(str(warning))
        return " ".join(parts)

    if data.get("success") is True and "suggested_offer" in data:
        return (
            f"To target {data['target_probability']:.0%} acceptance, "
            f"the lowest searched offer is {data['suggested_offer']} LPA "
            f"with predicted acceptance of {data['achieved_probability']:.0%}."
        )

    return json.dumps(data, indent=2)












