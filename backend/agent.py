from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from .data import city_tier
from .model import acceptance_curve, predict_acceptance
from .recommender import flexible_percentiles


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MIN_BENCHMARK_RECORDS = 8

SYSTEM_INSTRUCTION = """
You are an HR recruiter assistant inside the CTC Offer Intelligence platform.
You help recruiters understand historical offers, benchmark CTC ranges, and predicted acceptance probability.

Use tools for calculations instead of guessing. Treat model outputs as decision support, not as an automatic compensation decision.
Explain uncertainty clearly when benchmark support is broad or sample counts are low.

Useful behavior:
- Use simulator when asked about a candidate profile, acceptance probability, P20/P50/P80, or whether a given offer is competitive.
- Use optimize_offer when asked what CTC is needed for a target acceptance probability.
- Use kpis when asked about overall offer funnel, acceptance rate, or dashboard-level metrics.
- Use filter_ui when asked to open or show dashboard, simulator, or recent offers.

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
        """Simulate an offer and return probability plus benchmark percentiles."""
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
        probability = predict_acceptance(state["model"], candidate, offered_ctc)
        percentiles = flexible_percentiles(
            state["df"],
            candidate,
            flexibility=flexibility,
            min_records=MIN_BENCHMARK_RECORDS,
        )
        curve = acceptance_curve(state["model"], candidate)
        target_offer = next(
            (point["offered_ctc"] for point in curve if point["acceptance_probability"] >= 0.70),
            None,
        )

        return json.dumps({
            "acceptance_probability": round(probability, 3),
            "target_70_probability_offer": target_offer,
            "benchmark_filters": percentiles.get("filters_used"),
            "benchmark_rule": percentiles.get("similarity_rule"),
            "similar_records": percentiles.get("similar_records"),
            "accepted_joined_records": percentiles.get("accepted_similar_records"),
            "benchmark_confidence": percentiles.get("confidence"),
            "benchmark_percentiles": {
                "p20": percentiles.get("p20_offered_ctc"),
                "p50": percentiles.get("p50_offered_ctc"),
                "p80": percentiles.get("p80_offered_ctc"),
            },
            "warning": percentiles.get("warning"),
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
    ) -> str:
        """Find the lowest searched CTC that reaches a target acceptance probability."""
        target_probability = max(0.01, min(float(target_probability), 0.99))
        candidate = _candidate_payload(
            state,
            current_ctc,
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
        curve_candidate = dict(candidate)
        curve_candidate["offered_ctc"] = max(current_ctc, expected_ctc)
        curve = acceptance_curve(state["model"], curve_candidate, points=60)
        match = next((point for point in curve if point["acceptance_probability"] >= target_probability), None)

        if match is None:
            last = curve[-1]
            return json.dumps({
                "success": False,
                "message": (
                    f"Could not reach {target_probability:.0%} within the searched range. "
                    f"At {last['offered_ctc']} LPA, probability is {last['acceptance_probability']:.0%}."
                ),
                "max_searched_offer": last["offered_ctc"],
                "probability_at_max": last["acceptance_probability"],
            })

        return json.dumps({
            "success": True,
            "target_probability": target_probability,
            "suggested_offer": match["offered_ctc"],
            "achieved_probability": match["acceptance_probability"],
        })

    @tool
    def filter_ui(tab: str, filter_column: str | None = None, filter_value: str | None = None) -> str:
        """Switch UI tab. tab must be dashboard, simulator, or table."""
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

            final_msg = llm_with_tools.invoke(lc_messages)
            final_content = str(final_msg.content or "").strip()
            if final_content:
                return final_content, ui_actions_captured
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

    if {"total_offers", "accepted_or_joined", "acceptance_rate"}.issubset(data):
        return (
            f"There are {data['total_offers']} historical offers. "
            f"{data['accepted_or_joined']} were accepted or joined, "
            f"for an overall success rate of {data['acceptance_rate']:.0%}."
        )

    if "acceptance_probability" in data:
        parts = [f"Predicted acceptance probability is {data['acceptance_probability']:.0%}."]
        percentiles = data.get("benchmark_percentiles") or {}
        if percentiles:
            parts.append(
                "Benchmark CTC range: "
                f"P20 {percentiles.get('p20')} LPA, "
                f"P50 {percentiles.get('p50')} LPA, "
                f"P80 {percentiles.get('p80')} LPA."
            )
        if data.get("target_70_probability_offer") is not None:
            parts.append(f"The model reaches 70% near {data['target_70_probability_offer']} LPA.")
        if data.get("warning"):
            parts.append(str(data["warning"]))
        return " ".join(parts)

    if data.get("success") is True and "suggested_offer" in data:
        return (
            f"To target {data['target_probability']:.0%} acceptance, "
            f"the lowest searched offer is {data['suggested_offer']} LPA "
            f"with predicted acceptance of {data['achieved_probability']:.0%}."
        )

    return json.dumps(data, indent=2)
