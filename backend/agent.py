import json
import os
from dotenv import load_dotenv

from .model import predict_acceptance
from .recommender import flexible_percentiles

# Load environment variables from backend/.env if it exists
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

SYSTEM_INSTRUCTION = """
You are an expert HR Recruiter Assistant for the CTC Offer Intelligence platform.
You help recruiters determine competitive CTCs and predict offer acceptance probabilities.
Use the `simulator` tool when you need to calculate acceptance probabilities or find benchmark CTCs for a specific candidate profile.
Use the `optimize_offer` tool when you are asked to find the *minimum* or *optimal* CTC to achieve a specific target acceptance rate.
Use the `kpis` tool when asked about general offer trends, overall acceptance rates, or historical summaries.
Use the `filter_ui` tool when the user asks to "show me", "navigate to", or "filter" visual data (e.g., "show me Java developers in the dashboard").
If the user asks you to draft an offer letter, use the candidate's profile, benchmark data, and your persuasive writing skills to draft an engaging, personalized offer email.
Be concise and helpful. Format your response in clean markdown.
"""

def get_llm():
    try:
        from langchain_openai import AzureChatOpenAI
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not key:
            return None, "AZURE_OPENAI_API_KEY is not configured in backend/.env"
        
        return AzureChatOpenAI(
            azure_deployment=deployment,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=key,
            temperature=0.1
        ), None
    except Exception as e:
        return None, f"Azure OpenAI Error: {str(e)}"

def chat_with_agent(messages: list[dict], state: dict) -> tuple[str, list[dict]]:
    llm, err = get_llm()
    if err:
        return f"Error: {err}", []

    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool

    ui_actions_captured = []

    @tool
    def kpis() -> str:
        """Returns overall Key Performance Indicators for historical offers."""
        df = state["df"]
        total = len(df)
        accepted = int(df["accepted"].sum())
        return json.dumps({
            "total_offers": total,
            "accepted": accepted,
            "acceptance_rate": round(accepted / total, 3)
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
        relocation: int = 0
    ) -> str:
        """Simulates an offer and returns the acceptance probability and benchmark CTC percentiles."""
        from .data import city_tier
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
            "offer_year": 2026,
            "offer_month": 7,
            "offer_quarter": 3,
        }
        
        prob = predict_acceptance(state["model"], candidate, offered_ctc)
        percentiles = flexible_percentiles(state["df"], candidate, flexibility="balanced")
        
        return json.dumps({
            "acceptance_probability": round(prob, 3),
            "benchmark_percentiles": {
                "p20": percentiles.get("p20_offered_ctc"),
                "p50": percentiles.get("p50_offered_ctc"),
                "p80": percentiles.get("p80_offered_ctc")
            },
            "benchmark_confidence": percentiles.get("confidence")
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
        relocation: int = 0
    ) -> str:
        """Finds the minimum CTC required to achieve a specific target acceptance probability (e.g. 0.8)."""
        from .data import city_tier
        candidate = {
            "current_ctc": current_ctc,
            "expected_ctc": expected_ctc,
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
            "offer_year": 2026,
            "offer_month": 7,
            "offer_quarter": 3,
        }
        
        # Iterative search from current_ctc up to expected_ctc + 15
        best_ctc = None
        best_prob = 0
        search_ctc = current_ctc
        max_ctc = expected_ctc + 15.0
        
        while search_ctc <= max_ctc:
            prob = predict_acceptance(state["model"], candidate, search_ctc)
            if prob >= target_probability:
                best_ctc = search_ctc
                best_prob = prob
                break
            search_ctc += 0.5
            
        if best_ctc is None:
            return json.dumps({
                "success": False,
                "message": f"Could not achieve target probability of {target_probability} even at {max_ctc} LPA."
            })
            
        return json.dumps({
            "success": True,
            "optimal_ctc": round(best_ctc, 2),
            "achieved_probability": round(best_prob, 3),
            "target_probability": target_probability
        })

    @tool
    def filter_ui(
        tab: str,
        filter_column: str = None,
        filter_value: str = None
    ) -> str:
        """Use this to visually filter the UI dashboard or switch tabs based on user requests. 
        tab: 'dashboard', 'simulator', or 'table'.
        filter_column: e.g., 'primary_skill', 'lob'.
        filter_value: e.g., 'Java Spring'."""
        action = {
            "type": "FILTER_UI",
            "tab": tab,
            "column": filter_column,
            "value": filter_value
        }
        ui_actions_captured.append(action)
        return json.dumps({"status": "success", "ui_updated": True, "action": action})

    tools = [simulator, kpis, optimize_offer, filter_ui]
    llm_with_tools = llm.bind_tools(tools)

    # Convert history
    lc_messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "model" or m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
            
    # Step 1: Call LLM
    try:
        ai_msg = llm_with_tools.invoke(lc_messages)
        lc_messages.append(ai_msg)
        
        # Step 2: Handle tool calls if any
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                selected_tool = next((t for t in tools if t.name == tool_call["name"]), None)
                if selected_tool:
                    tool_output = selected_tool.invoke(tool_call["args"])
                    lc_messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
            
            # Step 3: Call LLM again with tool outputs
            final_msg = llm_with_tools.invoke(lc_messages)
            return final_msg.content, ui_actions_captured
        else:
            return ai_msg.content, ui_actions_captured
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Sorry, an error occurred while generating the response: {str(e)}", []
