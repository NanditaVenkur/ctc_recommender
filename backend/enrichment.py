"""External-tool integrations for the PACT offer terminal.

- GitHub Talent Scanner: live GitHub REST API (no key, 60 req/hr unauthenticated).
- Market Wire: live INR FX rates (open.er-api.com, no key) and optional Adzuna
  job-market salary data (free key via ADZUNA_APP_ID / ADZUNA_APP_KEY in .env).
- Offer Letter Forge: LLM-drafted personalized offer letter via the configured
  chat provider.

Every integration fails soft: network errors return an explicit "unavailable"
payload instead of fabricated data.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from datetime import datetime, timezone

import requests

GITHUB_API = "https://api.github.com"
FX_URL = "https://open.er-api.com/v6/latest/INR"
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"

_fx_cache: dict = {"ts": 0.0, "payload": None}
_adzuna_cache: dict = {}

# Which repo languages count as evidence for a claimed skill.
SKILL_LANGUAGE_HINTS = {
    "java": ["java", "kotlin"],
    "spring": ["java", "kotlin"],
    "python": ["python", "jupyter notebook"],
    "django": ["python"],
    "data": ["python", "jupyter notebook", "r", "scala"],
    "ml": ["python", "jupyter notebook"],
    "react": ["javascript", "typescript"],
    "node": ["javascript", "typescript"],
    "angular": ["typescript", "javascript"],
    "javascript": ["javascript", "typescript"],
    "frontend": ["javascript", "typescript", "html", "css", "vue"],
    ".net": ["c#"],
    "dotnet": ["c#"],
    "c#": ["c#"],
    "golang": ["go"],
    "go ": ["go"],
    "rust": ["rust"],
    "php": ["php"],
    "ruby": ["ruby"],
    "ios": ["swift", "objective-c"],
    "android": ["kotlin", "java"],
    "mobile": ["kotlin", "swift", "dart", "java"],
    "flutter": ["dart"],
    "devops": ["shell", "go", "python", "hcl", "dockerfile", "makefile"],
    "cloud": ["shell", "go", "python", "hcl", "dockerfile"],
    "aws": ["shell", "python", "hcl"],
    "sql": ["sql", "tsql", "plsql", "plpgsql"],
    "testing": ["java", "python", "javascript", "typescript"],
    "c++": ["c++", "c"],
    "embedded": ["c", "c++"],
}


def _language_hints_for_skill(skill: str) -> list[str]:
    lowered = (skill or "").lower()
    hints: list[str] = []
    for token, languages in SKILL_LANGUAGE_HINTS.items():
        if token in lowered:
            for language in languages:
                if language not in hints:
                    hints.append(language)
    return hints


def github_scan(username: str, claimed_skill: str = "") -> dict:
    """Aggregate a candidate's public GitHub footprint into a hiring signal."""
    username = (username or "").strip().lstrip("@")
    if not username:
        return {"available": False, "reason": "No username provided."}

    headers = {"Accept": "application/vnd.github+json", "User-Agent": "pact-offer-terminal"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        user_resp = requests.get(f"{GITHUB_API}/users/{username}", headers=headers, timeout=8)
        if user_resp.status_code == 404:
            return {"available": False, "reason": f"GitHub user '{username}' not found."}
        if user_resp.status_code == 403:
            return {"available": False, "reason": "GitHub API rate limit reached — try again in a few minutes."}
        user_resp.raise_for_status()
        user = user_resp.json()

        repos_resp = requests.get(
            f"{GITHUB_API}/users/{username}/repos",
            params={"sort": "pushed", "per_page": 100, "type": "owner"},
            headers=headers,
            timeout=8,
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []
    except requests.RequestException as exc:
        return {"available": False, "reason": f"GitHub unreachable: {exc.__class__.__name__}."}

    if not isinstance(repos, list):
        repos = []

    source_repos = [repo for repo in repos if not repo.get("fork")]
    languages = Counter(repo["language"] for repo in source_repos if repo.get("language"))
    total_stars = sum(int(repo.get("stargazers_count") or 0) for repo in source_repos)

    now = datetime.now(timezone.utc)
    recent_pushes = 0
    for repo in source_repos:
        pushed_at = repo.get("pushed_at")
        if not pushed_at:
            continue
        try:
            pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if (now - pushed).days <= 90:
            recent_pushes += 1

    created_at = user.get("created_at", "")
    account_years = None
    if created_at:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            account_years = round((now - created).days / 365.25, 1)
        except ValueError:
            pass

    top_languages = [{"language": lang, "repos": count} for lang, count in languages.most_common(6)]
    hints = _language_hints_for_skill(claimed_skill)
    matched = [item["language"] for item in top_languages if item["language"].lower() in hints]

    if not source_repos:
        verdict, verdict_note = "no_signal", "No public source repositories to evaluate."
    elif not hints:
        verdict, verdict_note = "unmapped", f"No language mapping for '{claimed_skill}'; showing raw footprint."
    elif top_languages and top_languages[0]["language"].lower() in hints:
        verdict, verdict_note = "strong_match", f"Primary public language ({top_languages[0]['language']}) matches the claimed skill."
    elif matched:
        verdict, verdict_note = "partial_match", f"Claimed skill appears in the footprint ({', '.join(matched)}), but is not the dominant language."
    else:
        langs = ", ".join(item["language"] for item in top_languages[:3]) or "none"
        verdict, verdict_note = "mismatch", f"Public repos are mostly {langs} — no visible evidence of the claimed skill."

    return {
        "available": True,
        "username": user.get("login", username),
        "name": user.get("name"),
        "profile_url": user.get("html_url"),
        "avatar_url": user.get("avatar_url"),
        "public_repos": user.get("public_repos", 0),
        "followers": user.get("followers", 0),
        "account_years": account_years,
        "source_repos": len(source_repos),
        "total_stars": total_stars,
        "recent_active_repos_90d": recent_pushes,
        "top_languages": top_languages,
        "claimed_skill": claimed_skill,
        "verdict": verdict,
        "verdict_note": verdict_note,
    }


def _fx_rates() -> dict | None:
    """INR-based FX rates, cached for 6 hours."""
    if _fx_cache["payload"] and time.time() - _fx_cache["ts"] < 6 * 3600:
        return _fx_cache["payload"]
    try:
        resp = requests.get(FX_URL, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates") or {}
        payload = {
            "usd": rates.get("USD"),
            "eur": rates.get("EUR"),
            "gbp": rates.get("GBP"),
            "aed": rates.get("AED"),
            "sgd": rates.get("SGD"),
            "updated": data.get("time_last_update_utc"),
        }
        if payload["usd"]:
            _fx_cache["ts"] = time.time()
            _fx_cache["payload"] = payload
            return payload
    except requests.RequestException:
        pass
    return _fx_cache["payload"]


def _adzuna_snapshot(skill: str, location: str) -> dict | None:
    """Live job-market snapshot from Adzuna (optional; needs free API keys)."""
    app_id = os.environ.get("ADZUNA_APP_ID", "").strip()
    app_key = os.environ.get("ADZUNA_APP_KEY", "").strip()
    if not app_id or not app_key:
        return None

    cache_key = f"{skill}|{location}".lower()
    cached = _adzuna_cache.get(cache_key)
    if cached and time.time() - cached["ts"] < 3600:
        return cached["payload"]

    try:
        resp = requests.get(
            ADZUNA_URL,
            params={
                "app_id": app_id,
                "app_key": app_key,
                "what": skill,
                "where": location,
                "results_per_page": 20,
                "content-type": "application/json",
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None

    results = data.get("results") or []
    salaries = []
    for job in results:
        low = job.get("salary_min")
        high = job.get("salary_max")
        if low and high:
            salaries.append((float(low) + float(high)) / 2)

    payload = {
        "openings_found": int(data.get("count") or 0),
        "postings_sampled": len(results),
        "avg_advertised_salary_lpa": round(sum(salaries) / len(salaries) / 100000, 2) if salaries else None,
        "source": "Adzuna live job postings (India)",
    }
    _adzuna_cache[cache_key] = {"ts": time.time(), "payload": payload}
    return payload


def market_wire(lpa: float | None = None, skill: str = "", location: str = "") -> dict:
    """FX conversions for a CTC plus an optional live job-market snapshot."""
    rates = _fx_rates()
    conversions = None
    if rates and lpa:
        inr = float(lpa) * 100000
        conversions = {
            code: round(inr * rate)
            for code, rate in (("usd", rates["usd"]), ("eur", rates["eur"]), ("gbp", rates["gbp"]))
            if rate
        }

    adzuna = _adzuna_snapshot(skill, location) if skill else None
    return {
        "fx_available": rates is not None,
        "rates": rates,
        "lpa": lpa,
        "conversions": conversions,
        "job_market": adzuna,
        "job_market_available": adzuna is not None,
    }


def forge_offer_letter(candidate: dict, quote: dict) -> dict:
    """Draft a personalized offer letter with the configured LLM."""
    from .agent import get_llm

    llm, err = get_llm()
    if err:
        return {"success": False, "reason": err}

    from langchain_core.messages import HumanMessage, SystemMessage

    fx = market_wire(lpa=quote.get("suggested_ctc") or candidate.get("offered_ctc"))
    conversions = fx.get("conversions") or {}
    usd_line = f" (approximately USD {conversions['usd']:,} per annum)" if conversions.get("usd") else ""

    candidate_name = str(candidate.get("candidate_name") or "").strip()
    greeting_rule = (
        f"Address the candidate exactly as '{candidate_name}'."
        if candidate_name
        else "No candidate name was provided: open with 'Dear Candidate,' and never invent a name."
    )

    system = (
        "You are an expert HR communications writer. Draft a warm, professional, persuasive offer letter. "
        "Rules: plain text only (no markdown, no placeholders like [Name] left unfilled), 220-320 words, "
        "structured as: greeting, congratulations naming the role context, compensation paragraph with the exact "
        "CTC figure, one paragraph tailored to the candidate's profile (experience, skill, location), a warm "
        "closing asking them to confirm by the joining date. Sign as 'Talent Acquisition Desk'. "
        f"{greeting_rule} "
        "Do not invent benefits that were not provided. Do not mention probabilities or internal models."
    )
    user = (
        f"Candidate profile: {candidate.get('relevant_experience_years')} years of experience in "
        f"{candidate.get('primary_skill')}, based in {candidate.get('location')}, band {candidate.get('offered_band')}, "
        f"business line {candidate.get('lob')}. Current CTC {candidate.get('current_ctc')} LPA, "
        f"expected {candidate.get('expected_ctc')} LPA.\n"
        f"Final offered CTC: {quote.get('suggested_ctc') or candidate.get('offered_ctc')} LPA{usd_line}.\n"
        f"Extras: joining bonus {'included' if candidate.get('joining_bonus') else 'not included'}, "
        f"relocation support {'included' if candidate.get('relocation') else 'not included'}.\n"
        f"Notice period: {candidate.get('notice_period_days')} days."
    )

    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        letter = str(response.content or "").strip()
    except Exception as exc:
        return {"success": False, "reason": f"LLM error: {exc}"}

    if not letter:
        return {"success": False, "reason": "The LLM returned an empty letter."}

    return {
        "success": True,
        "letter": letter,
        "ctc": quote.get("suggested_ctc") or candidate.get("offered_ctc"),
        "conversions": conversions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
