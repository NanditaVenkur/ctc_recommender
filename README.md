# PACT · The Offer Terminal

> **Every offer is a trade. This is the terminal.**

Compensation is a market — candidates are priced, offers are bids, declines are
losses. PACT (**P**redict · **A**nalyze · **C**lose · **T**alent) treats hiring
exactly that way: a Bloomberg-style trading terminal for job offers, powered by
an acceptance-probability model and three cooperating agents.

## The pitch (30 seconds)

Recruiters guess salaries; candidates ghost; companies overpay or lose the hire.
PACT replaces the guess with a market view:

1. **Price the candidate** like an asset — get a quoted CTC with its win probability.
2. **Stress-test the deal** — two AI agents negotiate it round-by-round before a human ever does.
3. **Never lose silently** — an autonomous radar re-scores live offers and drafts escalations for deals about to fall through.

## The desk (five screens)

| Screen | What it does |
| --- | --- |
| **01 · Pulse** | Live market overview: win rate, price bands, outcome book, engine health, scrolling offer ticker. |
| **02 · Offer Studio** | Enter a profile → get a **deal ticket**: suggested CTC, acceptance gauge, market band (P20/P50/P80), price↔probability curve, and the comparable winning offers behind it. |
| **03 · The Arena** | A Recruiter Agent (guards budget) vs a Candidate Agent (chases value) negotiate live, round-by-round, grounded in the same acceptance model. Watch offers converge. |
| **04 · Risk Radar** | Autonomous agent sweeps the offer queue on a radar scope — blips closest to center are deals about to be lost — and drafts the escalation alert itself. |
| **05 · The Ledger** | Every trade on the book: recent offers, pricing, and how each settled. |

Plus: a **Desk Copilot** chat drawer (ask questions, upload a CSV/Excel dataset
to retrain the engine live) and a **command palette** (`Ctrl+K`, keys `1–5`
jump between screens).

## Live external integrations

| Feature | External tool | Key needed |
| --- | --- | --- |
| **Voice Desk** | Browser Web Speech API — click VOICE (topbar) or the mic in the Copilot, speak a request, the agent answers out loud | none |
| **GitHub Talent Scanner** | Live GitHub REST API — enter a candidate's GitHub username in the Offer Studio and get a verified skill signal (languages, stars, activity) with a match/mismatch verdict against the claimed skill | none (optional `GITHUB_TOKEN` in `backend/.env` raises the rate limit from 60 to 5,000 req/hr) |
| **Live Market Wire** | open.er-api.com FX rates — every quote shows real USD/EUR/GBP equivalents; the ticker carries live USD/EUR/GBP-INR rates | none |
| **Job-market snapshot** | Adzuna job postings API (optional) | free `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` in `backend/.env` |
| **Offer Letter Forge** | LLM-drafted personalized offer letter + QR code (api.qrserver.com), printable to PDF from a CLEARED deal ticket | uses the existing chat LLM key |

All integrations fail soft: if an external service is unreachable, the UI says
so honestly instead of fabricating data.

## The copilot operates the desk

The Desk Copilot is not a Q&A box — it drives the terminal:

- **Chat drives the desk** — "price a 6-year Python dev in Pune at 18 LPA" makes
  the agent fill the Offer Studio console on screen and run the full quote
  (`PREFILL_SIMULATOR` action). Combined with the Voice Desk this is a fully
  voice-operated terminal.
- **Rich answer cards** — simulation, KPI, and briefing answers render as mini
  visual cards (stat strips, P20/P50/P80 band bars) inside the chat bubbles.
- **Daily Desk Brief** — the "☀ BRIEF ME" chip (or asking for a briefing)
  generates a grounded morning brief: trend direction, strongest/weakest skill
  segment, live at-risk count from the risk agent, and one recommended action.
- **Screen-aware context** — every message silently carries what you're viewing
  and your last quote/negotiation, so follow-ups like "why was this escalated?"
  work without re-typing the candidate profile.

## The engine

- Logistic regression classifier predicting `P(accept | candidate profile, offered CTC)`,
  reported with ROC AUC and Brier score on a held-out set.
- CTC benchmarks from historical accepted-offer percentiles with a transparent
  fallback ladder (strict → broad similarity), shown in the UI as the
  "benchmark search trail".
- Guardrails, not black boxes: every quote ships with its evidence coverage,
  conflicting-signal warnings, and escalation statuses.

## Run

From this folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

## Data

The app loads `datasets/synthetic_hr_offer_acceptance_dataset.csv` and creates a
local SQLite database (`ctc_recommender.sqlite3`) at startup. Upload a new
CSV/Excel through the Desk Copilot to retrain the model without restarting.

## Stack

FastAPI + pandas + scikit-learn backend · React 18 (no build step) frontend ·
hand-rolled SVG charts (validated colorblind-safe palette) · LLM-backed copilot
and negotiation agents.
