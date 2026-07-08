# CTC Offer Intelligence

Prototype HR dashboard and CTC offer recommender.

## What It Shows

- Offer outcome funnel: joined, accepted, declined, no-show.
- Acceptance trend over time.
- Accepted CTC percentiles: P20, P50, P80.
- Offer performance by band and candidate source.
- Recent offers table.
- Candidate offer simulator with:
  - suggested CTC
  - acceptance probability at current offer
  - acceptance probability curve
  - fallback benchmark filters used
  - warnings when the benchmark is broad

## Model

The app uses a logistic regression classifier trained on the historical offer dataset.

The model predicts:

```text
P(accept | candidate profile, offered CTC)
```

The CTC benchmark uses historical accepted-offer percentiles with flexible fallback levels.

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

The app loads:

```text
datasets/synthetic_hr_offer_acceptance_dataset.csv
```

and creates a local SQLite database:

```text
ctc_recommender.sqlite3
```

The database is generated at app startup and can be recreated from the CSV.

