# Analyses database

## Architecture: One-to-Many

One analysis (row in `analyses`) has many model results (rows in `model_results`).

SQLite file: `data/analyses.db`.

---

## Tables

### 1. `analyses` — analysis header (aggregated result)

| Column           | Type     | Description |
|------------------|----------|-------------|
| id               | INTEGER  | Primary key |
| text             | TEXT     | Input text analysed |
| url              | TEXT     | URL extracted from text (if any) |
| final_verdict    | TEXT     | `'phishing'` or `'safe'` |
| final_confidence | REAL     | Overall confidence (e.g. 70.3) |
| phishing_score   | REAL     | Weighted score for phishing |
| safety_score     | REAL     | Weighted score for safe |
| total_weight     | REAL     | Sum of model weights used |
| feedback         | TEXT     | User feedback: 'phishing', 'safe', 'not_sure', NULL |
| created_at       | TIMESTAMP| When the analysis was run |

### 2. `model_results` — per-model verdicts

| Column       | Type    | Description |
|--------------|---------|-------------|
| id           | INTEGER | Primary key |
| analysis_id  | INTEGER | FK → analyses.id |
| model_name   | TEXT    | e.g. "random_forest" |
| model_verdict| INTEGER | 1 = phishing, 0 = safe |
| confidence   | REAL    | Model confidence |
| weight_impact| REAL    | Model weight in final score |
| error        | TEXT    | Error message if prediction failed |

---

## Migration from old schema

On first run with an existing DB that still has the old schema (`analyses` with `is_phishing`, plus `model_votes` and `ensemble_results`), data is migrated automatically:

- `analyses` + `ensemble_results` → new `analyses` (final_verdict, final_confidence, phishing_score, safety_score, total_weight).
- `model_votes` → `model_results` (model_verdict, confidence, weight_impact).
- Old tables are dropped.

---

## API endpoints

- **POST /api/predict** — Saves analysis and model_results; returns `analysis_id`.
- **POST /api/feedback** — Updates `feedback` on an analysis.
- **GET /api/analyses** — List analyses (optional `limit`, `offset`, `feedback`).
- **GET /api/analyses/<id>** — One analysis with model_results (exposed as `model_votes` and `ensemble` for frontend).
- **GET /api/statistics** — Counts and distributions.

---

## Viewing data

`view_database.py`:

- `python view_database.py --analyses` — recent analyses
- `python view_database.py --votes` — model_results
- `python view_database.py --ensemble` — aggregated scores from analyses
- `python view_database.py --detail <id>` — full analysis by ID

See `DATABASE_VIEWER_GUIDE.md` for more.
