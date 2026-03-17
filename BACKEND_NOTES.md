# 🔧 Backend – Person B Handoff Notes

## Files you own (replace originals in the repo with these)

| File | What changed |
|------|-------------|
| `ml/predict.py` | **Lazy loading** (won't crash on start), **absolute paths** (works on Vercel), **mock fallback** (demo works before Person A trains model) |
| `ml/segmentation.py` | Absolute paths, cleaner fallback |
| `api/__init__.py` | **NEW** – required for Python package imports on Vercel |
| `api/routes/__init__.py` | Kept as-is |
| `vercel.json` | Fixed routes for static assets (`/assets/` and `/components/`) |
| `requirements.txt` | Unchanged – already correct |
| Everything else | Unchanged – `index.py`, `schemas.py`, all routes are already correct |

---

## How to run locally

```bash
# From project root
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

Then open: http://localhost:8000/docs  ← interactive API explorer

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/api/predict` | Single customer churn prediction |
| POST | `/api/simulate` | What-if comparison (two scenarios) |
| GET | `/api/dashboard` | Aggregate stats |
| GET | `/api/dashboard/customers` | Customer list with risk labels |

---

## What happens before Person A trains the model?

`ml/predict.py` now has a **mock mode**. If `.pkl` files don't exist yet:
- The API starts up fine (no crash)
- `/api/predict` returns realistic rule-based scores
- The entire frontend works for demo purposes

Once Person A runs `notebooks/03_model_training.ipynb` and commits the `.pkl` files to `ml/model_artifacts/`, the API automatically switches to real predictions on next deploy.

---

## Deploy to Vercel

```bash
npm install -g vercel
vercel login
vercel --prod
```

After deploy, give the live URL to Person C so they can update `API_BASE` in the 3 HTML files.

---

## Running tests

```bash
# Requires model artifacts OR works in mock mode
pytest tests/ -v
```

Tests cover: health, root, predict shape, probability range, validation errors, simulate delta, dashboard stats.
