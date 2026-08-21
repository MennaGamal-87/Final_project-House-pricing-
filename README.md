# Bhavan — House Price Prediction (End-to-End ML Web App)

An end-to-end machine-learning product: a Jupyter notebook that cleans real-world, messy Indian
property listing data and trains a regression model, a FastAPI backend that serves it, and a React
+ TypeScript frontend where a user enters property details and gets an instant price estimate.

> **On the dataset:** the notebook is trained on the real Kaggle **House Price** dataset by Juhi
> Bhojani (187,531 property listings across India). A small `generate_synthetic_dataset.py` script
> is also included for reference — it was used only for an early pipeline smoke test in a sandbox
> without network access, and is not part of the final trained model. If your own CSV ever needs to
> replace `notebooks/data/house_prices.csv`, just re-run `Kernel → Restart & Run All`.

## Overview

1. A Jupyter notebook loads the raw CSV, explores it, cleans seven categories of messiness, trains
   and compares three regression models, and exports the winning model as a single
   scikit-learn `Pipeline` (`house_price.pkl`) — so all preprocessing travels with the model.
2. A FastAPI backend loads that pickle once at startup and exposes `POST /predict`.
3. A React + TypeScript (Vite) frontend collects property details in a form and displays the
   prediction.

## Architecture

```
                    ┌─────────────────────┐
 kaggle.com  ─────▶ │  house_prices.csv    │
 (or synthetic gen) │  (raw, messy data)   │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  notebooks/                │
                 │  house_price_model.ipynb   │
                 │  clean → EDA → train →     │
                 │  evaluate → export         │
                 └──────────┬─────────────────┘
                             │ house_price.pkl + locations.json
                             ▼
        ┌───────────────────────────────────┐
        │  backend/ (FastAPI, port 8000)     │
        │  loads pipeline once at startup    │
        │  GET  /health                      │
        │  POST /predict                     │
        └──────────────┬─────────────────────┘
                        │ JSON over HTTP (CORS)
                        ▼
        ┌───────────────────────────────────┐
        │  frontend/ (React + TS + Vite,     │
        │  port 5173)                        │
        │  form → API call → result page     │
        └───────────────────────────────────┘
```

## Tech stack

| Layer      | Technology                                                              |
| ---------- | ------------------------------------------------------------------------ |
| Data / ML  | Python 3.11, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib     |
| Backend    | FastAPI, Pydantic v2, pydantic-settings, uvicorn, pytest, httpx           |
| Frontend   | React 19, TypeScript, Vite, react-router-dom                             |
| Packaging  | Docker (backend)                                                         |

## Project structure

```
house-price-project/
├── notebooks/
│   ├── house_price_model.ipynb      # the full data-science notebook (see Phase 2)
│   ├── generate_synthetic_dataset.py # produces the stand-in dataset (see note above)
│   ├── house_price.pkl               # exported model (also copied into backend/models/)
│   ├── locations.json                # allowed locations for the frontend dropdown
│   ├── model_metadata.json           # which model won + whether it was trained on log(price)
│   └── data/
│       └── house_prices.csv          # dataset (synthetic here — see Dataset section)
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI app, CORS, model loaded at startup (lifespan)
│   │   ├── api/routes/prediction.py  # GET /health, POST /predict
│   │   ├── core/config.py            # Settings from .env (pydantic-settings)
│   │   ├── schemas/prediction.py     # PredictionRequest / PredictionResponse
│   │   ├── services/
│   │   │   ├── preprocessing.py      # turns a request into a one-row DataFrame
│   │   │   └── inference.py          # loads .pkl, runs predict, handles log-target inversion
│   │   └── utils/logging_config.py
│   ├── models/                       # house_price.pkl + locations.json + model_metadata.json
│   ├── tests/test_prediction.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   └── src/
│       ├── api/predictionClient.ts   # fetch wrapper, base URL from VITE_API_BASE_URL
│       ├── components/PredictionForm.tsx
│       ├── pages/{HomePage,ResultPage,NotFoundPage}.tsx
│       ├── types/prediction.ts       # TS types mirroring the backend schema
│       ├── locations.json            # copied from the notebook export
│       └── App.tsx                   # routes: / , /result , * (404)
├── .gitignore
└── README.md
```

## Dataset

**House Price** by Juhi Bhojani — https://www.kaggle.com/datasets/juhibhojani/house-price
(~187,000 real property listings across India).

### Download the real dataset

**Option A — manual:** open the link above, click **Download**, unzip, and place
`house_prices.csv` in `notebooks/data/`.

**Option B — Kaggle CLI (recommended):**

```bash
pip install kaggle
# Get your API token: Kaggle → Settings → API → "Create New Token"
# Place kaggle.json in C:\Users\<you>\.kaggle\ (Windows) or ~/.kaggle/ (macOS/Linux)
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

Then re-run `notebooks/house_price_model.ipynb` top-to-bottom (`Kernel → Restart & Run All`) and
copy the freshly exported `house_price.pkl`, `locations.json`, and `model_metadata.json` into
`backend/models/` and `frontend/src/`.

## Setup

### Prerequisites

| Tool         | Minimum version | Check with       |
| ------------ | ---------------- | ------------------ |
| Python       | 3.11              | `python --version` |
| Node.js + npm | 18               | `node --version`   |
| Git          | any recent        | `git --version`    |

### 1. Notebook

```bash
cd notebooks
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib nbformat

# (optional) regenerate the synthetic dataset, or download the real one - see Dataset section
python generate_synthetic_dataset.py

jupyter notebook house_price_model.ipynb
# Run all cells. This produces house_price.pkl, locations.json, model_metadata.json.
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# make sure models/house_price.pkl, models/locations.json, models/model_metadata.json exist
# (already included in this repo, or copy fresh ones from notebooks/ after retraining)

uvicorn app.main:app --reload
# open http://localhost:8000/docs and try /predict from the Swagger UI
```

Run the tests:

```bash
pytest
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# open http://localhost:5173
```

With the backend running on port 8000 and the frontend dev server on port 5173, fill out the form
and submit — you'll be routed to `/result` with a real prediction from the model.

## Environment variables

### `backend/.env`

| Variable              | Default                         | Description                                  |
| ---------------------- | -------------------------------- | ----------------------------------------------- |
| `CORS_ORIGINS`         | `http://localhost:5173`          | Comma-separated list of allowed frontend origins |
| `MODEL_PATH`            | `models/house_price.pkl`         | Path to the exported model pipeline             |
| `MODEL_METADATA_PATH`  | `models/model_metadata.json`     | Path to metadata (which model, log vs raw target) |
| `LOCATIONS_PATH`        | `models/locations.json`          | Path to the list of locations known at train time |

### `frontend/.env`

| Variable               | Default                  | Description                        |
| ------------------------ | -------------------------- | ------------------------------------- |
| `VITE_API_BASE_URL`      | `http://localhost:8000`    | Base URL of the FastAPI backend       |

## API reference

### `GET /health`

Returns `{"status": "ok"}` once the model has loaded.

### `POST /predict`

Request body (`location` is a lowercase city slug — see `models/locations.json` for the full
list, e.g. `bangalore`, `new-delhi`, `greater-noida`):

```json
{
  "location": "bangalore",
  "carpet_area_sqft": 1200,
  "floor_num": 3,
  "bathroom": 2,
  "balcony": 1,
  "furnishing": "Semi-Furnished",
  "transaction": "Resale",
  "ownership": "Freehold",
  "facing": "East"
}
```

Response:

```json
{ "predicted_price": 6160508.82 }
```

curl example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "bangalore",
    "carpet_area_sqft": 1200,
    "floor_num": 3,
    "bathroom": 2,
    "balcony": 1,
    "furnishing": "Semi-Furnished",
    "transaction": "Resale",
    "ownership": "Freehold",
    "facing": "East"
  }'
```

Unknown locations are automatically mapped to `"other"`, matching how the model was trained.
Invalid input (missing fields, non-positive area, etc.) returns `422 Unprocessable Entity` with
details on which field failed validation.

## Model metrics

Three models were trained on the real dataset (174,708 rows after cleaning) and compared on a
held-out 20% test split (see notebook section 2.5). The winner is selected **programmatically** —
whichever model has the lowest test-set RMSE — not assumed in advance:

| Model                            | MAE (₹)   | RMSE (₹)   | R²    |
| ----------------------------------- | ----------- | ------------ | ------- |
| **Random Forest (log-target)** ✅    | 1,405,997   | 6,103,659    | 0.791   |
| Random Forest (raw target)          | 1,554,044   | 6,193,558    | 0.785   |
| Linear Regression (log-target)      | 3,955,267   | 11,348,077   | 0.277   |

**Random Forest trained on `np.log1p(price)`** won clearly, with more than double the R² of the
linear baseline. A 3-fold cross-validation check on the winning model (on a 25,000-row subsample —
see the runtime note in the notebook) gave a mean CV MAE of ~0.142 in log-space, consistent with
the held-out test result, confirming the result is stable rather than a lucky split.

Because the winner was trained on the log-transformed price, the backend inverts predictions with
`np.expm1(...)`. One extra safeguard is applied on both the notebook's evaluation and the backend's
inference: log-space predictions are **clipped to the range actually seen in training**
(`log_pred_min` / `log_pred_max` in `model_metadata.json`) before calling `expm1`. This matters
because a model's raw log-space output is technically unbounded, and `expm1` grows exponentially —
so even a small overshoot in log-space can otherwise explode into an absurd, meaningless price for
an unusual combination of inputs. Random Forest predictions are naturally close to bounded already
(they're an average of training leaf values), but the clip is applied uniformly for safety.

Two extra cleaning steps were needed for the real data beyond what the base guide describes,
because the raw CSV contains a small number of genuine data-entry errors: a handful of listings
with an obviously wrong carpet area (e.g. `"709222 sqft"` for what is clearly a normal
apartment) and a few listings with prices in the hundreds of crores for a small flat. Both are
handled in notebook section 2.3.7 by bounding carpet area to a plausible 100–20,000 sqft range
*before* the price-per-sqft percentile trim, which by itself wasn't enough to catch these
high-leverage rows.

## Screenshots
![Uploading image.png…]()

The running app: a form for entering property details, and a result page showing the estimated
price in the ₹ Lac/Cr convention used across Indian real-estate listings.

*(Add screenshots here after running the app locally — `npm run dev` on the frontend and
`uvicorn app.main:app --reload` on the backend, both described above.)*

## Common pitfalls this build avoided

- Committing `.env` or the raw dataset CSV (both are git-ignored).
- scikit-learn version mismatch between notebook and backend — pinned to `scikit-learn==1.8.0` in
  `backend/requirements.txt`, matching the version printed at export time in the notebook.
- Hard-coding `http://localhost:8000` in frontend components — the base URL comes from
  `VITE_API_BASE_URL`.
- Reporting metrics on the training set instead of the held-out test set (section 2.5 evaluates
  strictly on `X_test` / `y_test`).
