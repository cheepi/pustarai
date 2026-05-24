# Pustara AI — Recommendation Server

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-proprietary-lightgrey)](#license)

Concise, production-ready FastAPI service that provides hybrid book recommendations combining content, collaborative, social, trending and demographic signals.

## Table of contents
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [Rebuild models](#rebuild-models)
- [Docker](#docker)
- [Files](#files)

## Quick start

Requirements: Python 3.10+, pip.

Install dependencies:

```bash
pip install -r requirements.txt
```

Rebuild models first (requires `DATABASE_URL`):

```bash
python rebuild_models.py
```

Run locally:

```bash
uvicorn pustarai.fastapi_server:app --host 0.0.0.0 --port 8001 --reload
```

Server expects model artifacts under `./pustara_models` and a catalog fallback at `./pustara_books_100.csv` unless configured via env vars.

## Configuration

Set these environment variables as needed:

- `MAIN_BACKEND_URL` — optional backend URL to fetch catalog `/books`
- `DATABASE_URL` — Postgres DSN for analytics/catalog
- `REDIS_URL` — Redis DSN for social/trending/rate-limiting
- `MODEL_DIR` — models directory (default `./pustara_models`)
- `BOOKS_CSV` — fallback CSV (default `./pustara_books_100.csv`)
- `CORS_ORIGINS` — comma-separated allowed origins (default `http://localhost:3001`)
- `ADMIN_CONTACT_EMAIL` — contact email used in logs
- `HEALTH_SECRET` — optional secret for `GET /health`

Use a `.env` file or export variables in your shell prior to running the server.

## API

Selected endpoints (paths only):

- `GET /` — health/status
- `GET /health` — health check (optional secret)
- `GET /search/semantic` — semantic content search
- `GET /recommendations/similar-users`
- `GET /recommendations/cold-start`
- `GET /recommendations/trending`
- `POST /recommendations/direct`
- `POST /recommendations/chat`
- `POST /activity`
- `POST /reindex` — trigger model rebuild
- `POST /reload-catalog` — reload catalog from backend/DB/CSV

Example chat recommendation request:

```bash
curl -X POST http://localhost:8001/recommendations/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"<user-id>", "query":"recommend me modern indonesian fiction"}'
```

## Rebuild models

Model artifacts are not included in this repository. Run the included script to generate them. `DATABASE_URL` must be set.

```bash
python rebuild_models.py
```

This writes artifacts to `pustara_models/` which is excluded from version control. Alternatively, trigger a rebuild via the API after the server is running:

```bash
curl -X POST http://localhost:8001/reindex \
  -H "Content-Type: application/json" \
  -d '{"secret":"<RI_SECRET>"}'
```

## Docker

Build and run (example):

```bash
docker build -t pustarai:latest .
docker run --rm -p 8001:8001 -e PORT=8001 pustarai:latest
```

## Files

# Pustara AI — Recommendation Server

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-proprietary-lightgrey)](#license)

Concise, production-ready FastAPI service that provides hybrid book recommendations combining content, collaborative, social, trending and demographic signals.

## Table of contents
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [Rebuild models](#rebuild-models)
- [Docker](#docker)
- [Files](#files)

## Quick start

Requirements: Python 3.10+, pip.

Install dependencies:

```bash
pip install -r requirements.txt
```

Rebuild models first (requires `DATABASE_URL`):

```bash
python rebuild_models.py
```

Run locally:

```bash
uvicorn pustarai.fastapi_server:app --host 0.0.0.0 --port 8001 --reload
```

Server expects model artifacts under `./pustara_models` and a catalog fallback at `./pustara_books_100.csv` unless configured via env vars.

## Configuration

Set these environment variables as needed:

- `MAIN_BACKEND_URL` — optional backend URL to fetch catalog `/books`
- `DATABASE_URL` — Postgres DSN for analytics/catalog
- `REDIS_URL` — Redis DSN for social/trending/rate-limiting
- `MODEL_DIR` — models directory (default `./pustara_models`)
- `BOOKS_CSV` — fallback CSV (default `./pustara_books_100.csv`)
- `CORS_ORIGINS` — comma-separated allowed origins (default `http://localhost:3001`)
- `ADMIN_CONTACT_EMAIL` — contact email used in logs
- `HEALTH_SECRET` — optional secret for `GET /health`

Use a `.env` file or export variables in your shell prior to running the server.

## API

Selected endpoints (paths only):

- `GET /` — health/status
- `GET /health` — health check (optional secret)
- `GET /search/semantic` — semantic content search
- `GET /recommendations/similar-users`
- `GET /recommendations/cold-start`
- `GET /recommendations/trending`
- `POST /recommendations/direct`
- `POST /recommendations/chat`
- `POST /activity`
- `POST /reindex` — trigger model rebuild
- `POST /reload-catalog` — reload catalog from backend/DB/CSV

Example chat recommendation request:

```bash
curl -X POST http://localhost:8001/recommendations/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"<user-id>", "query":"recommend me modern indonesian fiction"}'
```

## Rebuild models

Model artifacts are not included in this repository. Run the included script to generate them. `DATABASE_URL` must be set.

```bash
python rebuild_models.py
```

This writes artifacts to `pustara_models/` which is excluded from version control. Alternatively, trigger a rebuild via the API after the server is running:

```bash
curl -X POST http://localhost:8001/reindex \
  -H "Content-Type: application/json" \
  -d '{"secret":"<RI_SECRET>"}'
```

## Docker

Build and run (example):

```bash
docker build -t pustarai:latest .
docker run --rm -p 8001:8001 -e PORT=8001 pustarai:latest
```

## Files

- [fastapi_server.py](fastapi_server.py)
- [rebuild_models.py](rebuild_models.py)
- [pustara_books_100.csv](pustara_books_100.csv)
- [requirements.txt](requirements.txt)
- [Dockerfile](Dockerfile)

## Repository hygiene

- Add a `.gitignore` that excludes `pustara_models/`, `*.pkl`, `logs/`, `.env`, and other large data files. Example entries:

```
# models and logs
pustara_models/
*.pkl
logs/

# local env
.env

# OS
.DS_Store
thumbs.db
```

- Add a `.env.example` (no secrets) listing required env vars; users copy it to `.env` and fill in secrets.

## Security note

- Never commit credentials, secrets, or production model artifacts. Use environment variables or a secrets manager.
- If you publish the repo, keep `pustara_models/` out of version control or use a private repo.

## API examples

Detailed example for `POST /recommendations/chat` (request + minimal response):

Request:
```bash
curl -X POST http://localhost:8001/recommendations/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-123","query":"recommend me modern indonesian fiction","n":5}'
```

Minimal successful response (example):

```json
{
  "query": "recommend me modern indonesian fiction",
  "recommendations": [
    {"book_id":"b1","title":"Novel A","final_score":0.92},
    {"book_id":"b2","title":"Novel B","final_score":0.86}
  ]
}
```

Example for `POST /reindex` (admin trigger):

```bash
curl -X POST http://localhost:8001/reindex \
  -H "Content-Type: application/json" \
  -d '{"secret":"<RI_SECRET>"}'
```

Response (example):

```json
{"status":"ok","message":"reindex started"}
```

## Run & debug tips

- Use `uvicorn ... --reload` for code changes during development.
- Check logs: the server prints startup messages and warnings about missing models/Redis/DB.
- If you plan to persist models built inside the container, mount `pustara_models` from host:

```bash
docker run --rm -p 8001:8001 -v $(pwd)/pustarai/pustara_models:/pustarai/pustara_models pustarai:local
```

## CI / smoke tests

- Consider a simple GitHub Actions workflow that builds the Docker image and runs a smoke test (curl `/`), ensuring PRs don't break basic startup.

## Contributing

- If others will collaborate, add `CONTRIBUTING.md` describing coding style, tests, and PR expectations.
- No license included per project preference.