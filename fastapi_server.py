"""
PUSTARA AI — FastAPI Server v5.2
Hybrid Recommender: Content (TF-IDF) + Collaborative + Social (Redis) + Trending (Redis)
                  + Demographic (Model C: gender + age group)
Port: 8001
"""

import asyncio
import os, json, pickle, logging, difflib, requests, uuid
import re
import warnings
from typing import Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
import psycopg2.pool
import random

# Load .env automatically for local runs
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"), override=False)
except Exception:
    pass

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis-py not installed. Social + Trending signals = 0.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pustara.ai")


def _read_sql_compat(query: str, conn):
    # Pandas warns for raw psycopg2 connections; keep behavior and silence noisy warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable.*",
            category=UserWarning,
        )
        return pd.read_sql_query(query, conn)

MAIN_BACKEND_URL = os.getenv("MAIN_BACKEND_URL", "").strip()
NEON_DB_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./pustara_models"))
BOOKS_CSV = Path(os.getenv("BOOKS_CSV", "./pustara_books_100.csv"))
MAIN_BACKEND_PAGE_LIMIT = int(os.getenv("MAIN_BACKEND_PAGE_LIMIT", "200"))
MAIN_BACKEND_MAX_PAGES = int(os.getenv("MAIN_BACKEND_MAX_PAGES", "25"))
MAIN_BACKEND_TIMEOUT_S = int(os.getenv("MAIN_BACKEND_TIMEOUT_S", "15"))
MODEL_A_PATH = MODEL_DIR / "model_a_prod.pkl"
MODEL_B_PATH = MODEL_DIR / "model_b_collaborative.pkl"
MODEL_C_PATH = MODEL_DIR / "model_c_demographic.pkl"
CACHE_TTL = int(os.getenv("REC_CACHE_TTL", "21600"))   # 6 jam
TRENDING_TTL_S = 8 * 24 * 3600                               # 8 hari
TRENDING_SNAPSHOT_KEY = "trending:last_snapshot"
USER_INTERACTION_COUNT_PREFIX = "user:interaction_count:"

# CORS — ambil dari env, fallback localhost dev
_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3001")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# Email admin — jangan hardcode, ambil dari env
ADMIN_CONTACT_EMAIL = os.getenv("ADMIN_CONTACT_EMAIL", "admin@pustara.id")

# Health check secret (opsional — kalau tidak di-set, /health terbuka)
HEALTH_SECRET = os.getenv("HEALTH_SECRET", "")

# Chat rate limiting: max N request per user per window (detik)
CHAT_RATE_LIMIT_MAX = int(os.getenv("CHAT_RATE_LIMIT_MAX", "10")) # 10 req
CHAT_RATE_LIMIT_WINDOW = int(os.getenv("CHAT_RATE_LIMIT_WINDOW", "60")) # per menit

ACTION_WEIGHTS = {"view": 1, "read": 3, "like": 5, "bookmark": 4, "share": 2, "review": 8}

_reindex_lock = asyncio.Lock()

# Global state
state: dict = {
    "model_a": None,
    "model_b": None,
    "model_c": None,
    "catalog": None,
    "redis": None,
    "db_pool": None, # psycopg2 ThreadedConnectionPool
    "last_reindex": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Pustara AI v5.2 starting up …")
    _init_redis()
    _init_db_pool()
    _load_models()
    _load_catalog()
    _ensure_analytics_table()
    log.info("✅ Startup complete.")
    yield
    log.info("👋 Shutting down Pustara AI.")
    _close_db_pool()


app = FastAPI(title="Pustara AI", version="5.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS, 
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    allow_credentials=True,
)

@app.get("/")
def home():
    return {"status": "Pustara AI is running!", "version": "5.2.0"}


def _init_redis():
    if not REDIS_AVAILABLE or not REDIS_URL:
        log.warning("Redis not configured. Social/Trending signals disabled.")
        return
    try:
        r = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=3)
        r.ping()
        state["redis"] = r
        log.info("✅ Redis connected: %s", REDIS_URL[:40])
    except Exception as e:
        log.warning("Redis connection failed: %s — signals disabled.", e)


def _init_db_pool():
    """Inisialisasi ThreadedConnectionPool — tidak buka koneksi baru tiap request."""
    if not NEON_DB_URL:
        log.warning("DATABASE_URL not set — DB features disabled.")
        return
    try:
        pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=10,
            dsn=NEON_DB_URL,
        )
        state["db_pool"] = pool
        log.info("✅ DB connection pool initialized (min=1, max=10).")
    except Exception as e:
        log.warning("DB pool init failed: %s", e)


def _close_db_pool():
    pool = state.get("db_pool")
    if pool:
        try:
            pool.closeall()
            log.info("DB pool closed.")
        except Exception:
            pass


def _get_db_conn():
    """Ambil koneksi dari pool. Caller wajib pool.putconn(conn) setelah selesai."""
    pool = state.get("db_pool")
    if not pool:
        # Fallback kalau pool belum ready: buka koneksi langsung
        if not NEON_DB_URL:
            return None
        try:
            return psycopg2.connect(NEON_DB_URL)
        except Exception as e:
            log.warning("DB fallback connection error: %s", e)
            return None
    try:
        return pool.getconn()
    except Exception as e:
        log.warning("DB pool getconn error: %s", e)
        return None


def _put_db_conn(conn):
    """Kembalikan koneksi ke pool (atau close kalau tidak ada pool)."""
    pool = state.get("db_pool")
    if pool and conn:
        try:
            pool.putconn(conn)
        except Exception:
            pass
    elif conn:
        try:
            conn.close()
        except Exception:
            pass


def _get_redis_user_interactions(user_id: Optional[str]) -> Optional[int]:
    if not user_id:
        return None

    r = state.get("redis")
    if r is None:
        return None

    try:
        raw_value = r.get(f"{USER_INTERACTION_COUNT_PREFIX}{user_id}")
        if raw_value is None:
            return None
        return max(int(float(raw_value)), 0)
    except Exception as e:
        log.debug("Redis user interaction counter read error: %s", e)
        return None


def _ensure_analytics_table():
    """Buat tabel chat_analytics kalau belum ada."""
    conn = _get_db_conn()
    if not conn:
        log.warning("Analytics table skipped — no DB connection.")
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_analytics (
                id              BIGSERIAL PRIMARY KEY,
                user_id         TEXT,
                query           TEXT NOT NULL,
                intent          TEXT,
                detected_title  TEXT,
                detected_genre  TEXT,
                detected_author TEXT,
                detected_language TEXT,
                show_recommendations BOOLEAN DEFAULT FALSE,
                n_results       INTEGER DEFAULT 0,
                groq_used       BOOLEAN DEFAULT FALSE,
                ts              TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_chat_analytics_user_id ON chat_analytics(user_id);
            CREATE INDEX IF NOT EXISTS idx_chat_analytics_ts      ON chat_analytics(ts);
            CREATE INDEX IF NOT EXISTS idx_chat_analytics_intent  ON chat_analytics(intent);
        """)
        conn.commit()
        log.info("✅ chat_analytics table ensured.")
    except Exception as e:
        log.warning("Analytics table creation failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        _put_db_conn(conn)


def _log_chat_analytics(
    user_id: Optional[str],
    query: str,
    parsed: dict,
    show_recs: bool,
    n_results: int,
    groq_used: bool,
):
    """Catat satu baris ke chat_analytics secara best-effort (tidak boleh crash endpoint)."""
    conn = _get_db_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_analytics
                (user_id, query, intent, detected_title, detected_genre,
                 detected_author, detected_language, show_recommendations, n_results, groq_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id or None,
            query[:500], # truncate panjang
            parsed.get("intent"),
            parsed.get("title"),
            parsed.get("genre"),
            parsed.get("author"),
            parsed.get("language"),
            show_recs,
            n_results,
            groq_used,
        ))
        conn.commit()
    except Exception as e:
        log.debug("chat_analytics insert error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        _put_db_conn(conn)


def _check_chat_rate_limit(user_id: Optional[str], ip: Optional[str]) -> bool:
    """
    Return True kalau masih boleh lanjut, False kalau sudah over limit.
    Pakai sliding window per user_id (atau IP kalau tidak ada user_id).
    """
    r = state["redis"]
    if r is None:
        return True # Kalau Redis tidak ada, skip rate limit

    key_subject = user_id or ip or "anon"
    rk = f"ratelimit:chat:{key_subject}"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    window_ms = CHAT_RATE_LIMIT_WINDOW * 1000

    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(rk, 0, now_ms - window_ms)
        pipe.zadd(rk, {str(now_ms): now_ms})
        pipe.zcard(rk)
        pipe.expire(rk, CHAT_RATE_LIMIT_WINDOW + 5)
        results = pipe.execute()
        count = results[2]
        return count <= CHAT_RATE_LIMIT_MAX
    except Exception as e:
        log.debug("Rate limit check error: %s", e)
        return True # Fail open


def _normalize_model_a(m: dict) -> dict:
    if "book_ids" in m and "matrix" in m:
        return m
    df = m["dataframe"]
    book_ids = df["book_id"].astype(str).tolist()
    return {
        "vectorizer": m["vectorizer"],
        "tfidf_matrix": m["tfidf_matrix"],
        "matrix": m["cosine_sim"],
        "book_ids": book_ids,
        "_source": "rebuild_models",
    }


def _normalize_model_b(m: dict) -> dict:
    if "book_ids" in m and "matrix" in m:
        return m
    import numpy as np
    book_ids = [m["idx2book"][i] for i in sorted(m["idx2book"].keys())]
    user_ids = [u for u, _ in sorted(m["user2idx"].items(), key=lambda x: x[1])]
    ui = m["user_item"]
    ui_array = ui.toarray() if hasattr(ui, "toarray") else np.asarray(ui)
    item_sim = m["item_sim"]
    item_sim_dense = item_sim.toarray() if hasattr(item_sim, "toarray") else np.asarray(item_sim)
    return {
        "matrix": item_sim_dense,
        "book_ids": book_ids,
        "user_ids": user_ids,
        "user_item_matrix": ui_array,
        "_source": "rebuild_models",
    }


def _load_models():
    if MODEL_A_PATH.exists():
        with open(MODEL_A_PATH, "rb") as f:
            raw = pickle.load(f)
        state["model_a"] = _normalize_model_a(raw)
        log.info("✅ Model A loaded: %d books", len(state["model_a"]["book_ids"]))
    else:
        log.warning("Model A not found at %s — run /reindex first.", MODEL_A_PATH)

    if MODEL_B_PATH.exists():
        with open(MODEL_B_PATH, "rb") as f:
            raw = pickle.load(f)
        state["model_b"] = _normalize_model_b(raw)
        log.info("✅ Model B loaded: %d books", len(state["model_b"]["book_ids"]))
    else:
        log.warning("Model B not found at %s — collaborative disabled.", MODEL_B_PATH)

    model_c_candidates = [MODEL_C_PATH, MODEL_DIR / "model_c_demographic.pkl"]
    for path_c in model_c_candidates:
        if path_c.exists():
            with open(path_c, "rb") as f:
                state["model_c"] = pickle.load(f)
            log.info("✅ Model C loaded: %d books profiled", state["model_c"].get("n_books", 0))
            break
    if state["model_c"] is None:
        log.warning("Model C not found — demographic signals disabled (run /reindex).")
    else:
        log.info("✅ Model C active.")


def _load_catalog() -> pd.DataFrame:
    df = None
    strict_backend_mode = bool(MAIN_BACKEND_URL)

    def _listish_to_csv(value) -> str:
        if isinstance(value, list):
            return ", ".join(str(v).strip() for v in value if str(v).strip())
        text = "" if value is None else str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return ", ".join(str(v).strip() for v in parsed if str(v).strip())
            except Exception:
                pass
        return text

    if MAIN_BACKEND_URL:
        try:
            rows = []
            page = 1
            limit = max(1, MAIN_BACKEND_PAGE_LIMIT)
            max_pages = max(1, MAIN_BACKEND_MAX_PAGES)

            while True:
                if page > max_pages:
                    log.warning("Catalog bootstrap capped at %d pages.", max_pages)
                    break

                resp = requests.get(
                    MAIN_BACKEND_URL,
                    params={"page": page, "limit": limit},
                    timeout=max(3, MAIN_BACKEND_TIMEOUT_S),
                )
                resp.raise_for_status()
                payload = resp.json()

                if isinstance(payload, list):
                    rows.extend(payload)
                    break

                if not isinstance(payload, dict):
                    break

                batch = payload.get("data")
                if not isinstance(batch, list):
                    batch = payload.get("books") if isinstance(payload.get("books"), list) else []

                rows.extend(batch)

                pagination = payload.get("pagination") if isinstance(payload.get("pagination"), dict) else {}
                total_pages = int(pagination.get("total_pages") or pagination.get("pages") or 1)
                if not batch or page >= total_pages:
                    break
                page += 1

            if rows:
                df = pd.DataFrame(rows)
                if "book_id" not in df.columns and "id" in df.columns:
                    df["book_id"] = df["id"].astype(str)

                for col in [
                    "book_id", "title", "authors", "genres", "description", "year", "pages",
                    "language", "avg_rating", "rating_count", "total_stock", "cover_url"
                ]:
                    if col not in df.columns:
                        df[col] = ""

                df["authors"] = df["authors"].apply(_listish_to_csv)
                df["genres"] = df["genres"].apply(_listish_to_csv)
                df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(0)
                df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)
                df["total_stock"] = pd.to_numeric(df["total_stock"], errors="coerce").fillna(1)

                log.info("✅ Catalog loaded from backend /books: %d books", len(df))
            else:
                log.error("Backend /books returned no data.")
        except Exception as e:
            log.error("Backend /books unavailable (%s).", e)

    if df is None and not strict_backend_mode and NEON_DB_URL:
        try:
            conn = _get_db_conn()
            if conn:
                df = _read_sql_compat(
                    """SELECT id::text AS book_id, title,
                              array_to_string(authors, ', ') AS authors,
                              array_to_string(genres, ', ') AS genres,
                              description, year, pages, language,
                              avg_rating, rating_count, total_stock,
                              COALESCE(cover_url, '') AS cover_url
                       FROM books WHERE is_active = TRUE ORDER BY title""",
                    conn,
                )
                _put_db_conn(conn)
                log.info("✅ Catalog loaded from Neon: %d books", len(df))
        except Exception as e:
            log.warning("Neon unavailable (%s) — falling back to CSV.", e)

    if df is None and not strict_backend_mode and BOOKS_CSV.exists():
        df = pd.read_csv(BOOKS_CSV, dtype=str)
        df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(0)
        df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)
        df["total_stock"] = pd.to_numeric(df["total_stock"], errors="coerce").fillna(1)
        log.info("✅ Catalog loaded from CSV: %d books", len(df))

    if df is None:
        df = pd.DataFrame(columns=["book_id","title","authors","genres",
                                    "description","year","pages","language",
                                    "avg_rating","rating_count","cover_url"])
        if strict_backend_mode:
            log.error("⚠️  Strict backend mode: catalog load failed.")
        else:
            log.error("⚠️  No catalog source found!")

    df = df.fillna("")
    df["soup"] = (
        df["genres"].str.replace(",", " ").str.lower() + " " +
        df["authors"].str.lower() + " " +
        df["title"].str.lower()
    )
    state["catalog"] = df
    return df


def _is_uuid_like(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        uuid.UUID(str(value))
        return True
    except Exception:
        return False


def _resolve_user_uuid(user_ref: Optional[str]) -> Optional[str]:
    if not user_ref:
        return None
    ref = str(user_ref).strip()
    if _is_uuid_like(ref):
        return ref

    conn = _get_db_conn()
    if not conn:
        return ref

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='users'"
        )
        cols = {r[0] for r in cur.fetchall()}
        for col in ["firebase_uid", "uid", "auth_uid", "external_uid", "user_uid"]:
            if col in cols:
                cur.execute(f"SELECT id::text FROM users WHERE {col} = %s LIMIT 1", (ref,))
                row = cur.fetchone()
                if row and row[0]:
                    return str(row[0])
        cur.execute("SELECT id::text FROM users WHERE id::text = %s LIMIT 1", (ref,))
        row = cur.fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception as e:
        log.debug("resolve user uuid error: %s", e)
    finally:
        _put_db_conn(conn)

    return ref


def _resolve_book_uuid(book_ref: Optional[str]) -> Optional[str]:
    if not book_ref:
        return None
    ref = str(book_ref).strip()
    if _is_uuid_like(ref):
        return ref

    conn = _get_db_conn()
    if not conn:
        return ref

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='books'"
        )
        cols = {r[0] for r in cur.fetchall()}

        if "external_key" in cols:
            cur.execute("SELECT id::text FROM books WHERE lower(external_key)=lower(%s) LIMIT 1", (ref,))
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])

        cur.execute("SELECT id::text FROM books WHERE lower(title)=lower(%s) LIMIT 1", (ref,))
        row = cur.fetchone()
        if row and row[0]:
            return str(row[0])

        cur.execute("SELECT id::text FROM books WHERE id::text=%s LIMIT 1", (ref,))
        row = cur.fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception as e:
        log.debug("resolve book uuid error: %s", e)
    finally:
        _put_db_conn(conn)

    return ref


def _content_scores(query_book_id: Optional[str], query_text: Optional[str]) -> dict:
    m = state["model_a"]
    if m is None:
        return {}
    book_ids = m["book_ids"]
    vectorizer = m["vectorizer"]
    sim_matrix = m["matrix"]
    tfidf_matrix = m.get("tfidf_matrix")

    if query_book_id and query_book_id in book_ids:
        idx = book_ids.index(query_book_id)
        row = np.asarray(sim_matrix[idx]).flatten()
        return {bid: float(s) for bid, s in zip(book_ids, row)}

    if query_text and tfidf_matrix is not None:
        from sklearn.metrics.pairwise import linear_kernel
        vec = vectorizer.transform([query_text])
        sims = linear_kernel(vec, tfidf_matrix).flatten()
        return {bid: float(s) for bid, s in zip(book_ids, sims)}

    return {}


def _profile_content_scores(user_id: Optional[str], top_k: int = 5) -> dict:
    m = state["model_a"]
    if m is None or not user_id:
        return {}

    book_ids = m["book_ids"]
    sim_matrix = m["matrix"]
    conn = _get_db_conn()
    if not conn:
        return {}

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            "SELECT book_id::text AS book_id FROM user_book_scores "
            "WHERE user_id=%s ORDER BY score DESC, updated_at DESC LIMIT %s",
            (user_id, top_k),
        )
        seeds = [str(r["book_id"]) for r in cur.fetchall() if r.get("book_id")]
    except Exception as e:
        log.debug("Profile content seed error: %s", e)
        return {}
    finally:
        _put_db_conn(conn)

    seed_indices = [book_ids.index(seed) for seed in seeds if seed in book_ids]
    if not seed_indices:
        return {}

    try:
        rows = [np.asarray(sim_matrix[idx]).flatten() for idx in seed_indices]
        avg_row = np.mean(rows, axis=0)
        avg_row = np.maximum(avg_row, 0)
        mx = float(np.max(avg_row)) if len(avg_row) > 0 else 0.0
        if mx <= 0:
            return {}
        norm = avg_row / mx
        return {bid: float(score) for bid, score in zip(book_ids, norm)}
    except Exception as e:
        log.debug("Profile content score error: %s", e)
        return {}


def _collab_scores(user_id: Optional[str]) -> dict:
    m = state["model_b"]
    if m is None or not user_id:
        return {}
    try:
        user_ids = m.get("user_ids", [])
        book_ids = m["book_ids"]
        sim_matrix = m["matrix"]
        if user_id not in user_ids:
            return {}
        u_idx = user_ids.index(user_id)
        user_vec = m.get("user_item_matrix")
        if user_vec is None:
            return {}
        row = np.asarray(user_vec[u_idx]).flatten()
        scores = row @ sim_matrix
      
        mx = float(scores.max()) if len(scores) > 0 else 0.0
        if mx > 0:
            scores = scores / mx
        return {bid: float(s) for bid, s in zip(book_ids, scores)}
    except Exception as e:
        log.debug("Collab score error: %s", e)
        return {}


def _social_scores(followed_ids: list, catalog_ids: list) -> dict:
    r = state["redis"]
    if r is None or not followed_ids:
        return {}
    try:
        cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=14)).timestamp() * 1000)
        entries = r.xrange("activity:stream", min=cutoff_ms, count=5000)
        counts: dict = {}
        for _, fields in entries:
            uid = fields.get("user_id", "")
            bid = fields.get("book_id", "")
            act = fields.get("action", "view")
            if uid in followed_ids and bid in catalog_ids:
                w = ACTION_WEIGHTS.get(act, 1)
                counts[bid] = counts.get(bid, 0) + w
        if not counts:
            return {}
        mx = max(counts.values())
        return {bid: v / mx for bid, v in counts.items()}
    except Exception as e:
        log.debug("Social score error: %s", e)
        return {}


def _trending_scores(catalog_ids: list) -> dict:
    r = state["redis"]
    if r is None:
        return {}
    try:
        pairs = r.zrevrange("trending:books:7d", 0, 199, withscores=True)
        raw = {bid: s for bid, s in pairs if bid in catalog_ids}
        if not raw:
            return {}
        mx = max(raw.values())
        return {bid: v / mx for bid, v in raw.items()}
    except Exception as e:
        log.debug("Trending score error: %s", e)
        return {}


def _demographic_scores(
    gender: Optional[str],
    age_group: Optional[str],
    preferred_genres: Optional[list],
    catalog_ids: list,
) -> dict:
    mc = state["model_c"]
    if mc is None:
        return {}
    if not gender and not age_group and not preferred_genres:
        return {}

    book_demo = mc.get("book_demographic", {})
    genre_age = mc.get("genre_age_weights", {})
    result = {}

    for bid in catalog_ids:
        demo = book_demo.get(bid, {})
        scores = []
        if gender and gender in demo:
            scores.append(demo[gender])
        if age_group and age_group in demo:
            scores.append(demo[age_group])
        base = sum(scores) / len(scores) if scores else 0.5

        genre_boost = 0.0
        if preferred_genres and age_group:
            age_map = {"<20": 0, "21-30": 1, "31-40": 2, ">40": 3}
            age_idx = age_map.get(age_group, 1)
            for pg in preferred_genres:
                aw = genre_age.get(pg)
                if aw and aw[age_idx] >= 0.65:
                    genre_boost = min(genre_boost + 0.10, 0.30)

        result[bid] = min(round(base + genre_boost, 4), 1.0)

    return result


def _dynamic_weights(n_interactions: int):
    if n_interactions < 5:
        # Cold: content dominan, demographic/trending sebagai cold-start signal
        return 0.70, 0.00, 0.15, 0.15
    elif n_interactions < 20:
        # Mid: content floor 0.35, gaboleh hilang meski collab mulai aktif
        t = (n_interactions - 5) / 15.0
        alpha = max(0.35, 0.70 - t * 0.30)
        beta = min(0.40, t * 0.40)
        gamma = 0.15 - t * 0.05
        delta = 0.15 - t * 0.05
        total = alpha + beta + gamma + delta
        return round(alpha/total, 3), round(beta/total, 3), round(gamma/total, 3), round(delta/total, 3)
    else:
        # Warm: content 40% tetap hadir, collab 35% — balance tidak jomplang
        return 0.40, 0.35, 0.15, 0.10


def _fallback_popular(n: int = 8, exclude_ids: list = None) -> list:
    df = state["catalog"]
    if df is None or len(df) == 0:
        return []
    exclude_ids = set(exclude_ids or [])
    df2 = df[~df["book_id"].isin(exclude_ids)].copy()
    df2["pop_score"] = (
        df2["avg_rating"].astype(float) *
        np.log1p(df2["rating_count"].astype(float))
    )
    top_candidates = df2.sort_values("pop_score", ascending=False).head(40)
    if len(top_candidates) > n:
        df2 = top_candidates.sample(n=n)
    else:
        df2 = top_candidates

    results = []
    for _, row in df2.iterrows():
        results.append({
            "book_id": row["book_id"],
            "title": row["title"],
            "authors": row["authors"],
            "genres": row["genres"],
            "description": str(row.get("description", ""))[:200],
            "year": row.get("year", ""),
            "language": str(row.get("language", "")),
            "avg_rating": float(row.get("avg_rating", 0)),
            "rating_count": int(row.get("rating_count", 0)),
            "cover_url": str(row.get("cover_url", "")),
            "content_score": 0.0,
            "collab_score": 0.0,
            "social_score": 0.0,
            "trending_score": 0.0,
            "final_score": float(row.get("pop_score", 0)),
            "n_interactions": 0,
            "weights": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 1.0},
            "reason_primary": "Buku Populer Pilihan Pustara 🔥",
            "reason_secondary": "",
            "signals": [],
        })
    return results


GENRE_MAP = {
    "fiksi": "fiksi", "fiction": "fiksi", "novel": "fiksi",
    "romance": "romance", "cinta": "romance", "romantis": "romance",
    "misteri": "misteri", "mystery": "misteri", "thriller": "thriller",
    "sains": "sains", "science": "sains", "ilmu": "sains",
    "sejarah": "sejarah", "history": "sejarah", "historis": "sejarah",
    "inspiratif": "inspiratif", "motivasi": "inspiratif", "inspiring": "inspiratif",
    "self-help": "self-help", "pengembangan diri": "self-help",
    "fantasi": "fantasi", "fantasy": "fantasi", "magic": "fantasi",
    "filsafat": "filsafat", "philosophy": "filsafat",
    "biografi": "biografi", "biography": "biografi", "memoir": "biografi",
    "distopia": "distopia", "dystopia": "distopia",
    "horror": "horror", "horor": "horror",
    "petualangan": "petualangan", "adventure": "petualangan",
    "humor": "humor", "lucu": "humor", "comedy": "humor",
    "indonesia": "id", "indonesia banget": "id", "lokal": "id",
    "internasional": "en", "inggris": "en",
}

SMALLTALK_PATTERNS = [
    "hai", "halo", "hello", "hi", "hey", "hei",
    "apa kabar", "assalamualaikum", "selamat pagi", "selamat siang",
    "selamat malam", "good morning", "good night",
    "siapa kamu", "kamu siapa", "who are you",
    "terima kasih", "makasih", "thanks", "thank you",
    "hehe", "haha", "wkwk", "lol",
]


def _normalize_lookup_text(text: str) -> str:
    lowered = str(text or "").lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(cleaned.split())


def _score_title_match(query_norm: str, title_norm: str) -> float:
    if not query_norm or not title_norm:
        return 0.0
    if title_norm in query_norm:
        return 1.0
    if query_norm in title_norm and len(query_norm) >= 6:
        return 0.92
    q_tokens = set(query_norm.split())
    t_tokens = title_norm.split()
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(set(t_tokens)))
    overlap_ratio = overlap / len(t_tokens)
    token_score = overlap_ratio if overlap >= 2 else overlap_ratio * 0.5
    fuzzy_score = difflib.SequenceMatcher(None, query_norm, title_norm).ratio()
    return max(token_score, fuzzy_score)


def _extract_title_from_query(raw_query: str, catalog_df: Optional[pd.DataFrame]) -> Optional[str]:
    if catalog_df is None or len(catalog_df) == 0:
        return None
    query_norm = _normalize_lookup_text(raw_query)
    if len(query_norm) < 3:
        return None
    best_title = None
    best_score = 0.0
    for title in catalog_df["title"].dropna().astype(str).tolist():
        title_norm = _normalize_lookup_text(title)
        score = _score_title_match(query_norm, title_norm)
        if score > best_score:
            best_score = score
            best_title = title
    return best_title if best_score >= 0.72 else None


def _extract_quoted_title(raw_query: str) -> Optional[str]:
    if not raw_query:
        return None
    # Ambil judul yang ditulis dalam tanda kutip ganda/typographic quote.
    matches = re.findall(r'["“](.+?)["”]', str(raw_query))
    for candidate in matches:
        title = " ".join(str(candidate).split()).strip(" .,!?:;")
        if len(title) >= 3:
            return title
    return None


def _is_title_in_catalog(title: str) -> bool:
    if not title:
        return False
    df = state.get("catalog")
    if df is None or len(df) == 0:
        return False

    qn = _normalize_lookup_text(title)
    if not qn:
        return False

    for raw_title in df["title"].dropna().astype(str).tolist():
        tn = _normalize_lookup_text(raw_title)
        if not tn:
            continue
        if qn == tn:
            return True
        if len(qn) >= 6 and (qn in tn or tn in qn):
            return True
        if difflib.SequenceMatcher(None, qn, tn).ratio() >= 0.90:
            return True

    return False


def _detect_missing_catalog_title(raw_query: str) -> Optional[str]:
    requested = _extract_quoted_title(raw_query)
    if requested and not _is_title_in_catalog(requested):
        return requested
    return None


def parse_query(raw: str) -> dict:
    q = raw.strip().lower()

    for pat in SMALLTALK_PATTERNS:
        if pat in q:
            return {"intent": "smalltalk", "genre": None, "author": None,
                    "title": None, "language": None, "query_text": ""}

    result = {"intent": "general", "genre": None, "author": None,
              "title": None, "language": None, "query_text": q}

    for kw, genre in GENRE_MAP.items():
        if kw in q:
            if genre in ("id", "en"):
                result["language"] = genre
            else:
                result["genre"] = genre
                result["intent"] = "genre"

    df = state["catalog"]
    if df is not None and len(df) > 0:
        title_from_query = _extract_title_from_query(raw, df)
        if title_from_query:
            result["title"] = title_from_query
            result["intent"] = "title"

        authors_flat = []
        for a in df["authors"].dropna():
            for name in str(a).split(","):
                name = name.strip().lower()
                if len(name) > 3:
                    authors_flat.append(name)

        matches = difflib.get_close_matches(q, authors_flat, n=1, cutoff=0.55)
        if matches:
            result["author"] = matches[0]
            result["intent"] = "author"

        if not result["title"]:
            titles = [str(t) for t in df["title"].dropna()]
            t_matches = difflib.get_close_matches(raw, titles, n=1, cutoff=0.45)
            if t_matches:
                result["title"] = t_matches[0]
                result["intent"] = "title"

    return result


def hybrid_recommend(
    user_id: Optional[str],
    query_book_id: Optional[str],
    query_text: Optional[str],
    n: int = 8,
    exclude_ids: list = None,
    language_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
) -> list:
    df = state["catalog"]
    if df is None or len(df) == 0:
        return []

    catalog_ids = df["book_id"].tolist()
    exclude_ids = set(exclude_ids or [])

    n_interactions = 0
    followed_ids = []
    if user_id:
        redis_interactions = _get_redis_user_interactions(user_id)
        if redis_interactions is not None:
            n_interactions = redis_interactions

        conn = _get_db_conn()
        if conn:
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                if redis_interactions is None:
                    cur.execute(
                        "SELECT COALESCE(SUM(" \
                        "COALESCE(views,0)+COALESCE(reads,0)+COALESCE(likes,0)+" \
                        "COALESCE(bookmarks,0)+COALESCE(shares,0)+COALESCE(review_cnt,0)" \
                        "),0) "
                        "FROM user_book_scores WHERE user_id=%s", (user_id,)
                    )
                    row = cur.fetchone()
                    n_interactions = int(row[0]) if row else 0
                cur.execute("SELECT following_id::text FROM follows WHERE follower_id=%s", (user_id,))
                followed_ids = [r[0] for r in cur.fetchall()]
            except Exception as e:
                log.debug("DB query error in hybrid_recommend: %s", e)
            finally:
                _put_db_conn(conn)

    α, β, γ, δ = _dynamic_weights(n_interactions)

    c_scores = _content_scores(query_book_id, query_text)
    p_scores = _profile_content_scores(user_id)
    if p_scores:
        if c_scores:
            c_scores = {
                bid: (0.70 * float(c_scores.get(bid, 0.0)) + 0.30 * float(p_scores.get(bid, 0.0)))
                for bid in catalog_ids
            }
        else:
            c_scores = p_scores
    b_scores = _collab_scores(user_id)
    s_scores = _social_scores(followed_ids, catalog_ids)
    t_scores = _trending_scores(catalog_ids)

    max_rating = df["avg_rating"].max() if len(df) > 0 else 5.0
    if max_rating == 0:
        max_rating = 5.0
    max_rc = df["rating_count"].max() if len(df) > 0 else 1.0
    if max_rc == 0:
        max_rc = 1.0

    results = []
    for _, row in df.iterrows():
        bid = row["book_id"]
        if bid in exclude_ids:
            continue
        lang = str(row.get("language", "")).strip()
        if language_filter and lang and lang != language_filter:
            continue
        if genre_filter:
            genres_str = str(row.get("genres", "")).lower()
            if genre_filter.lower() not in genres_str:
                continue

        cs = c_scores.get(bid, 0.0)
        bs = b_scores.get(bid, 0.0)
        ss = s_scores.get(bid, 0.0)
        ts = t_scores.get(bid, 0.0)

        rating_boost = float(row.get("avg_rating", 0)) / max_rating * 0.05
        rc_boost = min(float(row.get("rating_count", 0)) / max_rc, 1.0) * 0.03

        final = α*cs + β*bs + γ*ss + δ*ts + rating_boost + rc_boost

        results.append({
            "book_id": bid,
            "title": row["title"],
            "authors": row["authors"],
            "genres": row["genres"],
            "description": str(row.get("description", ""))[:200],
            "year": row.get("year", ""),
            "language": lang,
            "avg_rating": float(row.get("avg_rating", 0)),
            "rating_count": int(row.get("rating_count", 0)),
            "cover_url": str(row.get("cover_url", "")),
            "content_score": round(cs, 4),
            "collab_score": round(bs, 4),
            "social_score": round(ss, 4),
            "trending_score":round(ts, 4),
            "final_score": round(final, 4),
            "n_interactions": n_interactions,
            "weights": {"alpha": α, "beta": β, "gamma": γ, "delta": δ},
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:n]


def _catalog_book_payload(book_id: str) -> Optional[dict]:
    df = state["catalog"]
    if df is None or len(df) == 0:
        return None
    found = df[df["book_id"].astype(str) == str(book_id)]
    if found.empty:
        return None
    row = found.iloc[0]
    return {
        "book_id": str(row.get("book_id", "")),
        "title": str(row.get("title", "")),
        "authors": str(row.get("authors", "")),
        "genres": str(row.get("genres", "")),
        "description": str(row.get("description", ""))[:200],
        "year": row.get("year", ""),
        "language": str(row.get("language", "")),
        "avg_rating": float(row.get("avg_rating", 0) or 0),
        "rating_count": int(row.get("rating_count", 0) or 0),
        "cover_url": str(row.get("cover_url", "")),
        "content_score": 1.0,
        "collab_score": 0.0,
        "social_score": 0.0,
        "trending_score": 0.0,
        "final_score": 1.0,
        "n_interactions": 0,
        "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
    }


def build_justification(rows: list, parsed: dict) -> list:
    out = []
    for row in rows:
        cs = max(0.0, min(1.0, float(row.get("content_score", 0.0) or 0.0)))
        bs = max(0.0, min(1.0, float(row.get("collab_score", 0.0) or 0.0)))
        ss = max(0.0, min(1.0, float(row.get("social_score", 0.0) or 0.0)))
        ts = max(0.0, min(1.0, float(row.get("trending_score", 0.0) or 0.0)))
        final_score = max(0.0, min(1.0, float(row.get("final_score", 0.0) or 0.0)))
        n_interactions = int(row.get("n_interactions", 0) or 0)

        if n_interactions < 5:
            phase = "❄️ Cold"
        elif n_interactions < 20:
            phase = "🌡️ Mid"
        else:
            phase = "🔥 Warm"

        dominant_signal = max(
            {"content": cs, "collab": bs, "social": ss, "trending": ts},
            key=lambda k: {"content": cs, "collab": bs, "social": ss, "trending": ts}[k],
        )

        if cs >= 0.3:
            reason_primary = f"Cocok dengan topik yang kamu cari ({row['genres'].split(',')[0].strip()})"
        elif bs >= 0.3:
            reason_primary = "Disukai pembaca dengan selera mirip kamu"
        elif ss >= 0.3:
            reason_primary = "Sedang dibaca oleh orang yang kamu ikuti"
        elif ts >= 0.3:
            reason_primary = "Sedang trending di Pustara minggu ini"
        elif row["avg_rating"] >= 4.5:
            reason_primary = "Sangat direkomendasikan oleh pembaca Pustara"
        else:
            reason_primary = "Pilihan kurator Pustara untuk kamu"

        parts = []
        if row["year"]:
            parts.append(f"📅 {row['year']}")
        lang = "🇮🇩 Indonesia" if row.get("language") == "id" else "🌐 English"
        parts.append(lang)
        reason_secondary = " · ".join(parts)

        signals = [
            {"label": f"Kemiripan konten: {cs:.0%}", "value": cs},
            {"label": f"Collaborative: {bs:.0%}", "value": bs},
            {"label": f"Sosial: {ss:.0%}", "value": ss},
            {"label": f"Trending: {ts:.0%}", "value": ts},
        ]

        signal_map = {
            "content": {"score": cs, "weight": float((row.get("weights") or {}).get("alpha", 0.0) or 0.0), "label": f"Kemiripan konten: {cs:.0%}"},
            "collab": {"score": bs, "weight": float((row.get("weights") or {}).get("beta",  0.0) or 0.0), "label": f"Skor komunitas: {bs:.0%}"},
            "social": {"score": ss, "weight": float((row.get("weights") or {}).get("gamma", 0.0) or 0.0), "label": f"Sinyal sosial: {ss:.0%}"},
            "trending": {"score": ts, "weight": float((row.get("weights") or {}).get("delta", 0.0) or 0.0), "label": f"Sinyal trending: {ts:.0%}"},
        }

        out.append({
            **row,
            "reason_primary": reason_primary,
            "reason_secondary": reason_secondary,
            "dominant_signal": dominant_signal,
            "hybrid_score": round(final_score, 4),
            "phase": phase,
            "signals_map": signal_map,
            "signals": signals,
        })
    return out


def _safe_float(value, default=0.0) -> float:
    try:
        n = float(value)
        if np.isfinite(n):
            return n
    except Exception:
        pass
    return float(default)


def _json_safe(value):
    """Convert numpy/pandas containers and scalars into JSON-serializable Python values."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, float) and not np.isfinite(value):
        return 0.0

    return value


def _normalize_phase(phase_raw: object, n_interactions: int) -> str:
    token = str(phase_raw or "").lower()
    if "warm" in token or "hot" in token or "🔥" in token:
        return "🔥 Warm"
    if "mid" in token or "🌡" in token:
        return "🌡️ Mid"
    if "cold" in token or "❄" in token:
        return "❄️ Cold"
    if n_interactions < 5:
        return "❄️ Cold"
    if n_interactions < 20:
        return "🌡️ Mid"
    return "🔥 Warm"


def _subkey_float(obj: object, names: list, default=0.0) -> float:
    if not isinstance(obj, dict):
        return float(default)
    for name in names:
        if name in obj:
            return _safe_float(obj.get(name), default)
    return float(default)


def _sanitize_recommendation_item(item: dict) -> dict:
    item = _json_safe(item)
    signal_map_raw = item.get("signals_map") if isinstance(item.get("signals_map"), dict) else {}
    weights = item.get("weights") if isinstance(item.get("weights"), dict) else {}

    content_raw = signal_map_raw.get("content") if isinstance(signal_map_raw.get("content"),  dict) else {}
    collab_raw = signal_map_raw.get("collab") if isinstance(signal_map_raw.get("collab"),   dict) else {}
    social_raw = signal_map_raw.get("social") if isinstance(signal_map_raw.get("social"),   dict) else {}
    trending_raw = signal_map_raw.get("trending") if isinstance(signal_map_raw.get("trending"), dict) else {}

    cs = max(0.0, min(1.0, _safe_float(item.get("content_score", _subkey_float(content_raw,  ["score"], 0.0)))))
    bs = max(0.0, min(1.0, _safe_float(item.get("collab_score", _subkey_float(collab_raw,   ["score"], 0.0)))))
    ss = max(0.0, min(1.0, _safe_float(item.get("social_score", _subkey_float(social_raw,   ["score"], 0.0)))))
    ts = max(0.0, min(1.0, _safe_float(item.get("trending_score", _subkey_float(trending_raw, ["score"], 0.0)))))

    alpha = max(0.0, min(1.0, _safe_float(weights.get("alpha", _subkey_float(content_raw,  ["weight"], 0.0)))))
    beta = max(0.0, min(1.0, _safe_float(weights.get("beta", _subkey_float(collab_raw,   ["weight"], 0.0)))))
    gamma = max(0.0, min(1.0, _safe_float(weights.get("gamma", _subkey_float(social_raw,   ["weight"], 0.0)))))
    delta = max(0.0, min(1.0, _safe_float(weights.get("delta", _subkey_float(trending_raw, ["weight"], 0.0)))))

    dominant_signal = max(
        {"content": cs, "collab": bs, "social": ss, "trending": ts},
        key=lambda k: {"content": cs, "collab": bs, "social": ss, "trending": ts}[k],
    )

    n_interactions = int(item.get("n_interactions", 0) or 0)
    phase = _normalize_phase(item.get("phase"), n_interactions)

    clean_signals_map = {
        "content": {"score": cs, "weight": alpha, "label": f"Kemiripan konten: {cs:.0%}"},
        "collab": {"score": bs, "weight": beta, "label": f"Skor komunitas: {bs:.0%}"},
        "social": {"score": ss, "weight": gamma, "label": f"Sinyal sosial: {ss:.0%}"},
        "trending": {"score": ts, "weight": delta, "label": f"Sinyal trending: {ts:.0%}"},
    }

    clean_signals = [
        {"label": f"Kemiripan konten: {cs:.0%}", "value": cs},
        {"label": f"Collaborative: {bs:.0%}", "value": bs},
        {"label": f"Sosial: {ss:.0%}", "value": ss},
        {"label": f"Trending: {ts:.0%}", "value": ts},
    ]

    return _json_safe({
        **item,
        "content_score": round(cs, 4),
        "collab_score": round(bs, 4),
        "social_score": round(ss, 4),
        "trending_score": round(ts, 4),
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "dominant_signal": item.get("dominant_signal") or dominant_signal,
        "phase": phase,
        "signals_map": clean_signals_map,
        "signals": clean_signals,
    })


def _sanitize_recommendations_payload(items: list) -> list:
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(_sanitize_recommendation_item(item))
    return out


def _enforce_missing_catalog_notice(
    reply: str,
    missing_catalog_title: Optional[str],
    recommended_books: list,
    admin_email: str,
) -> str:
    if not missing_catalog_title:
        return reply

    text = str(reply or "").strip()
    lowered = text.lower()

    explicit_unavailable = "tapi sayangnya buku itu belum tersedia di pustara"
    has_unavailable_notice = (
        explicit_unavailable in lowered
        or "belum tersedia di pustara" in lowered
    )

    lines = []
    if recommended_books:
        alt = str((recommended_books[0] or {}).get("title") or "")
        if alt:
            has_mirip_phrase = "mirip banget" in lowered
            if not has_mirip_phrase:
                if has_unavailable_notice:
                    lines.append(
                        f'Buku "{alt}" ini tuh mirip banget sama "{missing_catalog_title}".'
                    )
                else:
                    lines.append(
                        f'Buku "{alt}" ini tuh mirip banget sama "{missing_catalog_title}", tapi sayangnya buku itu belum tersedia di Pustara.'
                    )

    if not has_unavailable_notice and not lines:
        lines.append(
            f'Tapi sayangnya buku "{missing_catalog_title}" belum tersedia di Pustara.'
        )

    has_request_admin = ("request ke admin" in lowered) or (admin_email.lower() in lowered)
    if not has_request_admin:
        lines.append(f'Kalau kamu mau, request ke admin di {admin_email} ya.')

    if not lines:
        return text

    if text:
        return " ".join(lines + [text])
    return " ".join(lines)


def _strip_false_unavailable_notice(reply: str, missing_catalog_title: Optional[str]) -> str:
    """If no missing catalog title is detected, remove hallucinated unavailable claims."""
    text = str(reply or "").strip()
    if not text or missing_catalog_title:
        return text

    lowered = text.lower()
    if "belum tersedia di pustara" not in lowered:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if "belum tersedia di pustara" not in s.lower()]
    cleaned = " ".join(s.strip() for s in kept if s and s.strip())
    if cleaned:
        return cleaned

    return "Kalau kamu mau, aku bisa kasih alternatif lain yang tersedia di katalog Pustara."


def _strip_false_admin_request(
    reply: str,
    missing_catalog_title: Optional[str],
    query: str,
    admin_email: str,
) -> str:
    """Remove stray request-admin lines on follow-up chat when no title is missing."""
    text = str(reply or "").strip()
    if not text or missing_catalog_title:
        return text

    q = str(query or "").lower()
    follow_up_markers = [
        "udah baca", "sudah baca", "udh baca", "udah tamat", "sudah tamat",
        "udah pernah baca", "sudah pernah baca", "yang lain", "lainnya", "next", "selanjutnya",
    ]
    if not any(marker in q for marker in follow_up_markers):
        return text

    lowered = text.lower()
    if ("request ke admin" not in lowered) and (admin_email.lower() not in lowered):
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    filtered = []
    for s in sentences:
        ls = s.lower()
        if ("request ke admin" in ls) or (admin_email.lower() in ls):
            continue
        filtered.append(s)

    cleaned = " ".join(s.strip() for s in filtered if s and s.strip())
    if cleaned:
        return cleaned
    return "Siap, kalau itu sudah kamu baca, aku bisa cariin alternatif lain yang vibes-nya mirip dari katalog Pustara."


def _enforce_attached_context_guard(reply: str, admin_email: str) -> str:
    """For attached-book mode, remove any forbidden unavailable/admin guidance from model output."""
    text = str(reply or "").strip()
    if not text:
        return text

    blocked_markers = [
        "belum tersedia di pustara",
        "tidak tersedia di pustara",
        "tidak ada di katalog",
        "nggak ada di katalog",
        "request ke admin",
        "hubungi admin",
        admin_email.lower(),
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in blocked_markers):
            continue
        kept.append(sentence)

    cleaned = " ".join(s.strip() for s in kept if s and s.strip())
    if cleaned:
        return cleaned
    return "Siap! Buku ini tersedia di katalog Pustara. Aku langsung bantu sesuai permintaanmu."


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    context: Optional[dict] = None
    n: int = 8
    attached_book_title: Optional[str] = None
    attached_book_desc: Optional[str] = None
    user_gender: Optional[str] = None
    user_age: Optional[str] = None
    user_age_group: Optional[str] = None
    preferred_genres: Optional[list] = None
    chat_history: Optional[list] = None

class DirectRequest(BaseModel):
    book_id: str
    n: int = 8
    user_id: Optional[str] = None
    user_gender: Optional[str] = None
    user_age_group: Optional[str] = None
    preferred_genres: Optional[list] = None

class ActivityRequest(BaseModel):
    user_id: str
    book_id: str
    action: str

class ReindexRequest(BaseModel):
    secret: str # Wajib ada — tidak bisa kosong


def generate_ai_reply(
    query: str,
    recommended_books: list,
    attached_title: str = None,
    attached_desc: str = None,
    user_gender: str = None,
    user_age: str = None,
    chat_history: list = None,
    missing_catalog_title: Optional[str] = None,
) -> tuple[str, bool]:
    """Return (reply_text, groq_was_used)."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        log.warning("GROQ_API_KEY not set — using fallback reply.")
        if attached_title:
            return f"Wah, tentang {attached_title} ya? Coba cek detail yang PustarAI temukan di bawah ini!", False
        return f"Ini nih beberapa rekomendasi buku yang pas buat pencarian '{query}', selamat membaca!", False

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}

    demografi = f"User kamu adalah seorang {user_gender} berumur {user_age}." if user_gender and user_age else ""
    admin_email = ADMIN_CONTACT_EMAIL

    base_persona = f"""Kamu adalah PustarAI, asisten perpustakaan digital Pustara yang gaul, asik, helpful, dan such a storyteller. {demografi}
Jawab dengan bahasa Indonesia santai seperti teman yang asik. Maksimal 4-5 kalimat. Jangan kaku kayak robot.

SCOPE WAJIB: Kamu HANYA membantu hal yang berkaitan dengan buku — rekomendasi, sinopsis, genre, penulis, cerita, tema, ulasan, dan sejenisnya.
Kalau user nanya hal di luar buku (coding, matematika, trivia umum, tebak angka, dll), TOLAK dengan ramah: "Aku lebih fokus bantu soal buku dan bacaan nih! Mau cari rekomendasi atau bahas buku tertentu?" — jangan dijawab walau kamu tahu jawabannya.

EMPATI DULU: Kalau user curhat atau ungkapkan perasaan (capek, sedih, stress, bete, insomnia, dll), acknowledge dulu dengan hangat (1 kalimat), BARU tawarkan buku yang relevan. Jangan langsung loncat ke rekomendasi.

IDIOM & EKSPRESI INDONESIA: Pahami idiom dengan benar. "Mati suri" = sangat capek/vakum/ga produktif, BUKAN query buku bertema kematian. "Mau nangis" = frustrasi/overwhelmed. Tangkap maksud sebenarnya, jangan literal.

SERI & LANJUTAN: Kalau buku yang dibahas atau direkomendasikan adalah bagian dari seri, WAJIB sebut urutan dan judulnya. Contoh: "Ini buku pertama dari trilogi Poppy War — lanjutannya The Dragon Republic, lalu The Burning God." Jangan biarkan user tidak tahu ada kelanjutannya.

ANTI-HALLUSINASI: JANGAN membuat daftar panjang. Kalau menyebut judul seri/lanjutan, maksimal 3-4 judul saja. Kalau tidak yakin judul lanjutannya, cukup bilang "ada beberapa lanjutannya" tanpa mengarang judul. Lebih baik jujur tidak tahu daripada mengulang atau mengarang."""

    if attached_title:
        # CASE 1: User lagi nanya soal buku tertentu yang sudah ada di card (diklik).
        # → Sinopsis/info boleh dari pengetahuan Groq sendiri — open knowledge.
        # → Tapi kalau user minta rekomendasi buku lain, arahkan ke daftar Pustara.
        db_hint = f"\n\nInfo dari database Pustara tentang buku ini:\n{str(attached_desc)[:400]}" if attached_desc and len(str(attached_desc).strip()) > 20 else ""
        system_msg = f"""{base_persona}

Buku yang sedang dibahas: "{attached_title}"{db_hint}

ATURAN MUTLAK (SYSTEM OVERRIDE):
1. KESADARAN KONTEKS: User menyertakan konteks buku spesifik dari sistem. Buku ini 100% tersedia di katalog Pustara.
2. DILARANG MENOLAK KONTEKS: DILARANG mengatakan buku ini tidak tersedia / belum tersedia / tidak ada di katalog.
3. DILARANG MENYURUH HUBUNGI ADMIN: DILARANG menyuruh user request atau hubungi admin untuk buku konteks ini.
4. EKSEKUSI LANGSUNG: Segera penuhi permintaan user berdasarkan buku konteks ini (misalnya ringkasan, tema, karakter, dll).
5. CEGAH PENGULANGAN: Jika user bilang "sudah baca", jangan rekomendasikan judul yang sama lagi di sesi ini.

ATURAN:
1. Kamu boleh menjawab pertanyaan tentang buku ini (sinopsis, tokoh, tema, dll) menggunakan pengetahuanmu sendiri — ini diperbolehkan.
2. Kalau info dari database tersedia di atas, prioritaskan itu. Tapi kalau tidak ada atau kurang, pakai pengetahuanmu — jangan ngarang kalau memang tidak tahu.
3. Kalau user minta rekomendasi buku LAIN, jawab: "Untuk rekomendasi, coba ketik di kolom chat ya!" — jangan sebut judul buku lain di luar konteks ini.
4. Tetap asik dan storytelling mode ON kalau diminta cerita sinopsis."""

    elif recommended_books:
        # CASE 2: Ada daftar buku dari DB — mode rekomendasi.
        # → Rekomendasi WAJIB dari daftar ini saja.
        # → Kalau tidak ada yang cocok, boleh bilang jujur + sarankan request ke admin.
        top_titles = []
        for b in recommended_books[:5]:
            clue = str(b.get("description", "") or "").strip()
            clue_short = clue.split(".")[0].strip() if clue else ""
            clue_short = (clue_short[:120] + "...") if len(clue_short) > 120 else clue_short
            if clue_short:
                top_titles.append(
                    f"- \"{b.get('title')}\" karya {b.get('authors')} (genre: {b.get('genres', '-')}) | clue isi: {clue_short}"
                )
            else:
                top_titles.append(
                    f"- \"{b.get('title')}\" karya {b.get('authors')} (genre: {b.get('genres', '-')})"
                )
        daftar = "\n".join(top_titles)
        missing_title_context = ""
        if missing_catalog_title:
            missing_title_context = (
                f"\n\nKONTEKS WAJIB:\n"
                f"User menanyakan buku \"{missing_catalog_title}\" dan buku itu TIDAK ADA di katalog Pustara."
            )
        system_msg = f"""{base_persona}
{missing_title_context}

ATURAN REKOMENDASI — WAJIB DIIKUTI:
1. Kamu HANYA boleh merekomendasikan buku dari [DAFTAR BUKU PUSTARA] di bawah. Jangan sebut buku lain.
2. Kalau user cari buku yang tidak ada di daftar:
   - Akui jujur dengan kalimat ini: "Tapi sayangnya buku itu belum tersedia di Pustara."
    - Kalau ada yang mirip di daftar, sebut dengan gaya: "Buku ini tuh mirip banget sama [judul user]".
   - Sarankan request ke admin: {admin_email}
3. Kalau tidak ada yang relevan sama sekali, bilang jujur dan minta user coba kata kunci lain.
4. Untuk pertanyaan sinopsis/info tentang buku yang ADA di daftar, kamu boleh jawab dari pengetahuanmu sendiri.
5. Gaya bahasa harus natural dan tidak kaku. Setelah memberi rekomendasi, selalu beri clue singkat isi buku (1 kalimat) agar user kebayang bukunya.
6. Jangan cuma sebut judul. Untuk setiap buku, jelaskan singkat kenapa relevan + clue isi buku.
7. Jangan pernah bilang "belum tersedia di Pustara" kecuali KONTEKS WAJIB di atas menyatakan buku user memang tidak ada di katalog.

[DAFTAR BUKU PUSTARA — hanya ini yang boleh direkomendasikan]:
{daftar}"""

    else:
        # CASE 3: Tidak ada buku relevan sama sekali di DB.
        # → Jujur tidak ada, sarankan request ke admin.
        system_msg = f"""{base_persona}

Saat ini tidak ada buku yang cocok di katalog Pustara untuk permintaan ini.

ATURAN:
1. Bilang jujur bahwa buku yang dicari belum tersedia di Pustara.
2. Sarankan user untuk request buku ke admin via email: {admin_email}
3. Boleh menyebut judul buku yang dimaksud user (untuk konfirmasi), tapi JANGAN merekomendasikan buku lain yang tidak ada di Pustara.
4. Tetap ramah dan bantu user dengan kata kunci pencarian alternatif kalau memungkinkan."""

    messages = [{"role": "system", "content": system_msg}]
    if chat_history:
        for h in chat_history[-6:]:
            role = h.get("role")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.4,
        "stream": False,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if reply:
                if attached_title:
                    reply = _enforce_attached_context_guard(reply, admin_email)
                else:
                    reply = _strip_false_unavailable_notice(reply, missing_catalog_title)
                    reply = _strip_false_admin_request(reply, missing_catalog_title, query, admin_email)
                    reply = _enforce_missing_catalog_notice(reply, missing_catalog_title, recommended_books, admin_email)
                return reply, True
        else:
            log.warning("Groq API error %s: %s", response.status_code, response.text[:200])
    except Exception as e:
        log.warning("Groq exception: %s", e)

    if attached_title:
        return f"Wah, tentang {attached_title} ya? Coba cek detail yang PustarAI temukan di bawah ini!", False
    fallback_reply = f"Ini nih beberapa rekomendasi buku yang pas buat pencarian '{query}', selamat membaca!"
    if attached_title:
        fallback_reply = _enforce_attached_context_guard(fallback_reply, admin_email)
    else:
        fallback_reply = _enforce_missing_catalog_notice(fallback_reply, missing_catalog_title, recommended_books, admin_email)
    return fallback_reply, False


def _is_recommendation_request(message: str, parsed: dict) -> bool:
    if parsed["intent"] in ("genre", "author", "title"):
        return True
    q = message.lower()
    REC_KEYWORDS = [
        "rekomendasi", "rekomendasiin", "rekomen", "saran", "saranin",
        "suggest", "suggest me", "recommend",
        "buku apa", "buku yang", "mau baca", "pingin baca", "pengen baca",
        "cari buku", "cariin buku", "mirip", "seperti", "kayak buku",
        "selanjutnya", "lanjutan", "next book",
    ]
    if any(kw in q for kw in REC_KEYWORDS):
        return True
    QUESTION_KEYWORDS = [
        "apa itu", "apa sih", "ceritanya", "sinopsis", "ringkasan",
        "tentang apa", "siapa pengarang", "siapa penulis",
        "kapan terbit", "berapa halaman", "genre apa",
        "what is", "who wrote", "summary", "synopsis",
        "how many", "when was",
    ]
    if any(kw in q for kw in QUESTION_KEYWORDS):
        return False
    return True


@app.get("/health")
def health(request: Request):
    # Kalau HEALTH_SECRET di-set, wajib kirim header X-Health-Key
    if HEALTH_SECRET:
        client_key = request.headers.get("X-Health-Key", "")
        if client_key != HEALTH_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "status": "ok",
        "model_a": state["model_a"] is not None,
        "model_b": state["model_b"] is not None,
        "model_c": state["model_c"] is not None,
        "catalog_size": len(state["catalog"]) if state["catalog"] is not None else 0,
        "redis": state["redis"] is not None,
        "last_reindex": state["last_reindex"],
    }

@app.get("/search/semantic")
async def search_semantic(
    q: str,
    n: int = 10,
    language: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    Semantic search via TF-IDF — untuk search bar.
    Query bisa berupa judul, genre, mood ('buku sedih', 'novel seru petualangan'), dll.
    Jauh lebih cepat dari /recommendations/chat karena tidak pakai LLM.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' tidak boleh kosong.")

    # Rate limit — per user_id atau anonymous, 30 req/menit
    r = state["redis"]
    if r is not None:
        try:
            key_subject = user_id or "anon"
            rk = f"ratelimit:search:{key_subject}"
            max_req = int(os.getenv("SEARCH_RATE_LIMIT_MAX", "30"))
            window_s = int(os.getenv("SEARCH_RATE_LIMIT_WINDOW", "60"))
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            pipe = r.pipeline()
            pipe.zremrangebyscore(rk, 0, now_ms - window_s * 1000)
            pipe.zadd(rk, {str(now_ms): now_ms})
            pipe.zcard(rk)
            pipe.expire(rk, window_s + 5)
            count = pipe.execute()[2]
            if count > max_req:
                raise HTTPException(status_code=429, detail="Terlalu banyak request. Coba lagi sebentar.")
        except HTTPException:
            raise
        except Exception:
            pass # Redis error → fail open

    ma = state["model_a"]
    df = state["catalog"]
    if ma is None or df is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    try:
        from sklearn.metrics.pairwise import linear_kernel

        vec = ma["vectorizer"].transform([q.lower().strip()])
        sims = linear_kernel(vec, ma["tfidf_matrix"]).flatten()

        mx = float(sims.max())
        if mx > 0:
            sims = sims / mx

        book_ids = ma["book_ids"]
        scored = sorted(zip(book_ids, sims), key=lambda x: x[1], reverse=True)

        results = []
        for bid, score in scored:
            if score < 0.01:
                break
            rows = df[df["book_id"].astype(str) == str(bid)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            lang = str(row.get("language", "")).strip()
            if language and lang and lang != language:
                continue
            results.append(_json_safe({
                "book_id": str(row.get("book_id", "")),
                "title": str(row.get("title", "")),
                "authors": str(row.get("authors", "")),
                "genres": str(row.get("genres", "")),
                "description": str(row.get("description", ""))[:200],
                "year": row.get("year", ""),
                "language": lang,
                "avg_rating": float(row.get("avg_rating", 0) or 0),
                "rating_count": int(row.get("rating_count", 0) or 0),
                "cover_url": str(row.get("cover_url", "")),
                "search_score": round(float(score), 4),
            }))
            if len(results) >= n:
                break

        return _json_safe({"ok": True, "query": q, "results": results, "n": len(results)})

    except Exception as e:
        log.error("search_semantic error: %s", e)
        raise HTTPException(status_code=500, detail="Search gagal.")


@app.get("/recommendations/similar-users")
async def recommendations_similar_users(
    user_id: str,
    n: int = 8,
):
    """
    Rekomendasi berdasarkan 'pengguna yang mirip kamu'.
    Cari user dengan taste paling mirip via cosine similarity
    di user-item space, lalu ambil buku yang mereka suka tapi
    belum dibaca user ini.
    """
    mb = state["model_b"]
    df = state["catalog"]
    if mb is None or df is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    resolved = _resolve_user_uuid(user_id)
    user_ids = mb.get("user_ids", [])
    book_ids = mb.get("book_ids", [])
    ui_matrix = mb.get("user_item_matrix")

    if resolved not in user_ids or ui_matrix is None:
        return _json_safe({
            "ok": True, "recommendations": [], "similar_users": 0,
            "source": "similar-users", "reason": "user tidak ditemukan di model",
        })

    u_idx = user_ids.index(resolved)
    u_vec = np.asarray(ui_matrix[u_idx]).flatten()
    u_norm = float(np.linalg.norm(u_vec))

    if u_norm == 0:
        return _json_safe({
            "ok": True, "recommendations": [], "similar_users": 0,
            "source": "similar-users", "reason": "user belum punya interaksi",
        })

    # Cosine similarity ke semua user lain
    all_vecs = np.asarray(ui_matrix)
    norms = np.linalg.norm(all_vecs, axis=1)
    norms[norms == 0] = 1e-9
    dots = all_vecs @ u_vec
    sims = dots / (norms * u_norm)
    sims[u_idx] = -1 # exclude diri sendiri

    top_users = np.argsort(sims)[::-1][:10]

    # Buku yang sudah diinteraksi user ini
    interacted = {book_ids[i] for i, v in enumerate(u_vec) if v > 0}

    # Agregasi skor dari buku yang disukai similar users
    book_scores: dict = {}
    for uid_idx in top_users:
        if sims[uid_idx] <= 0.05:
            continue
        sim_weight = float(sims[uid_idx])
        sim_vec = np.asarray(all_vecs[uid_idx]).flatten()
        for b_idx, score in enumerate(sim_vec):
            if score <= 0:
                continue
            bid = book_ids[b_idx]
            if bid in interacted:
                continue
            book_scores[bid] = book_scores.get(bid, 0.0) + float(score) * sim_weight

    if not book_scores:
        return _json_safe({
            "ok": True, "recommendations": [], "similar_users": int((sims[top_users] > 0.05).sum()),
            "source": "similar-users", "reason": "tidak ada buku baru dari similar users",
        })

    mx_score = max(book_scores.values())
    book_scores = {bid: v / mx_score for bid, v in book_scores.items()}

    catalog_map = {str(r["book_id"]): r for _, r in df.iterrows()}
    results = []
    for bid, score in sorted(book_scores.items(), key=lambda x: x[1], reverse=True):
        row = catalog_map.get(str(bid))
        if row is None:
            continue
        results.append(_json_safe({
            "book_id": str(row.get("book_id", "")),
            "title": str(row.get("title", "")),
            "authors": str(row.get("authors", "")),
            "genres": str(row.get("genres", "")),
            "description": str(row.get("description", ""))[:200],
            "year": row.get("year", ""),
            "language": str(row.get("language", "")),
            "avg_rating": float(row.get("avg_rating", 0) or 0),
            "rating_count": int(row.get("rating_count", 0) or 0),
            "cover_url": str(row.get("cover_url", "")),
            "similar_user_score": round(score, 4),
            "reason_primary": "Disukai pembaca dengan selera mirip kamu",
            "reason_secondary": "",
        }))
        if len(results) >= n:
            break

    return _json_safe({
        "ok": True,
        "recommendations": results,
        "similar_users": int((sims[top_users] > 0.05).sum()),
        "source": "similar-users",
    })


@app.get("/recommendations/cold-start")
async def recommendations_cold_start(
    user_id: Optional[str] = None,
    genres: Optional[str] = None,
    n: int = 10,
    top_n: int = 10,
    gender: Optional[str] = None,
    age_group: Optional[str] = None,
):
    """Cold-start recommendations — untuk user baru yang belum punya history."""
    resolved_user_id = _resolve_user_uuid(user_id)
    df = state["catalog"]
    catalog_ids = df["book_id"].tolist() if df is not None else []

    genre_list = [g.strip() for g in genres.split(",")] if genres else []

    # Demographic scores dari Model C
    demo_scores = _demographic_scores(
        gender=gender,
        age_group=age_group,
        preferred_genres=genre_list,
        catalog_ids=catalog_ids,
    )

    # Content scores dari genre yang dipilih user
    query_text = " ".join(genre_list) if genre_list else None
    c_scores = _content_scores(None, query_text) if query_text else {}

    real_n = max(n, top_n)
    recs = hybrid_recommend(
        user_id=resolved_user_id,
        query_book_id=None,
        query_text=query_text,
        n=real_n,
        exclude_ids=[],
        genre_filter=genre_list[0] if len(genre_list) == 1 else None,
    )

    if not recs:
        recs = _fallback_popular(n=real_n)

    # Boost dengan demographic score
    if demo_scores:
        for r in recs:
            boost = demo_scores.get(r["book_id"], 0.5)
            r["final_score"] = round(r["final_score"] * (0.80 + 0.20 * boost), 4)
        recs.sort(key=lambda x: x["final_score"], reverse=True)

    parsed = {"intent": "cold-start", "genre": genre_list[0] if genre_list else None,
              "author": None, "title": None, "language": None}
    recs_out = _sanitize_recommendations_payload(build_justification(recs[:real_n], parsed))
    return _json_safe({"ok": True, "recommendations": recs_out, "source": "cold-start"})


@app.get("/recommendations/trending")
async def recommendations_trending(
    n: int = 6,
    top_n: int = 6,
    gender: Optional[str] = None,
    age_group: Optional[str] = None,
):
    """Trending recommendations — buku yang lagi populer minggu ini."""
    df = state["catalog"]
    if df is None or len(df) == 0:
        return _json_safe({"ok": True, "recommendations": [], "source": "trending"})

    real_n = max(n, top_n)

    r = state["redis"]
    cache_key = f"trending:global:{real_n}"
    if r is not None:
        try:
            cached = r.get(cache_key)
            if cached:
                return _json_safe(json.loads(cached))
        except Exception:
            pass

    catalog_ids = df["book_id"].tolist()
    t_scores = _trending_scores(catalog_ids)

    # Kalau Redis tidak ada / trending kosong, fallback ke popular
    if not t_scores:
        recs = _fallback_popular(n=real_n)
        parsed = {"intent": "trending", "genre": None, "author": None,
                  "title": None, "language": None}
        recs_out = _sanitize_recommendations_payload(build_justification(recs, parsed))
        return _json_safe({"ok": True, "recommendations": recs_out, "source": "popular_fallback"})

    # Demographic boost opsional
    demo_scores = _demographic_scores(gender, age_group, None, catalog_ids) if (gender or age_group) else {}

    results = []
    max_rating = float(df["avg_rating"].max() or 5.0)
    for _, row in df.iterrows():
        bid = row["book_id"]
        ts = t_scores.get(bid, 0.0)
        if ts <= 0:
            continue
        rating_boost = float(row.get("avg_rating", 0)) / max_rating * 0.05
        demo_boost = demo_scores.get(bid, 0.5) * 0.10 if demo_scores else 0.0
        final = ts + rating_boost + demo_boost
        results.append({
            "book_id": bid,
            "title": row["title"],
            "authors": row["authors"],
            "genres": row["genres"],
            "description": str(row.get("description", ""))[:200],
            "year": row.get("year", ""),
            "language": str(row.get("language", "")),
            "avg_rating": float(row.get("avg_rating", 0)),
            "rating_count": int(row.get("rating_count", 0)),
            "cover_url": str(row.get("cover_url", "")),
            "content_score": 0.0,
            "collab_score": 0.0,
            "social_score": 0.0,
            "trending_score": round(ts, 4),
            "final_score": round(final, 4),
            "n_interactions": 0,
            "weights": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 1.0},
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    parsed = {"intent": "trending", "genre": None, "author": None,
              "title": None, "language": None}
    recs_out = _sanitize_recommendations_payload(
        build_justification(results[:real_n], parsed)
    )
    result_payload = _json_safe({"ok": True, "recommendations": recs_out, "source": "trending"})

    # Tulis ke Redis cache 6 jam
    if r is not None:
        try:
            r.setex(cache_key, 6 * 3600, json.dumps(result_payload))
        except Exception:
            pass

    return result_payload

@app.post("/recommendations/direct")
async def recommendations_direct(req: DirectRequest):
    resolved_user_id = _resolve_user_uuid(req.user_id)
    resolved_book_id = _resolve_book_uuid(req.book_id)

    recs = hybrid_recommend(
        user_id=resolved_user_id,
        query_book_id=resolved_book_id if _is_uuid_like(resolved_book_id) else None,
        query_text=None if _is_uuid_like(resolved_book_id) else str(req.book_id),
        n=req.n * 3,
        exclude_ids=[resolved_book_id] if resolved_book_id else [],
    )

    if not recs:
        recs = _fallback_popular(n=req.n, exclude_ids=[resolved_book_id] if resolved_book_id else [])

    # Demographic rerank
    if req.user_gender or req.user_age_group:
        df = state["catalog"]
        catalog_ids = df["book_id"].tolist() if df is not None else []
        demo_scores = _demographic_scores(
            req.user_gender, req.user_age_group, req.preferred_genres, catalog_ids
        )
        if demo_scores:
            for r in recs:
                boost = demo_scores.get(r["book_id"], 0.5)
                r["final_score"] = round(r["final_score"] * (0.85 + 0.15 * boost), 4)
            recs.sort(key=lambda x: x["final_score"], reverse=True)

    parsed = {"intent": "direct", "genre": None, "author": None, "title": None, "language": None}
    recs_out = _sanitize_recommendations_payload(build_justification(recs[:req.n], parsed))
    return _json_safe({"ok": True, "recommendations": recs_out, "book_id": resolved_book_id})


@app.post("/recommendations/chat")
async def recommendations_chat(req: ChatRequest, request: Request):
    """
    Main endpoint — dipanggil FE chat AI.
    Includes: rate limiting, chat analytics logging.
    """
    client_ip = request.client.host if request.client else None
    if not _check_chat_rate_limit(req.user_id, client_ip):
        raise HTTPException(
            status_code=429,
            detail="Terlalu banyak request. Coba lagi sebentar ya! 😅",
        )

    parsed = parse_query(req.message)
    user_id = req.user_id
    user_id_db = _resolve_user_uuid(user_id)
    ctx = req.context or {}
    n = req.n
    attached_book_title = (req.attached_book_title or "").strip()
    missing_catalog_title = None if attached_book_title else _detect_missing_catalog_title(req.message)

    if parsed["intent"] == "smalltalk":
        ai_text, groq_used = generate_ai_reply(
            query=req.message,
            recommended_books=[],
            user_gender=req.user_gender,
            user_age=req.user_age,
            chat_history=req.chat_history,
        )
        _log_chat_analytics(user_id, req.message, parsed, False, 0, groq_used)
        return {
            "response_text": ai_text,
            "intent": "smalltalk",
            "recommendations": [],
            "show_recommendations": False,
        }

    search_keywords = [req.message]
    if parsed["genre"]: search_keywords.append(parsed["genre"])
    if parsed["author"]: search_keywords.append(parsed["author"])
    if parsed["title"]: search_keywords.append(parsed["title"])
    query_text = " ".join(set(search_keywords))

    last_book_id_raw = ctx.get("last_book_id") or ctx.get("book_id")
    last_book_id = _resolve_book_uuid(last_book_id_raw)
    recent_books_raw = ctx.get("recent_books", [])
    recent_books = [_resolve_book_uuid(b) for b in recent_books_raw[:5] if b]

    parsed_title_book_id = None
    if parsed.get("title"):
        candidate = _resolve_book_uuid(parsed.get("title"))
        if _is_uuid_like(candidate):
            parsed_title_book_id = candidate

    exclude_ids = list({last_book_id, *recent_books} - {None})
    if parsed_title_book_id:
        exclude_ids = [bid for bid in exclude_ids if bid != parsed_title_book_id]

    seed_book_id = parsed_title_book_id or (last_book_id if _is_uuid_like(last_book_id) else None)

    recs = hybrid_recommend(
        user_id=user_id_db,
        query_book_id=seed_book_id,
        query_text=query_text,
        n=n,
        exclude_ids=exclude_ids,
        language_filter=parsed.get("language"),
        genre_filter=parsed.get("genre"),
    )

    if not recs:
        recs = _fallback_popular(n=n, exclude_ids=exclude_ids)
        intent = "fallback"
    else:
        intent = parsed["intent"]

    # Untuk intent title/genre, buang kandidat dengan skor terlalu jauh dari top result.
    if intent in ("title", "genre") and recs:
        max_score = max(float(r.get("final_score", 0.0) or 0.0) for r in recs)
        threshold = max_score * 0.25 # minimal 25% dari skor tertinggi
        recs_filtered = [
            r for r in recs
            if float(r.get("final_score", 0.0) or 0.0) >= threshold
        ]
        # Kalau filter terlalu agresif (< 3 hasil), tetap pakai semua.
        recs = recs_filtered if len(recs_filtered) >= 3 else recs

    # Paksa buku yang disebut masuk ke kartu teratas
    if parsed_title_book_id:
        existing = next((r for r in recs if str(r.get("book_id")) == parsed_title_book_id), None)
        if existing:
            existing["content_score"] = max(float(existing.get("content_score", 0.0) or 0.0), 1.0)
            existing["final_score"] = max(float(existing.get("final_score",   0.0) or 0.0), 1.0)
            seed_item = existing
        else:
            seed_item = _catalog_book_payload(parsed_title_book_id)

        if seed_item:
            recs = [seed_item] + [r for r in recs if str(r.get("book_id")) != parsed_title_book_id]
            recs = recs[:n]

    recs_with_reason = _sanitize_recommendations_payload(build_justification(recs, parsed))

    show_recs = _is_recommendation_request(req.message, parsed)
    if recs and recs[0]["final_score"] > 0.15:
        show_recs = True

        ai_text, groq_used = generate_ai_reply(
        query=req.message,
        recommended_books=recs_with_reason if show_recs else [],
        attached_title=req.attached_book_title,
        attached_desc=req.attached_book_desc,
        user_gender=req.user_gender,
        user_age=req.user_age,
        chat_history=req.chat_history,
        missing_catalog_title=missing_catalog_title,
    )

        _log_chat_analytics(
        user_id=user_id,
        query=req.message,
        parsed=parsed,
        show_recs=show_recs,
        n_results=len(recs_with_reason),
        groq_used=groq_used,
    )

        r = state["redis"]
    if r and user_id:
        try:
            r.setex(f"rec:cache:{user_id}", CACHE_TTL, json.dumps(recs_with_reason))
            if show_recs and recs_with_reason and recs_with_reason[0]["final_score"] > 0.1:
                top_book_id = _resolve_book_uuid(recs_with_reason[0]["book_id"])
                stream_user_id = user_id_db or user_id
                ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                r.xadd(
                    "activity:stream",
                    {"user_id": stream_user_id, "book_id": top_book_id,
                     "action": "search_intent", "ts": str(ts_ms)},
                    maxlen=100_000, approximate=True,
                )
                r.zincrby("trending:books:7d", 0.5, top_book_id)
        except Exception as e:
            log.warning("Redis chat signal error: %s", e)

    return _json_safe({
        "response_text": ai_text,
        "intent": intent,
        "recommendations": recs_with_reason if show_recs else [],
        "show_recommendations": show_recs,
        "parsed_query": parsed,
    })


@app.get("/analytics/chat/summary")
async def chat_analytics_summary(
    request: Request,
    user_id: Optional[str] = None,
    days: int = 30,
):
    """
    Ringkasan analytics chat.
    - Tanpa user_id: summary global (butuh HEALTH_SECRET header)
    - Dengan user_id: summary untuk satu user (butuh HEALTH_SECRET header juga)
    
    Returns:
      - top_intents: distribusi intent (genre/title/author/general/fallback)
      - top_genres: genre paling sering ditanya
      - top_titles: judul buku paling sering disebut
      - top_authors: penulis paling sering disebut
      - daily_volume: jumlah chat per hari
      - groq_usage_rate: persentase request yang sampai ke Groq
    """
    # Endpoint ini butuh autentikasi (HEALTH_SECRET)
    if HEALTH_SECRET:
        client_key = request.headers.get("X-Health-Key", "")
        if client_key != HEALTH_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    conn = _get_db_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="DB tidak tersedia")

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        base_where = "ts >= %s"
        base_params: list = [cutoff]

        if user_id:
            base_where += " AND user_id = %s"
            base_params.append(user_id)

        # Total queries
        cur.execute(f"SELECT COUNT(*) AS total FROM chat_analytics WHERE {base_where}", base_params)
        total = int((cur.fetchone() or {}).get("total", 0))

        # Top intents
        cur.execute(
            f"SELECT intent, COUNT(*) AS cnt FROM chat_analytics "
            f"WHERE {base_where} AND intent IS NOT NULL "
            f"GROUP BY intent ORDER BY cnt DESC LIMIT 10",
            base_params,
        )
        top_intents = [{"intent": r["intent"], "count": int(r["cnt"])} for r in cur.fetchall()]

        # Top genres
        cur.execute(
            f"SELECT detected_genre, COUNT(*) AS cnt FROM chat_analytics "
            f"WHERE {base_where} AND detected_genre IS NOT NULL "
            f"GROUP BY detected_genre ORDER BY cnt DESC LIMIT 10",
            base_params,
        )
        top_genres = [{"genre": r["detected_genre"], "count": int(r["cnt"])} for r in cur.fetchall()]

        # Top titles
        cur.execute(
            f"SELECT detected_title, COUNT(*) AS cnt FROM chat_analytics "
            f"WHERE {base_where} AND detected_title IS NOT NULL "
            f"GROUP BY detected_title ORDER BY cnt DESC LIMIT 10",
            base_params,
        )
        top_titles = [{"title": r["detected_title"], "count": int(r["cnt"])} for r in cur.fetchall()]

        # Top authors
        cur.execute(
            f"SELECT detected_author, COUNT(*) AS cnt FROM chat_analytics "
            f"WHERE {base_where} AND detected_author IS NOT NULL "
            f"GROUP BY detected_author ORDER BY cnt DESC LIMIT 10",
            base_params,
        )
        top_authors = [{"author": r["detected_author"], "count": int(r["cnt"])} for r in cur.fetchall()]

        # Daily volume (last N days)
        cur.execute(
            f"SELECT DATE(ts) AS day, COUNT(*) AS cnt FROM chat_analytics "
            f"WHERE {base_where} GROUP BY DATE(ts) ORDER BY day DESC LIMIT {days}",
            base_params,
        )
        daily_volume = [{"day": str(r["day"]), "count": int(r["cnt"])} for r in cur.fetchall()]

        # Groq usage rate
        cur.execute(
            f"SELECT COUNT(*) AS groq_cnt FROM chat_analytics "
            f"WHERE {base_where} AND groq_used = TRUE",
            base_params,
        )
        groq_cnt = int((cur.fetchone() or {}).get("groq_cnt", 0))
        groq_rate = round(groq_cnt / total, 4) if total > 0 else 0.0

        return _json_safe({
            "ok": True,
            "period_days": days,
            "user_id": user_id,
            "total_queries": total,
            "top_intents": top_intents,
            "top_genres": top_genres,
            "top_titles": top_titles,
            "top_authors": top_authors,
            "daily_volume": daily_volume,
            "groq_usage_rate": groq_rate,
        })

    except Exception as e:
        log.warning("Analytics query error: %s", e)
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")
    finally:
        _put_db_conn(conn)


@app.post("/activity")
async def push_activity(req: ActivityRequest):
    valid_actions = set(ACTION_WEIGHTS.keys())
    if req.action not in valid_actions:
        raise HTTPException(400, f"Invalid action. Valid: {sorted(valid_actions)}")

    weight = ACTION_WEIGHTS[req.action]
    user_id_db = _resolve_user_uuid(req.user_id)
    book_id_db = _resolve_book_uuid(req.book_id)
    stream_user_id = user_id_db or req.user_id
    stream_book_id = book_id_db or req.book_id
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    r = state["redis"]
    if r:
        try:
            pipe = r.pipeline()
            pipe.xadd(
                "activity:stream",
                {"user_id": stream_user_id, "book_id": stream_book_id,
                 "action": req.action, "ts": str(ts_ms)},
                maxlen=100_000, approximate=True,
            )
            pipe.zincrby("trending:books:7d", weight, stream_book_id)
            pipe.expire("trending:books:7d", TRENDING_TTL_S)
            pipe.incr(f"{USER_INTERACTION_COUNT_PREFIX}{stream_user_id}")
            pipe.execute()
            r.delete(f"rec:cache:{req.user_id}")
        except Exception as e:
            log.warning("Redis push_activity error: %s", e)

    return {"ok": True, "action": req.action, "weight": weight}


@app.post("/reindex")
async def reindex(req: ReindexRequest):
    """
    Rebuild model A (content) dari katalog Neon / CSV.
    Dipanggil oleh cron job.
    
    POST /reindex
    Body: { "secret": "UR_SECRET_YEAH" }
    
    SEC: Secret di body, bukan query param — tidak bocor ke server log.
    SEC: Async lock — tidak bisa jalan paralel.
    """
    EXPECTED_KEY = os.getenv("RI_SECRET", "pustara-default-secret")
    if req.secret != EXPECTED_KEY:
        log.warning("🚫 Unauthorized reindex attempt")
        raise HTTPException(status_code=403, detail="Gak punya kunci gak boleh masuk!")

    # Lock — cegah parallel rebuild
    if _reindex_lock.locked():
        raise HTTPException(status_code=409, detail="Reindex sedang berjalan. Tunggu sebentar!")

    async with _reindex_lock:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        import scipy.sparse as sp

        log.info("🔄 Reindex started …")

        df = _load_catalog()
        if len(df) == 0:
            raise HTTPException(503, "Empty catalog — cannot reindex")

        soups = (
            df["genres"].str.replace(",", " ") * 3 + " " +
            df["authors"] * 2 + " " +
            df["title"] + " " +
            df["description"].fillna("")
        )
        soups = soups.str.lower()

        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, sublinear_tf=True, min_df=1)
        tfidf_matrix = vect.fit_transform(soups)
        sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

        model_a = {
            "vectorizer": vect,
            "matrix": sim_matrix,
            "tfidf_matrix": tfidf_matrix,
            "book_ids": df["book_id"].tolist(),
        }

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_A_PATH, "wb") as f:
            pickle.dump(model_a, f)
        state["model_a"] = model_a

        conn = _get_db_conn()
        model_b_status = "skipped (no DB)"
        if conn:
            try:
                ubs = _read_sql_compat(
                    "SELECT user_id::text, book_id::text, score FROM user_book_scores WHERE score > 0",
                    conn,
                )
                if len(ubs) > 5:
                    from scipy.sparse import csr_matrix
                    from sklearn.metrics.pairwise import cosine_similarity
                    u_ids = ubs["user_id"].unique().tolist()
                    b_ids = ubs["book_id"].unique().tolist()
                    u_map = {u: i for i, u in enumerate(u_ids)}
                    b_map = {b: i for i, b in enumerate(b_ids)}
                    rows_ = ubs["user_id"].map(u_map).tolist()
                    cols_ = ubs["book_id"].map(b_map).tolist()
                    data_ = ubs["score"].tolist()
                    ui_matrix = csr_matrix((data_, (rows_, cols_)), shape=(len(u_ids), len(b_ids)))
                    bb_sim = cosine_similarity(ui_matrix.T)
                    model_b = {
                        "matrix": bb_sim,
                        "book_ids": b_ids,
                        "user_ids": u_ids,
                        "user_item_matrix": ui_matrix.toarray(),
                    }
                    with open(MODEL_B_PATH, "wb") as f:
                        pickle.dump(model_b, f)
                    state["model_b"] = model_b
                    model_b_status = f"rebuilt ({len(b_ids)} books, {len(u_ids)} users)"
                else:
                    model_b_status = "skipped (too few interactions)"
            except Exception as e:
                model_b_status = f"error: {e}"
            finally:
                _put_db_conn(conn)

        model_c_status = "skipped"
        try:
            import importlib.util
            rebuild_path = Path("rebuild_models.py")
            if rebuild_path.exists():
                spec = importlib.util.spec_from_file_location("rebuild_models", rebuild_path)
                rm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rm)
                model_c = rm.build_model_c(df)
            else:
                model_c = _rebuild_model_c_inline(df)

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_C_PATH, "wb") as f:
                pickle.dump(model_c, f)
            state["model_c"] = model_c
            model_c_status = f"rebuilt ({model_c['n_books']} books)"
            log.info("✅ Model C rebuilt: %s", model_c_status)
        except Exception as e:
            model_c_status = f"error: {e}"
            log.warning("Model C rebuild failed: %s", e)

        state["last_reindex"] = datetime.now(timezone.utc).isoformat()
        log.info("✅ Reindex complete. Model A: %d books. Model B: %s. Model C: %s",
                 len(df), model_b_status, model_c_status)

        return {
            "ok": True,
            "catalog_size": len(df),
            "model_a_books": len(df),
            "model_b": model_b_status,
            "model_c": model_c_status,
            "timestamp": state["last_reindex"],
        }


@app.post("/reload-catalog")
async def reload_catalog(req: ReindexRequest):
    """
    Refresh catalog dari backend/DB tanpa rebuild ML model.
    Lebih cepat dari /reindex — cocok dipanggil setelah nambah buku baru.
    POST /reload-catalog — Body: { "secret": "UR_SECRET" }
    """
    EXPECTED_KEY = os.getenv("RI_SECRET", "pustara-default-secret")
    if req.secret != EXPECTED_KEY:
        raise HTTPException(status_code=403, detail="Gak punya kunci gak boleh masuk!")

    df = _load_catalog()
    if len(df) == 0:
        raise HTTPException(status_code=503, detail="Catalog kosong setelah reload.")

    # Sync book_ids di model_a supaya content scoring tetap akurat
    ma = state["model_a"]
    if ma is not None:
        existing_ids = set(ma["book_ids"])
        new_ids = set(df["book_id"].tolist())
        added = new_ids - existing_ids
        if added:
            log.info("🆕 %d buku baru ditemukan — model_a belum include ini, jalankan /reindex untuk full rebuild.", len(added))

    log.info("✅ Catalog reloaded: %d books", len(df))
    return _json_safe({
        "ok": True,
        "catalog_size": len(df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Catalog di-refresh. Jalankan /reindex untuk rebuild ML model jika ada buku baru.",
    })


def _rebuild_model_c_inline(catalog_df) -> dict:
    from datetime import datetime as _dt

    GENRE_GENDER = {
        "Romance":(0.20,0.90,0.50),"Coming-of-age":(0.40,0.75,0.55),
        "Drama":(0.45,0.70,0.55),"Keluarga":(0.45,0.75,0.55),
        "Feminisme":(0.20,0.95,0.50),"Diary":(0.25,0.85,0.50),
        "Puisi":(0.35,0.75,0.50),"Persahabatan":(0.45,0.70,0.55),
        "Memoir":(0.40,0.70,0.55),"Biografi":(0.60,0.55,0.55),
        "Inspiratif":(0.60,0.65,0.65),"Self-help":(0.55,0.70,0.65),
        "Produktivitas":(0.65,0.55,0.60),"Keuangan":(0.70,0.45,0.55),
        "Sains":(0.70,0.50,0.55),"Filsafat":(0.75,0.45,0.55),
        "Politik":(0.75,0.45,0.55),"Thriller":(0.70,0.55,0.60),
        "Misteri":(0.60,0.65,0.60),"Kriminal":(0.70,0.50,0.55),
        "Petualangan":(0.70,0.55,0.60),"Fiksi Ilmiah":(0.72,0.52,0.58),
        "Distopia":(0.60,0.60,0.60),"Fantasi":(0.60,0.60,0.60),
        "Humor":(0.65,0.60,0.62),"Humor Gelap":(0.70,0.50,0.55),
        "Komedi":(0.60,0.60,0.58),"Sastra":(0.55,0.60,0.58),
        "Fiksi":(0.50,0.60,0.55),"Non-fiksi":(0.60,0.50,0.55),
        "Sejarah":(0.65,0.50,0.55),"Fiksi Sejarah":(0.60,0.55,0.55),
        "Kolonialisme":(0.58,0.52,0.55),"Realisme Magis":(0.50,0.62,0.55),
        "Religi":(0.55,0.65,0.60),"Spiritualitas":(0.50,0.70,0.58),
        "Psikologi":(0.55,0.65,0.60),"Sosial":(0.55,0.60,0.58),
        "Alam":(0.65,0.55,0.58),"Esai":(0.60,0.52,0.55),
        "Cerpen":(0.50,0.60,0.55),"Diaspora":(0.50,0.60,0.55),
        "Perang":(0.75,0.40,0.52),"Tradisi":(0.50,0.60,0.55),
        "Pendidikan":(0.55,0.60,0.57),"Fabel":(0.50,0.55,0.52),
        "Fiksi Anak":(0.50,0.55,0.52),"Klasik":(0.55,0.55,0.55),
        "Realisme":(0.55,0.55,0.55),
    }
    GENRE_AGE = {
        "Romance":(0.80,0.85,0.50,0.35),"Coming-of-age":(0.90,0.75,0.40,0.30),
        "Diary":(0.85,0.70,0.40,0.30),"Puisi":(0.60,0.70,0.55,0.50),
        "Persahabatan":(0.80,0.70,0.50,0.40),"Fiksi Anak":(0.85,0.45,0.35,0.30),
        "Fantasi":(0.80,0.75,0.55,0.40),"Distopia":(0.75,0.80,0.55,0.40),
        "Fiksi Ilmiah":(0.65,0.80,0.65,0.55),"Thriller":(0.55,0.80,0.70,0.60),
        "Misteri":(0.55,0.75,0.70,0.65),"Kriminal":(0.45,0.70,0.70,0.65),
        "Self-help":(0.50,0.80,0.75,0.65),"Produktivitas":(0.40,0.80,0.80,0.65),
        "Keuangan":(0.30,0.70,0.85,0.80),"Biografi":(0.40,0.65,0.75,0.80),
        "Memoir":(0.40,0.60,0.70,0.80),"Filsafat":(0.45,0.75,0.70,0.65),
        "Politik":(0.40,0.70,0.75,0.70),"Sejarah":(0.45,0.65,0.70,0.75),
        "Fiksi Sejarah":(0.50,0.65,0.70,0.70),"Sastra":(0.50,0.65,0.65,0.70),
        "Non-fiksi":(0.45,0.65,0.70,0.70),"Sains":(0.55,0.70,0.65,0.60),
        "Religi":(0.50,0.60,0.65,0.75),"Spiritualitas":(0.45,0.60,0.65,0.75),
        "Inspiratif":(0.55,0.70,0.65,0.60),"Drama":(0.60,0.70,0.60,0.55),
        "Humor":(0.65,0.75,0.65,0.55),"Humor Gelap":(0.50,0.75,0.65,0.50),
        "Petualangan":(0.75,0.70,0.55,0.45),"Psikologi":(0.50,0.75,0.70,0.60),
        "Sosial":(0.50,0.70,0.65,0.60),"Feminisme":(0.55,0.80,0.65,0.50),
        "Komedi":(0.65,0.70,0.60,0.50),"Fiksi":(0.65,0.70,0.60,0.55),
        "Keluarga":(0.50,0.65,0.70,0.70),"Klasik":(0.45,0.60,0.65,0.70),
        "Realisme":(0.45,0.60,0.65,0.65),"Realisme Magis":(0.50,0.65,0.60,0.55),
        "Alam":(0.50,0.60,0.65,0.65),"Kolonialisme":(0.45,0.65,0.65,0.65),
        "Tradisi":(0.45,0.55,0.60,0.70),"Pendidikan":(0.55,0.65,0.65,0.60),
        "Esai":(0.40,0.60,0.65,0.65),"Cerpen":(0.55,0.60,0.55,0.55),
        "Diaspora":(0.45,0.65,0.60,0.55),"Perang":(0.50,0.65,0.65,0.65),
        "Fabel":(0.70,0.50,0.40,0.40),
    }
    DG = (0.55, 0.55, 0.55)
    DA = (0.55, 0.60, 0.60, 0.55)

    book_demographic = {}
    for _, row in catalog_df.iterrows():
        book_id = str(row["book_id"])
        genres = [g.strip() for g in str(row.get("genres", "")).split(",") if g.strip()] or ["Fiksi"]
        gs = {"L": 0.0, "P": 0.0, "X": 0.0}
        ags = {"<20": 0.0, "21-30": 0.0, "31-40": 0.0, ">40": 0.0}
        for genre in genres:
            gw = GENRE_GENDER.get(genre, DG)
            aw = GENRE_AGE.get(genre, DA)
            gs["L"] += gw[0]; gs["P"] += gw[1]; gs["X"] += gw[2]
            ags["<20"] += aw[0]; ags["21-30"] += aw[1]
            ags["31-40"] += aw[2]; ags[">40"] += aw[3]
        n = len(genres)
        book_demographic[book_id] = {
            "L": round(gs["L"]/n,4), "P": round(gs["P"]/n,4), "X": round(gs["X"]/n,4),
            "<20": round(ags["<20"]/n,4), "21-30": round(ags["21-30"]/n,4),
            "31-40": round(ags["31-40"]/n,4), ">40": round(ags[">40"]/n,4),
        }

    return {
        "genre_gender_weights": GENRE_GENDER,
        "genre_age_weights": GENRE_AGE,
        "book_demographic": book_demographic,
        "n_books": len(catalog_df),
        "built_at": _dt.now().isoformat(),
        "version": "1.0",
    }