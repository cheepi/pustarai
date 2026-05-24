"""Rebuild Pustara model artifacts (Model A/B/C).

Run: python rebuild_models.py
Requires: DATABASE_URL in environment.
"""
import os, sys, json, pickle, logging, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

Path("logs").mkdir(exist_ok=True)
Path("pustara_models").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/rebuild.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("rebuild")


MOOD_VOCAB = {
    "Romance":        "sedih cinta romantis perasaan hati",
    "Drama":          "sedih haru emosional perasaan menyentuh",
    "Coming-of-age":  "tumbuh dewasa perjalanan remaja muda",
    "Thriller":       "tegang menegangkan misteri gelap seru",
    "Misteri":        "penasaran teka-teki misteri gelap detective",
    "Kriminal":       "gelap kejahatan berbahaya misteri tegang",
    "Humor":          "lucu seru santai ringan menghibur kocak",
    "Komedi":         "lucu seru santai ringan menghibur kocak",
    "Petualangan":    "seru aksi petualangan mengasyikkan semangat",
    "Fantasi":        "ajaib magical petualangan seru imajinasi dunia lain",
    "Fiksi Ilmiah":   "futuristik teknologi luar angkasa ilmiah seru",
    "Distopia":       "gelap futuristik kritis sosial berat",
    "Inspiratif":     "motivasi semangat menginspirasi positif kuat",
    "Self-help":      "produktif berkembang motivasi diri positif",
    "Biografi":       "nyata kisah hidup perjalanan inspirasi perjuangan",
    "Sejarah":        "masa lalu fakta sejarah peristiwa berat",
    "Filsafat":       "berat mendalam pemikiran renungan filosofis",
    "Psikologi":      "pikiran perasaan mental dalam refleksi berat",
    "Spiritualitas":  "damai tenang jiwa rohani renungan",
    "Religi":         "iman damai rohani spiritual tenang",
    "Sains":          "ilmiah fakta pengetahuan teknologi informatif",
    "Sastra":         "indah bahasa puitis mendalam klasik",
    "Puisi":          "indah puitis liris bahasa perasaan",
    "Keuangan":       "uang investasi finansial praktis bisnis",
    "Produktivitas":  "efisien kerja karir pengembangan praktis",
    "Keluarga":       "hangat keluarga haru bersamaan kasih sayang",
    "Persahabatan":   "hangat pertemanan bersama kebersamaan haru",
    "Perang":         "berat gelap perjuangan konflik serius",
    "Feminisme":      "perempuan kesetaraan hak sosial kritis",
    "Sosial":         "masyarakat kritis sosial refleksi",
    "Realisme Magis": "ajaib nyata indah puitis magis",
    "Fiksi Sejarah":  "masa lalu sejarah fiksi petualangan",
    "Klasik":         "abadi timeless mendalam sastra berat",
}


def build_soup_v2(row) -> str:
    """Construct a text 'soup' used by the TF-IDF vectorizer."""
    genres_raw = str(row.get("genres", "")).replace(",", " ")
    authors_raw = str(row.get("authors", "")).replace(",", " ")
    title_raw = str(row.get("title", ""))
    desc_raw = str(row.get("description", ""))
    lang = str(row.get("language", "id"))

    genres_list = [g.strip() for g in str(row.get("genres", "")).split(",") if g.strip()]
    mood_words = " ".join(MOOD_VOCAB.get(g, "") for g in genres_list)

    soup = (
        f"{genres_raw} {genres_raw} {genres_raw} {genres_raw} "
        f"{title_raw} {title_raw} "
        f"{authors_raw} {authors_raw} "
        f"{desc_raw} {desc_raw} "
        f"{mood_words} {mood_words} "
        f"{lang}"
    )
    return re.sub(r"\s+", " ", soup).strip().lower()


# MODEL C: DEMOGRAPHIC PROFILER — full 49-genre catalog priors
GENRE_GENDER = {
    "Romance":        (0.18, 0.92, 0.48), "Coming-of-age": (0.38, 0.78, 0.52),
    "Drama":          (0.42, 0.72, 0.52), "Keluarga":      (0.42, 0.78, 0.52),
    "Feminisme":      (0.18, 0.96, 0.48), "Diary":         (0.22, 0.88, 0.48),
    "Puisi":          (0.32, 0.78, 0.48), "Persahabatan":  (0.42, 0.72, 0.52),
    "Memoir":         (0.38, 0.72, 0.52), "Biografi":      (0.58, 0.58, 0.52),
    "Inspiratif":     (0.58, 0.68, 0.62), "Self-help":     (0.52, 0.72, 0.62),
    "Produktivitas":  (0.62, 0.58, 0.58), "Keuangan":      (0.72, 0.42, 0.52),
    "Sains":          (0.72, 0.48, 0.52), "Filsafat":      (0.78, 0.42, 0.52),
    "Politik":        (0.78, 0.42, 0.52), "Thriller":      (0.68, 0.58, 0.58),
    "Misteri":        (0.58, 0.68, 0.58), "Kriminal":      (0.68, 0.48, 0.52),
    "Petualangan":    (0.68, 0.52, 0.58), "Fiksi Ilmiah":  (0.74, 0.48, 0.56),
    "Distopia":       (0.58, 0.62, 0.58), "Fantasi":       (0.56, 0.64, 0.58),
    "Humor":          (0.62, 0.62, 0.60), "Humor Gelap":   (0.68, 0.52, 0.52),
    "Komedi":         (0.58, 0.62, 0.56), "Sastra":        (0.52, 0.62, 0.56),
    "Fiksi":          (0.48, 0.62, 0.52), "Non-fiksi":     (0.58, 0.52, 0.52),
    "Sejarah":        (0.62, 0.52, 0.52), "Fiksi Sejarah": (0.58, 0.56, 0.52),
    "Kolonialisme":   (0.56, 0.54, 0.52), "Realisme Magis":(0.48, 0.64, 0.52),
    "Religi":         (0.52, 0.68, 0.58), "Spiritualitas": (0.48, 0.72, 0.56),
    "Psikologi":      (0.52, 0.68, 0.58), "Sosial":        (0.52, 0.62, 0.56),
    "Alam":           (0.62, 0.58, 0.56), "Esai":          (0.58, 0.54, 0.52),
    "Cerpen":         (0.48, 0.62, 0.52), "Diaspora":      (0.48, 0.62, 0.52),
    "Perang":         (0.78, 0.38, 0.50), "Tradisi":       (0.48, 0.62, 0.52),
    "Pendidikan":     (0.52, 0.62, 0.54), "Fabel":         (0.50, 0.56, 0.50),
    "Fiksi Anak":     (0.50, 0.56, 0.50), "Klasik":        (0.52, 0.58, 0.52),
    "Realisme":       (0.52, 0.58, 0.52),
}
GENRE_AGE = {
    "Romance":        (0.82, 0.88, 0.48, 0.32), "Coming-of-age": (0.92, 0.78, 0.38, 0.28),
    "Diary":          (0.88, 0.72, 0.38, 0.28), "Puisi":         (0.58, 0.72, 0.54, 0.48),
    "Persahabatan":   (0.82, 0.72, 0.48, 0.38), "Fiksi Anak":    (0.88, 0.42, 0.32, 0.28),
    "Fantasi":        (0.82, 0.78, 0.52, 0.38), "Distopia":      (0.78, 0.82, 0.52, 0.38),
    "Fiksi Ilmiah":   (0.62, 0.82, 0.62, 0.52), "Thriller":      (0.52, 0.82, 0.72, 0.58),
    "Misteri":        (0.52, 0.78, 0.72, 0.62), "Kriminal":      (0.42, 0.72, 0.72, 0.62),
    "Self-help":      (0.48, 0.82, 0.78, 0.62), "Produktivitas": (0.38, 0.82, 0.82, 0.62),
    "Keuangan":       (0.28, 0.72, 0.88, 0.82), "Biografi":      (0.38, 0.68, 0.78, 0.82),
    "Memoir":         (0.38, 0.62, 0.72, 0.82), "Filsafat":      (0.42, 0.78, 0.72, 0.62),
    "Politik":        (0.38, 0.72, 0.78, 0.72), "Sejarah":       (0.42, 0.68, 0.72, 0.78),
    "Fiksi Sejarah":  (0.48, 0.68, 0.72, 0.68), "Sastra":        (0.48, 0.68, 0.68, 0.72),
    "Non-fiksi":      (0.42, 0.68, 0.72, 0.72), "Sains":         (0.52, 0.72, 0.68, 0.58),
    "Religi":         (0.48, 0.62, 0.68, 0.78), "Spiritualitas": (0.42, 0.62, 0.68, 0.78),
    "Inspiratif":     (0.52, 0.72, 0.68, 0.58), "Drama":         (0.58, 0.72, 0.58, 0.52),
    "Humor":          (0.62, 0.78, 0.62, 0.52), "Humor Gelap":   (0.48, 0.78, 0.62, 0.48),
    "Petualangan":    (0.78, 0.72, 0.52, 0.42), "Psikologi":     (0.48, 0.78, 0.72, 0.58),
    "Sosial":         (0.48, 0.72, 0.68, 0.58), "Feminisme":     (0.52, 0.82, 0.68, 0.48),
    "Komedi":         (0.62, 0.72, 0.58, 0.48), "Fiksi":         (0.62, 0.72, 0.58, 0.52),
    "Keluarga":       (0.48, 0.68, 0.72, 0.72), "Klasik":        (0.42, 0.62, 0.68, 0.72),
    "Realisme":       (0.42, 0.62, 0.68, 0.68), "Realisme Magis":(0.48, 0.68, 0.62, 0.52),
    "Alam":           (0.48, 0.62, 0.68, 0.68), "Kolonialisme":  (0.42, 0.68, 0.68, 0.68),
    "Tradisi":        (0.42, 0.58, 0.62, 0.72), "Pendidikan":    (0.52, 0.68, 0.68, 0.58),
    "Esai":           (0.38, 0.62, 0.68, 0.68), "Cerpen":        (0.52, 0.62, 0.54, 0.52),
    "Diaspora":       (0.42, 0.68, 0.62, 0.52), "Perang":        (0.48, 0.68, 0.68, 0.68),
    "Fabel":          (0.72, 0.52, 0.38, 0.38),
}
_DEFAULT_GENDER = (0.54, 0.56, 0.54)
_DEFAULT_AGE    = (0.54, 0.62, 0.62, 0.54)


def build_model_c(df_books) -> dict:
    """Build demographic profiler from catalog priors."""
    import numpy as np
    log.info("Building Model C (demographic)")

    book_demographic = {}
    for _, row in df_books.iterrows():
        book_id = str(row["book_id"])
        genres = [g.strip() for g in str(row.get("genres", "")).split(",") if g.strip()]
        if not genres:
            genres = ["Fiksi"]

        gs = np.zeros(3)
        ag = np.zeros(4)
        for genre in genres:
            gs += np.array(GENRE_GENDER.get(genre, _DEFAULT_GENDER))
            ag += np.array(GENRE_AGE.get(genre, _DEFAULT_AGE))
        n = len(genres)
        gs = np.round(gs / n, 4)
        ag = np.round(ag / n, 4)
        book_demographic[book_id] = {
            "L": float(gs[0]), "P": float(gs[1]), "X": float(gs[2]),
            "<20": float(ag[0]), "21-30": float(ag[1]),
            "31-40": float(ag[2]), ">40": float(ag[3]),
        }

    model_c = {
        "genre_gender_weights": GENRE_GENDER,
        "genre_age_weights": GENRE_AGE,
        "book_demographic": book_demographic,
        "n_books": len(df_books),
        "built_at": datetime.now().isoformat(),
        "version": "v5.0-catalog-priors",
        "training_stats": {"source": "catalog_priors_indonesia", "n_books": len(df_books)},
    }
    log.info(f"Model C: {len(book_demographic)} books profiled")
    return model_c


def run():
    start = datetime.now()
    log.info("🔁 Starting model rebuild v5.0...")

    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix
    import psycopg2

    DB_URL = os.environ.get("DATABASE_URL")
    if not DB_URL:
        log.error("DATABASE_URL not set! Aborting.")
        sys.exit(1)

    # 1. Fetch books dari DB 
    log.info("📚 Fetching books from Neon...")
    conn = psycopg2.connect(DB_URL)
    df_books = pd.read_sql("""
        SELECT id::text as book_id,
               title,
               array_to_string(authors, ', ') as authors,
               array_to_string(genres, ', ') as genres,
               COALESCE(description, '') as description,
               COALESCE(avg_rating, 4.0) as avg_rating,
               COALESCE(language, 'id') as language,
               COALESCE(cover_url, '') as cover_url
        FROM books WHERE is_active = TRUE
    """, conn)
    log.info(f"   {len(df_books)} books loaded")

    # 2. Fetch tracking dari DB 
    log.info("📊 Fetching tracking data...")
    df_tracking = pd.read_sql("""
        SELECT user_id::text, book_id::text, score
        FROM user_book_scores WHERE score > 0
    """, conn)
    conn.close()
    log.info(f"   {len(df_tracking)} tracking records")

    # 3. Build soup v2 (dengan mood augmentation) 
    df_books["soup"] = df_books.apply(build_soup_v2, axis=1)

    # 4. Build Model A (Content TF-IDF v2) 
    log.info("🧠 Building Model A (TF-IDF v2 + mood vocab)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        max_features=10_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = tfidf.fit_transform(df_books["soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    title_index = {}
    bookid_index = {}
    for idx, row in df_books.iterrows():
        t = str(row["title"]).lower().strip()
        title_index.setdefault(t, []).append(idx)
        bookid_index[str(row["book_id"])] = idx

    book_ids = df_books["book_id"].astype(str).tolist()

    model_a = {
        "vectorizer": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "cosine_sim": cosine_sim,
        "matrix": cosine_sim, # alias untuk server
        "title_index": title_index,
        "bookid_index": bookid_index,
        "book_ids": book_ids,
        "dataframe": df_books.reset_index(drop=True),
        "n_books": len(df_books),
        "built_at": datetime.now().isoformat(),
        "version": "v5.0-mood-augmented",
    }
    with open("pustara_models/model_a_content.pkl", "wb") as f:
        pickle.dump(model_a, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Server load dari model_a_prod.pkl
    with open("pustara_models/model_a_prod.pkl", "wb") as f:
        pickle.dump(model_a, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("✅ Model A saved")

    # 5. Build Model B (Collaborative v2 — normalized)
    if len(df_tracking) > 0:
        log.info("🤝 Building Model B (Collaborative v2 — per-user normalized)...")
        valid_ids = set(df_books["book_id"].astype(str))
        df_t = df_tracking[df_tracking["book_id"].astype(str).isin(valid_ids)].copy()
        df_agg = df_t.groupby(["user_id", "book_id"])["score"].sum().reset_index()

        # normalize score per-user sebelum build matrix
        # Raw scores bisa 1-27+; tanpa normalisasi collab output bisa 56x lipat content
        def _norm_user(grp):
            mx = grp["score"].max()
            grp = grp.copy()
            grp["norm_score"] = grp["score"] / mx if mx > 0 else grp["score"]
            return grp

        df_agg = df_agg.groupby("user_id", group_keys=False).apply(
            _norm_user, include_groups=False
        ).reset_index()
        # Fallback jika include_groups tidak didukung pandas lama
        if "norm_score" not in df_agg.columns:
            df_agg = df_t.groupby(["user_id", "book_id"])["score"].sum().reset_index()
            df_agg["norm_score"] = df_agg.groupby("user_id")["score"].transform(
                lambda x: x / x.max() if x.max() > 0 else x
            )

        users = df_agg["user_id"].unique()
        books = df_agg["book_id"].unique()
        u2i = {u: i for i, u in enumerate(users)}
        b2i = {b: i for i, b in enumerate(books)}

        mat = csr_matrix(
            (df_agg["norm_score"],
             (df_agg["user_id"].map(u2i), df_agg["book_id"].map(b2i))),
            shape=(len(users), len(books))
        )
        item_sim = cosine_similarity(mat.T, dense_output=False)

        model_b = {
            "user_item": mat,
            "item_sim": item_sim,
            "user2idx": u2i,
            "book2idx": b2i,
            "idx2book": {i: b for b, i in b2i.items()},
            "idx2user": {i: u for u, i in u2i.items()},
            "n_users": len(users),
            "n_books": len(books),
            "built_at": datetime.now().isoformat(),
            "version": "v5.0-normalized",
            "normalization": "per_user_minmax",
        }
        with open("pustara_models/model_b_collaborative.pkl", "wb") as f:
            pickle.dump(model_b, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"✅ Model B saved: {len(users)} users × {len(books)} books")
    else:
        log.warning("No tracking data — Model B skipped")

    # 6. Build Model C (Demographic v2) 
    model_c = build_model_c(df_books)
    with open("pustara_models/model_c_demographic.pkl", "wb") as f:
        pickle.dump(model_c, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("✅ Model C saved")

    # 7. Update metadata 
    duration = (datetime.now() - start).total_seconds()
    meta = {
        "rebuilt_at": datetime.now().isoformat(),
        "version": "5.0",
        "duration_seconds": duration,
        "n_books": len(df_books),
        "n_tracking": len(df_tracking),
        "model_a_path": "pustara_models/model_a_content.pkl",
        "model_a_prod": "pustara_models/model_a_prod.pkl",
        "model_b_path": "pustara_models/model_b_collaborative.pkl",
        "model_c_path": "pustara_models/model_c_demographic.pkl",
    }
    with open("pustara_models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"✅ Rebuild selesai dalam {duration:.1f}s")


if __name__ == "__main__":
    run()
