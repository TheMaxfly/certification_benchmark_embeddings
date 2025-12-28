import os
import uuid
import json
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import psycopg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# ----------------------------
# Config
# ----------------------------
load_dotenv()
DB_DSN = os.environ.get(
    "APIMANGA_DSN",
    "dbname=apimanga user=postgres password=postgres host=localhost port=5432",
)

OUT_DIR = os.environ.get("BENCH_OUT_DIR", "./bench_out")
TOP_K = int(os.environ.get("BENCH_TOP_K", "8"))
CHUNKING_NAME = "char_1200_overlap_200"
EMBED_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
]


# ----------------------------
# Helpers
# ----------------------------
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def chunk_text_chars(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, chunk_text)."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [(0, len(text), text)]
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == len(text):
            break
        start += step
    return chunks


@dataclass
class RunInfo:
    run_id: str
    model_name: str
    dim: int
    chunking_id: int
    model_id: int


# ----------------------------
# DB ops
# ----------------------------
def fetch_one(conn, sql: str, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()


def fetch_all(conn, sql: str, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def ensure_out_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "faiss"), exist_ok=True)


def get_ids(conn) -> Tuple[int, int, int, int]:
    # chunking_id
    row = fetch_one(
        conn,
        "SELECT chunking_id, chunk_size, chunk_overlap FROM bench.chunking_strategies WHERE name=%s",
        (CHUNKING_NAME,),
    )
    if not row:
        raise RuntimeError(f"chunking strategy not found: {CHUNKING_NAME}")
    chunking_id, chunk_size, chunk_overlap = row

    # model ids
    model_rows = fetch_all(
        conn,
        "SELECT model_id, model_name, dim FROM bench.embedding_models WHERE model_name = ANY(%s)",
        (EMBED_MODELS,),
    )
    model_map = {name: (mid, dim) for (mid, name, dim) in model_rows}
    for m in EMBED_MODELS:
        if m not in model_map:
            raise RuntimeError(f"embedding model not found in bench.embedding_models: {m}")

    return chunking_id, chunk_size, chunk_overlap, model_map


def build_chunks_if_needed(conn, chunk_size: int, overlap: int, limit_docs: int | None = None):
    # Check if chunks exist
    n_chunks = fetch_one(conn, "SELECT COUNT(*) FROM bench.corpus_chunks")[0]
    if n_chunks > 0:
        print(f"[OK] bench.corpus_chunks already has {n_chunks} rows. Skipping chunk build.")
        return

    print("[RUN] Building chunks into bench.corpus_chunks ...")
    sql_docs = """
        SELECT doc_key, doc_text
        FROM bench.corpus_docs
        WHERE doc_text IS NOT NULL AND length(doc_text) >= 50
        ORDER BY doc_key
    """
    if limit_docs:
        sql_docs += f" LIMIT {int(limit_docs)}"

    docs = fetch_all(conn, sql_docs)
    print(f"[INFO] Docs to chunk: {len(docs)}")

    rows_to_insert = []
    for doc_key, doc_text in tqdm(docs, desc="Chunking"):
        chunks = chunk_text_chars(doc_text, chunk_size=chunk_size, overlap=overlap)
        for idx, (cs, ce, ctext) in enumerate(chunks):
            rows_to_insert.append(
                (doc_key, idx, ctext, cs, ce, None, sha1_text(ctext))
            )

        # batch insert to avoid huge memory
        if len(rows_to_insert) >= 5000:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO bench.corpus_chunks
                      (doc_key, chunk_index, chunk_text, char_start, char_end, token_count, chunk_hash)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (doc_key, chunk_index) DO NOTHING
                    """,
                    rows_to_insert,
                )
            conn.commit()
            rows_to_insert.clear()

    if rows_to_insert:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO bench.corpus_chunks
                  (doc_key, chunk_index, chunk_text, char_start, char_end, token_count, chunk_hash)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (doc_key, chunk_index) DO NOTHING
                """,
                rows_to_insert,
            )
        conn.commit()

    n_chunks2 = fetch_one(conn, "SELECT COUNT(*) FROM bench.corpus_chunks")[0]
    print(f"[OK] Built chunks: {n_chunks2}")


def create_run(conn, model_name: str, model_id: int, dim: int, chunking_id: int) -> str:
    run_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bench.embedding_runs (run_id, model_id, chunking_id, notes)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, model_id, chunking_id, f"benchmark embedding: {model_name}"),
        )
    conn.commit()
    return run_id


def load_chunks(conn) -> pd.DataFrame:
    # Load chunk_id, doc_key, chunk_text
    rows = fetch_all(
        conn,
        """
        SELECT c.chunk_id, c.doc_key, c.chunk_text
        FROM bench.corpus_chunks c
        ORDER BY c.chunk_id
        """,
    )
    df = pd.DataFrame(rows, columns=["chunk_id", "doc_key", "chunk_text"])
    return df


def save_faiss_meta(conn, run_id: str, index_path: str, meta_path: str, index_type="FlatIP", metric="cosine"):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bench.faiss_indexes
              (run_id, index_path, meta_path, index_type, metric, notes)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id) DO UPDATE
              SET index_path = EXCLUDED.index_path,
                  meta_path  = EXCLUDED.meta_path,
                  index_type = EXCLUDED.index_type,
                  metric     = EXCLUDED.metric,
                  built_at   = now()
            """,
            (run_id, index_path, meta_path, index_type, metric, "FAISS built by bench_embeddings.py"),
        )
    conn.commit()


def load_queries_qrels(conn) -> Tuple[pd.DataFrame, Dict[int, set]]:
    q = fetch_all(conn, "SELECT query_id, query_text FROM bench.queries WHERE split='test' ORDER BY query_id")
    queries = pd.DataFrame(q, columns=["query_id", "query_text"])

    # qrels: query_id -> set(doc_key)
    rel_rows = fetch_all(conn, "SELECT query_id, doc_key FROM bench.qrels WHERE relevance >= 1")
    qrels: Dict[int, set] = {}
    for qid, doc_key in rel_rows:
        qrels.setdefault(qid, set()).add(doc_key)
    return queries, qrels


def store_retrieval_results(conn, run_id: str, query_id: int, ranks: List[Tuple[int, str, int, float]]):
    # ranks: list of (rank, doc_key, chunk_id, score)
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO bench.retrieval_results (run_id, query_id, rank, doc_key, chunk_id, score)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, query_id, rank) DO UPDATE
              SET doc_key = EXCLUDED.doc_key,
                  chunk_id = EXCLUDED.chunk_id,
                  score = EXCLUDED.score,
                  created_at = now()
            """,
            [(run_id, query_id, r, dk, cid, sc) for (r, dk, cid, sc) in ranks],
        )
    conn.commit()


def store_metrics(conn, run_id: str, recall_at_k: float, mrr: float):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bench.metrics (run_id, metric_name, metric_value)
            VALUES (%s,'recall@%s',%s)
            ON CONFLICT (run_id, metric_name) DO UPDATE SET metric_value=EXCLUDED.metric_value, created_at=now()
            """,
            (run_id, TOP_K, recall_at_k),
        )
        cur.execute(
            """
            INSERT INTO bench.metrics (run_id, metric_name, metric_value)
            VALUES (%s,'mrr',%s)
            ON CONFLICT (run_id, metric_name) DO UPDATE SET metric_value=EXCLUDED.metric_value, created_at=now()
            """,
            (run_id, mrr),
        )
    conn.commit()


# ----------------------------
# Benchmark core
# ----------------------------
def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # -> cosine ready (FlatIP)
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_faiss_index(emb: np.ndarray) -> faiss.Index:
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(emb)
    return index


def evaluate_run(
    conn,
    model_name: str,
    model_id: int,
    dim: int,
    chunking_id: int,
    chunks_df: pd.DataFrame,
):
    print(f"\n=== RUN {model_name} ===")
    run_id = create_run(conn, model_name, model_id, dim, chunking_id)

    # Load embedding model (CPU)
    st_model = SentenceTransformer(model_name, device="cpu")

    # Encode all chunks
    texts = chunks_df["chunk_text"].tolist()
    embeddings = encode_texts(st_model, texts, batch_size=64)
    assert embeddings.shape[1] == dim, f"dim mismatch: got {embeddings.shape[1]}, expected {dim}"

    # Build FAISS
    index = build_faiss_index(embeddings)
    faiss_dir = os.path.join(OUT_DIR, "faiss", run_id)
    os.makedirs(faiss_dir, exist_ok=True)
    index_path = os.path.join(faiss_dir, "index.faiss")
    faiss.write_index(index, index_path)

    # Meta mapping rowid -> chunk_id/doc_key
    meta = chunks_df[["chunk_id", "doc_key"]].to_dict(orient="records")
    meta_path = os.path.join(faiss_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_faiss_meta(conn, run_id, index_path=index_path, meta_path=meta_path)

    # Eval on queries if qrels exist
    queries, qrels = load_queries_qrels(conn)
    if len(queries) == 0:
        print("[WARN] No queries in bench.queries. Skipping evaluation.")
        return
    if len(qrels) == 0:
        print("[WARN] No qrels in bench.qrels. Add qrels to compute metrics. Skipping evaluation.")
        return

    # Prepare quick lookup
    chunk_id_arr = chunks_df["chunk_id"].to_numpy()
    doc_key_arr = chunks_df["doc_key"].to_numpy()

    hits = 0
    reciprocal_ranks = []

    for _, row in queries.iterrows():
        qid = int(row["query_id"])
        qtext = row["query_text"]

        q_emb = encode_texts(st_model, [qtext], batch_size=1)
        scores, idxs = index.search(q_emb, TOP_K)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        ranks_out = []
        found_rank = None
        rel_set = qrels.get(qid, set())

        for rank, (ix, sc) in enumerate(zip(idxs, scores), start=1):
            if ix < 0:
                continue
            chunk_id = int(chunk_id_arr[ix])
            doc_key = str(doc_key_arr[ix])
            ranks_out.append((rank, doc_key, chunk_id, float(sc)))
            if found_rank is None and doc_key in rel_set:
                found_rank = rank

        store_retrieval_results(conn, run_id, qid, ranks_out)

        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = hits / len(queries)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    store_metrics(conn, run_id, recall_at_k=recall_at_k, mrr=mrr)

    print(f"[RESULT] run_id={run_id}")
    print(f"[RESULT] recall@{TOP_K}={recall_at_k:.3f} | mrr={mrr:.3f}")


def main():
    ensure_out_dirs()
    with psycopg.connect(DB_DSN) as conn:
        conn.autocommit = False

        chunking_id, chunk_size, chunk_overlap, model_map = get_ids(conn)

        # Build chunks if missing (you can set limit_docs for fast smoke test)
        build_chunks_if_needed(conn, chunk_size=chunk_size, overlap=chunk_overlap, limit_docs=None)

        chunks_df = load_chunks(conn)
        print(f"[INFO] chunks loaded: {len(chunks_df)}")

        for model_name in EMBED_MODELS:
            model_id, dim = model_map[model_name]
            evaluate_run(conn, model_name=model_name, model_id=model_id, dim=dim,
                         chunking_id=chunking_id, chunks_df=chunks_df)


if __name__ == "__main__":
    main()

"""
-- Example queries for benchmark (French manga recommendations)
-- 1) Humour / school / léger (Takagi)
('eval', 'Je cherche un manga dans le même esprit que "Quand Takagi me taquine" : collège, taquineries, humour léger, pas de violence.', 'fr', 'ref_takagi_humour_school'),

-- 2) Romance / comédie (Kaguya)
('eval', 'Je veux une romance comique avec jeu psychologique entre deux lycéens, comme "Kaguya-sama: Love Is War".', 'fr', 'ref_kaguya_romcom'),

-- 3) Romance feel-good (Shikimori)
('eval', 'Une romance lycée feel-good, douce et drôle, dans le style de "Shikimori n’est pas juste mignonne".', 'fr', 'ref_shikimori_romance_school'),

-- 4) Shonen super-héros (My Hero Academia)
('eval', 'Un shonen d’action et de super-héros dans la veine de "My Hero Academia" (académie, progression, combats).', 'fr', 'ref_mha_shonen'),

-- 5) Shonen gangs / delinquants (Tokyo Revengers)
('eval', 'Un manga avec des gangs/affrontements, tension et drame, proche de "Tokyo Revengers".', 'fr', 'ref_tokyo_revengers_gangs'),

-- 6) Sport (Blue Lock)
('eval', 'Je cherche un manga de sport centré sur le football, compétition, mental et dépassement de soi, comme "Blue Lock".', 'fr', 'ref_blue_lock_football'),

-- 7) Slice of life adulte / long cours (Space Brothers)
('eval', 'Un manga réaliste et motivant sur une ambition de vie, carrière/projet au long cours, comme "Space Brothers".', 'fr', 'ref_space_brothers_realiste'),

-- 8) Nocturne / surnaturel / romance (Call of the Night)
('eval', 'Ambiance nocturne, surnaturel léger et romance, dans le style de "Call of the Night".', 'fr', 'ref_call_of_the_night_nocturne'),

-- 9) Poétique / contemplatif (Du mouvement de la Terre)
('eval', 'Un récit profond et contemplatif, avec une dimension historique/philosophique, comme "Du mouvement de la Terre".', 'fr', 'ref_du_mouvement_de_la_terre_philo'),

-- 10) Imaginaire / onirique (Les Enfants de la Baleine)
('eval', 'Un univers onirique et mélancolique, fantasy/aventure avec poésie, proche de "Les Enfants de la Baleine".', 'fr', 'ref_enfants_de_la_baleine_onirique'),

-- 11) Compétition / dépassement (Chihayafuru)
('eval', 'Une série centrée sur la compétition et le dépassement de soi, avec un groupe de personnages, comme "Chihayafuru".', 'fr', 'ref_chihayafuru_competition'),

-- 12) Query “pont” (shonen référence Naruto + MHA)
('eval', 'Je veux un shonen proche de Naruto et My Hero Academia : progression, amitiés, rivalités, combats et valeurs positives.', 'fr', 'ref_shonen_naruto_mha_bridge');

"""
