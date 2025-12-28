import os
import json
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg
import faiss
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

TOP_K = int(os.environ.get("BENCH_TOP_K", "8"))
DEBUG = os.environ.get("BENCH_DEBUG", "1") == "1"

RUN_IDS = [
    "bac7e306-583c-4db8-afd5-0d35d3964e08",  # paraphrase-multilingual-MiniLM-L12-v2
    "6e2f572f-66c9-46d1-8aec-7bced65c7820",  # intfloat/multilingual-e5-small
]
RUN_IDS = [rid.strip() for rid in RUN_IDS]


# ----------------------------
# DB helpers
# ----------------------------
def fetch_all(conn, sql: str, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def fetch_one(conn, sql: str, params=None):
    rows = fetch_all(conn, sql, params)
    return rows[0] if rows else None


def load_queries(conn) -> pd.DataFrame:
    q = fetch_all(
        conn,
        "SELECT query_id, query_text FROM bench.queries WHERE split='eval' ORDER BY query_id",
    )
    return pd.DataFrame(q, columns=["query_id", "query_text"])


def entity_key(series_id: Optional[int], kitsu_id: Optional[int]) -> Optional[str]:
    if series_id is not None:
        return f"S:{int(series_id)}"
    if kitsu_id is not None:
        return f"K:{int(kitsu_id)}"
    return None


def load_dockey_to_entity(conn) -> Dict[str, str]:
    """
    Map doc_key -> entity_key (S:series_id or K:kitsu_id).
    Important: This makes evaluation robust across ms_review/ms_hybrid/kitsu.
    """
    rows = fetch_all(
        conn,
        """
        SELECT doc_key, series_id, kitsu_id
        FROM bench.corpus_docs
        """,
    )
    m: Dict[str, str] = {}
    missing = 0
    for dk, sid, kid in rows:
        dk = str(dk)
        ek = entity_key(sid, kid)
        if ek is None:
            missing += 1
            continue
        m[dk] = ek

    if DEBUG:
        print(f"[DEBUG] doc_key->entity mapping size={len(m)} | missing(no series_id/kitsu_id)={missing}")
    return m


def load_qrels_entities(conn, dockey_to_entity: Dict[str, str]) -> Dict[int, Set[str]]:
    """
    Convert existing bench.qrels (doc_key-based) into entity-based ground truth.
    query_id -> set(entity_key)
    """
    rel_rows = fetch_all(conn, "SELECT query_id, doc_key FROM bench.qrels WHERE relevance >= 1")
    qrels_ent: Dict[int, Set[str]] = {}
    missing = 0
    for qid, doc_key in rel_rows:
        qid = int(qid)
        dk = str(doc_key)
        ek = dockey_to_entity.get(dk)
        if ek is None:
            missing += 1
            continue
        qrels_ent.setdefault(qid, set()).add(ek)

    if DEBUG:
        print(f"[DEBUG] qrels entity size (queries)={len(qrels_ent)} | missing doc_key in map={missing}")
    return qrels_ent


def store_metrics(conn, run_id: str, recall_at_k: float, mrr: float, prefix: str):
    """
    prefix lets us store both doc_key metrics and entity metrics if needed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bench.metrics (run_id, metric_name, metric_value)
            VALUES (%s, %s, %s)
            ON CONFLICT (run_id, metric_name) DO UPDATE
              SET metric_value = EXCLUDED.metric_value,
                  created_at = now()
            """,
            (run_id, f"{prefix}recall@{TOP_K}", float(recall_at_k)),
        )
        cur.execute(
            """
            INSERT INTO bench.metrics (run_id, metric_name, metric_value)
            VALUES (%s, %s, %s)
            ON CONFLICT (run_id, metric_name) DO UPDATE
              SET metric_value = EXCLUDED.metric_value,
                  created_at = now()
            """,
            (run_id, f"{prefix}mrr", float(mrr)),
        )
    conn.commit()


def list_known_faiss_run_ids(conn) -> List[str]:
    rows = fetch_all(conn, "SELECT run_id::text FROM bench.faiss_indexes ORDER BY built_at DESC")
    return [r[0] for r in rows]


def load_index_and_meta(conn, run_id: str):
    row = fetch_one(
        conn,
        """
        SELECT index_path, meta_path
        FROM bench.faiss_indexes
        WHERE run_id = %s::uuid
        """,
        (run_id,),
    )
    if not row:
        known = list_known_faiss_run_ids(conn)
        raise RuntimeError(
            f"No FAISS index registered in DB for run_id={run_id!r}.\n"
            f"Known run_ids in bench.faiss_indexes: {known[:10]}"
        )
    index_path, meta_path = row

    row2 = fetch_one(
        conn,
        """
        SELECT em.model_name
        FROM bench.embedding_runs r
        JOIN bench.embedding_models em ON em.model_id = r.model_id
        WHERE r.run_id = %s::uuid
        """,
        (run_id,),
    )
    if not row2:
        raise RuntimeError(f"run_id={run_id!r} exists in faiss_indexes but not in embedding_runs/models join.")
    model_name = row2[0]

    if not os.path.exists(index_path):
        raise RuntimeError(f"index_path not found on disk: {index_path}")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"meta_path not found on disk: {meta_path}")

    index = faiss.read_index(index_path)

    chunk_ids: List[int] = []
    doc_keys: List[str] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_ids.append(int(obj["chunk_id"]))
            doc_keys.append(str(obj["doc_key"]))

    chunk_id_arr = np.array(chunk_ids, dtype=np.int64)
    doc_key_arr = np.array(doc_keys, dtype=object)

    return model_name, index, chunk_id_arr, doc_key_arr, index_path, meta_path


# ----------------------------
# EVAL: entity-level (series_id/kitsu_id)
# ----------------------------
def evaluate_run_entity_only(conn, run_id: str):
    model_name, index, chunk_id_arr, doc_key_arr, index_path, meta_path = load_index_and_meta(conn, run_id)

    print(f"\n=== EVAL (ENTITY) {model_name} ===")
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] index={index_path}")
    print(f"[INFO] meta ={meta_path}")

    queries = load_queries(conn)
    if len(queries) == 0:
        raise RuntimeError("No queries in bench.queries for split='eval'.")

    dockey_to_entity = load_dockey_to_entity(conn)
    qrels_ent = load_qrels_entities(conn, dockey_to_entity)
    if len(qrels_ent) == 0:
        raise RuntimeError("No qrels in bench.qrels (or none could be mapped to entity keys).")

    st_model = SentenceTransformer(model_name, device="cpu")

    hits = 0
    reciprocal_ranks: List[float] = []

    for i, row in enumerate(queries.itertuples(index=False), start=1):
        qid = int(row.query_id)
        qtext = str(row.query_text)

        rel_set = qrels_ent.get(qid, set())
        if not rel_set:
            # query has no mapped qrels -> count as miss but keep going
            reciprocal_ranks.append(0.0)
            continue

        q_emb = st_model.encode([qtext], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, idxs = index.search(q_emb, TOP_K)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        found_rank = None
        top_entities: List[str] = []

        for rank, ix in enumerate(idxs, start=1):
            if ix < 0:
                continue
            dk = str(doc_key_arr[ix])
            ek = dockey_to_entity.get(dk)
            if ek is None:
                continue
            top_entities.append(ek)
            if found_rank is None and ek in rel_set:
                found_rank = rank

        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        if DEBUG and i <= 2:
            print(f"[DEBUG] qid={qid} text={qtext}")
            print(f"[DEBUG] rel_entities={sorted(list(rel_set))[:12]}")
            print(f"[DEBUG] top_entities={top_entities[:12]}")

    recall_at_k = hits / len(queries)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # Store entity-level metrics (prefix "entity_")
    store_metrics(conn, run_id, recall_at_k=recall_at_k, mrr=mrr, prefix="entity_")

    print(f"[RESULT] queries={len(queries)} | entity_recall@{TOP_K}={recall_at_k:.3f} | entity_mrr={mrr:.3f}")


def main():
    if DEBUG:
        print("[DEBUG] RUN_IDS (repr, len):")
        for rid in RUN_IDS:
            print("  ", repr(rid), "len=", len(rid))

    with psycopg.connect(DB_DSN) as conn:
        conn.autocommit = False
        if DEBUG:
            known = list_known_faiss_run_ids(conn)
            print("[DEBUG] Known FAISS run_ids in DB (bench.faiss_indexes):", known[:10])

        for run_id in RUN_IDS:
            evaluate_run_entity_only(conn, run_id)


if __name__ == "__main__":
    main()
