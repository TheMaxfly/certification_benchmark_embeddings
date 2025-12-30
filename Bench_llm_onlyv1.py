import os, json, time
import requests
import numpy as np
import faiss
import psycopg

DB_DSN = os.environ.get(
    "APIMANGA_DSN",
    "dbname=apimanga user=postgres password=postgres host=localhost port=5432",
)

# IMPORTANT: mettre ici le run_id embeddings (MiniLM) qui a l'index FAISS
RETRIEVAL_RUN_ID = os.environ.get("BENCH_RETRIEVAL_RUN_ID", "bac7e306-583c-4db8-afd5-0d35d3964e08")
SPLIT = os.environ.get("BENCH_SPLIT", "eval")
TOP_K_CHUNKS = int(os.environ.get("BENCH_TOP_K_CHUNKS", "30"))  # chunks récupérés
MAX_EVIDENCES = int(os.environ.get("BENCH_MAX_EVIDENCES", "10")) # extraits donnés au LLM
OUT_JSONL = os.environ.get("BENCH_LLM_OUT", "./bench_out/llm_results.jsonl")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

LLM_MODELS = [
    #"mistral:7b-instruct-v0.3-q4_K_M",
    #"phi3.5:3.8b-mini-instruct-q4_K_M",
    "ministral-3:3b-instruct-2512-q4_K_M",
]

OLLAMA_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_ctx": 2048,
    #"num_predict": 220,
}

SYSTEM_PROMPT = """Tu es un conseiller libraire manga pour une TPE.
Règles:
- Utilise UNIQUEMENT les éléments fournis dans EVIDENCES.
- Ne cite pas d’auteurs/tags/infos non présents.
- Donne 5 recommandations max, en français.
- Pour chaque recommandation: 1 justification courte + une citation [doc_key|chunk_id].
"""

def fetch_all(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()

def fetch_one(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()

def load_faiss_paths(conn, run_id: str):
    row = fetch_one(conn, """
        SELECT index_path, meta_path
        FROM bench.faiss_indexes
        WHERE run_id = %s
    """, (run_id,))
    if not row:
        raise RuntimeError(f"No FAISS index in bench.faiss_indexes for run_id={run_id}")
    return row[0], row[1]

def load_meta(meta_path: str):
    chunk_ids = []
    doc_keys = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_ids.append(int(obj["chunk_id"]))
            doc_keys.append(str(obj["doc_key"]))
    return np.array(chunk_ids), np.array(doc_keys)

def load_chunk_texts(conn, chunk_ids):
    # récupère le texte exact des chunks retournés
    rows = fetch_all(conn, """
        SELECT chunk_id, doc_key, chunk_text
        FROM bench.corpus_chunks
        WHERE chunk_id = ANY(%s)
    """, (list(map(int, chunk_ids)),))
    m = {int(cid): (dk, tx) for (cid, dk, tx) in rows}
    return m

def load_queries(conn):
    rows = fetch_all(conn, """
        SELECT query_id, intent, query_text
        FROM bench.queries
        WHERE split = %s
        ORDER BY query_id
    """, (SPLIT,))
    return rows

def call_ollama(model: str, prompt: str):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return data.get("response", ""), dt

def build_prompt(query_text: str, evidences: list[dict]) -> str:
    ev_lines = []
    for ev in evidences:
        ev_lines.append(
            f"- [{ev['doc_key']}|{ev['chunk_id']}] {ev['chunk_text']}"
        )
    ev_block = "\n".join(ev_lines)
    return f"{SYSTEM_PROMPT}\n\nDEMANDE UTILISATEUR:\n{query_text}\n\nEVIDENCES:\n{ev_block}\n\nRéponse:"

def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    with psycopg.connect(DB_DSN) as conn:
        index_path, meta_path = load_faiss_paths(conn, RETRIEVAL_RUN_ID)
        index = faiss.read_index(index_path)
        chunk_id_arr, doc_key_arr = load_meta(meta_path)

        queries = load_queries(conn)
        if not queries:
            raise RuntimeError(f"Aucune query dans bench.queries avec split='{SPLIT}'")

        # Important: ici, tu dois réutiliser le même modèle d'embedding que celui de l'index.
        # Comme tu benchmarkes le LLM, pas le retrieval, on réutilise l'index tel quel.
        # -> On ne ré-encode pas la query ici : on suppose que tu as déjà un pipeline retrieval
        # Dans ton projet, tu as déjà la fonction encode(query). Ici, on fait une version simple:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

        with open(OUT_JSONL, "w", encoding="utf-8") as out:
            for model in LLM_MODELS:
                for (qid, intent, qtext) in queries:
                    q_emb = emb_model.encode([qtext], normalize_embeddings=True)
                    scores, idxs = index.search(np.asarray(q_emb, dtype=np.float32), TOP_K_CHUNKS)
                    idxs = idxs[0].tolist()

                    # récupérer chunks + textes (limiter à MAX_EVIDENCES)
                    selected = []
                    for ix in idxs:
                        if ix < 0:
                            continue
                        chunk_id = int(chunk_id_arr[ix])
                        doc_key = str(doc_key_arr[ix])
                        selected.append((chunk_id, doc_key))
                        if len(selected) >= MAX_EVIDENCES:
                            break

                    chunk_map = load_chunk_texts(conn, [c for (c, _) in selected])
                    evidences = []
                    for (chunk_id, doc_key) in selected:
                        dk, tx = chunk_map.get(chunk_id, (doc_key, None))
                        if tx:
                            evidences.append({"chunk_id": chunk_id, "doc_key": dk, "chunk_text": tx})

                    prompt = build_prompt(qtext, evidences)
                    answer, latency_s = call_ollama(model, prompt)

                    rec = {
                        "llm_model": model,
                        "query_id": int(qid),
                        "intent": intent,
                        "query_text": qtext,
                        "latency_s": latency_s,
                        "evidences": [{"doc_key": e["doc_key"], "chunk_id": e["chunk_id"]} for e in evidences],
                        "answer": answer,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[OK] {model} qid={qid} intent={intent} latency={latency_s:.2f}s")

if __name__ == "__main__":
    main()
