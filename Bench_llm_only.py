#!/usr/bin/env python3
# Bench_llm_only.py — V2 constrained (FR + 3 recos max + citations obligatoires)
#
# Usage (exemple):
#   export APIMANGA_DSN="dbname=apimanga user=postgres password=postgres host=localhost port=5432"
#   export BENCH_RETRIEVAL_RUN_ID="bac7e306-583c-4db8-afd5-0d35d3964e08"
#   export BENCH_SPLIT="eval"
#   python3 Bench_llm_only.py
#
# Résultats: ./bench_out/llm_results_v2_constrained.jsonl (par défaut)

import os
import re
import json
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import psycopg
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer


# ----------------------------
# Config via env
# ----------------------------
DB_DSN = os.environ.get(
    "APIMANGA_DSN",
    "dbname=apimanga user=postgres password=postgres host=localhost port=5432",
)

# Run id FAISS (embeddings) à utiliser pour le retrieval
BENCH_RETRIEVAL_RUN_ID = os.environ.get("BENCH_RETRIEVAL_RUN_ID", "").strip()
if not BENCH_RETRIEVAL_RUN_ID:
    raise SystemExit("ERROR: BENCH_RETRIEVAL_RUN_ID is required (export BENCH_RETRIEVAL_RUN_ID=...)")

# Split de queries
BENCH_SPLIT = os.environ.get("BENCH_SPLIT", "eval")

# Combien de chunks on récupère dans FAISS
TOP_K_CHUNKS = int(os.environ.get("BENCH_TOP_K_CHUNKS", "30"))
# Combien de chunks on envoie au LLM
MAX_EVIDENCES = int(os.environ.get("BENCH_MAX_EVIDENCES", "10"))

# Embedding model pour vectoriser la requête (doit matcher le run FAISS)
# NOTE: Dans tes runs, le retrieval run_id correspond à MiniLM.
RETRIEVAL_EMBED_MODEL = os.environ.get("BENCH_RETRIEVAL_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

# Modèles LLM à benchmarker
# (tu peux surcharger via env BENCH_LLM_MODELS="a,b,c")
_llm_env = os.environ.get("BENCH_LLM_MODELS", "").strip()
if _llm_env:
    LLM_MODELS = [m.strip() for m in _llm_env.split(",") if m.strip()]
else:
    LLM_MODELS = [
        "phi3.5:3.8b-mini-instruct-q4_K_M",
        "ministral-3:3b-instruct-2512-q4_K_M",
        "mistral:7b-instruct-v0.3-q4_K_M",
    ]

# Contraintes generation
NUM_CTX = int(os.environ.get("BENCH_NUM_CTX", "2048"))
NUM_PREDICT = int(os.environ.get("BENCH_NUM_PREDICT", "220"))  # important pour éviter les romans / outliers
TEMPERATURE = float(os.environ.get("BENCH_TEMPERATURE", "0.2"))
TOP_P = float(os.environ.get("BENCH_TOP_P", "0.9"))
REPEAT_PENALTY = float(os.environ.get("BENCH_REPEAT_PENALTY", "1.1"))

# Output
OUT_DIR = os.environ.get("BENCH_OUT_DIR", "./bench_out")
OUT_JSONL = os.environ.get("BENCH_LLM_OUT", os.path.join(OUT_DIR, "llm_results_v2_constrained.jsonl"))

# Debug
DEBUG = os.environ.get("BENCH_DEBUG", "1") == "1"
DEBUG_SHOW_EVIDENCES_CHARS = int(os.environ.get("BENCH_DEBUG_EVID_CHARS", "240"))


# ----------------------------
# Prompts (V2 contraint)
# ----------------------------
SYSTEM_PROMPT = """Tu es un conseiller libraire manga. Tu réponds UNIQUEMENT en français.

RÈGLES STRICTES (obligatoires):
- Donne AU MAXIMUM 3 recommandations.
- Format obligatoire: exactement des lignes numérotées "1) ...", "2) ...", "3) ..." (si tu as moins de 3, tu peux t'arrêter).
- Une seule phrase de justification par recommandation.
- Chaque ligne DOIT se terminer par une citation au format [doc_key|chunk_id] en réutilisant UNIQUEMENT les sources fournies.
- N'invente aucune information (auteurs, tomes, genres, dates) si ce n'est pas présent dans les sources.
- Si les sources ne suffisent pas pour recommander, écris UNE ligne:
  1) Je n'ai pas assez d'informations dans les sources fournies. [doc_key|chunk_id]
"""

USER_PROMPT_TEMPLATE = """Demande client:
{query_text}

SOURCES (tu dois citer exactement au format [doc_key|chunk_id] sur chaque ligne):
{evidences_block}

Réponds maintenant en respectant STRICTEMENT les règles.
"""

REPAIR_PROMPT = """Tu n'as pas respecté les règles. Corrige maintenant.

Rappel STRICT:
- Français uniquement
- 3 lignes maximum: 1) 2) 3)
- 1 phrase par ligne
- Chaque ligne se termine par une citation [doc_key|chunk_id] issue des sources
- Aucun texte hors de ces lignes

Réécris la réponse complète maintenant.
"""


# ----------------------------
# Validation heuristiques
# ----------------------------
CIT_RE = re.compile(r"\[[^\[\]\|]+\|[0-9]+\]")  # [doc_key|123]
NUM_LINE_RE = re.compile(r"^\s*([1-3])\)\s+", re.M)

def extract_numbered_lines(ans: str) -> List[str]:
    lines = []
    for raw in ans.splitlines():
        if re.match(r"^\s*[1-3]\)\s+", raw):
            lines.append(raw.strip())
    return lines

def count_citations(ans: str) -> int:
    return len(CIT_RE.findall(ans))

def citations_per_numbered_line_ok(ans: str) -> bool:
    for line in extract_numbered_lines(ans):
        if not CIT_RE.search(line):
            return False
    return True

def has_valid_format(ans: str) -> bool:
    lines = extract_numbered_lines(ans)
    if not (1 <= len(lines) <= 3):
        return False
    # must be 1) then 2) then 3) in order (if present)
    seen = []
    for ln in lines:
        m = re.match(r"^\s*([1-3])\)\s+", ln)
        if m:
            seen.append(int(m.group(1)))
    # ensure strictly increasing starting at 1
    if seen[0] != 1:
        return False
    if any(seen[i] <= seen[i-1] for i in range(1, len(seen))):
        return False
    return True

def looks_french(ans: str) -> bool:
    # Simple heuristique: présence de mots fréquents FR et peu de marqueurs EN
    low = ans.lower()
    fr_hits = sum(w in low for w in [" je ", " tu ", " avec ", " pour ", " pas ", " une ", " des ", " et ", "ce ", "ça "])
    en_hits = sum(w in low for w in [" the ", " and ", " with ", "because", "recommend", "recommendation", "you should"])
    # autorise FR même si pas d'espaces (début/fin)
    fr_hits += sum(low.startswith(w.strip()) for w in ["je", "tu", "avec", "pour", "pas", "une", "des", "et"])
    return fr_hits >= 3 and en_hits <= 1

def validate_answer(ans: str) -> Dict[str, Any]:
    lines = extract_numbered_lines(ans)
    ok_format = has_valid_format(ans)
    ok_citations = citations_per_numbered_line_ok(ans) and count_citations(ans) >= len(lines) >= 1
    ok_french = looks_french(ans)
    n_recos = len(lines)
    n_cit = count_citations(ans)
    reasons = []
    if not ok_french:
        reasons.append("not_french")
    if not ok_format:
        reasons.append("bad_format")
    if not ok_citations:
        reasons.append("missing_citations")
    return {
        "ok_french": ok_french,
        "ok_format": ok_format,
        "ok_citations": ok_citations,
        "n_recos": n_recos,
        "n_citations": n_cit,
        "fail_reasons": reasons,
    }


# ----------------------------
# DB helpers
# ----------------------------
def fetch_one(conn, sql: str, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()

def fetch_all(conn, sql: str, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def load_queries(conn) -> List[Dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT query_id, intent, query_text
        FROM bench.queries
        WHERE split = %s
        ORDER BY query_id
        """,
        (BENCH_SPLIT,),
    )
    return [{"query_id": int(qid), "intent": intent, "query_text": qtxt} for (qid, intent, qtxt) in rows]

def load_faiss_paths(conn, run_id: str) -> Tuple[str, str]:
    row = fetch_one(
        conn,
        """
        SELECT index_path, meta_path
        FROM bench.faiss_indexes
        WHERE run_id = %s
        """,
        (run_id,),
    )
    if not row:
        raise RuntimeError(f"No FAISS index registered in DB for run_id={run_id!r}")
    return row[0], row[1]

def load_meta_arrays(meta_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # meta.jsonl lines: {"chunk_id": ..., "doc_key": ...}
    chunk_ids = []
    doc_keys = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_ids.append(int(obj["chunk_id"]))
            doc_keys.append(str(obj["doc_key"]))
    return np.asarray(chunk_ids, dtype=np.int64), np.asarray(doc_keys, dtype=object)

def fetch_chunks_text(conn, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    # returns chunk_id -> {doc_key, chunk_text}
    if not chunk_ids:
        return {}
    rows = fetch_all(
        conn,
        """
        SELECT chunk_id, doc_key, chunk_text
        FROM bench.corpus_chunks
        WHERE chunk_id = ANY(%s)
        """,
        (chunk_ids,),
    )
    out: Dict[int, Dict[str, Any]] = {}
    for cid, dk, txt in rows:
        out[int(cid)] = {"doc_key": str(dk), "chunk_text": str(txt)}
    return out


# ----------------------------
# Retrieval (FAISS)
# ----------------------------
@dataclass
class Evidence:
    doc_key: str
    chunk_id: int
    score: float
    chunk_text: str

def build_evidences_block(evidences: List[Evidence]) -> str:
    lines = []
    for ev in evidences:
        # IMPORTANT: on met le tag de citation dès le début pour obliger le LLM à l'utiliser.
        lines.append(f"- [{ev.doc_key}|{ev.chunk_id}] {ev.chunk_text}")
    return "\n".join(lines)


# ----------------------------
# Ollama call
# ----------------------------
def ollama_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "repeat_penalty": REPEAT_PENALTY,
        },
    }
    r = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    # Ollama returns: {"message":{"role":"assistant","content":"..."}...}
    return (data.get("message") or {}).get("content", "") or ""


# ----------------------------
# Main benchmark
# ----------------------------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def summarize_model_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lat = [float(x["latency_s"]) for x in rows]
    ok_fr = sum(1 for x in rows if x["ok_french"]) / len(rows) if rows else 0.0
    ok_fmt = sum(1 for x in rows if x["ok_format"]) / len(rows) if rows else 0.0
    ok_cit = sum(1 for x in rows if x["ok_citations"]) / len(rows) if rows else 0.0
    ok_all = sum(1 for x in rows if (x["ok_french"] and x["ok_format"] and x["ok_citations"])) / len(rows) if rows else 0.0
    retries = sum(1 for x in rows if x.get("retry_used"))
    n_recos_avg = statistics.mean([x.get("n_recos", 0) for x in rows]) if rows else 0.0

    def pct(p): return round(100.0 * p, 1)

    return {
        "n": len(rows),
        "latency_mean_s": round(statistics.mean(lat), 2) if rows else None,
        "latency_median_s": round(statistics.median(lat), 2) if rows else None,
        "latency_max_s": round(max(lat), 2) if rows else None,
        "ok_french_pct": pct(ok_fr),
        "ok_format_pct": pct(ok_fmt),
        "ok_citations_pct": pct(ok_cit),
        "ok_all_pct": pct(ok_all),
        "retries": retries,
        "avg_recos": round(n_recos_avg, 2),
    }

def run_benchmark():
    ensure_out_dir()

    with psycopg.connect(DB_DSN) as conn:
        # Load queries
        queries = load_queries(conn)
        if not queries:
            raise RuntimeError(f"No queries found in bench.queries for split={BENCH_SPLIT!r}")

        # Load FAISS + meta from DB for the retrieval run
        index_path, meta_path = load_faiss_paths(conn, BENCH_RETRIEVAL_RUN_ID)
        if DEBUG:
            print(f"[INFO] retrieval_run_id={BENCH_RETRIEVAL_RUN_ID}")
            print(f"[INFO] index_path={index_path}")
            print(f"[INFO] meta_path ={meta_path}")
            print(f"[INFO] retrieval_embed_model={RETRIEVAL_EMBED_MODEL}")
            print(f"[INFO] TOP_K_CHUNKS={TOP_K_CHUNKS} | MAX_EVIDENCES={MAX_EVIDENCES}")
            print(f"[INFO] num_ctx={NUM_CTX} | num_predict={NUM_PREDICT}")

        index = faiss.read_index(index_path)
        chunk_id_arr, doc_key_arr = load_meta_arrays(meta_path)

        # Retrieval embed model (CPU)
        emb_model = SentenceTransformer(RETRIEVAL_EMBED_MODEL, device="cpu")

        # Output JSONL
        with open(OUT_JSONL, "w", encoding="utf-8") as out:
            all_rows_by_model: Dict[str, List[Dict[str, Any]]] = {}

            for llm_model in LLM_MODELS:
                print(f"\n=== LLM BENCH (V2 constrained) {llm_model} ===")
                rows_model: List[Dict[str, Any]] = []

                for q in queries:
                    qid = q["query_id"]
                    intent = q["intent"]
                    qtext = q["query_text"]

                    # 1) Embed query
                    q_emb = emb_model.encode([qtext], normalize_embeddings=True)
                    q_emb = np.asarray(q_emb, dtype=np.float32)

                    # 2) Search FAISS
                    scores, idxs = index.search(q_emb, TOP_K_CHUNKS)
                    scores = scores[0].tolist()
                    idxs = idxs[0].tolist()

                    # map to chunk_ids
                    top_chunk_ids: List[int] = []
                    top_pairs: List[Tuple[int, float]] = []
                    for ix, sc in zip(idxs, scores):
                        if ix < 0:
                            continue
                        cid = int(chunk_id_arr[ix])
                        top_pairs.append((cid, float(sc)))
                        top_chunk_ids.append(cid)

                    # 3) Fetch chunk texts from DB (limit to MAX_EVIDENCES after fetch)
                    chunk_map = fetch_chunks_text(conn, top_chunk_ids)

                    evidences: List[Evidence] = []
                    for cid, sc in top_pairs:
                        rec = chunk_map.get(cid)
                        if not rec:
                            continue
                        evidences.append(Evidence(
                            doc_key=rec["doc_key"],
                            chunk_id=cid,
                            score=sc,
                            chunk_text=rec["chunk_text"].strip(),
                        ))
                        if len(evidences) >= MAX_EVIDENCES:
                            break

                    if not evidences:
                        # fallback minimal
                        evidences_block = "- [unknown|0] (Aucune source récupérée)"
                    else:
                        evidences_block = build_evidences_block(evidences)

                    if DEBUG:
                        preview = evidences_block[:DEBUG_SHOW_EVIDENCES_CHARS].replace("\n", " ")
                        print(f"[DEBUG] qid={qid} intent={intent} evidences_preview={preview!r}")

                    # 4) Call LLM (with 1 retry if needed)
                    user_prompt = USER_PROMPT_TEMPLATE.format(query_text=qtext, evidences_block=evidences_block)

                    t0 = time.time()
                    answer = ollama_chat(llm_model, SYSTEM_PROMPT, user_prompt)
                    latency = time.time() - t0

                    v1 = validate_answer(answer)
                    retry_used = False
                    answer_repaired = ""
                    v2 = None

                    if v1["fail_reasons"]:
                        # Retry once with repair prompt
                        retry_used = True
                        t1 = time.time()
                        answer_repaired = ollama_chat(
                            llm_model,
                            SYSTEM_PROMPT,
                            user_prompt + "\n\n" + REPAIR_PROMPT
                        )
                        latency += (time.time() - t1)
                        v2 = validate_answer(answer_repaired)

                        # choose best between answer and repaired (prefer one that satisfies more constraints)
                        def score_valid(v):
                            return int(v["ok_french"]) + int(v["ok_format"]) + int(v["ok_citations"])

                        if score_valid(v2) >= score_valid(v1):
                            answer = answer_repaired
                            v1 = v2

                    row = {
                        "split": BENCH_SPLIT,
                        "retrieval_run_id": BENCH_RETRIEVAL_RUN_ID,
                        "retrieval_embed_model": RETRIEVAL_EMBED_MODEL,
                        "top_k_chunks": TOP_K_CHUNKS,
                        "max_evidences": MAX_EVIDENCES,
                        "llm_model": llm_model,
                        "query_id": qid,
                        "intent": intent,
                        "query_text": qtext,
                        "latency_s": round(latency, 3),
                        "retry_used": retry_used,
                        "ok_french": v1["ok_french"],
                        "ok_format": v1["ok_format"],
                        "ok_citations": v1["ok_citations"],
                        "n_recos": v1["n_recos"],
                        "n_citations": v1["n_citations"],
                        "fail_reasons": v1["fail_reasons"],
                        # keep a compact evidence list for auditability
                        "evidences": [
                            {"doc_key": e.doc_key, "chunk_id": e.chunk_id, "score": round(e.score, 6)}
                            for e in evidences
                        ],
                        "answer": answer.strip(),
                    }

                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()

                    rows_model.append(row)
                    status = "OK" if (row["ok_french"] and row["ok_format"] and row["ok_citations"]) else "WARN"
                    print(f"[{status}] {llm_model} qid={qid} intent={intent} latency={row['latency_s']}s retry={int(retry_used)}")

                all_rows_by_model[llm_model] = rows_model

            # Final summary
            print("\n=== SUMMARY (V2 constrained) ===")
            for llm_model, rows in all_rows_by_model.items():
                s = summarize_model_stats(rows)
                print(
                    f"- {llm_model} | n={s['n']}"
                    f" | mean={s['latency_mean_s']}s med={s['latency_median_s']}s max={s['latency_max_s']}s"
                    f" | ok_all={s['ok_all_pct']}% (FR={s['ok_french_pct']}% fmt={s['ok_format_pct']}% cit={s['ok_citations_pct']}%)"
                    f" | retries={s['retries']} | avg_recos={s['avg_recos']}"
                )

            print(f"\n[OUT] {OUT_JSONL}")

def main():
    run_benchmark()

if __name__ == "__main__":
    main()
