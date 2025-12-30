#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bench_llm_only_v3.py

V3 goals:
- FR-first evidence filtering (reduce English pollution)
- LLM-friendly prompt (1–3 recos, intro optional, citations anywhere per reco)
- Parse titles -> map to entities (series_id/kitsu_id)
- Score: EntityHit@3, Recall@3, CitationValidity (+ basic FR rate, latency)
- Export JSONL + console summary

Expected DB objects:
- bench.queries (query_id, query_text, intent, split)
- bench.qrels (query_id, doc_key, relevance)
- bench.corpus_docs (doc_key, source, series_id, kitsu_id, title, metadata_json, doc_text)
- bench.corpus_chunks (chunk_id, doc_key, chunk_text)
- bench.embedding_runs, bench.embedding_models, bench.faiss_indexes

Requires:
pip install psycopg sentence-transformers faiss-cpu requests numpy
"""

import os
import re
import json
import time
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import psycopg
import faiss
import requests
from sentence_transformers import SentenceTransformer


# ----------------------------
# Config (env)
# ----------------------------
DB_DSN = os.environ.get(
    "APIMANGA_DSN",
    "dbname=apimanga user=postgres password=postgres host=localhost port=5432",
)

RETRIEVAL_RUN_ID = os.environ.get("BENCH_RETRIEVAL_RUN_ID", "").strip()
SPLIT = os.environ.get("BENCH_SPLIT", "eval").strip()

TOP_K_CHUNKS = int(os.environ.get("BENCH_TOP_K_CHUNKS", "30"))
MAX_EVIDENCES = int(os.environ.get("BENCH_MAX_EVIDENCES", "10"))
TOP_RECOS = int(os.environ.get("BENCH_TOP_RECOS", "3"))

OUT_DIR = os.environ.get("BENCH_OUT_DIR", "./bench_out")
OUT_JSONL = os.environ.get("BENCH_OUT_JSONL", os.path.join(OUT_DIR, "llm_results_v3.jsonl"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_URL}/api/chat"

# If BENCH_LLM_MODELS is set, it overrides defaults (comma-separated).
DEFAULT_LLM_MODELS = [
    "phi3.5:3.8b-mini-instruct-q4_K_M",
    "ministral-3:3b-instruct-2512-q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M",
]
_llm_env = os.environ.get("BENCH_LLM_MODELS", "").strip()
LLM_MODELS = [m.strip() for m in _llm_env.split(",") if m.strip()] if _llm_env else DEFAULT_LLM_MODELS

# Ollama options (can be overridden by env)
OLLAMA_OPTIONS = {
    "temperature": float(os.environ.get("BENCH_TEMP", "0.2")),
    "top_p": float(os.environ.get("BENCH_TOP_P", "0.9")),
    "num_ctx": int(os.environ.get("BENCH_NUM_CTX", "2048")),
    "num_predict": int(os.environ.get("BENCH_NUM_PREDICT", "220")),
}

# ----------------------------
# Prompts (V3 - less strict, more natural)
# ----------------------------
SYSTEM_PROMPT = """Tu es un conseiller libraire manga pour une petite librairie indépendante.
Tu réponds UNIQUEMENT en français.
Tu t'appuies UNIQUEMENT sur les sources fournies (SOURCES). Ne fais aucune supposition non présente dans les sources.
Si les sources ne permettent pas de répondre correctement, dis-le explicitement et propose une question de clarification.
Objectif: recommander des mangas pertinents avec une justification courte et des citations.
"""

USER_PROMPT_TEMPLATE = """Demande client:
{query_text}

SOURCES (extraits RAG). Tu dois citer des preuves en utilisant exactement le format [doc_key|chunk_id] :
{evidences_block}

Consignes de réponse (format naturel, pas trop strict):
- Réponds en français.
- Tu peux écrire 0 à 1 phrase d'introduction (optionnel).
- Propose 1 à {top_recos} recommandations maximum.
- Pour chaque recommandation:
  * Donne le titre (ou le meilleur intitulé disponible dans les sources).
  * Donne 1 à 2 phrases de justification basées sur les SOURCES.
  * Ajoute au moins 1 citation au format [doc_key|chunk_id] (dans la phrase ou à la fin).
- Évite l'anglais: si un extrait source est en anglais, reformule-le en français dans ta justification.
- N'invente pas d'informations (auteurs, tomes, âge, thèmes) si ce n'est pas dans les SOURCES.

Réponds maintenant.
"""

REPAIR_PROMPT = """Ta réponse n'est pas conforme. Corrige-la.

Rappel (contraintes V3):
- Français uniquement (réforme/paraphrase en FR si la source est en anglais).
- 1 à {top_recos} recommandations maximum.
- Chaque recommandation contient: titre + 1 à 2 phrases de justification + au moins 1 citation [doc_key|chunk_id].
- Les citations doivent provenir des SOURCES fournies.
- Tu peux garder 0 à 1 phrase d'introduction, mais pas plus.
- N'ajoute pas de contenu non justifié par les SOURCES.

Réécris la réponse complète maintenant, en restant clair et concis.
"""


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class QueryItem:
    query_id: int
    intent: str
    query_text: str


@dataclass
class Evidence:
    doc_key: str
    chunk_id: int
    score: float
    source: str
    entity: str  # "S:<series_id>" or "K:<kitsu_id>"
    chunk_text: str
    is_fr: bool


# ----------------------------
# Small helpers
# ----------------------------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def fetch_all(conn, sql: str, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def fetch_one(conn, sql: str, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


# ----------------------------
# Language heuristics (very simple, fast)
# ----------------------------
FR_TOKENS = {
    "le", "la", "les", "des", "un", "une", "et", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quoi", "dans", "sur", "avec", "sans", "pour", "pas", "plus",
    "très", "trop", "comme", "au", "aux", "du", "de", "ce", "cet", "cette", "ces",
    "est", "sont", "été", "être", "fait", "faire", "a", "ont", "avait", "avons",
}
EN_TOKENS = {
    "the", "and", "with", "without", "this", "that", "these", "those", "is", "are",
    "was", "were", "be", "been", "to", "from", "of", "in", "on", "for", "not", "more",
    "very", "too", "like", "as", "into", "about", "story", "volume",
}

_word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']{2,}")


def fr_likeness_score(text: str) -> float:
    toks = [t.lower() for t in _word_re.findall(text)]
    if not toks:
        return 0.0
    fr = sum(1 for t in toks if t in FR_TOKENS)
    en = sum(1 for t in toks if t in EN_TOKENS)
    return (fr - en) / max(1, len(toks))


def is_mostly_french(text: str) -> bool:
    return fr_likeness_score(text) >= 0.01


# ----------------------------
# Load FAISS + meta + retrieval embed model
# ----------------------------
def load_faiss_bundle(conn, run_id: str):
    row = fetch_one(
        conn,
        """
        SELECT em.model_name, fi.index_path, fi.meta_path
        FROM bench.embedding_runs r
        JOIN bench.embedding_models em ON em.model_id = r.model_id
        JOIN bench.faiss_indexes fi ON fi.run_id = r.run_id
        WHERE r.run_id = %s
        """,
        (run_id,),
    )
    if not row:
        raise RuntimeError(f"Cannot find retrieval run/index in DB for run_id={run_id!r}. Check bench.faiss_indexes.")

    embed_model_name, index_path, meta_path = row

    if not os.path.exists(index_path):
        raise RuntimeError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"FAISS meta file not found: {meta_path}")

    index = faiss.read_index(index_path)

    chunk_ids: List[int] = []
    doc_keys: List[str] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_ids.append(int(obj["chunk_id"]))
            doc_keys.append(str(obj["doc_key"]))

    chunk_id_arr = np.asarray(chunk_ids, dtype=np.int64)
    doc_key_arr = np.asarray(doc_keys, dtype=object)
    return embed_model_name, index, chunk_id_arr, doc_key_arr, index_path, meta_path


# ----------------------------
# DB: maps and query/qrels
# ----------------------------
def load_doc_key_to_entity(conn) -> Dict[str, str]:
    rows = fetch_all(
        conn,
        """
        SELECT doc_key, series_id, kitsu_id
        FROM bench.corpus_docs
        """,
    )
    m: Dict[str, str] = {}
    for doc_key, series_id, kitsu_id in rows:
        if series_id is not None:
            m[str(doc_key)] = f"S:{int(series_id)}"
        elif kitsu_id is not None:
            m[str(doc_key)] = f"K:{int(kitsu_id)}"
        else:
            m[str(doc_key)] = "NA"
    return m


def load_doc_key_to_source(conn) -> Dict[str, str]:
    rows = fetch_all(conn, "SELECT doc_key, source FROM bench.corpus_docs")
    return {str(k): str(s) for (k, s) in rows}


def load_queries(conn, split: str) -> List[QueryItem]:
    rows = fetch_all(
        conn,
        """
        SELECT query_id, COALESCE(intent,''), query_text
        FROM bench.queries
        WHERE split = %s
        ORDER BY query_id
        """,
        (split,),
    )
    return [QueryItem(int(qid), str(intent), str(qtext)) for (qid, intent, qtext) in rows]


def load_qrels_entities(conn, doc_key_to_entity: Dict[str, str], split: str) -> Dict[int, Set[str]]:
    # join queries to enforce same split
    rows = fetch_all(
        conn,
        """
        SELECT q.query_id, r.doc_key
        FROM bench.queries q
        JOIN bench.qrels r ON r.query_id = q.query_id
        WHERE q.split = %s AND r.relevance >= 1
        """,
        (split,),
    )
    out: Dict[int, Set[str]] = {}
    for qid, doc_key in rows:
        ent = doc_key_to_entity.get(str(doc_key), "NA")
        if ent != "NA":
            out.setdefault(int(qid), set()).add(ent)
    return out


# ----------------------------
# Retrieval: fetch chunk texts for candidates
# ----------------------------
def fetch_chunks_by_ids(conn, chunk_ids: List[int]) -> Dict[int, str]:
    if not chunk_ids:
        return {}
    # chunk_ids can be large; keep IN list manageable
    out: Dict[int, str] = {}
    CHUNK = 2000
    for i in range(0, len(chunk_ids), CHUNK):
        part = chunk_ids[i:i + CHUNK]
        rows = fetch_all(
            conn,
            """
            SELECT chunk_id, chunk_text
            FROM bench.corpus_chunks
            WHERE chunk_id = ANY(%s)
            """,
            (part,),
        )
        for cid, txt in rows:
            out[int(cid)] = str(txt)
    return out


# ----------------------------
# Evidence selection: FR-first + source preference + diversity
# ----------------------------
SOURCE_PRIORITY = {
    "ms_review": 3,
    "ms_hybrid": 2,
    "kitsu_synopsis": 1,
}

def build_evidences(
    conn,
    query_emb: np.ndarray,
    index: faiss.Index,
    chunk_id_arr: np.ndarray,
    doc_key_arr: np.ndarray,
    doc_key_to_entity: Dict[str, str],
    doc_key_to_source: Dict[str, str],
    top_k_chunks: int,
    max_evidences: int,
) -> List[Evidence]:
    scores, idxs = index.search(query_emb, top_k_chunks)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    # collect candidate chunk ids
    cand_chunk_ids: List[int] = []
    cand: List[Tuple[str, int, float]] = []  # doc_key, chunk_id, score
    for ix, sc in zip(idxs, scores):
        if ix < 0:
            continue
        chunk_id = int(chunk_id_arr[ix])
        doc_key = str(doc_key_arr[ix])
        cand.append((doc_key, chunk_id, float(sc)))
        cand_chunk_ids.append(chunk_id)

    chunk_text_map = fetch_chunks_by_ids(conn, cand_chunk_ids)

    evidences: List[Evidence] = []
    for doc_key, chunk_id, sc in cand:
        txt = chunk_text_map.get(chunk_id, "")
        src = doc_key_to_source.get(doc_key, "unknown")
        ent = doc_key_to_entity.get(doc_key, "NA")
        fr_flag = is_mostly_french(txt)
        evidences.append(Evidence(
            doc_key=doc_key,
            chunk_id=chunk_id,
            score=sc,
            source=src,
            entity=ent,
            chunk_text=txt,
            is_fr=fr_flag,
        ))

    # Sort:
    # - FR first
    # - source priority
    # - then score
    evidences.sort(
        key=lambda e: (
            1 if e.is_fr else 0,
            SOURCE_PRIORITY.get(e.source, 0),
            e.score,
        ),
        reverse=True,
    )

    # Diversity: avoid too many from same entity/doc_key
    picked: List[Evidence] = []
    used_entities: Set[str] = set()
    used_doc_keys: Set[str] = set()

    for e in evidences:
        if len(picked) >= max_evidences:
            break
        if not e.chunk_text:
            continue

        # Always allow at least 1, then diversify
        if e.entity != "NA" and e.entity in used_entities and len(picked) >= 3:
            continue
        if e.doc_key in used_doc_keys and len(picked) >= 2:
            continue

        picked.append(e)
        used_doc_keys.add(e.doc_key)
        if e.entity != "NA":
            used_entities.add(e.entity)

    return picked


def evidences_to_block(evidences: List[Evidence], max_chars: int = 320) -> str:
    lines = []
    for e in evidences:
        snippet = e.chunk_text.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "…"
        lines.append(f"- [{e.doc_key}|{e.chunk_id}] {snippet}")
    return "\n".join(lines)


def evidence_pairs(evidences: List[Evidence]) -> Set[Tuple[str, int]]:
    return {(e.doc_key, e.chunk_id) for e in evidences}


# ----------------------------
# Ollama call
# ----------------------------
def ollama_chat(model: str, system: str, user: str, options: dict) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
        "stream": False,
    }
    r = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # Ollama returns: {"message":{"role":"assistant","content":"..."}, ...}
    return str(data.get("message", {}).get("content", "")).strip()


# ----------------------------
# Parsing output: citations, recommendations, titles -> entities
# ----------------------------
CIT_RE = re.compile(r"\[([^\[\]\|]+)\|(\d+)\]")

def extract_citations(text: str) -> List[Tuple[str, int]]:
    out = []
    for m in CIT_RE.finditer(text):
        out.append((m.group(1).strip(), int(m.group(2))))
    return out


def split_recommendations(text: str) -> List[str]:
    """
    Loose split:
    - prefer bullet / numbering lines
    - otherwise split by blank lines, keep up to TOP_RECOS
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bulletish = [ln for ln in lines if re.match(r"^(\-|\*|•|\d+[\)\.])\s+", ln)]
    if bulletish:
        return bulletish[:TOP_RECOS]

    # group by blank lines (original text)
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if parts:
        # if first part looks like intro, keep it separate
        recos: List[str] = []
        for p in parts:
            recos.append(p.replace("\n", " ").strip())
        # skip intro if too short and no citations
        if recos and len(recos) > 1 and len(extract_citations(recos[0])) == 0 and len(recos[0].split()) <= 18:
            recos = recos[1:]
        return recos[:TOP_RECOS]

    return [text.strip()] if text.strip() else []


def extract_title_candidate(reco_line: str) -> str:
    """
    Heuristic:
    - if quoted "..." or « ... », take it
    - else take the prefix until ':' or ' — ' or ' - ' or first sentence
    - strip numbering/bullets
    """
    s = reco_line.strip()
    s = re.sub(r"^(\-|\*|•|\d+[\)\.])\s+", "", s).strip()

    m = re.search(r"«([^»]{2,120})»", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"\"([^\"]{2,120})\"", s)
    if m:
        return m.group(1).strip()

    # up to separators
    for sep in [":", " — ", " - ", " —", " – ", " –", " —"]:
        if sep in s:
            left = s.split(sep, 1)[0].strip()
            if 2 <= len(left) <= 120:
                return left

    # first sentence chunk
    s2 = re.split(r"[\.!\?]\s+", s, maxsplit=1)[0].strip()
    if 2 <= len(s2) <= 120:
        return s2

    return s[:120].strip()


def map_title_to_entity(conn, title: str) -> Optional[str]:
    """
    Map a recommended title to an entity using bench.corpus_docs.title (preferred).
    Returns "S:<id>" or "K:<id>" or None.
    """
    title = title.strip()
    if not title or len(title) < 2:
        return None

    # Exact-ish match on title column first
    rows = fetch_all(
        conn,
        """
        SELECT series_id, kitsu_id, COUNT(*) AS n
        FROM bench.corpus_docs
        WHERE title IS NOT NULL AND title ILIKE %s
        GROUP BY series_id, kitsu_id
        ORDER BY n DESC
        LIMIT 5
        """,
        (f"%{title}%",),
    )
    if rows:
        series_id, kitsu_id, _n = rows[0]
        if series_id is not None:
            return f"S:{int(series_id)}"
        if kitsu_id is not None:
            return f"K:{int(kitsu_id)}"

    # Fallback: search metadata_json->>'title'
    rows = fetch_all(
        conn,
        """
        SELECT series_id, kitsu_id, COUNT(*) AS n
        FROM bench.corpus_docs
        WHERE (metadata_json->>'title') ILIKE %s
        GROUP BY series_id, kitsu_id
        ORDER BY n DESC
        LIMIT 5
        """,
        (f"%{title}%",),
    )
    if rows:
        series_id, kitsu_id, _n = rows[0]
        if series_id is not None:
            return f"S:{int(series_id)}"
        if kitsu_id is not None:
            return f"K:{int(kitsu_id)}"

    return None


# ----------------------------
# Validations (soft)
# ----------------------------
def check_response_soft(
    response: str,
    evid_pairs: Set[Tuple[str, int]],
    top_recos: int,
) -> Tuple[bool, Dict[str, float]]:
    """
    Soft compliance:
    - French-ish
    - 1..top_recos recommendations (approx)
    - at least 1 valid citation overall
    - citations should be valid pairs (preferably)
    """
    recos = split_recommendations(response)
    reco_ok = 1 <= len(recos) <= top_recos

    fr_ok = is_mostly_french(response)

    cits = extract_citations(response)
    if not cits:
        cit_any_ok = False
        cit_valid_rate = 0.0
    else:
        cit_any_ok = True
        valid = sum(1 for dk, cid in cits if (dk, cid) in evid_pairs)
        cit_valid_rate = valid / len(cits)

    ok = fr_ok and reco_ok and cit_any_ok

    return ok, {
        "fr_ok": 1.0 if fr_ok else 0.0,
        "reco_ok": 1.0 if reco_ok else 0.0,
        "cit_any_ok": 1.0 if cit_any_ok else 0.0,
        "cit_valid_rate": float(cit_valid_rate),
        "n_recos": float(len(recos)),
        "n_citations": float(len(cits)),
    }


# ----------------------------
# Scoring: EntityHit@3 / Recall@3 / CitationValidity
# ----------------------------
def score_entity_metrics(
    rec_entities: List[str],
    rel_entities: Set[str],
) -> Tuple[float, float]:
    rec_set = {e for e in rec_entities if e}
    if not rel_entities:
        return 0.0, 0.0
    hit = 1.0 if (rec_set & rel_entities) else 0.0
    recall = len(rec_set & rel_entities) / len(rel_entities)
    return hit, recall


# ----------------------------
# Main bench loop
# ----------------------------
def main():
    if not RETRIEVAL_RUN_ID:
        raise SystemExit("ERROR: BENCH_RETRIEVAL_RUN_ID is required (export BENCH_RETRIEVAL_RUN_ID=...).")

    ensure_out_dir()

    with psycopg.connect(DB_DSN) as conn:
        conn.autocommit = True

        # Load retrieval bundle
        retrieval_embed_model, index, chunk_id_arr, doc_key_arr, index_path, meta_path = load_faiss_bundle(conn, RETRIEVAL_RUN_ID)
        print(f"[INFO] retrieval_run_id={RETRIEVAL_RUN_ID}")
        print(f"[INFO] index_path={index_path}")
        print(f"[INFO] meta_path ={meta_path}")
        print(f"[INFO] retrieval_embed_model={retrieval_embed_model}")
        print(f"[INFO] TOP_K_CHUNKS={TOP_K_CHUNKS} | MAX_EVIDENCES={MAX_EVIDENCES}")
        print(f"[INFO] num_ctx={OLLAMA_OPTIONS['num_ctx']} | num_predict={OLLAMA_OPTIONS['num_predict']}")
        print(f"[INFO] split={SPLIT} | top_recos={TOP_RECOS}")
        print(f"[INFO] llm_models={LLM_MODELS}")

        # Load embedder (same as retrieval run)
        embedder = SentenceTransformer(retrieval_embed_model, device="cpu")

        # Load maps
        doc_key_to_entity = load_doc_key_to_entity(conn)
        doc_key_to_source = load_doc_key_to_source(conn)

        # Load queries + qrels (entities)
        queries = load_queries(conn, SPLIT)
        qrels_entities = load_qrels_entities(conn, doc_key_to_entity, SPLIT)
        if not queries:
            raise SystemExit(f"ERROR: No queries found for split={SPLIT!r} in bench.queries.")
        if not qrels_entities:
            print("[WARN] No qrels entities found for this split. Entity metrics will be 0. (Check bench.qrels).")

        # Output file
        out_f = open(OUT_JSONL, "w", encoding="utf-8")

        # Accumulators per model
        model_stats: Dict[str, dict] = {m: {
            "latencies": [],
            "entity_hit": [],
            "entity_recall": [],
            "cit_valid": [],
            "fr_ok": [],
            "soft_ok": [],
        } for m in LLM_MODELS}

        for model in LLM_MODELS:
            print(f"\n=== LLM BENCH (V3) {model} ===")
            for qi in queries:
                # Retrieve top chunks for this query
                q_emb = embedder.encode([qi.query_text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                evids = build_evidences(
                    conn=conn,
                    query_emb=q_emb,
                    index=index,
                    chunk_id_arr=chunk_id_arr,
                    doc_key_arr=doc_key_arr,
                    doc_key_to_entity=doc_key_to_entity,
                    doc_key_to_source=doc_key_to_source,
                    top_k_chunks=TOP_K_CHUNKS,
                    max_evidences=MAX_EVIDENCES,
                )
                evid_block = evidences_to_block(evids)
                evid_pairs = evidence_pairs(evids)

                # Prepare prompt
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    query_text=qi.query_text,
                    evidences_block=evid_block,
                    top_recos=TOP_RECOS,
                )

                # Call model (1 try + optional repair)
                t0 = time.time()
                try:
                    resp = ollama_chat(model, SYSTEM_PROMPT, user_prompt, OLLAMA_OPTIONS)
                except Exception as e:
                    latency = time.time() - t0
                    rec = {
                        "version": "v3",
                        "model": model,
                        "query_id": qi.query_id,
                        "intent": qi.intent,
                        "query_text": qi.query_text,
                        "latency_s": latency,
                        "error": str(e),
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(f"[ERR] {model} qid={qi.query_id} intent={qi.intent} error={e}")
                    continue

                ok, checks = check_response_soft(resp, evid_pairs, TOP_RECOS)
                retry = 0

                if not ok:
                    retry = 1
                    repair = REPAIR_PROMPT.format(top_recos=TOP_RECOS)
                    # Provide original response + repair instruction as user content
                    repair_user = (
                        f"{repair}\n\n"
                        f"Demande client:\n{qi.query_text}\n\n"
                        f"SOURCES:\n{evid_block}\n\n"
                        f"Ta réponse précédente:\n{resp}\n"
                    )
                    t1 = time.time()
                    resp2 = ollama_chat(model, SYSTEM_PROMPT, repair_user, OLLAMA_OPTIONS)
                    latency = (time.time() - t0)  # total including repair time
                    resp = resp2
                    ok, checks = check_response_soft(resp, evid_pairs, TOP_RECOS)
                else:
                    latency = time.time() - t0

                # Parse recommendations -> titles -> entities
                recos = split_recommendations(resp)
                titles = [extract_title_candidate(r) for r in recos][:TOP_RECOS]
                rec_entities: List[str] = []
                for t in titles:
                    ent = map_title_to_entity(conn, t)
                    if ent:
                        rec_entities.append(ent)

                rel_ents = sorted(list(qrels_entities.get(qi.query_id, set())))
                hit3, recall3 = score_entity_metrics(rec_entities, set(rel_ents))

                # Citation validity
                cits = extract_citations(resp)
                if cits:
                    valid = sum(1 for dk, cid in cits if (dk, cid) in evid_pairs)
                    cit_valid_rate = valid / len(cits)
                else:
                    cit_valid_rate = 0.0

                # Collect stats
                st = model_stats[model]
                st["latencies"].append(float(latency))
                st["entity_hit"].append(float(hit3))
                st["entity_recall"].append(float(recall3))
                st["cit_valid"].append(float(cit_valid_rate))
                st["fr_ok"].append(float(checks.get("fr_ok", 0.0)))
                st["soft_ok"].append(1.0 if ok else 0.0)

                # Write record
                rec = {
                    "version": "v3",
                    "model": model,
                    "query_id": qi.query_id,
                    "intent": qi.intent,
                    "query_text": qi.query_text,
                    "split": SPLIT,
                    "retrieval_run_id": RETRIEVAL_RUN_ID,
                    "retrieval_embed_model": retrieval_embed_model,
                    "top_k_chunks": TOP_K_CHUNKS,
                    "max_evidences": MAX_EVIDENCES,
                    "latency_s": float(latency),
                    "retry": int(retry),
                    "evidences_sent": [
                        {
                            "doc_key": e.doc_key,
                            "chunk_id": e.chunk_id,
                            "source": e.source,
                            "entity": e.entity,
                            "score": e.score,
                            "is_fr": e.is_fr,
                        }
                        for e in evids
                    ],
                    "response": resp,
                    "parsed_recos": recos[:TOP_RECOS],
                    "parsed_titles": titles,
                    "rec_entities": rec_entities,
                    "rel_entities": rel_ents,
                    "metrics": {
                        "entity_hit@3": float(hit3),
                        "entity_recall@3": float(recall3),
                        "citation_valid_rate": float(cit_valid_rate),
                        "soft_ok": bool(ok),
                        "fr_ok": bool(checks.get("fr_ok", 0.0) >= 1.0),
                        "reco_ok": bool(checks.get("reco_ok", 0.0) >= 1.0),
                        "n_recos": int(checks.get("n_recos", 0)),
                        "n_citations": int(checks.get("n_citations", 0)),
                    },
                    "citations": [{"doc_key": dk, "chunk_id": cid, "valid": (dk, cid) in evid_pairs} for dk, cid in cits],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()

                status = "OK" if ok else "WARN"
                print(f"[{status}] {model} qid={qi.query_id} intent={qi.intent} "
                      f"latency={latency:.2f}s hit@3={hit3:.2f} recall@3={recall3:.2f} cit_valid={cit_valid_rate:.2f} retry={retry}")

        out_f.close()

        # Summary
        print("\n=== SUMMARY (V3) ===")
        for model in LLM_MODELS:
            st = model_stats[model]
            n = len(st["latencies"])
            if n == 0:
                print(f"- {model} | n=0")
                continue

            mean_lat = statistics.mean(st["latencies"])
            med_lat = statistics.median(st["latencies"])
            p95_lat = percentile(st["latencies"], 95)

            mean_hit = statistics.mean(st["entity_hit"]) if st["entity_hit"] else 0.0
            mean_rec = statistics.mean(st["entity_recall"]) if st["entity_recall"] else 0.0
            mean_citv = statistics.mean(st["cit_valid"]) if st["cit_valid"] else 0.0

            soft_ok = statistics.mean(st["soft_ok"]) if st["soft_ok"] else 0.0
            fr_ok = statistics.mean(st["fr_ok"]) if st["fr_ok"] else 0.0

            print(
                f"- {model} | n={n} | "
                f"lat(mean/med/p95)={mean_lat:.2f}/{med_lat:.2f}/{p95_lat:.2f}s | "
                f"EntityHit@3={mean_hit:.3f} | Recall@3={mean_rec:.3f} | "
                f"CitValid={mean_citv:.3f} | soft_ok={soft_ok:.3f} | fr_ok={fr_ok:.3f}"
            )

        print(f"\n[OUT] {OUT_JSONL}")


if __name__ == "__main__":
    main()
