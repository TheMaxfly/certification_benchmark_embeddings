#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import statistics
from string import Template
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import psycopg
import faiss
import requests
from sentence_transformers import SentenceTransformer

# -------------------------
# Env / Config
# -------------------------
DB_DSN = os.environ.get(
    "APIMANGA_DSN",
    "dbname=apimanga user=postgres password=postgres host=localhost port=5432",
)

OUT_DIR = os.environ.get("BENCH_OUT_DIR", "./bench_out")

RETRIEVAL_RUN_ID = os.environ.get("BENCH_RETRIEVAL_RUN_ID", "").strip()
SPLIT = os.environ.get("BENCH_SPLIT", "eval").strip()

TOP_K_CHUNKS = int(os.environ.get("BENCH_TOP_K_CHUNKS", "30"))       # FAISS search top-k
MAX_EVIDENCES = int(os.environ.get("BENCH_MAX_EVIDENCES", "10"))     # evidence lines in prompt
TOP_RECOS = int(os.environ.get("BENCH_TOP_RECOS", "3"))              # 1..3 lines expected

LLM_MODELS_ENV = os.environ.get(
    "BENCH_LLM_MODELS",
    "phi3.5:3.8b-mini-instruct-q4_K_M,ministral-3:3b-instruct-2512-q4_K_M,mistral:7b-instruct-v0.3-q4_K_M"
)
LLM_MODELS = [m.strip() for m in LLM_MODELS_ENV.split(",") if m.strip()]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

OLLAMA_OPTIONS = {
    "temperature": float(os.environ.get("BENCH_TEMP", "0.2")),
    "top_p": float(os.environ.get("BENCH_TOP_P", "0.9")),
    "num_ctx": int(os.environ.get("BENCH_NUM_CTX", "2048")),
    "num_predict": int(os.environ.get("BENCH_NUM_PREDICT", "220")),
}

# -------------------------
# Prompt (V4: mix FR/EN + format strict “3 lignes”)
# -------------------------
SYSTEM_PROMPT = """Tu es un conseiller libraire manga.
Tu dois répondre en français.
Les sources peuvent être en anglais : tu dois traduire/paraphraser en français ce qui est utile.
Base-toi uniquement sur les sources fournies. Si les sources ne suffisent pas, dis-le en respectant le format.
"""

# V6: citations robustes via E-tags (le modèle ne recopie plus doc_key|chunk_id)
USER_PROMPT_TEMPLATE = """Demande client:
${query_text}

SOURCES (ne recopie pas le texte des sources ; utilise uniquement les balises [E#] à la fin des lignes):
${evidences_block}

Règles STRICTES de sortie (aucune exception) :
- EXACTEMENT ${top_recos} lignes.
- Chaque ligne commence par "1) ", "2) ", "3) " (selon le nombre de lignes).
- 1 phrase par ligne (max 3 lignes de texte, pas de liste à puces).
- Chaque ligne se termine par UNE SEULE citation au format [E#] (ex: [E2]).
- Aucun texte en dehors de ces lignes (pas d'intro, pas de conclusion).

Exemple (format attendu) :
1) Une comédie scolaire douce et pleine de taquineries. [E2]
2) Une romance légère portée par des personnages attachants. [E5]
3) Un slice of life apaisant, idéal si tu veux du feel-good. [E1]
"""

REPAIR_PROMPT = """Tu n'as pas respecté les règles. Corrige maintenant.

Rappel STRICT:
- Français uniquement (tu peux paraphraser/traduire des sources en anglais)
- EXACTEMENT ${top_recos} lignes.
- Chaque ligne commence par "1) ", "2) ", "3) "
- 1 phrase par ligne
- Chaque ligne se termine par UNE SEULE citation [E#]
- Aucun texte hors de ces lignes
- Interdit d'utiliser des puces

Réécris la réponse complète maintenant.
"""

# -------------------------
# Small FR/EN heuristics
# -------------------------
FR_STOP = {
    "le","la","les","un","une","des","du","de","d","et","ou","mais","donc","or","ni","car",
    "dans","sur","avec","sans","pour","par","au","aux","ce","cet","cette","ces","son","sa","ses",
    "est","sont","être","été","avoir","a","ont","que","qui","quoi","dont","où","plus","moins",
    "très","pas","comme","en","se","s","il","elle","ils","elles","on","vous","tu","je","nous"
}
EN_STOP = {
    "the","and","or","but","so","in","on","with","without","for","by","to","of","a","an","is","are",
    "was","were","be","been","have","has","had","that","which","who","what","where","more","less",
    "very","not","as","it","you","we","they","he","she","them","his","her"
}

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9']{2,}", (s or "").lower())

def lang_is_frenchish(text: str) -> bool:
    toks = _tokenize(text)
    if not toks:
        return True
    fr = sum(1 for t in toks if t in FR_STOP) + sum(1 for ch in text if ch in "éèêëàâäùûüîïôöç")
    en = sum(1 for t in toks if t in EN_STOP)
    return fr >= en

def response_is_frenchish(text: str) -> bool:
    # Same heuristic but stricter threshold: require FR dominance
    toks = _tokenize(text)
    if not toks:
        return False
    fr = sum(1 for t in toks if t in FR_STOP) + sum(1 for ch in text if ch in "éèêëàâäùûüîïôöç")
    en = sum(1 for t in toks if t in EN_STOP)
    return fr >= (en + 2)

def extract_strong_tokens(query: str) -> Set[str]:
    # Keep longish words, remove stopwords, keep quoted phrases as tokens too
    q = (query or "").strip()
    quoted = re.findall(r'"([^"]+)"', q)
    toks = set()
    for ph in quoted:
        ph = ph.strip().lower()
        if len(ph) >= 4:
            toks.add(ph)
    for t in _tokenize(q):
        if len(t) >= 5 and t not in FR_STOP and t not in EN_STOP:
            toks.add(t)
    return toks

# -------------------------
# DB structures
# -------------------------
@dataclass(frozen=True)
class Entity:
    kind: str  # "S" or "K"
    id: int

    def key(self) -> str:
        return f"{self.kind}:{self.id}"

@dataclass
class QueryRow:
    query_id: int
    intent: str
    query_text: str

@dataclass
class ChunkRow:
    chunk_id: int
    doc_key: str
    chunk_text: str
    series_id: Optional[int]
    kitsu_id: Optional[int]
    title: Optional[str]

    def entity(self) -> Entity:
        if self.series_id is not None:
            return Entity("S", int(self.series_id))
        if self.kitsu_id is not None:
            return Entity("K", int(self.kitsu_id))
        # In your dataset you said mapping missing=0; keep fallback
        return Entity("S", -1)

# -------------------------
# IO helpers
# -------------------------
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

# -------------------------
# Load retrieval run (FAISS + meta) + embed model
# -------------------------
def load_retrieval_assets(conn, run_id: str):
    row = fetch_one(
        conn,
        """
        SELECT r.run_id, em.model_name, fi.index_path, fi.meta_path
        FROM bench.embedding_runs r
        JOIN bench.embedding_models em ON em.model_id = r.model_id
        JOIN bench.faiss_indexes fi ON fi.run_id = r.run_id
        WHERE r.run_id = %s
        """,
        (run_id,),
    )
    if not row:
        raise RuntimeError(f"Retrieval run_id not found or missing faiss index: {run_id!r}")

    _, embed_model_name, index_path, meta_path = row
    if not os.path.exists(index_path):
        raise RuntimeError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"FAISS meta file not found: {meta_path}")

    index = faiss.read_index(index_path)

    # meta.jsonl: one row per FAISS vector, mapping rowid -> {chunk_id, doc_key}
    chunk_ids = []
    doc_keys = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_ids.append(int(obj["chunk_id"]))
            doc_keys.append(str(obj["doc_key"]))

    chunk_id_arr = np.asarray(chunk_ids, dtype=np.int64)
    doc_key_arr = np.asarray(doc_keys, dtype=object)

    return embed_model_name, index_path, meta_path, index, chunk_id_arr, doc_key_arr

# -------------------------
# Load queries & qrels -> entities
# -------------------------
def load_queries(conn, split: str) -> List[QueryRow]:
    rows = fetch_all(
        conn,
        "SELECT query_id, COALESCE(intent,''), query_text FROM bench.queries WHERE split=%s ORDER BY query_id",
        (split,),
    )
    return [QueryRow(int(qid), str(intent), str(qtext)) for (qid, intent, qtext) in rows]

def build_doc_key_to_entity(conn) -> Dict[str, Entity]:
    # From corpus_docs: doc_key -> entity (prefer series_id, else kitsu_id)
    rows = fetch_all(
        conn,
        "SELECT doc_key, series_id, kitsu_id FROM bench.corpus_docs"
    )
    mp: Dict[str, Entity] = {}
    for doc_key, series_id, kitsu_id in rows:
        if series_id is not None:
            mp[str(doc_key)] = Entity("S", int(series_id))
        elif kitsu_id is not None:
            mp[str(doc_key)] = Entity("K", int(kitsu_id))
        else:
            mp[str(doc_key)] = Entity("S", -1)
    return mp

def load_qrels_entities(conn, split: str, doc_key_to_entity: Dict[str, Entity]) -> Dict[int, Set[Entity]]:
    # qrels table doesn't store split; join with queries to filter split
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
    mp: Dict[int, Set[Entity]] = {}
    missing = 0
    for qid, doc_key in rows:
        qid = int(qid)
        dk = str(doc_key)
        ent = doc_key_to_entity.get(dk)
        if ent is None:
            missing += 1
            continue
        mp.setdefault(qid, set()).add(ent)
    return mp

# -------------------------
# Chunk fetch for evidences
# -------------------------
def fetch_chunks(conn, chunk_ids: List[int]) -> Dict[int, ChunkRow]:
    if not chunk_ids:
        return {}
    rows = fetch_all(
        conn,
        """
        SELECT c.chunk_id, c.doc_key, c.chunk_text,
               d.series_id, d.kitsu_id, d.title
        FROM bench.corpus_chunks c
        JOIN bench.corpus_docs d ON d.doc_key = c.doc_key
        WHERE c.chunk_id = ANY(%s)
        """,
        (chunk_ids,),
    )
    out: Dict[int, ChunkRow] = {}
    for chunk_id, doc_key, chunk_text, series_id, kitsu_id, title in rows:
        out[int(chunk_id)] = ChunkRow(
            chunk_id=int(chunk_id),
            doc_key=str(doc_key),
            chunk_text=str(chunk_text or ""),
            series_id=int(series_id) if series_id is not None else None,
            kitsu_id=int(kitsu_id) if kitsu_id is not None else None,
            title=str(title) if title is not None else None,
        )
    return out

# -------------------------
# Evidence selection (V4)
#   - keep rel entity evidence (if present) regardless of FR/EN
#   - then FR-first
#   - never discard strong-token matches even if EN
# -------------------------
def select_evidences(
    query_text: str,
    candidates: List[Tuple[int, float, ChunkRow]],
    max_evidences: int,
) -> List[Tuple[int, float, ChunkRow]]:
    """FR-first selection without killing the most relevant EN chunks.

    Strategy:
    - Always keep the top-1 by similarity.
    - Keep chunks that strongly match query tokens (e.g., quoted titles / distinctive words), even if EN.
    - Then prefer FR-ish chunks.
    - Deduplicate by doc_key when possible (diversify sources), while allowing token-hit duplicates.
    """
    if not candidates:
        return []

    strong = extract_strong_tokens(query_text)

    def token_hit(row: ChunkRow) -> bool:
        if not strong:
            return False
        hay = f"{row.title or ''} {row.chunk_text or ''}".lower()
        return any(t in hay for t in strong)

    # sort by similarity descending
    cands = sorted(candidates, key=lambda x: x[1], reverse=True)

    selected: List[Tuple[int, float, ChunkRow]] = []
    used_docs: set = set()

    def try_add(item, allow_dup_doc: bool = False):
        nonlocal selected, used_docs
        cid, sc, row = item
        if len(selected) >= max_evidences:
            return
        if (row.doc_key in used_docs) and not allow_dup_doc:
            return
        selected.append(item)
        used_docs.add(row.doc_key)

    # 1) Always keep top-1
    try_add(cands[0], allow_dup_doc=True)

    # 2) Token-hit (keep even if EN)
    for it in cands:
        cid, sc, row = it
        if token_hit(row):
            try_add(it, allow_dup_doc=True)

    # 3) FR-first fill
    for it in cands:
        cid, sc, row = it
        if lang_is_frenchish(row.chunk_text):
            try_add(it, allow_dup_doc=False)

    # 4) Fill remaining by similarity
    for it in cands:
        try_add(it, allow_dup_doc=False)

    return selected[:max_evidences]


def build_evidences_block(evids: List[Tuple[int, float, ChunkRow]]) -> Tuple[str, Dict[str, Tuple[str, int]]]:
    """Build evidence block with robust E-tags.
    Returns:
      - evidences_block: lines like 'E1 (doc=... chunk=... score=...): <snippet>'
      - etag_map: {'E1': (doc_key, chunk_id), ...}
    """
    lines: List[str] = []
    etag_map: Dict[str, Tuple[str, int]] = {}
    for i, (cid, sc, row) in enumerate(evids, start=1):
        tag = f"E{i}"
        snippet = (row.chunk_text or "").replace("\n", " ").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        snippet = snippet[:420]
        # Avoid curly braces in prompt rendering safety
        snippet = snippet.replace("{", "(").replace("}", ")")

        title = (row.title or "").strip()
        title = re.sub(r"\s+", " ", title)[:120]
        title = title.replace("{", "(").replace("}", ")")

        # Keep doc/chunk visible for debugging, but model must cite only [E#]
        lines.append(f"{tag} (doc={row.doc_key} chunk={cid} score={sc:.3f}) {title} — {snippet}")
        etag_map[tag] = (row.doc_key, cid)

    return "\n".join(lines), etag_map


# -------------------------
# Ollama call
# -------------------------
def ollama_generate(model: str, system: str, prompt: str, options: dict) -> str:
    """Prefer /api/chat, fallback to /api/generate."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": options,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=600)

    if r.status_code == 404:
        url2 = f"{OLLAMA_HOST}/api/generate"
        payload2 = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "options": options,
            "stream": False,
        }
        r2 = requests.post(url2, json=payload2, timeout=600)
        r2.raise_for_status()
        return r2.json().get("response", "")

    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


# -------------------------
# Output validation & citation parsing
# -------------------------
LINE_RX = re.compile(r"^\s*([1-3])\s*\)\s+(.+)$")
CIT_END_RX = re.compile(r"\[(E\d+)\]\s*[\.!?]?\s*$")

def parse_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _nonempty_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

def extract_end_etags(answer_text: str, top_recos: int) -> List[Optional[str]]:
    """Extract one E-tag per output line (must be at end). Returns list length <= top_recos."""
    lines = _nonempty_lines(answer_text)
    lines = lines[:top_recos]
    out: List[Optional[str]] = []
    for ln in lines:
        m = CIT_END_RX.search(ln)
        out.append(m.group(1) if m else None)
    return out

def citations_validity_etag(etags: List[Optional[str]], etag_map: Dict[str, Tuple[str, int]]) -> float:
    if not etags:
        return 0.0
    ok = 0
    for t in etags:
        if t and t in etag_map:
            ok += 1
    return ok / len(etags)

def cited_entities_from_etags(
    etags: List[Optional[str]],
    etag_map: Dict[str, Tuple[str, int]],
    doc_key_to_entity: Dict[str, str],
) -> List[str]:
    """Map [E#] -> doc_key -> entity (S:<series_id> or K:<kitsu_id>), keeping order."""
    ents: List[str] = []
    for t in etags:
        if not t:
            continue
        tup = etag_map.get(t)
        if not tup:
            continue
        doc_key, _chunk_id = tup
        ent = doc_key_to_entity.get(doc_key)
        if ent:
            ents.append(ent)
    return ents
def check_format(lines: List[str], top_recos: int) -> bool:
    """Strict but LLM-friendly: exactly N lines, numbered 1)..N), each ends with exactly one [E#]."""
    if not lines:
        return False
    if len(lines) != top_recos:
        return False
    expected = 1
    for ln in lines:
        m = LINE_RX.match(ln)
        if not m:
            return False
        num = int(m.group(1))
        if num != expected:
            return False
        expected += 1

        m_end = CIT_END_RX.search(ln)
        if not m_end:
            return False

        tags = re.findall(r"\[(E\d+)\]", ln)
        if len(tags) != 1:
            return False
        if tags[0] != m_end.group(1):
            return False
    return True




def score_entity_metrics(
    rec_entities: List[Entity],
    rel_entities: Set[Entity]
) -> Tuple[float, float]:
    # EntityHit@3: any relevant entity in top recs
    if not rel_entities:
        return 0.0, 0.0
    top_ents = rec_entities[:TOP_RECOS]
    hit = 1.0 if any(e in rel_entities for e in top_ents) else 0.0
    covered = len(set(top_ents) & set(rel_entities))
    recall = covered / max(1, len(rel_entities))
    return hit, recall

# -------------------------
# Main evaluation loop
# -------------------------
def main():
    if not RETRIEVAL_RUN_ID:
        raise SystemExit("ERROR: BENCH_RETRIEVAL_RUN_ID is required (uuid from bench.embedding_runs).")

    ensure_out_dir()

    with psycopg.connect(DB_DSN) as conn:
        conn.autocommit = True

        # Load retrieval assets
        embed_model_name, index_path, meta_path, index, chunk_id_arr, doc_key_arr = load_retrieval_assets(conn, RETRIEVAL_RUN_ID)
        print(f"[INFO] retrieval_run_id={RETRIEVAL_RUN_ID}")
        print(f"[INFO] index_path={index_path}")
        print(f"[INFO] meta_path ={meta_path}")
        print(f"[INFO] retrieval_embed_model={embed_model_name}")
        print(f"[INFO] TOP_K_CHUNKS={TOP_K_CHUNKS} | MAX_EVIDENCES={MAX_EVIDENCES}")
        print(f"[INFO] num_ctx={OLLAMA_OPTIONS['num_ctx']} | num_predict={OLLAMA_OPTIONS['num_predict']}")
        print(f"[INFO] split={SPLIT} | top_recos={TOP_RECOS}")
        print(f"[INFO] llm_models={LLM_MODELS}")

        # SentenceTransformer for retrieval embedding
        st = SentenceTransformer(embed_model_name, device="cpu")

        # Precompute doc_key -> entity
        doc_key_to_entity = build_doc_key_to_entity(conn)
        chunkid_to_dockey = {int(chunk_id_arr[i]): str(doc_key_arr[i]) for i in range(len(chunk_id_arr))}

        # Load queries + qrels entities
        queries = load_queries(conn, SPLIT)
        qrels_entities = load_qrels_entities(conn, SPLIT, doc_key_to_entity)
        print(f"[INFO] queries={len(queries)} | qrels_entities(queries_with_rel)={len(qrels_entities)}")

        out_path = os.path.join(OUT_DIR, "llm_results_v6.jsonl")
        all_rows = []

        # Run per LLM
        for model in LLM_MODELS:
            print(f"\n=== LLM BENCH (V6 cite-based entity) {model} ===")
            latencies = []
            hits = []
            recalls = []
            cit_vals = []
            fmt_ok = 0
            fr_ok = 0
            retries_total = 0

            for q in queries:
                rel_ents = qrels_entities.get(q.query_id, set())

                # Retrieve top chunks via FAISS
                q_emb = st.encode([q.query_text], normalize_embeddings=True)
                q_emb = np.asarray(q_emb, dtype=np.float32)
                scores, idxs = index.search(q_emb, TOP_K_CHUNKS)
                idxs = idxs[0].tolist()
                scores = scores[0].tolist()

                # Map to chunk_ids
                retrieved_chunk_ids = []
                retrieved_scored = []
                for ix, sc in zip(idxs, scores):
                    if ix < 0:
                        continue
                    cid = int(chunk_id_arr[ix])
                    retrieved_chunk_ids.append(cid)
                    retrieved_scored.append((cid, float(sc)))

                # Fetch chunk texts/entities from DB
                chunk_map = fetch_chunks(conn, retrieved_chunk_ids)

                candidates = []
                for cid, sc in retrieved_scored:
                    row = chunk_map.get(cid)
                    if row is None:
                        continue
                    candidates.append((cid, sc, row))

                # Select evidences (FR-first but keep relevant entity and strong tokens)
                evids = select_evidences(q.query_text, candidates, MAX_EVIDENCES)
                evidences_block, etag_map = build_evidences_block(evids)

                # Render prompt safely (avoid str.format issues with braces in sources)
                user_prompt = Template(USER_PROMPT_TEMPLATE).safe_substitute(
                    query_text=q.query_text,
                    evidences_block=evidences_block,
                    top_recos=str(TOP_RECOS),
                )

                repair_prompt = Template(REPAIR_PROMPT).safe_substitute(top_recos=str(TOP_RECOS))


                # Call LLM with repair if needed
                t0 = time.time()
                retry = 0

                resp = ollama_generate(model, SYSTEM_PROMPT, user_prompt, OLLAMA_OPTIONS)

                lines = parse_lines(resp)
                ok_format = check_format(lines, TOP_RECOS)

                if not ok_format:
                    retry = 1
                    repair_user = repair_prompt + "\n\nRéponse à corriger:\n" + resp
                    resp = ollama_generate(model, SYSTEM_PROMPT, repair_user, OLLAMA_OPTIONS)
                    lines = parse_lines(resp)
                    ok_format = check_format(lines, TOP_RECOS)

                dt = time.time() - t0
                latencies.append(dt)
                retries_total += retry

                # Language check (response should be FR-ish)
                is_fr = response_is_frenchish(resp)
                fr_ok += 1 if is_fr else 0
                fmt_ok += 1 if ok_format else 0

                # Citation validity on end-citations only
                
                end_tags = extract_end_etags("\n".join(lines), TOP_RECOS) if lines else []
                cit_valid = citations_validity_etag(end_tags, etag_map) if end_tags else 0.0
                cit_vals.append(cit_valid)

                # V6 scoring: entities derived from cited E-tags
                rec_entities = cited_entities_from_etags(end_tags, etag_map, doc_key_to_entity) if end_tags else []
                hit, recall = score_entity_metrics(rec_entities, rel_ents)
                hits.append(hit)
                recalls.append(recall)
                status = "OK" if ok_format and is_fr else "WARN"
                print(f"[{status}] {model} qid={q.query_id} intent={q.intent} latency={dt:.2f}s hit@{TOP_RECOS}={hit:.2f} recall@{TOP_RECOS}={recall:.2f} cit_valid={cit_valid:.2f} retry={retry}")

                row = {
                    "version": "v5",
                    "retrieval_run_id": RETRIEVAL_RUN_ID,
                    "retrieval_embed_model": embed_model_name,
                    "llm_model": model,
                    "split": SPLIT,
                    "query_id": q.query_id,
                    "intent": q.intent,
                    "query_text": q.query_text,
                    "rel_entities": sorted([e.key() for e in rel_ents]),
                    "latency_s": dt,
                    "format_ok": bool(ok_format),
                    "fr_ok": bool(is_fr),
                    "citation_validity": cit_valid,
                    f"entity_hit@{TOP_RECOS}": hit,
                    f"entity_recall@{TOP_RECOS}": recall,
                    "retry": retry,
                    "response": resp,
                    "end_etags": end_tags,
                    "rec_entities_from_citations": [e.key() for e in rec_entities],
                    "evidences": [
                        {
                            "etag": f"E{i+1}",
                            "doc_key": r.doc_key,
                            "chunk_id": cid,
                            "score": float(sc),
                            "entity": (
                                doc_key_to_entity.get(r.doc_key).key()
                                if doc_key_to_entity.get(r.doc_key)
                                else None
                            ),
                            "is_fr": lang_is_frenchish(r.chunk_text),
                        }
                        for i, (cid, sc, r) in enumerate(evids)
                    ],
                }
                all_rows.append(row)

            # Summary per model
            n = len(latencies) if latencies else 0
            mean_lat = statistics.mean(latencies) if latencies else 0.0
            med_lat = statistics.median(latencies) if latencies else 0.0
            p95_lat = np.percentile(latencies, 95) if latencies else 0.0

            hit_mean = float(np.mean(hits)) if hits else 0.0
            recall_mean = float(np.mean(recalls)) if recalls else 0.0
            cit_mean = float(np.mean(cit_vals)) if cit_vals else 0.0

            print(f"[SUMMARY] {model} | n={n} | lat(mean/med/p95)={mean_lat:.2f}/{med_lat:.2f}/{p95_lat:.2f}s "
                  f"| EntityHit@{TOP_RECOS}={hit_mean:.3f} | Recall@{TOP_RECOS}={recall_mean:.3f} | CitValid={cit_mean:.3f} "
                  f"| fmt_ok={fmt_ok/n if n else 0:.3f} | fr_ok={fr_ok/n if n else 0:.3f} | retries={retries_total}")

        # Write JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[OUT] {out_path}")

if __name__ == "__main__":
    main()
