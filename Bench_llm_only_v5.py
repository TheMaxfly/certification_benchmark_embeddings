#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import statistics
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
Tu ne dois utiliser QUE les informations présentes dans les sources fournies.
Si les sources ne suffisent pas, tu dois le dire en respectant le format.
Important : les citations sont des identifiants à COPIER-COLLER exactement, elles peuvent pointer vers des sources en anglais.
"""

USER_PROMPT_TEMPLATE = """Demande client:
{query_text}

SOURCES:
{evidences_block}

CITATIONS AUTORISÉES (copie-colle exactement UNE citation par ligne, prise dans cette liste) :
{allowed_citations_block}

Règles de réponse (STRICTES) :
- Écris entre 1 et 3 lignes.
- 1 seule phrase par ligne.
- EXACTEMENT 1 citation par ligne, choisie dans "CITATIONS AUTORISÉES".
- La citation doit être le DERNIER élément de la ligne : la ligne doit se terminer par ] (aucun caractère après, pas de point).
- Aucun texte en dehors de ces lignes.
- Interdit : mettre deux citations sur la même ligne.
Réponds maintenant.
"""

REPAIR_PROMPT = """Tu n'as pas respecté les règles. Corrige maintenant.

CITATIONS AUTORISÉES (copie-colle exactement UNE citation par ligne) :
{allowed_citations_block}

Rappel STRICT :
- Français uniquement.
- 1 à 3 lignes.
- 1 seule phrase par ligne.
- EXACTEMENT 1 citation par ligne, prise dans la liste "CITATIONS AUTORISÉES".
- La citation est le DERNIER élément de la ligne : la ligne doit se terminer EXACTEMENT par ] (pas de point, pas d'espace après).
- Aucun texte hors de ces lignes.
- Interdit : mettre deux citations sur la même ligne.

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
    candidates: List[Tuple[int, float, ChunkRow]],   # (chunk_id, score, row)
    rel_entities: Set[Entity],
    max_evidences: int
) -> List[Tuple[int, float, ChunkRow]]:
    strong = extract_strong_tokens(query_text)

    def has_strong_token(txt: str) -> bool:
        low = (txt or "").lower()
        return any(tok in low for tok in strong)

    rel = []
    fr = []
    en = []
    keep_any = []

    for cid, sc, row in candidates:
        is_rel = (row.entity() in rel_entities) if rel_entities else False
        is_fr = lang_is_frenchish(row.chunk_text)
        strong_hit = has_strong_token(row.chunk_text)

        if is_rel:
            rel.append((cid, sc, row))
        elif strong_hit:
            keep_any.append((cid, sc, row))
        elif is_fr:
            fr.append((cid, sc, row))
        else:
            en.append((cid, sc, row))

    # Sort each group by score desc (FAISS IP)
    rel.sort(key=lambda x: x[1], reverse=True)
    keep_any.sort(key=lambda x: x[1], reverse=True)
    fr.sort(key=lambda x: x[1], reverse=True)
    en.sort(key=lambda x: x[1], reverse=True)

    chosen: List[Tuple[int, float, ChunkRow]] = []
    seen = set()

    def add_group(group):
        nonlocal chosen
        for item in group:
            if len(chosen) >= max_evidences:
                return
            if item[0] in seen:
                continue
            chosen.append(item)
            seen.add(item[0])

    # V4 policy:
    # 1) keep up to 2 “relevant entity” evidences if present
    add_group(rel[:2])
    # 2) fill with FR
    add_group(fr)
    # 3) then keep_any (strong-token hits) — not already in FR/rel
    add_group(keep_any)
    # 4) then EN
    add_group(en)

    return chosen[:max_evidences]

def build_evidences_block(evids: List[Tuple[int, float, ChunkRow]]) -> str:
    lines = []
    for cid, sc, row in evids:
        snippet = row.chunk_text.replace("\n", " ").strip()
        if len(snippet) > 420:
            snippet = snippet[:420].rstrip() + "…"
        lines.append(f"- [{row.doc_key}|{cid}] {snippet}")
    return "\n".join(lines)

# -------------------------
# Ollama call
# -------------------------
def ollama_generate(model: str, system: str, prompt: str, options: dict) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "options": options,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

# -------------------------
# Output validation & citation parsing
# -------------------------
LINE_RX = re.compile(r"^\s*([123])\)\s+(.+)$")
CIT_END_RX = re.compile(r"\[([^\|\]]+)\|(\d+)\]\s*$")
CIT_ANY_RX = re.compile(r"\[([^\|\]]+)\|(\d+)\]")

def parse_lines(text: str) -> List[str]:
    # Keep non-empty lines
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines

def check_format(lines: List[str], top_recos: int) -> bool:
    if not lines:
        return False
    if len(lines) > top_recos:
        return False
    # must be 1) 2) 3) sequence starting at 1
    expected = 1
    for ln in lines:
        m = LINE_RX.match(ln)
        if not m:
            return False
        num = int(m.group(1))
        if num != expected:
            return False
        expected += 1
        # must end with a citation
        if not CIT_END_RX.search(ln):
            return False
    return True

def extract_end_citations(lines: List[str]) -> List[Tuple[str,int]]:
    out = []
    for ln in lines:
        m = CIT_END_RX.search(ln)
        if not m:
            out.append(("", -1))
        else:
            out.append((m.group(1).strip(), int(m.group(2))))
    return out

def citations_validity(end_cits: List[Tuple[str,int]], chunkid_to_dockey: Dict[int,str]) -> float:
    if not end_cits:
        return 0.0
    ok = 0
    for doc_key, chunk_id in end_cits:
        if chunk_id in chunkid_to_dockey and chunkid_to_dockey[chunk_id] == doc_key:
            ok += 1
    return ok / len(end_cits)

def cited_entities_from_end_citations(
    end_cits: List[Tuple[str,int]],
    doc_key_to_entity: Dict[str, Entity]
) -> List[Entity]:
    ents = []
    for doc_key, _cid in end_cits:
        ent = doc_key_to_entity.get(doc_key)
        if ent is not None:
            ents.append(ent)
    return ents

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

        out_path = os.path.join(OUT_DIR, "llm_results_v5.jsonl")
        all_rows = []

        # Run per LLM
        for model in LLM_MODELS:
            print(f"\n=== LLM BENCH (V5 cite-based entity) {model} ===")
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
                evids = select_evidences(q.query_text, candidates, rel_ents, MAX_EVIDENCES)
                evidences_block = build_evidences_block(evids)

                # Build allowed citations block (one per line)
                allowed_citations = []
                seen_cits = set()
                for cid, _sc, r in evids:
                    cit = f"[{r.doc_key}|{cid}]"
                    if cit in seen_cits:
                        continue
                    seen_cits.add(cit)
                    allowed_citations.append(cit)
                allowed_citations_block = "\n".join(allowed_citations)

                user_prompt = USER_PROMPT_TEMPLATE.format(
                    query_text=q.query_text,
                    evidences_block=evidences_block,
                    allowed_citations_block=allowed_citations_block,
                )

                # Call LLM with repair if needed
                t0 = time.time()
                retry = 0
                resp = ollama_generate(model, SYSTEM_PROMPT, user_prompt, OLLAMA_OPTIONS)

                lines = parse_lines(resp)
                ok_format = check_format(lines, TOP_RECOS)

                if not ok_format:
                    retry = 1
                    resp2 = ollama_generate(model, SYSTEM_PROMPT, REPAIR_PROMPT.format(allowed_citations_block=allowed_citations_block) + "\n\n" + resp, OLLAMA_OPTIONS)
                    resp = resp2
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
                end_cits = extract_end_citations(lines) if lines else []
                cit_valid = citations_validity(end_cits, chunkid_to_dockey) if end_cits else 0.0
                cit_vals.append(cit_valid)

                # V4 scoring: entities derived from citations
                rec_entities = cited_entities_from_end_citations(end_cits, doc_key_to_entity) if end_cits else []
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
                    "end_citations": [{"doc_key": dk, "chunk_id": cid} for (dk, cid) in end_cits],
                    "rec_entities_from_citations": [e.key() for e in rec_entities],
                    "evidences": [
                        {"doc_key": r.doc_key, "chunk_id": cid, "score": sc, "entity": r.entity().key(), "is_fr": lang_is_frenchish(r.chunk_text)}
                        for (cid, sc, r) in evids
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
