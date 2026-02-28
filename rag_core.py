import json
from pathlib import Path
import numpy as np
import requests

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.npy"
META_PATH = DATA_DIR / "meta.json"

OLLAMA_BASE = "http://localhost:11434"
EMBED_URL = f"{OLLAMA_BASE}/api/embeddings"
CHAT_URL = f"{OLLAMA_BASE}/api/chat"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"

# CPU-safe caps (fast responses)
TOP_K = 2
FORCE_MAIN_CHARS = 800
MAX_CONTEXT_CHARS = 1800

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


_vecs = None
_meta = None
_repo_root = None


def _chunk_text(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def _ensure_index_loaded():
    global _vecs, _meta, _repo_root
    if _vecs is not None:
        return

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Index not found. Run ingest.py first.")

    _vecs = np.load(INDEX_PATH).astype(np.float32)
    _vecs = _vecs / (np.linalg.norm(_vecs, axis=1, keepdims=True) + 1e-12)

    meta_json = json.loads(META_PATH.read_text(encoding="utf-8"))
    _repo_root = Path(meta_json["repo_root"])
    _meta = meta_json["meta"]


def _embed_query(text: str) -> np.ndarray:
    r = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=240,
    )
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def _get_chunk(rel_path: str, chunk_index: int) -> str:
    fp = _repo_root / rel_path
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    chunks = _chunk_text(text)
    if 0 <= chunk_index < len(chunks):
        return chunks[chunk_index]
    return ""


def _build_context(idxs):
    blocks = []
    total = 0
    sources = []

    # force include main.py snippet if it exists
    main_fp = _repo_root / "main.py"
    if main_fp.exists():
        main_text = main_fp.read_text(encoding="utf-8", errors="ignore")
        block = f"FILE: main.py\n{main_text[:FORCE_MAIN_CHARS]}\n"
        blocks.append(block)
        total += len(block)
        sources.append("main.py")

    for i in idxs:
        m = _meta[i]
        rel_path = m["file"]
        chunk = _get_chunk(rel_path, m["chunk_index"])
        if not chunk.strip():
            continue

        chunk = chunk[:600]
        block = f"FILE: {rel_path}\n{chunk}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total += len(block)
        if rel_path not in sources:
            sources.append(rel_path)

    return "\n---\n".join(blocks), sources


def _ask_model(context: str, question: str) -> str:
    system = (
        "You are RepoGPT. Use ONLY the provided repo context. "
        "Do NOT guess. If missing, reply: Not found in repo context. "
        "Keep answer short and cite file paths."
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Repo context:\n{context}\n\nQuestion:\n{question}"},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 1536,
            "num_predict": 220,
        },
    }

    r = requests.post(CHAT_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def answer_question(question: str):
    _ensure_index_loaded()

    qv = _embed_query(question)
    sims = _vecs @ qv
    idxs = np.argsort(-sims)[:min(TOP_K, len(_meta))]

    context, sources = _build_context(idxs)
    if not context.strip():
        return {"answer": "Not found in repo context.", "sources": []}

    answer = _ask_model(context, question)
    return {"answer": answer, "sources": sources}


def health_check():
    # verify ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        r.raise_for_status()
        ok = True
    except Exception as e:
        ok = False
        return {"ok": False, "error": str(e)}

    return {"ok": ok, "models": r.json().get("models", [])}