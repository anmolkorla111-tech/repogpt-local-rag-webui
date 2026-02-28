import json
from pathlib import Path
import numpy as np
import requests

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.npy"
META_PATH = DATA_DIR / "meta.json"

EMBED_URL = "http://localhost:11434/api/embeddings"
CHAT_URL = "http://localhost:11434/api/chat"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"

# 🔥 ULTRA CPU-SAFE LIMITS (instant response target)
TOP_K = 2
FORCE_MAIN_CHARS = 800
MAX_CONTEXT_CHARS = 1800

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def chunk_text(text: str):
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


def embed_query(text: str) -> np.ndarray:
    r = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=240,
    )
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Index not found. Run ingest.py first.")

    vecs = np.load(INDEX_PATH).astype(np.float32)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    meta_json = json.loads(META_PATH.read_text(encoding="utf-8"))
    repo_root = Path(meta_json["repo_root"])
    meta = meta_json["meta"]
    return vecs, meta, repo_root


def get_chunk_from_file(repo_root: Path, rel_path: str, chunk_index: int) -> str:
    fp = repo_root / rel_path
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    chunks = chunk_text(text)
    if 0 <= chunk_index < len(chunks):
        return chunks[chunk_index]
    return ""


def build_context(repo_root: Path, meta, idxs):
    blocks = []
    total = 0

    # Force include main.py (small slice)
    main_fp = repo_root / "main.py"
    if main_fp.exists():
        main_text = main_fp.read_text(encoding="utf-8", errors="ignore")
        main_chunk = main_text[:FORCE_MAIN_CHARS]
        block = f"FILE: main.py\n{main_chunk}\n"
        blocks.append(block)
        total += len(block)

    # Add small retrieved chunks (hard cap)
    for i in idxs:
        m = meta[i]
        rel_path = m["file"]
        chunk = get_chunk_from_file(repo_root, rel_path, m["chunk_index"])
        if not chunk.strip():
            continue

        # keep each retrieved chunk tiny
        chunk = chunk[:600]
        block = f"FILE: {rel_path}\n{chunk}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total += len(block)

    return "\n---\n".join(blocks)


def chat_stream(context: str, question: str):
    system = (
        "You are RepoGPT. Answer using ONLY the provided repo context. "
        "Do NOT guess/infer. "
        "If missing, reply exactly: Not found in repo context. "
        "Keep answer SHORT (3-8 lines). Always cite file paths."
    )

    user = (
        f"Repo context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Rules: Use only context, no guessing, short answer, cite files."
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 1536,     # smaller ctx -> faster
            "num_predict": 220,  # hard output limit -> faster
        },
    }

    with requests.post(CHAT_URL, json=payload, stream=True, timeout=1200) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "message" in obj and "content" in obj["message"]:
                yield obj["message"]["content"]
            if obj.get("done") is True:
                break


def main():
    vecs, meta, repo_root = load_index()
    print("✅ RepoGPT Chat Ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        qv = embed_query(q)
        sims = vecs @ qv
        idxs = np.argsort(-sims)[:min(TOP_K, len(meta))].tolist()

        context = build_context(repo_root, meta, idxs)
        print(f"[debug] context_chars={len(context)}")

        if not context.strip():
            print("\nRepoGPT:\nNot found in repo context.\n")
            continue

        print("\nRepoGPT:")
        for token in chat_stream(context, q):
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()