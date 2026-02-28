import os
import json
from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.npy"
META_PATH = DATA_DIR / "meta.json"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Smaller chunks = faster + enough for your repo size
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Skip heavy/unwanted folders
SKIP_DIRS = {
    "venv", ".venv", "__pycache__", ".git", "node_modules",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "screenshots"
}

# Only index these file types
ALLOWED_EXTS = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".env"}


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


def embed_text(text: str):
    r = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def process_repo(repo_path: str):
    repo_path = Path(repo_path).resolve()

    all_chunks = []
    metadata = []

    for root, dirs, files in os.walk(repo_path):
        # prune skip dirs (important)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            fp = Path(root) / file
            if fp.suffix.lower() not in ALLOWED_EXTS:
                continue

            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            chunks = chunk_text(content)

            rel_path = str(fp.relative_to(repo_path)).replace("\\", "/")
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "file": rel_path,       # store relative path (clean)
                    "chunk_index": i
                })

    if not all_chunks:
        raise SystemExit("No indexable files found. Check path/extensions.")

    print(f"Files/chunks to embed: {len(all_chunks)}")
    print("Generating embeddings...")

    vectors = []
    for chunk in tqdm(all_chunks):
        vectors.append(embed_text(chunk))

    vectors = np.array(vectors, dtype=np.float32)

    DATA_DIR.mkdir(exist_ok=True)
    np.save(INDEX_PATH, vectors)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"repo_root": str(repo_path), "meta": metadata}, f, indent=2)

    print("✅ Index created successfully!")
    print(f"Saved: {INDEX_PATH} and {META_PATH}")


if __name__ == "__main__":
    repo = input("Enter full path of repo: ").strip().strip('"')
    process_repo(repo)