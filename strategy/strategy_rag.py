from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.chunking import RecursiveChunker
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "data" / "data_group" / "processed_data"
METADATA_PATH = PROJECT_ROOT / "strategy" / "metadata.json"
COLLECTION_NAME = "tamanh_medical_kb"
CHUNK_SIZE = 500
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def make_qwen_embedder(model_name: str = EMBEDDING_MODEL):
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(text: str) -> list[float]:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    return embed


def load_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    if not metadata_path.exists():
        print(f"  ⚠ metadata.json not found at {metadata_path}")
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def load_and_chunk(
    md_dir: Path,
    chunk_size: int = CHUNK_SIZE,
) -> list[Document]:
    chunker = RecursiveChunker(chunk_size=chunk_size)
    documents: list[Document] = []
    meta_map = load_metadata()

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {md_dir}")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = chunker.chunk(text)
        file_meta = meta_map.get(md_file.stem, {})

        for idx, chunk in enumerate(chunks):
            doc = Document(
                id=f"{md_file.stem}",
                content=chunk,
                metadata={
                    "source_file": md_file.name,
                    "doc_id": md_file.stem,
                    "chunk_index": idx,
                    "disease_name": file_meta.get("disease_name", ""),
                    "category": file_meta.get("category", ""),
                },
            )
            documents.append(doc)

        print(f"  {md_file.name}: {len(chunks)} chunks  "
              f"[{file_meta.get('disease_name', '?')} | {file_meta.get('category', '?')}]")

    return documents


def build_store(
    documents: list[Document],
    embedding_fn,
    collection_name: str = COLLECTION_NAME,
) -> EmbeddingStore:
    store = EmbeddingStore(
        collection_name=collection_name,
        embedding_fn=embedding_fn,
    )

    BATCH_SIZE = 32
    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        store.add_documents(batch)
        print(f"  Indexed {min(start + BATCH_SIZE, total)}/{total} chunks")

    print(f"Collection '{collection_name}' size: {store.get_collection_size()}")
    return store


def openai_llm(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý y tế thông minh. "
                    "Trả lời bằng ngôn ngữ phù hợp, chỉ dựa trên ngữ cảnh được cung cấp. "
                    "Nếu không đủ thông tin, hãy nói rõ."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def _build_category_index(meta_map: dict) -> list[tuple[str, str]]:
    """
    Build a list of (keyword_lower, category) pairs for matching.

    For each entry we add:
      - The full category name  (e.g. "tiêu hóa - gan mật")
      - Sub-parts split by " - " (e.g. "tiêu hóa", "gan mật")
      - The disease_name        (e.g. "ăn không tiêu")
        → maps back to that disease's category

    Sorted longest-first so "tiêu hóa - hậu môn trực tràng" matches
    before plain "tiêu hóa".
    """
    index: list[tuple[str, str]] = []
    seen = set()

    for _stem, info in meta_map.items():
        category = info.get("category", "")
        disease = info.get("disease_name", "")

        if category:
            key = category.lower()
            if key not in seen:
                index.append((key, category))
                seen.add(key)
            for part in category.split(" - "):
                part_key = part.strip().lower()
                if part_key and part_key not in seen:
                    index.append((part_key, category))
                    seen.add(part_key)

        if disease and category:
            d_key = disease.lower()
            if d_key not in seen:
                index.append((d_key, category))
                seen.add(d_key)

    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


def infer_category_filter(question: str, meta_map: dict) -> dict | None:
    q_lower = question.lower()
    cat_index = _build_category_index(meta_map)

    for keyword, category in cat_index:
        if keyword in q_lower:
            return {"category": category}

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Medical RAG pipeline")
    parser.add_argument(
        "--md-dir", type=str, default=str(PROCESSED_DIR),
        help="Directory containing preprocessed .md files",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", type=str, default=None, help="Single question mode")
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    meta_map = load_metadata()

    embed_fn = make_qwen_embedder()

    print(f"\nLoading & chunking .md files from: {md_dir}")
    documents = load_and_chunk(md_dir, chunk_size=args.chunk_size)
    print(f"Total chunks: {len(documents)}\n")

    if meta_map:
        print("Auto-filter targets (detected from metadata.json):")
        for stem, info in meta_map.items():
            print(f"  {info.get('disease_name', stem):30s}  [{info.get('category', '')}]")
        print()

    print("Building vector store (ChromaDB) ...")
    store = build_store(documents, embed_fn)
    print()

    agent = KnowledgeBaseAgent(store=store, llm_fn=openai_llm)

    if args.question:
        auto_filter = infer_category_filter(args.question, meta_map)
        if auto_filter:
            print(f"  Auto-filter detected: {auto_filter}")
        answer = agent.answer(args.question, top_k=args.top_k, metadata_filter=auto_filter)
        print(f"\nAnswer:\n{answer}")
        return

    print("Medical Knowledge Base — Interactive Mode")
    print("Filter is auto-detected from your question")
    print("Type 'quit' / 'exit' / 'q' to exit")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        auto_filter = infer_category_filter(question, meta_map)
        if auto_filter:
            print(f"Auto-filter: {auto_filter}")
        else:
            print("No specific filter detected — searching all documents")

        print("\nRetrieving relevant chunks ...")
        if auto_filter:
            results = store.search_with_filter(question, top_k=args.top_k, metadata_filter=auto_filter)
        else:
            results = store.search(question, top_k=args.top_k)

        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            disease = meta.get("disease_name", "")
            cat = meta.get("category", "")
            score = r.get("score", 0)
            snippet = r["content"][:120].replace("\n", " ")
            print(f"  [{i}] {disease} | {cat} | score={score:.4f}")
            print(f"       {snippet}...")

        print("\nGenerating answer ...")
        answer = agent.answer(question, top_k=args.top_k, metadata_filter=auto_filter)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
