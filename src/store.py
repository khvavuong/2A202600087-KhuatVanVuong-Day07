from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self._client = chromadb.Client()

            # Delete any stale collection with the same name to start fresh
            try:
                self._client.delete_collection(name=self._collection_name)
            except Exception:
                pass

            self._collection = self._client.create_collection(
                name=self._collection_name
            )

            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        if not doc or not doc.content.strip():
            raise ValueError("Document must have non-empty text")

        text = doc.content.strip()

        embedding = self._embedding_fn(text)

        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("Embedding function must return list[float]")

        doc_id = f"doc_{self._next_index}"
        self._next_index += 1

        metadata = doc.metadata.copy() if getattr(doc, "metadata", None) else {}
        if hasattr(doc, "id") and doc.id:
            metadata["doc_id"] = doc.id

        return {
            "id": doc_id,
            "content": text,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        if not records:
            return []

        query_embedding = self._embedding_fn(query)

        scored_records = []
        for record in records:
            embedding = record.get("embedding")

            if not embedding:
                continue

            score = compute_similarity(query_embedding, embedding)

            scored_record = {
                **record,
                "score": score,
            }
            scored_records.append(scored_record)

        scored_records.sort(key=lambda x: x["score"], reverse=True)

        return scored_records[:max(1, top_k)]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma and self._collection is not None:
            ids = [r["id"] for r in records]
            documents = [r["content"] for r in records]
            embeddings = [r["embedding"] for r in records]
            metadatas = [r["metadata"] for r in records]

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if not query or not query.strip():
            return []

        if self._use_chroma and self._collection is not None:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
            )

            output = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if "distances" in results else 0.0
                score = 1.0 / (1.0 + distance)
                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output

        # In-memory fallback
        return self._search_records(query, self._store, top_k)
    
    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()

        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None
    ) -> list[dict]:
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma and self._collection is not None:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter
            )

            output = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if "distances" in results else 0.0
                score = 1.0 / (1.0 + distance)
                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output

        filtered = []
        for record in self._store:
            metadata = record.get("metadata", {})

            match = all(metadata.get(k) == v for k, v in metadata_filter.items())
            if match:
                filtered.append(record)

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if not doc_id:
            return False

        removed = False

        # ChromaDB
        if self._use_chroma and self._collection is not None:
            count_before = self._collection.count()
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                return False
            count_after = self._collection.count()
            return count_after < count_before

        # In-memory
        new_store = []
        for record in self._store:
            metadata = record.get("metadata", {})

            if metadata.get("doc_id") == doc_id:
                removed = True
                continue

            new_store.append(record)

        self._store = new_store
        return removed