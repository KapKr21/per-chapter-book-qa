# src/retriever.py
import faiss
import numpy as np

class ChapterRestrictedRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chapter_map = []  # maps faiss row -> chapter_id

    def build_index(self, chapters):
        """
        Index all chapters (we apply spoiler constraint at retrieval time).
        """
        embeddings = self.embedder.embed_chapters(chapters)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.chapter_map = list(range(len(chapters)))

    def retrieve_safe_context(self, question, max_allowed_chapter_idx, top_k=3):
        """
        Returns safe chapter IDs (<= max_allowed_chapter_idx).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index(chapters) first.")

        query_vec = self.embedder.embed_query(question)
        query_vec = np.array(query_vec).astype("float32")

        # Search more than needed then filter.
        distances, indices = self.index.search(query_vec, self.index.ntotal)

        safe_ids = []
        for idx in indices[0]:
            # FAISS can return -1 in some edge cases
            if idx < 0:
                continue

            chapter_id = self.chapter_map[idx]
            if chapter_id <= max_allowed_chapter_idx:
                safe_ids.append(chapter_id)

            if len(safe_ids) >= top_k:
                break

        return safe_ids