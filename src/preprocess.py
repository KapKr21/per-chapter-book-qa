import re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from datasets import load_dataset

from src.embedder import BookEmbedder

IDK_FALLBACK = "I don't know based on the given text."


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _norm_lower(s: str) -> str:
    return _norm_ws(s).lower()


class BookPreprocessor:
    """
    LiteraryQA-based preprocessor.

    LiteraryQA item schema (key parts):
      - document_id: str
      - title: str
      - text: str  (full book text)
      - qas: list[{question: str, answers: list[str], ...}]
    Source: HuggingFace dataset card. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        narrative_split: str = "train",   # kept for compatibility with your main.py arg name
        booksum_split: str = "train",     # kept for compatibility (unused here)
        dataset_name: str = "sapienzanlp/LiteraryQA",
        chunk_chars: int = 6000,
        chunk_overlap: int = 400,
        semantic_fallback: bool = True,
    ):
        print("Loading datasets...")
        # NOTE: LiteraryQA downloads and preprocesses Gutenberg texts; requires extra deps per dataset card. :contentReference[oaicite:2]{index=2}
        self.ds = load_dataset(dataset_name, split=narrative_split)

        self.chunk_chars = int(chunk_chars)
        self.chunk_overlap = int(chunk_overlap)
        self.semantic_fallback = bool(semantic_fallback)

        # Used only for semantic fallback when answers aren't exact spans
        self.embedder = BookEmbedder()
        try:
            self.embedder.device = "cpu"
            self.embedder.model = self.embedder.model.to("cpu")
        except Exception:
            pass

    def list_available_bids(self, limit: int = 20) -> List[str]:
        """For compatibility with your code: returns document_id list."""
        out, seen = [], set()
        for ex in self.ds:
            did = str(ex.get("document_id") or "")
            if not did or did in seen:
                continue
            out.append(did)
            seen.add(did)
            if len(out) >= limit:
                break
        return out

    def _chunk_text(self, text: str) -> List[str]:
        """
        Sequential chunks = "chapters" for your spoiler constraint.
        Keeps order so 'k' is meaningful.
        """
        t = _norm_ws(text)
        if not t:
            return [""]

        n = len(t)
        chunks = []
        step = max(1, self.chunk_chars - self.chunk_overlap)

        start = 0
        while start < n:
            end = min(n, start + self.chunk_chars)
            chunks.append(t[start:end])
            if end == n:
                break
            start += step

        return chunks

    def _find_k_by_exact_answer(self, answers: List[str], chunks: List[str]) -> Optional[int]:
        """
        Return earliest chunk index that contains any answer string (normalized).
        """
        if not answers or not chunks:
            return None

        chunks_norm = [_norm_lower(c) for c in chunks]
        for a in answers:
            a_norm = _norm_lower(a)
            if not a_norm or len(a_norm) < 4:
                continue
            # try exact substring match
            for idx, c in enumerate(chunks_norm):
                if a_norm in c:
                    return idx
        return None

    def _find_k_by_semantic_question(self, question: str, chunks: List[str], min_sim: float = 0.20) -> Optional[int]:
        """
        Fallback when answers are abstractive: pick earliest chunk whose embedding is similar to question.
        """
        if not question or not chunks:
            return None

        q_vec = self.embedder.embed([question])[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        # embed chunk prefixes (cheap + works well)
        prefixes = [c[:2000] for c in chunks]
        c_vecs = self.embedder.embed(prefixes)
        c_vecs = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-8)

        sims = (c_vecs @ q_vec)
        best = int(np.argmax(sims))
        best_sim = float(sims[best])

        if best_sim < min_sim:
            return None

        # earliest chunk above threshold (spoiler-safe earliest reveal)
        for idx, s in enumerate(sims):
            if float(s) >= min_sim:
                return idx
        return best

    def align_questions_to_chapters(self, book_bid: str, max_questions: int = 25) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Here `book_bid` is treated as LiteraryQA `document_id`.
        Returns:
          aligned_questions: [{question, gold_answer, max_chapter_idx, unanswerable}]
          chapters_text: list[str] (sequential chunks)
        """
        # Find the doc
        ex = None
        for row in self.ds:
            if str(row.get("document_id")) == str(book_bid):
                ex = row
                break

        if ex is None:
            return [], []

        full_text = ex.get("text") or ""
        chapters_text = self._chunk_text(full_text)

        qas = ex.get("qas") or []
        aligned = []

        for qa in qas[: max_questions]:
            q = (qa.get("question") or "").strip()
            answers = qa.get("answers") or []
            answers = [a.strip() for a in answers if isinstance(a, str) and a.strip()]

            if not q:
                continue

            # choose first reference answer as gold (keep list for k search)
            gold = answers[0] if answers else IDK_FALLBACK

            k = self._find_k_by_exact_answer(answers, chapters_text)

            if k is None and self.semantic_fallback:
                k = self._find_k_by_semantic_question(q, chapters_text)

            if k is None:
                aligned.append({
                    "question": q,
                    "gold_answer": IDK_FALLBACK,
                    "max_chapter_idx": -1,
                    "unanswerable": True,
                })
            else:
                aligned.append({
                    "question": q,
                    "gold_answer": gold,
                    "max_chapter_idx": int(k),
                    "unanswerable": False,
                })

        return aligned, chapters_text