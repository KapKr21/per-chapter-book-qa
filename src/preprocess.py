# src/preprocess.py

import re
from datasets import load_dataset

IDK_FALLBACK = "I don't know based on the given text."


class BookPreprocessor:
    def __init__(self, narrative_split="train[:2000]", booksum_split="train[:2000]"):
        print("Loading datasets...")
        # Keep your dataset names consistent with your environment:
        # - Some environments use "google/narrativeqa", others "narrativeqa"
        # Use the one that works for you.
        try:
            self.narrative_qa = load_dataset("google/narrativeqa", split=narrative_split)
        except Exception:
            self.narrative_qa = load_dataset("narrativeqa", split=narrative_split)

        self.booksum = load_dataset("kmfoda/booksum", split=booksum_split)

    def list_available_bids(self, limit=20):
        """Return first N unique BookSum bids with non-empty chapter text."""
        bids = []
        seen = set()
        for ex in self.booksum:
            bid = ex.get("bid")
            chapter = ex.get("chapter", "")
            if bid is None or bid in seen or not chapter:
                continue
            bids.append(bid)
            seen.add(bid)
            if len(bids) >= limit:
                break
        return bids

    # ----------------------------
    # NarrativeQA helpers
    # ----------------------------
    def _extract_question_text(self, ex) -> str:
        q = ex.get("question")
        if isinstance(q, dict):
            return (q.get("text") or "").strip()
        return (q or "").strip()

    def _extract_answer_text(self, ex) -> str:
        answers = ex.get("answers", [])
        if isinstance(answers, list) and answers:
            a0 = answers[0]
            if isinstance(a0, dict):
                return (a0.get("text") or a0.get("answer") or "").strip()
            if isinstance(a0, str):
                return a0.strip()
        if isinstance(answers, dict):
            return (answers.get("text") or "").strip()
        return ""

    def _doc_id(self, ex) -> str:
        doc = ex.get("document", {}) or {}
        return str(doc.get("id") or "").strip()

    def _doc_text(self, ex) -> str:
        doc = ex.get("document", {}) or {}
        return doc.get("text") or ""

    # ----------------------------
    # Robust doc matching
    # ----------------------------
    def _normalize_ws(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _pick_snippets(self, text: str, n=6, span=180):
        """
        Choose a handful of fixed-length snippets from early in chapter 1.
        Exact snippet hits in NarrativeQA document text are a strong alignment signal.
        """
        t = self._normalize_ws(text)
        if not t:
            return []
        if len(t) < span + 20:
            return [t]

        # Early offsets; stay near the beginning to reduce spoiler risk
        positions = [0, 250, 650, 1100, 1600, 2200, 3000]
        snippets = []
        for p in positions[:n]:
            if p + span <= len(t):
                snippets.append(t[p : p + span])

        # unique
        out = []
        for s in snippets:
            if s and s not in out:
                out.append(s)
        return out

    def _match_nqa_document_by_snippets(self, chapters_text):
        """
        Return (best_doc_id, best_score, second_doc_id, second_score).
        Score is number of exact snippet hits in document text.
        """
        seed = chapters_text[0] if chapters_text else ""
        snips = self._pick_snippets(seed, n=6, span=180)
        if not snips:
            return None, 0, None, 0

        best_id, best_score = None, -1
        second_id, second_score = None, -1

        for ex in self.narrative_qa:
            doc_id = self._doc_id(ex)
            if not doc_id:
                continue
            doc_text = self._doc_text(ex)
            if not doc_text:
                continue

            doc_text_norm = self._normalize_ws(doc_text)
            score = sum(1 for s in snips if s and s in doc_text_norm)

            if score > best_score:
                second_id, second_score = best_id, best_score
                best_id, best_score = doc_id, score
            elif score > second_score:
                second_id, second_score = doc_id, score

        return best_id, best_score, second_id, second_score

    # ----------------------------
    # Chapter reveal heuristic (improved)
    # ----------------------------
    def _find_first_revealing_chapter(self, answer: str, chapters_text: list[str]):
        """
        Find earliest chapter where answer appears.
        - Try exact normalized phrase match first (best for names).
        - Then token overlap heuristic.
        """
        ans = self._normalize_ws(str(answer)).lower()
        if not ans:
            return None

        # 1) exact match (normalized whitespace)
        for idx, ch in enumerate(chapters_text):
            t = self._normalize_ws(ch).lower()
            if ans in t:
                return idx

        # 2) token overlap heuristic (less strict than your old version)
        toks = [w for w in re.findall(r"[a-zA-Z']+", ans) if len(w) >= 4]
        toks = toks[:12]
        if not toks:
            return None

        for idx, ch in enumerate(chapters_text):
            t = self._normalize_ws(ch).lower()
            hits = sum(1 for w in toks[:8] if w in t)
            if hits >= 2:
                return idx

        return None

    # ----------------------------
    # Main alignment
    # ----------------------------
    def align_questions_to_chapters(self, book_bid: str, max_questions: int = 25):
        """
        1) Load BookSum chapters for bid
        2) Match correct NarrativeQA document using exact snippet hits
        3) Take questions ONLY from that doc
        4) Compute max_chapter_idx by locating answer in chapters
        """

        # BookSum chapters for bid
        book_chapters = [b for b in self.booksum if str(b.get("bid")) == str(book_bid)]
        if not book_chapters:
            return [], []

        chapters_text = [c.get("chapter", "") for c in book_chapters if c.get("chapter")]
        if not chapters_text:
            return [], []

        # Match NarrativeQA doc by snippets
        best_id, best_score, second_id, second_score = self._match_nqa_document_by_snippets(chapters_text)

        # Require confidence: at least 2 snippet hits
        if not best_id or best_score < 2:
            print(f"[warn] Could not confidently match a NarrativeQA document for bid={book_bid}. best_score={best_score}")
            # Return all as unanswerable (or return [] to force you to pick another bid)
            aligned = []
            for ex in self.narrative_qa.select(range(min(max_questions, len(self.narrative_qa)))):
                q = self._extract_question_text(ex)
                if not q:
                    continue
                aligned.append({
                    "question": q,
                    "gold_answer": IDK_FALLBACK,
                    "max_chapter_idx": -1,
                    "unanswerable": True
                })
                if len(aligned) >= max_questions:
                    break
            return aligned, chapters_text

        print(f"[info] Matched NarrativeQA document id='{best_id}' for bid={book_bid} (snippet_hits={best_score})")
        if second_id is not None:
            print(f"[info] Runner-up id='{second_id}' (snippet_hits={second_score})")

        # Select only QA for matched doc
        matched = [ex for ex in self.narrative_qa if self._doc_id(ex) == best_id]

        if matched:
            doc_text = (matched[0].get("document", {}) or {}).get("text", "")
            print("\n[debug] doc_text_head:\n", doc_text[:300])
            print("\n[debug] booksum_ch1_head:\n", chapters_text[0][:300])

        if not matched:
            return [], chapters_text

        aligned_data = []
        for ex in matched:
            q_text = self._extract_question_text(ex)
            if not q_text:
                continue

            answer_text = self._extract_answer_text(ex)

            if not answer_text:
                aligned_data.append({
                    "question": q_text,
                    "gold_answer": IDK_FALLBACK,
                    "max_chapter_idx": -1,
                    "unanswerable": True
                })
            else:
                k = self._find_first_revealing_chapter(answer_text, chapters_text)
                if k is None:
                    aligned_data.append({
                        "question": q_text,
                        "gold_answer": IDK_FALLBACK,
                        "max_chapter_idx": -1,
                        "unanswerable": True
                    })
                else:
                    aligned_data.append({
                        "question": q_text,
                        "gold_answer": answer_text,
                        "max_chapter_idx": k,
                        "unanswerable": False
                    })

            if len(aligned_data) >= max_questions:
                break

        return aligned_data, chapters_text