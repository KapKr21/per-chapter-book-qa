from datasets import load_dataset

IDK_FALLBACK = "I don't know based on the given text."

class BookPreprocessor:
    def __init__(self, narrative_split="train[:2000]", booksum_split="train[:2000]"):
        print("Loading datasets...")
        # NOTE: your current code uses "narrativeqa" (not "google/narrativeqa")
        self.narrative_qa = load_dataset("narrativeqa", split=narrative_split)
        self.booksum = load_dataset("kmfoda/booksum", split=booksum_split)

    def list_available_bids(self, limit=20):
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

    def align_questions_to_chapters(self, book_bid, max_questions=50, require_match=False):
        """
        Align NarrativeQA questions to BookSum chapters.

        Matching order:
          1) NarrativeQA document id == book_bid  (best if it exists)
          2) NarrativeQA document title == BookSum title
          3) Fallback: scan entire NarrativeQA slice (warn)

        require_match:
          - If True, raise when (1) and (2) fail.
          - If False, allow fallback scanning (default).
        """

        def _match_bid(x):
            b = x.get("bid")
            return b is not None and str(b) == str(book_bid)

        book_chapters = self.booksum.filter(_match_bid)

        if len(book_chapters) == 0:
            sample_bids = list({str(self.booksum[i]["bid"]) for i in range(min(20, len(self.booksum)))})
            raise ValueError(
                f"No chapters found for bid={book_bid} in loaded BookSum slice.\n"
                f"Example bids in this slice: {sample_bids}\n"
                f"Try increasing --booksum_split (e.g., train[:20000])."
            )

        chapters_text = [c.get("chapter", "") for c in book_chapters if c.get("chapter")]
        if not chapters_text:
            raise ValueError(f"Chapters exist for bid={book_bid}, but chapter text is empty.")

        book_title = (book_chapters[0].get("title") or "").strip()
        book_questions = None

        # ---------- 1) Try doc-id match ----------
        def _get_doc_id(ex):
            # NarrativeQA variants store this differently; try several common shapes
            doc = ex.get("document", {}) or {}
            for key in ("id", "doc_id", "document_id", "story_id"):
                if key in doc and doc[key] is not None:
                    return str(doc[key]).strip()
            # sometimes it's at top-level
            for key in ("document_id", "doc_id", "story_id"):
                if key in ex and ex[key] is not None:
                    return str(ex[key]).strip()
            return ""

        bid_str = str(book_bid).strip()

        try:
            filtered = self.narrative_qa.filter(lambda ex: _get_doc_id(ex) == bid_str)
            if len(filtered) > 0:
                book_questions = filtered
        except Exception:
            book_questions = None

        # ---------- 2) Try title match ----------
        if book_questions is None and book_title:
            def _match_title(ex):
                doc = ex.get("document", {}) or {}
                doc_title = (doc.get("title") or doc.get("name") or "").strip().lower()
                return doc_title == book_title.lower()

            try:
                filtered = self.narrative_qa.filter(_match_title)
                if len(filtered) > 0:
                    book_questions = filtered
            except Exception:
                book_questions = None

        # ---------- 3) Decide what to do if still no match ----------
        if book_questions is None:
            msg = (
                f"No NarrativeQA questions matched this book (bid={book_bid}).\n"
                f"BookSum title was: '{book_title}'.\n"
                f"Falling back to scanning NarrativeQA slice (likely mismatched Q/A)."
            )
            if require_match:
                raise ValueError(msg)
            else:
                print("[warn]", msg)

        aligned_data = []

        def _extract_question_text(ex):
            q = ex.get("question")
            if isinstance(q, dict):
                return (q.get("text") or "").strip()
            return (q or "").strip()

        def _extract_answer_text(ex):
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

        def _add_example(q_text, answer_text):
            if not answer_text:
                aligned_data.append({
                    "question": q_text,
                    "gold_answer": IDK_FALLBACK,
                    "max_chapter_idx": -1,
                    "unanswerable": True
                })
                return

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

        source_iter = book_questions if book_questions is not None else self.narrative_qa

        for ex in source_iter:
            q_text = _extract_question_text(ex)
            if not q_text:
                continue
            answer_text = _extract_answer_text(ex)
            _add_example(q_text, answer_text)
            if len(aligned_data) >= max_questions:
                break

        if not aligned_data:
            raise ValueError(
                "No usable questions found in NarrativeQA slice after scanning.\n"
                "Try increasing --narrative_split (e.g., train[:20000])."
            )

        return aligned_data, chapters_text

    def _find_first_revealing_chapter(self, answer, chapters):
        kws = [w.lower() for w in str(answer).split() if len(w) > 3]
        kws = kws[:8]
        if not kws:
            return None

        for idx, text in enumerate(chapters):
            t = text.lower()
            hits = sum(1 for k in kws[:3] if k in t)
            if hits >= 2:
                return idx
        return None