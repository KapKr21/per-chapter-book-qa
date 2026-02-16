from datasets import load_dataset

IDK_FALLBACK = "I don't know based on the given text."

class BookPreprocessor:
    def __init__(self, 
                 narrative_split="train[:2000]", 
                 booksum_split="train[:2000]"):
        """
        Pascal-friendly: default to slices so you can iterate fast.
        Increase to full train later.
        """
        print("Loading datasets...")
        self.narrative_qa = load_dataset("narrativeqa", split=narrative_split)
        self.booksum = load_dataset("kmfoda/booksum", split=booksum_split)

    def list_available_bids(self, limit=20):
        """Return bids that have chapter text in the loaded BookSum slice."""
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

    def align_questions_to_chapters(self, book_bid, max_questions=50):
        """
        Align NarrativeQA questions to BookSum chapters.

        IMPORTANT:
        - NarrativeQA IDs generally do NOT match BookSum bids.
        - Title matching may or may not work depending on fields present in your BookSum slice.
        - This function will:
            1) try title matching if possible
            2) otherwise (or if it yields nothing) fall back to scanning NarrativeQA slice
               until it collects max_questions.
        - Unlocatable / missing-answer pairs become:
            max_chapter_idx = -1, unanswerable=True, gold_answer=IDK_FALLBACK
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

        def _match_docid(ex):
            doc = ex.get("document", {}) or {}
            return str(doc.get("id", "")).strip() == str(book_bid)

        try:
            filtered = self.narrative_qa.filter(_match_docid)
            if len(filtered) > 0:
                book_questions = filtered
        except Exception:
            pass

        if book_title:
            def _match_nqa(ex):
                doc = ex.get("document", {}) or {}
                doc_title = (doc.get("title") or "").strip().lower()
                return doc_title == book_title.lower()

            try:
                filtered = self.narrative_qa.filter(_match_nqa)
                if len(filtered) > 0:
                    book_questions = filtered
            except Exception:
                book_questions = None

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

            k = self._find_first_revealing_chapter(answer_text, 
                                                   chapters_text)
            if k is None:
                # Skip unrelated / unlocatable Q/A so your experiment isn't dominated by IDK
                return
            aligned_data.append({
                "question": q_text,
                "gold_answer": answer_text,
                "max_chapter_idx": k,
                "unanswerable": False
            })

        if book_questions is None:
            raise ValueError(
                f"No NarrativeQA questions matched this book (bid={book_bid}).\n"
                f"BookSum title was: '{book_title}'.\n"
                f"This means your BookSum book is not aligned to NarrativeQA in this slice.\n"
                f"Pick a different bid or increase --narrative_split to find matches."
            )
        
        source_iter = book_questions

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
        """
        Heuristic: find earliest chapter containing enough keyword evidence from the answer.
        Returns chapter index, or None if answer cannot be located in the book.
        """
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