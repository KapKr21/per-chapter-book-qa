# src/preprocess.py
from datasets import load_dataset

class BookPreprocessor:
    def __init__(self, narrative_split="train[:2000]", booksum_split="train[:2000]"):
        """
        Pascal-friendly: default to slices so you can iterate fast.
        Increase to full train later.
        """
        print("Loading datasets...")
        self.narrative_qa = load_dataset("narrativeqa", split=narrative_split)
        self.booksum = load_dataset("kmfoda/booksum", split=booksum_split)

    def list_available_bids(self, limit=20):
        """
        Return bids that have chapter text in the loaded BookSum slice.
        """
        bids = []
        seen = set()
        for ex in self.booksum:
            bid = ex.get("bid")
            chapter = ex.get("chapter", "")
            if bid is None:
                continue
            if bid in seen:
                continue
            if not chapter:
                continue
            bids.append(bid)
            seen.add(bid)
            if len(bids) >= limit:
                break
        return bids

    def align_questions_to_chapters(self, book_bid, max_questions=50):
        """
        Align NarrativeQA questions to BookSum chapters.

        - Uses BookSum column: 'chapter'
        - Robust to int/str bid mismatch
        - Does NOT assume NarrativeQA IDs match BookSum bids
        """

        # ---------- 1) Match BookSum chapters ----------
        def _match_bid(x):
            b = x.get("bid")
            if b is None:
                return False
            return str(b) == str(book_bid)

        book_chapters = self.booksum.filter(_match_bid)

        if len(book_chapters) == 0:
            sample_bids = list({str(self.booksum[i]["bid"]) for i in range(min(20, len(self.booksum)))})
            raise ValueError(
                f"No chapters found for bid={book_bid} in loaded BookSum slice.\n"
                f"Example bids in this slice: {sample_bids}\n"
                f"Try increasing --booksum_split (e.g., train[:20000])."
            )

        # BookSum text column is 'chapter'
        chapters_text = [
            c.get("chapter", "")
            for c in book_chapters
            if c.get("chapter")
        ]

        if not chapters_text:
            raise ValueError(f"Chapters exist for bid={book_bid}, but chapter text is empty.")

        # ---------- 2) Select NarrativeQA questions ----------
        # NOTE:
        # NarrativeQA document IDs DO NOT reliably match BookSum bids.
        # So we simply take a slice of NarrativeQA questions.
        # pseudo-fix: find BookSum title for this bid, then filter narrativeqa by title/doc id
        book_title = book_chapters[0].get("title")  # if available

        def _match_nqa(ex):
            doc = ex.get("document", {}) or {}
            # adapt keys based on dataset schema
            return str(doc.get("title","")).strip().lower() == str(book_title).strip().lower()

        book_questions = self.narrative_qa.filter(_match_nqa)

        aligned_data = []

        for q in book_questions:
            answers = q.get("answers", [])
            if not answers:
                continue

            answer = answers[0].get("text", "")
            if not answer:
                continue

            k = self._find_first_revealing_chapter(answer, chapters_text)
            if k is None:
                aligned_data.append({
                    "question": q.get("question", {}).get("text", ""),
                    "gold_answer": answer,
                    "max_chapter_idx": -1,
                    "unanswerable": True
                })
            else:
                aligned_data.append({
                    "question": q.get("question", {}).get("text", ""),
                    "gold_answer": answer,
                    "max_chapter_idx": k,
                    "unanswerable": False
                })

        if not aligned_data:
            raise ValueError("No valid questions found in NarrativeQA slice.")

        return aligned_data, chapters_text

    def _find_first_revealing_chapter(self, answer, chapters):
        kws = [w.lower() for w in answer.split() if len(w) > 3]
        kws = kws[:8]  # cap to avoid noise
        if not kws:
            return None

        for idx, text in enumerate(chapters):
            t = text.lower()
            hits = sum(1 for k in kws[:3] if k in t)
            if hits >= 2:   # require stronger evidence than "any one keyword"
                return idx

        return None  # <--- IMPORTANT CHANGE
