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
        Return bids that are guaranteed to have >=1 chapter text in the loaded slice.
        """
        bids = []
        seen = set()
        for ex in self.booksum:
            bid = ex.get("bid")
            txt = ex.get("text", "")
            if bid is None:
                continue
            if bid in seen:
                continue
            if not txt:
                continue
            bids.append(bid)
            seen.add(bid)
            if len(bids) >= limit:
                break
        return bids

    def align_questions_to_chapters(self, book_bid, max_questions=50):
        """
        Align questions to the chapter where the answer is first revealed.
        Returns: aligned_data, chapters_text
        """
        # ---- Robust bid normalization (handles "27681" vs 27681) ----
        try:
            bid_int = int(book_bid)
        except Exception:
            bid_int = None

        # Try both comparisons (some datasets store as int, some as str)
        def _match_bid(x):
            b = x.get("bid")
            if b is None:
                return False
            if bid_int is not None and b == bid_int:
                return True
            return str(b) == str(book_bid)

        book_chapters = self.booksum.filter(_match_bid)

        # Debug help if filter fails
        if len(book_chapters) == 0:
            # show a few bids from the currently loaded slice
            sample = [self.booksum[i].get("bid") for i in range(min(20, len(self.booksum)))]
            raise ValueError(
                f"No chapters found for bid={book_bid} in CURRENT BookSum slice.\n"
                f"Example bids in loaded slice: {sample}\n"
                f"Tip: increase booksum_split (e.g., train[:20000]) or choose a bid from this sample."
            )

        chapters_text = [c.get("text", "") for c in book_chapters if c.get("text")]
        if not chapters_text:
            raise ValueError(f"Chapters found for bid={book_bid} but chapter text is empty.")

        # NOTE: NarrativeQA and BookSum IDs do NOT match reliably.
        # For now, just use a small set of NarrativeQA questions without trying to match ids.
        book_questions = self.narrative_qa.select(range(min(max_questions, len(self.narrative_qa))))

        aligned_data = []
        n = min(max_questions, len(book_questions))
        for q in book_questions.select(range(n)):
            answers = q.get("answers", [])
            if not answers:
                continue

            answer = answers[0].get("text", "")
            if not answer:
                continue

            k = self._find_first_revealing_chapter(answer, chapters_text)
            aligned_data.append({
                "question": q.get("question", {}).get("text", ""),
                "gold_answer": answer,
                "max_chapter_idx": k
            })

        return aligned_data, chapters_text


    def _find_first_revealing_chapter(self, answer, chapters):
        # simple keyword heuristic (good enough for v1)
        kws = [w.lower() for w in answer.split() if len(w) > 3]
        for idx, text in enumerate(chapters):
            t = text.lower()
            if any(k in t for k in kws[:3]):
                return idx
        return len(chapters) - 1