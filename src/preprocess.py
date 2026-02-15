# src/preprocess.py
from datasets import load_dataset

class BookPreprocessor:
    def __init__(self, narrative_split="train[:2000]", booksum_split="train[:2000]"):
        """
        Pascal-friendly: default to slices so you can iterate fast.
        Increase to full train later.
        """
        print("Loading datasets...")
        self.narrative_qa = load_dataset("google/narrativeqa", split=narrative_split)
        self.booksum = load_dataset("kmfoda/booksum", split=booksum_split)

    def list_available_bids(self, limit=20):
        bids = []
        for ex in self.booksum:
            bid = ex.get("bid")
            if bid and bid not in bids:
                bids.append(bid)
            if len(bids) >= limit:
                break
        return bids

    def align_questions_to_chapters(self, book_bid, max_questions=50):
        """
        Align each question to the chapter where the answer is first revealed.
        Returns: aligned_data, chapters_text
        """
        book_chapters = self.booksum.filter(lambda x: x.get("bid") == book_bid)
        chapters_text = [c.get("text", "") for c in book_chapters if c.get("text")]

        if not chapters_text:
            raise ValueError(f"No chapters found for bid={book_bid} in BookSum split.")

        book_questions = self.narrative_qa.filter(
            lambda x: x.get("document", {}).get("id") == book_bid
        )

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