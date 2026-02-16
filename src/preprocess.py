from datasets import load_dataset
import re

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

    def align_questions_to_chapters(self, book_bid: str, max_questions: int = 25):
        # 1) Get BookSum chapters for bid
        book_chapters = [b for b in self.booksum if str(b.get("bid")) == str(book_bid)]
        if not book_chapters:
            return [], []

        chapters_text = [c.get("chapter", "") for c in book_chapters if c.get("chapter")]
        if not chapters_text:
            return [], []

        # 2) Identify the matching NarrativeQA document by text overlap
        # Use a short fingerprint from early BookSum text
        fingerprint = " ".join(chapters_text[0].split())[:1200].lower()

        # score documents by how many fingerprint words appear
        fp_words = [w for w in re.findall(r"[a-zA-Z']+", fingerprint) if len(w) >= 6]
        fp_words = fp_words[:30]  # cap

        def _doc_score(ex):
            doc = ex.get("document", {}) or {}
            txt = (doc.get("text") or "").lower()
            if not txt:
                return 0
            return sum(1 for w in fp_words if w in txt)

        # compute best doc id in this slice
        best_id = None
        best_score = 0
        for ex in self.narrative_qa:
            s = _doc_score(ex)
            if s > best_score:
                best_score = s
                best_id = (ex.get("document", {}) or {}).get("id")

        if not best_id or best_score < 5:
            # Not enough evidence inside this narrative slice
            print(f"[warn] Could not confidently match NarrativeQA document for bid={book_bid}. best_score={best_score}")
            # fallback to returning unanswerables from random questions (old behavior)
            source_iter = self.narrative_qa
        else:
            print(f"[info] Matched NarrativeQA document id='{best_id}' for bid={book_bid} (score={best_score})")
            source_iter = [ex for ex in self.narrative_qa if (ex.get("document", {}) or {}).get("id") == best_id]

        # 3) Build aligned examples
        aligned = []
        for ex in source_iter:
            q_text = ex.get("question", {}).get("text", "") if isinstance(ex.get("question"), dict) else ex.get("question", "")
            q_text = (q_text or "").strip()
            if not q_text:
                continue

            answers = ex.get("answers", [])
            answer_text = ""
            if isinstance(answers, list) and answers:
                a0 = answers[0]
                if isinstance(a0, dict):
                    answer_text = (a0.get("text") or "").strip()
                elif isinstance(a0, str):
                    answer_text = a0.strip()

            if not answer_text:
                aligned.append({
                    "question": q_text,
                    "gold_answer": IDK_FALLBACK,
                    "max_chapter_idx": -1,
                    "unanswerable": True
                })
            else:
                k = self._find_first_revealing_chapter(answer_text, chapters_text)
                if k is None:
                    aligned.append({
                        "question": q_text,
                        "gold_answer": IDK_FALLBACK,
                        "max_chapter_idx": -1,
                        "unanswerable": True
                    })
                else:
                    aligned.append({
                        "question": q_text,
                        "gold_answer": answer_text,
                        "max_chapter_idx": k,
                        "unanswerable": False
                    })

            if len(aligned) >= max_questions:
                break

        return aligned, chapters_text

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