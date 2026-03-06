from datasets import load_dataset
import re

IDK_FALLBACK = "I don't know based on the given text."

class BookSumPreprocessor:
    """
    BookSum-only preprocessor for per-chapter QA.
    Generates questions from chapter summaries or uses them as queries.
    """
    def __init__(self, booksum_split="train[:5000]"):
        print("Loading BookSum dataset...")
        self.booksum = load_dataset("kmfoda/booksum", split=booksum_split)
        print(f"Loaded {len(self.booksum)} BookSum entries")

    def list_available_books(self, limit=50):
        """Return book IDs (bids) that have multiple chapters."""
        bid_counts = {}
        for ex in self.booksum:
            bid = ex.get("bid")
            chapter = ex.get("chapter", "")
            if bid and chapter:
                bid_counts[bid] = bid_counts.get(bid, 0) + 1
        
        # Return bids with at least 3 chapters
        books = [(bid, count) for bid, count in bid_counts.items() if count >= 3]
        books.sort(key=lambda x: x[1], reverse=True)  # Sort by chapter count
        return books[:limit]

    def get_book_info(self, book_bid):
        """Get book title and metadata for a given bid."""
        for ex in self.booksum:
            if str(ex.get("bid")) == str(book_bid):
                return {
                    "bid": book_bid,
                    "title": ex.get("summary_name", f"Book {book_bid}"),
                    "source": ex.get("source", "unknown")
                }
        return None

    def prepare_chapters_and_questions(self, book_bid, max_questions_per_chapter=3):
        """
        Prepare chapters and generate questions from summaries.
        
        For each chapter:
        - Extract chapter text (full content)
        - Extract summary (if available)
        - Generate simple questions from summary
        - Enforce spoiler-free constraint: questions for chapter k can only use chapters 0...k
        
        Returns:
            aligned_data: List of {question, gold_answer, max_chapter_idx, unanswerable}
            chapters_text: List of chapter texts
        """
        
        # Filter chapters for this book
        def _match_bid(x):
            return str(x.get("bid")) == str(book_bid)
        
        book_chapters = self.booksum.filter(_match_bid)
        
        if len(book_chapters) == 0:
            available_bids = self.list_available_books(10)
            raise ValueError(
                f"No chapters found for bid={book_bid}.\n"
                f"Available books (bid, chapter_count): {available_bids[:5]}\n"
                f"Use list_available_books() to see all options."
            )
        
        # Extract chapter texts and summaries
        chapters_data = []
        for ex in book_chapters:
            chapter_text = ex.get("chapter", "").strip()
            summary_text = ex.get("summary_text", "").strip()
            
            if chapter_text:  # Only include if chapter has content
                chapters_data.append({
                    "text": chapter_text,
                    "summary": summary_text,
                    "chapter_id": len(chapters_data)
                })
        
        if not chapters_data:
            raise ValueError(f"No valid chapters found for bid={book_bid}")
        
        print(f"Found {len(chapters_data)} chapters for book {book_bid}")
        
        # Generate questions from summaries
        aligned_data = []
        
        for chapter_idx, chapter_info in enumerate(chapters_data):
            summary = chapter_info["summary"]
            
            if not summary or len(summary) < 50:
                # Skip chapters without meaningful summaries
                continue
            
            # Generate questions from this chapter's summary
            questions = self._generate_questions_from_summary(
                summary, 
                chapter_idx,
                max_questions=max_questions_per_chapter
            )
            
            for q_data in questions:
                aligned_data.append({
                    "question": q_data["question"],
                    "gold_answer": q_data["answer"],
                    "max_chapter_idx": chapter_idx,  # Can only use chapters 0...chapter_idx
                    "unanswerable": False,
                    "chapter_summary": summary[:200]  # Store snippet for reference
                })
        
        if not aligned_data:
            raise ValueError(
                f"No questions generated for bid={book_bid}. "
                f"Chapters may lack summaries. Try a different book."
            )
        
        chapters_text = [ch["text"] for ch in chapters_data]
        
        print(f"Generated {len(aligned_data)} questions across {len(chapters_text)} chapters")
        
        return aligned_data, chapters_text

    def _generate_questions_from_summary(self, summary, chapter_idx, max_questions=3):
        """
        Generate simple questions from a chapter summary.
        
        Strategy:
        1. Extract key sentences from summary
        2. Convert statements to questions
        3. Use summary content as answers
        
        For a prototype, we use simple heuristics.
        For production, you could use an LLM to generate better questions.
        """
        questions = []
        
        # Split summary into sentences
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return questions
        
        # Generate questions from first few sentences
        for i, sentence in enumerate(sentences[:max_questions]):
            if len(questions) >= max_questions:
                break
            
            # Simple question generation heuristics
            question, answer = self._sentence_to_question(sentence, chapter_idx)
            
            if question and answer:
                questions.append({
                    "question": question,
                    "answer": answer
                })
        
        # If we didn't generate enough questions, add a generic one
        if len(questions) == 0 and len(summary) > 50:
            questions.append({
                "question": f"What happens in chapter {chapter_idx + 1}?",
                "answer": summary[:300]  # First 300 chars as answer
            })
        
        return questions

    def _sentence_to_question(self, sentence, chapter_idx):
        """
        Convert a statement sentence into a question.
        Simple heuristic-based approach for prototype.
        """
        sentence = sentence.strip()
        
        if len(sentence) < 20:
            return None, None
        
        # Pattern 1: "X does Y" -> "What does X do?"
        # Pattern 2: "X is Y" -> "What is X?" or "Who is X?"
        # Pattern 3: Generic -> "What happens regarding [key phrase]?"
        
        # Extract potential subjects (simple heuristic)
        words = sentence.split()
        
        # Look for character names (capitalized words)
        characters = [w for w in words if w[0].isupper() and len(w) > 2 and w not in ['The', 'A', 'An', 'In', 'At']]
        
        if characters:
            char = characters[0]
            # Generate character-focused question
            if ' is ' in sentence.lower() or ' was ' in sentence.lower():
                question = f"Who is {char} and what is their role?"
                answer = sentence
            elif ' does ' in sentence.lower() or ' did ' in sentence.lower():
                question = f"What does {char} do?"
                answer = sentence
            else:
                question = f"What happens with {char}?"
                answer = sentence
        else:
            # Generic question
            question = f"What is described in this part of the story?"
            answer = sentence
        
        return question, answer

    def _find_first_revealing_chapter(self, answer, chapters):
        """
        Find the earliest chapter that contains evidence for the answer.
        Used for validation and spoiler detection.
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