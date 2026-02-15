import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

class BookPreprocessor:
    def __init__(self):
        #Primary datasets for QA and segmentation
        print("Loading datasets...")
        self.narrative_qa = load_dataset("google/narrativeqa", 
                                         split='train')
        self.booksum = load_dataset("kmfoda/booksum", 
                                    split='train')

    def align_questions_to_chapters(self, book_bid):
        """
        Aligns each question to the chapter where the answer is first revealed.
        This defines the 'reading progress' for each question.
        """
        #Getting all chapters for this book from BOOKSUM
        book_chapters = self.booksum.filter(lambda x: x['bid'] == book_bid)
        chapters_text = [c['text'] for c in book_chapters]
        
        #Getting questions for this book from NarrativeQA
        #Note: 'bid' in BOOKSUM often maps to 'document.id' in NarrativeQA
        book_questions = self.narrative_qa.filter(lambda x: x['document']['id'] == book_bid)
        
        aligned_data = []
        for q in book_questions:
            answer = q['answers'][0]['text']
            #Finding the chapter index (k) where the answer first appears
            k = self._find_first_revealing_chapter(answer, chapters_text)
            
            aligned_data.append({
                "question": q['question']['text'],
                "gold_answer": answer,
                "max_chapter_idx": k
            })
            
        return aligned_data, chapters_text

    def _find_first_revealing_chapter(self, answer, chapters):
        """Simple keyword-based alignment to find the answer's appearance."""
        answer_keywords = [w.lower() for w in answer.split() if len(w) > 3]
        for idx, text in enumerate(chapters):
            if any(key in text.lower() for key in answer_keywords[:3]):
                return idx
        return len(chapters) - 1 #Default to last chapter if not found