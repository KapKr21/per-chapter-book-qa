from src.preprocess import BookPreprocessor
from src.generator import LongContextGenerator
from src.evaluator import BookEvaluator

def run_experiment(book_bid):
    prep = BookPreprocessor()
    gen = LongContextGenerator()
    evaluator = BookEvaluator()

    #1. Dataset Alignment
    aligned_questions, all_chapters = prep.align_questions_to_chapters(book_bid)

    for entry in aligned_questions:
        k = entry['max_chapter_idx']
        
        #2. Hard Boundary: Only provide chapters 0 through k
        allowed_context = all_chapters[:k+1]
        future_context = all_chapters[k+1:]
        
        #3. Long Context Generation (1M Token Window)
        ans = gen.generate_answer(entry['question'], allowed_context)
        
        #4. Evaluation
        results = evaluator.evaluate(ans, entry['gold_answer'], future_context)
        
        print(f"Q: {entry['question']} | Spoiler Safe: {not results['spoiler_violation']}")

if __name__ == "__main__":
    #Example: 'The Great Gatsby' or other BOOKSUM BID
    run_experiment("12345")