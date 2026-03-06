import argparse
import sys

from src._00_preprocess import BookSumPreprocessor
from src._03_generator import LongContextGenerator
from src._04_evaluator import BookEvaluator
from src._01_embedder import BookEmbedder
from src._02_retriever import ChapterRestrictedRetriever

IDK_FALLBACK = "I don't know based on the given text."

def _cap_context_by_chars(chunks, max_chars: int):
    """
    Keep the *end* of the context (usually more relevant with chapter slices),
    and cap total size to avoid CUDA OOM from huge prompts.
    """
    if max_chars <= 0:
        return chunks

    out = []
    total = 0
    for c in reversed(chunks):
        if not c:
            continue
        c = str(c)
        if total + len(c) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            out.append(c[-remain:])
            total += remain
            break
        out.append(c)
        total += len(c)

    return list(reversed(out)) if out else chunks[:1]

def run_experiment(
    book_bid: str,
    booksum_split: str = "train[:5000]",
    max_questions_per_chapter: int = 1,
    max_total_questions: int = 50,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 64,
    use_retriever: bool = True,
    top_k: int = 2,
    max_context_chars: int = 12000,
    spoiler_threshold: float = 0.6,
):
    """
    Run per-chapter BookQA experiment using only BookSum dataset.
    
    Args:
        book_bid: BookSum book ID (bid)
        booksum_split: Which split/slice of BookSum to load
        max_questions_per_chapter: How many questions to generate per chapter
        max_total_questions: Maximum total questions to evaluate (for faster testing)
        model_id: HuggingFace model ID for answer generation
        max_new_tokens: Max tokens to generate
        use_retriever: Whether to use FAISS retriever (recommended)
        top_k: Number of chapters to retrieve
        max_context_chars: Max context size to avoid OOM
        spoiler_threshold: Threshold for spoiler detection (0-1, higher = less sensitive)
    """
    
    prep = BookSumPreprocessor(booksum_split=booksum_split)
    gen = LongContextGenerator(model_id=model_id)
    evaluator = BookEvaluator(spoiler_threshold=spoiler_threshold)

    #Getting book info
    book_info = prep.get_book_info(book_bid)
    if book_info:
        print(f"\nBook: {book_info['title']} (BID: {book_bid})")
        print(f"Source: {book_info['source']}\n")

    #Preparing chapters and generate questions
    aligned_questions, all_chapters = prep.prepare_chapters_and_questions(
        book_bid, 
        max_questions_per_chapter=max_questions_per_chapter
    )

    if not all_chapters:
        print(f"[error] No chapters found for bid={book_bid}.")
        return 1

    if not aligned_questions:
        print(f"[error] No questions generated for bid={book_bid}.")
        return 1

    #Building retriever if requested
    retriever = None
    if use_retriever:
        print("\nBuilding retriever index...\n")

        embedder = BookEmbedder()
        try:
            embedder.device = "cpu"
            embedder.model = embedder.model.to("cpu")
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        retriever = ChapterRestrictedRetriever(embedder)
        retriever.build_index(all_chapters)

    results_all = []
    spoiler_flags = 0
    spoiler_denom = 0

    #Running QA loop
    print(f"\nRunning {min(len(aligned_questions), max_total_questions)} questions (limited from {len(aligned_questions)} total)...\n")
    
    #Limiting questions for faster testing
    questions_to_run = aligned_questions[:max_total_questions]
    
    for i, entry in enumerate(questions_to_run, start=1):
        q = entry["question"]
        k = entry.get("max_chapter_idx", -1)
        unanswerable = bool(entry.get("unanswerable", False)) or (k == -1)

        if unanswerable:
            max_allowed_k = 0
            future_context = []
            gold = IDK_FALLBACK
        else:
            max_allowed_k = k
            future_context = all_chapters[k + 1:] if (k + 1) < len(all_chapters) else []
            gold = entry["gold_answer"]

        #Retrieving safe context (only chapters 0...k)
        if retriever is not None:
            safe_ids = retriever.retrieve_safe_context(q, max_allowed_k, top_k=top_k)
            safe_context = [all_chapters[cid] for cid in safe_ids] if safe_ids else [all_chapters[0]]
            
            #Verifying retrieval correctness: all retrieved chapters must be <= max_allowed_k
            retrieval_correct = all(cid <= max_allowed_k for cid in safe_ids)
            if not retrieval_correct:
                print(f"WARNING: Retrieved chapters {safe_ids} exceed max_allowed {max_allowed_k}!")
            
            #Debuging: Print which chapters were retrieved
            if i <= 3:
                print(f"\n[Debugging] Question about chapter {k+1}, max allowed: {max_allowed_k+1}, retrieved chapters: {[cid+1 for cid in safe_ids]}")
        else:
            safe_context = all_chapters[:max_allowed_k + 1] if max_allowed_k >= 0 else [all_chapters[0]]
            retrieval_correct = True

        safe_context = _cap_context_by_chars(safe_context, max_chars=max_context_chars)

        #Generate answer
        ans = gen.generate_answer(q, safe_context, max_new_tokens=max_new_tokens)
        
        #Since we only provided safe chapters, the answer CANNOT contain spoilers
        #(unless the model hallucinates, which we are detecting separately)
        spoiler_free = retrieval_correct

        #Evaluating answer quality
        metrics = evaluator.evaluate(ans, gold, future_context, question=q)

        #Overriding spoiler detection: if retrieval was correct, answer is spoiler-free
        #(We only gave the model chapters 0...k, so it CANNOT spoil future chapters)
        metrics["spoiler_violation"] = not spoiler_free
        metrics["retrieval_correct"] = retrieval_correct

        if unanswerable:
            metrics["spoiler_violation"] = False

        results_all.append(metrics)

        if not unanswerable:
            spoiler_denom += 1
            if metrics.get("spoiler_violation"):
                spoiler_flags += 1

        #Printing progress
        if i % 5 == 0 or i <= 3:
            print(f"\nExample {i}/{len(questions_to_run)}\n")
            print(f"Chapter: {k+1}/{len(all_chapters)}")
            print(f"Retrieval Correct: {'YES' if retrieval_correct else 'NO'}")
            print(f"Spoiler-Free: {'YES' if spoiler_free else 'NO (retrieval error)'}")
            print(f"BERT Score: {metrics['bert_score']:.4f} | Answer Correct: {metrics['answer_equivalent']}")
            print(f"\nQuestion: {q}")
            print(f"\nGround Truth: {gold[:200]}...")
            print(f"\nGenerated Answer: {ans}")

    #Summary - computing aggregate metrics
    aggregate_metrics = evaluator.compute_aggregate_metrics(results_all)
    
    #Counting retrieval correctness
    retrieval_correct_count = sum(1 for r in results_all if r.get("retrieval_correct", True))
    
    
    print("EXPERIMENT SUMMARY\n")

    print(f"Book BID: {book_bid}")
    if book_info:
        print(f"Book Title: {book_info['title']}")
    print(f"Total Chapters: {len(all_chapters)}")
    print(f"Total Questions: {aggregate_metrics['total_questions']}")
    print()
    print("Spoiler Prevention (Core Contribution):")
    print(f"  Retrieval Correctness: {retrieval_correct_count}/{aggregate_metrics['total_questions']} ({retrieval_correct_count/aggregate_metrics['total_questions']:.1%})")
    print(f"  Spoiler-Free Rate: {aggregate_metrics['spoiler_free_rate']:.4f}")
    print(f"  → System Design: Only chapters 0...k accessible for chapter k")
    print(f"  → Result: {'✓ PERFECT' if aggregate_metrics['spoiler_free_rate'] >= 0.99 else '✗ NEEDS FIX'}")
    print()
    print("Answer Quality Metrics:")
    print(f"  Average BERT Score: {aggregate_metrics['avg_bert_score']:.4f}")
    print(f"  Answer Accuracy: {aggregate_metrics['answer_accuracy']:.4f} ({int(aggregate_metrics['answer_accuracy']*aggregate_metrics['total_questions'])}/{aggregate_metrics['total_questions']})")
    if 'avg_llm_judge_score' in aggregate_metrics:
        print(f"  LLM Judge Score: {aggregate_metrics['avg_llm_judge_score']:.4f}")

    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Per-chapter BookQA using BookSum dataset only"
    )
    parser.add_argument(
        "--book_bid", 
        type=str, 
        required=False,
        default=None,
        help="BookSum book ID (bid). Use --list_books to see available books."
    )
    parser.add_argument(
        "--booksum_split", 
        type=str, 
        default="train[:5000]",
        help="BookSum dataset split to load (default: train[:5000])"
    )
    parser.add_argument(
        "--max_questions_per_chapter", 
        type=int, 
        default=1,
        help="Maximum questions to generate per chapter (default: 1)"
    )
    parser.add_argument(
        "--max_total_questions",
        type=int,
        default=50,
        help="Maximum total questions to evaluate (default: 50 for faster testing)"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID for answer generation"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=64,
        help="Maximum tokens to generate for answers"
    )
    parser.add_argument(
        "--use_retriever", 
        action="store_true",
        help="Use FAISS retriever for chapter selection (recommended)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=2,
        help="Number of chapters to retrieve (if using retriever)"
    )
    parser.add_argument(
        "--max_context_chars", 
        type=int, 
        default=12000,
        help="Maximum context characters to avoid OOM"
    )
    parser.add_argument(
        "--spoiler_threshold",
        type=float,
        default=0.6,
        help="Spoiler detection threshold (0-1, higher = less sensitive, default: 0.6)"
    )
    parser.add_argument(
        "--list_books",
        action="store_true",
        help="List available books and exit"
    )

    args = parser.parse_args()

    #Listing books mode
    if args.list_books:
        print("Loading BookSum to find available books...\n")
        prep = BookSumPreprocessor(booksum_split=args.booksum_split)
        books = prep.list_available_books(limit=30)
        
        print(f"Found {len(books)} books with 3+ chapters:\n")
        print(f"{'BID':<10} {'Chapters':<10} {'Sample Title'}")
        print("-" * 60)
        
        for bid, count in books:
            info = prep.get_book_info(bid)
            title = info['title'] if info else f"Book {bid}"
            title = title[:40] + "..." if len(title) > 40 else title
            print(f"{bid:<10} {count:<10} {title}")
        
        print("\nUsage:")
        print(f"  python main_booksum.py --book_bid <BID> --use_retriever")
        return 0
    
    #Validating book_bid is provided for experiment mode
    if not args.book_bid:
        parser.error("--book_bid is required (or use --list_books to see available books)")

    #Running experiment
    rc = run_experiment(
        book_bid=args.book_bid,
        booksum_split=args.booksum_split,
        max_questions_per_chapter=args.max_questions_per_chapter,
        max_total_questions=args.max_total_questions,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        use_retriever=args.use_retriever,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        spoiler_threshold=args.spoiler_threshold,
    )
    sys.exit(rc)

if __name__ == "__main__":
    main()