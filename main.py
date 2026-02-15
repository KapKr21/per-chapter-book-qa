# main.py
import argparse
import sys

from src.preprocess import BookPreprocessor
from src.generator import LongContextGenerator
from src.evaluator import BookEvaluator

# Optional retrieval (only used if --use_retriever is passed)
from src.embedder import BookEmbedder
from src.retriever import ChapterRestrictedRetriever


IDK_FALLBACK = "I don't know based on the given text."


def run_experiment(
    book_bid: str,
    narrative_split: str = "train[:2000]",
    booksum_split: str = "train[:2000]",
    max_questions: int = 25,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct-1M",
    max_new_tokens: int = 64,
    use_retriever: bool = False,
    top_k: int = 3,
):
    # 0) Init components
    prep = BookPreprocessor(narrative_split=narrative_split, booksum_split=booksum_split)
    gen = LongContextGenerator(model_id=model_id)
    evaluator = BookEvaluator()

    # 1) Dataset Alignment
    aligned_questions, all_chapters = prep.align_questions_to_chapters(book_bid, max_questions=max_questions)

    if not all_chapters:
        print(f"[error] No chapters found for bid={book_bid}. Try a different --book_bid or increase --booksum_split.")
        return 1

    if not aligned_questions:
        print(f"[error] No aligned questions found for bid={book_bid}. Increase --narrative_split or try another --book_bid.")
        return 1

    # 1b) Optional retrieval setup
    retriever = None
    if use_retriever:
        print("Building retriever index...")
        embedder = BookEmbedder()
        retriever = ChapterRestrictedRetriever(embedder)
        retriever.build_index(all_chapters)

    results_all = []
    spoiler_flags = 0
    spoiler_denom = 0  # only count spoiler rate over answerable examples

    # 2) Run loop
    for i, entry in enumerate(aligned_questions, start=1):
        q = entry["question"]
        k = entry.get("max_chapter_idx", -1)
        unanswerable = bool(entry.get("unanswerable", False)) or (k == -1)

        # Decide allowed/future context
        if unanswerable:
            # Keep context minimal to avoid accidentally enabling a real answer/spoilers
            max_allowed_k = 0
            future_context = []  # do not evaluate spoilers on unrelated/unanswerable items
            gold = IDK_FALLBACK
        else:
            max_allowed_k = k
            future_context = all_chapters[k + 1 :] if (k + 1) < len(all_chapters) else []
            gold = entry["gold_answer"]

        # Build safe context
        if retriever is not None:
            safe_ids = retriever.retrieve_safe_context(q, max_allowed_k, top_k=top_k)
            safe_context = [all_chapters[cid] for cid in safe_ids] if safe_ids else [all_chapters[0]]
        else:
            # Hard boundary baseline (no retrieval)
            safe_context = all_chapters[: max_allowed_k + 1] if max_allowed_k >= 0 else [all_chapters[0]]

        # Generation
        ans = gen.generate_answer(q, safe_context, max_new_tokens=max_new_tokens)

        # Evaluation
        metrics = evaluator.evaluate(ans, gold, future_context)

        # If unanswerable, spoiler check is meaningless (Q/A not from this book)
        # Force spoiler_violation False to avoid polluting your spoiler rate.
        if unanswerable:
            metrics["spoiler_violation"] = False

        results_all.append(metrics)

        # Spoiler rate denominator should exclude unanswerable examples
        if not unanswerable:
            spoiler_denom += 1
            if metrics.get("spoiler_violation"):
                spoiler_flags += 1

        print(f"\n--- Example {i} ---")
        print(f"k={k} | unanswerable={unanswerable} | Spoiler Safe: {not metrics['spoiler_violation']} | ROUGE-L={metrics['rougeL']:.4f}")
        print(f"Q: {q}")
        print(f"A: {ans}")
        # helpful debug for retrieval
        if retriever is not None:
            print(f"safe_ids: {safe_ids}")

    # 3) Summary
    avg_rouge = sum(r["rougeL"] for r in results_all) / len(results_all)

    # Avoid divide-by-zero if everything is unanswerable
    spoiler_rate = (spoiler_flags / spoiler_denom) if spoiler_denom > 0 else 0.0

    print("\n=== Summary ===")
    print(f"book_bid: {book_bid}")
    print(f"num_examples: {len(results_all)}")
    print(f"avg_rougeL: {avg_rouge:.4f}")
    print(f"spoiler_rate(answerable_only): {spoiler_rate:.4f}")
    print(f"answerable_examples: {spoiler_denom}")
    print(f"unanswerable_examples: {len(results_all) - spoiler_denom}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Per-chapter BookQA experiment runner")
    parser.add_argument("--book_bid", type=str, required=True, help="BookSum bid / NarrativeQA document.id")
    parser.add_argument("--narrative_split", type=str, default="train[:2000]", help="HF split slice for NarrativeQA")
    parser.add_argument("--booksum_split", type=str, default="train[:2000]", help="HF split slice for BookSum")
    parser.add_argument("--max_questions", type=int, default=25, help="Max aligned questions to run")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M", help="HF model id")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation length")

    # Retrieval controls
    parser.add_argument("--use_retriever", action="store_true", help="Use FAISS retriever (recommended)")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k safe chapters to retrieve")

    args = parser.parse_args()

    rc = run_experiment(
        book_bid=args.book_bid,
        narrative_split=args.narrative_split,
        booksum_split=args.booksum_split,
        max_questions=args.max_questions,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        use_retriever=args.use_retriever,
        top_k=args.top_k,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()