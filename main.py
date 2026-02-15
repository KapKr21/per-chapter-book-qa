# main.py
import argparse
import sys

from src.preprocess import BookPreprocessor
from src.generator import LongContextGenerator
from src.evaluator import BookEvaluator


def run_experiment(
    book_bid: str,
    narrative_split: str = "train[:2000]",
    booksum_split: str = "train[:2000]",
    max_questions: int = 25,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct-1M",
    max_new_tokens: int = 200,
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

    results_all = []
    spoiler_flags = 0

    # 2) Run loop
    for i, entry in enumerate(aligned_questions, start=1):
        k = entry["max_chapter_idx"]

        # Hard Boundary: Only provide chapters 0..k
        allowed_context = all_chapters[: k + 1]
        future_context = all_chapters[k + 1 :]

        # Long Context Generation
        ans = gen.generate_answer(entry["question"], allowed_context, max_new_tokens=max_new_tokens)

        # Evaluation
        results = evaluator.evaluate(ans, entry["gold_answer"], future_context)
        results_all.append(results)

        if results.get("spoiler_violation"):
            spoiler_flags += 1

        print(f"\n--- Example {i} ---")
        print(f"k={k} | Spoiler Safe: {not results['spoiler_violation']} | ROUGE-L={results['rougeL']:.4f}")
        print(f"Q: {entry['question']}")
        print(f"A: {ans}")

    # 3) Summary
    avg_rouge = sum(r["rougeL"] for r in results_all) / len(results_all)
    spoiler_rate = spoiler_flags / len(results_all)

    print("\n=== Summary ===")
    print(f"book_bid: {book_bid}")
    print(f"num_examples: {len(results_all)}")
    print(f"avg_rougeL: {avg_rouge:.4f}")
    print(f"spoiler_rate: {spoiler_rate:.4f}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Per-chapter BookQA experiment runner")
    parser.add_argument("--book_bid", type=str, required=True, help="BookSum bid / NarrativeQA document.id")
    parser.add_argument("--narrative_split", type=str, default="train[:2000]", help="HF split slice for NarrativeQA")
    parser.add_argument("--booksum_split", type=str, default="train[:2000]", help="HF split slice for BookSum")
    parser.add_argument("--max_questions", type=int, default=25, help="Max aligned questions to run")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M", help="HF model id")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Generation length")
    args = parser.parse_args()

    rc = run_experiment(
        book_bid=args.book_bid,
        narrative_split=args.narrative_split,
        booksum_split=args.booksum_split,
        max_questions=args.max_questions,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()