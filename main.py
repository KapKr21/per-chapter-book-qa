import argparse
import sys

from src.preprocess import BookPreprocessor
from src.generator import LongContextGenerator
from src.evaluator import BookEvaluator

from src.embedder import BookEmbedder
from src.retriever import ChapterRestrictedRetriever

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
    narrative_split: str = "train[:2000]",
    booksum_split: str = "train[:2000]",
    max_questions: int = 25,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 64,
    use_retriever: bool = True,
    top_k: int = 2,
    max_context_chars: int = 12000,
):
    prep = BookPreprocessor(narrative_split=narrative_split, 
                            booksum_split=booksum_split)
    gen = LongContextGenerator(model_id=model_id)
    evaluator = BookEvaluator()

    aligned_questions, all_chapters = prep.align_questions_to_chapters(book_bid, 
                                                                       max_questions=max_questions)

    if not all_chapters:
        print(f"[error] No chapters found for bid={book_bid}. Try a different --book_bid or increase --booksum_split.")
        return 1

    if not aligned_questions:
        print(f"[error] No aligned questions found for bid={book_bid}. Increase --narrative_split or try another --book_bid.")
        return 1

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

    # Run loop
    for i, entry in enumerate(aligned_questions, start=1):
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

        if retriever is not None:
            safe_ids = retriever.retrieve_safe_context(q, max_allowed_k, top_k=top_k)
            safe_context = [all_chapters[cid] for cid in safe_ids] if safe_ids else [all_chapters[0]]
        else:
            safe_context = all_chapters[:max_allowed_k + 1] if max_allowed_k >= 0 else [all_chapters[0]]

        safe_context = _cap_context_by_chars(safe_context, max_chars=max_context_chars)

        ans = gen.generate_answer(q, safe_context, max_new_tokens=max_new_tokens)

        metrics = evaluator.evaluate(ans, gold, future_context)

        if unanswerable:
            metrics["spoiler_violation"] = False

        results_all.append(metrics)

        if not unanswerable:
            spoiler_denom += 1
            if metrics.get("spoiler_violation"):
                spoiler_flags += 1

        print(f"Example {i}\n")
        print(f"k = {k} | unanswerable = {unanswerable} | Spoiler Safe: {not metrics['spoiler_violation']} | ROUGE-L = {metrics['rougeL']:.4f}\n")
        print(f"Q: {q}")
        print(f"A: {ans}\n")
        if retriever is not None:
            #print(f"safe_ids: {safe_ids}")
            print(f"")

    avg_rouge = sum(r["rougeL"] for r in results_all) / len(results_all)
    spoiler_rate = (spoiler_flags / spoiler_denom) if spoiler_denom > 0 else 0.0

    print("Summary\n")
    print(f"book_bid: {book_bid}")
    print(f"num_examples: {len(results_all)}")
    print(f"avg_rougeL: {avg_rouge:.4f}")
    print(f"spoiler_rate(answerable_only): {spoiler_rate:.4f}")
    print(f"answerable_examples: {spoiler_denom}")
    print(f"unanswerable_examples: {len(results_all) - spoiler_denom}")

    return 0

def main():
    parser = argparse.ArgumentParser(description="Per-chapter BookQA experiment runner")
    parser.add_argument("--book_bid", type=str, required=True, help="BookSum bid")
    parser.add_argument("--narrative_split", type=str, default="train[:2000]")
    parser.add_argument("--booksum_split", type=str, default="train[:2000]")
    parser.add_argument("--max_questions", type=int, default=25)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--use_retriever", action="store_true", help="Use FAISS retriever (recommended)")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k retrieved chapters")
    parser.add_argument("--max_context_chars", type=int, default=12000, help="Cap total context chars to avoid OOM")

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
        max_context_chars=args.max_context_chars,
    )
    sys.exit(rc)

if __name__ == "__main__":
    main()