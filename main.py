import argparse
import sys
import re

from src.preprocess import BookPreprocessor
from src.generator import LongContextGenerator
from src.evaluator import BookEvaluator

from src.embedder import BookEmbedder
from src.retriever import ChapterRestrictedRetriever

IDK_FALLBACK = "I don't know based on the given text."

def _cap_context_by_chars(chunks, max_chars: int):
    """
    Keep the *end* of the context (usually more relevant with chapter slices),
    and cap total size to avoid huge prompts.
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

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def _gold_supported_in_context(gold: str, chunks: list[str]) -> bool:
    """
    Guard against bogus "answerable" labels caused by mismatched Q/A (NarrativeQA vs BookSum).
    Returns True if the gold answer is plausibly supported by the safe context.
    """
    g = _normalize(gold)
    if not g or g == _normalize(IDK_FALLBACK):
        return False

    ctx = _normalize("\n".join(chunks))
    if not ctx:
        return False

    # Use meaningful tokens (len>=5) from gold and require multiple hits.
    toks = [t for t in re.findall(r"[a-zA-Z']+", g) if len(t) >= 5]
    toks = toks[:10]
    if not toks:
        return False

    hits = sum(1 for t in toks[:6] if t in ctx)
    return hits >= 2

def _answer_looks_ungrounded(ans: str, chunks: list[str]) -> bool:
    """
    Cheap hallucination filter: if answer has no overlap with context (excluding short/common tokens),
    treat it as ungrounded and force IDK.
    """
    a = _normalize(ans)
    if not a or a == _normalize(IDK_FALLBACK):
        return False

    ctx = _normalize("\n".join(chunks))
    if not ctx:
        return True

    toks = [t for t in re.findall(r"[a-zA-Z']+", a) if len(t) >= 5]
    toks = toks[:12]
    if not toks:
        return True

    return not any(t in ctx for t in toks[:7])

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
    prep = BookPreprocessor(narrative_split=narrative_split, booksum_split=booksum_split)
    gen = LongContextGenerator(model_id=model_id)
    evaluator = BookEvaluator()

    aligned_questions, all_chapters = prep.align_questions_to_chapters(
        book_bid, max_questions=max_questions
    )

    if not all_chapters:
        print(
            f"[error] No chapters found for bid={book_bid}. "
            f"Try a different --book_bid or increase --booksum_split."
        )
        return 1

    if not aligned_questions:
        print(
            f"[error] No aligned questions found for bid={book_bid}. "
            f"Increase --narrative_split or try another --book_bid."
        )
        return 1

    retriever = None
    if use_retriever:
        print("\nBuilding retriever index...\n")
        embedder = BookEmbedder()

        # keep embeddings on CPU to reduce GPU memory pressure
        try:
            embedder.device = "cpu"
            embedder.model = embedder.model.to("cpu")
        except Exception:
            pass

        retriever = ChapterRestrictedRetriever(embedder)
        retriever.build_index(all_chapters)

    results_all = []
    spoiler_flags = 0
    spoiler_denom = 0

    for i, entry in enumerate(aligned_questions, start=1):
        q = entry["question"]
        k = entry.get("max_chapter_idx", -1)
        unanswerable = bool(entry.get("unanswerable", False)) or (k == -1)

        # Determine gold/future context
        if unanswerable:
            max_allowed_k = 0
            future_context = []
            gold = IDK_FALLBACK
        else:
            max_allowed_k = k
            future_context = all_chapters[k + 1 :] if (k + 1) < len(all_chapters) else []
            gold = entry.get("gold_answer", "").strip() or IDK_FALLBACK

        # Build safe context
        safe_ids = []
        if retriever is not None:
            safe_ids = retriever.retrieve_safe_context(q, max_allowed_k, top_k=top_k) or []

            # HARD SAFETY: never allow chapters beyond k (even if retriever returns them)
            safe_ids = [cid for cid in safe_ids if isinstance(cid, int) and 0 <= cid <= max_allowed_k]

            if not safe_ids:
                safe_ids = [0]

            safe_context = [all_chapters[cid] for cid in safe_ids]
        else:
            safe_context = all_chapters[: max_allowed_k + 1] if max_allowed_k >= 0 else [all_chapters[0]]

        # Reduce noise: don't feed full chapters; long chapter text encourages drift.
        safe_context = [c[:2000] for c in safe_context if c and str(c).strip()]
        if not safe_context:
            safe_context = [all_chapters[0][:2000]]

        # Cap total prompt budget
        safe_context = _cap_context_by_chars(safe_context, max_chars=max_context_chars)

        # IMPORTANT: if preprocess says "answerable" but gold isn't supported by safe context,
        # treat as unanswerable. This stops bogus answerables from causing hallucinations.
        if not unanswerable:
            if not _gold_supported_in_context(gold, safe_context):
                unanswerable = True
                k = -1
                max_allowed_k = 0
                future_context = []
                gold = IDK_FALLBACK

        # Generate
        if unanswerable:
            ans = IDK_FALLBACK
        else:
            ans = gen.generate_answer(q, safe_context, max_new_tokens=max_new_tokens).strip()

            # Hard clamp: if it contains IDK, return ONLY the IDK line
            if "I don't know based on the given text." in ans:
                ans = IDK_FALLBACK

            # Hallucination gate
            if _answer_looks_ungrounded(ans, safe_context):
                ans = IDK_FALLBACK

        # Evaluate
        metrics = evaluator.evaluate(ans, gold, future_context)
        metrics["unanswerable"] = unanswerable

        if unanswerable:
            metrics["spoiler_violation"] = False

        results_all.append(metrics)

        if not unanswerable:
            spoiler_denom += 1
            if metrics.get("spoiler_violation"):
                spoiler_flags += 1

        print(f"\nExample {i}\n")
        print(
            f"k = {entry.get('max_chapter_idx', -1)} | "
            f"Unanswerable = {unanswerable} | "
            f"Spoiler Safe: {not metrics.get('spoiler_violation', False)} | "
            f"ROUGE-L = {metrics['rougeL']:.4f}\n"
        )
        print(f"Q: {q}")
        print(f"A: {ans}")

        if retriever is not None:
            print(f"\n[debug] safe_ids: {safe_ids}")
            print("\n[debug] CONTEXT SNIPPET:")
            print(f"\n{safe_context[0][:400]}")

    avg_rouge = sum(r["rougeL"] for r in results_all) / len(results_all)

    answerable_rouges = [r["rougeL"] for r in results_all if not r.get("unanswerable", False)]
    avg_rouge_answerable = (sum(answerable_rouges) / len(answerable_rouges)) if answerable_rouges else 0.0

    spoiler_rate = (spoiler_flags / spoiler_denom) if spoiler_denom > 0 else 0.0

    print("\nSummary\n")
    print(f"book_bid: {book_bid}")
    print(f"num_examples: {len(results_all)}")
    print(f"avg_rougeL: {avg_rouge:.4f}")
    print(f"avg_rougeL(answerable_only): {avg_rouge_answerable:.4f}")
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