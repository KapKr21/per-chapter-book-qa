from sentence_transformers import SentenceTransformer, util
import torch
from typing import Optional

class BookEvaluator:
    """
    Evaluator for per-chapter BookQA with answer equivalence and spoiler detection.
    
    Uses:
    1. BERT-based semantic similarity for answer equivalence
    2. Optional LLM-as-a-judge for answer quality
    3. Semantic similarity for spoiler detection
    """
    
    def __init__(self, 
                 use_llm_judge: bool = False,
                 llm_judge_model: Optional[str] = None,
                 similarity_threshold: float = 0.5,
                 spoiler_threshold: float = 0.6):
        """
        Initialize evaluator.
        
        Args:
            use_llm_judge: Whether to use LLM-as-a-judge (requires API key)
            llm_judge_model: Model to use for LLM judging (e.g., "gpt-4", "claude-3")
            similarity_threshold: Threshold for answer equivalence (0-1)
            spoiler_threshold: Threshold for spoiler detection (0-1) - higher = less sensitive
        """
        print("Loading BERT model for semantic similarity...")
        # Use a model optimized for semantic similarity
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        self.spoiler_threshold = spoiler_threshold
        
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model
        
        if use_llm_judge and not llm_judge_model:
            print("Warning: use_llm_judge=True but no model specified. Disabling LLM judge.")
            self.use_llm_judge = False

    def evaluate(self, prediction, ground_truth, future_chapters, question=None):
        """
        Evaluate answer quality and check for spoilers.
        
        Args:
            prediction: Model-generated answer
            ground_truth: True answer
            future_chapters: List of chapter texts that should NOT be revealed
            question: Optional question text (used for LLM judge)
            
        Returns:
            dict with metrics:
                - bert_score: Semantic similarity between prediction and ground truth
                - answer_equivalent: Boolean, whether answers are semantically equivalent
                - llm_judge_score: Optional LLM judge score (0-1)
                - spoiler_violation: Boolean, whether prediction contains future info
                - spoiler_score: Max similarity with future chapters
        """
        
        # 1. BERT-based Answer Equivalence
        bert_score = self._compute_bert_similarity(prediction, ground_truth)
        answer_equivalent = bert_score >= self.similarity_threshold
        
        # 2. Spoiler Detection (semantic similarity with future chapters)
        spoiler_score, spoiler_violation = self._check_spoilers(prediction, future_chapters)
        
        # 3. Optional LLM-as-a-Judge
        llm_judge_score = None
        if self.use_llm_judge and question:
            llm_judge_score = self._llm_judge(question, prediction, ground_truth)
        
        return {
            "bert_score": bert_score,
            "answer_equivalent": answer_equivalent,
            "llm_judge_score": llm_judge_score,
            "spoiler_violation": spoiler_violation,
            "spoiler_score": spoiler_score,
        }

    def _compute_bert_similarity(self, text1, text2):
        """
        Compute semantic similarity using BERT embeddings.
        Returns cosine similarity score (0-1).
        """
        if not text1 or not text2:
            return 0.0
        
        # Encode texts
        emb1 = self.bert_model.encode(text1, convert_to_tensor=True)
        emb2 = self.bert_model.encode(text2, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.cos_sim(emb1, emb2).item()
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    def _check_spoilers(self, prediction, future_chapters):
        """
        Check if prediction contains information from future chapters.
        
        Returns:
            spoiler_score: Maximum similarity with any future chapter
            spoiler_violation: Boolean indicating if threshold exceeded
        """
        if not future_chapters or not prediction:
            return 0.0, False
        
        # Encode prediction once
        pred_emb = self.bert_model.encode(prediction, convert_to_tensor=True)
        
        max_similarity = 0.0
        
        # Check similarity with each future chapter
        for chapter in future_chapters:
            if not chapter or len(chapter.strip()) < 50:
                continue
            
            # For long chapters, sample chunks to avoid memory issues
            chapter_text = chapter[:5000]  # First 5000 chars
            
            chapter_emb = self.bert_model.encode(chapter_text, convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, chapter_emb).item()
            
            max_similarity = max(max_similarity, similarity)
        
        spoiler_violation = max_similarity > self.spoiler_threshold
        
        return max_similarity, spoiler_violation

    def _llm_judge(self, question, prediction, ground_truth):
        """
        Use LLM-as-a-judge to evaluate answer quality.
        
        Returns score between 0 and 1.
        
        Note: This is a placeholder. You need to implement actual LLM API calls.
        Options:
        - OpenAI API (GPT-4)
        - Anthropic API (Claude)
        - Local LLM (Llama, Mistral)
        """
        if not self.use_llm_judge:
            return None
        
        # Placeholder implementation
        # TODO: Implement actual LLM API call
        print("Warning: LLM judge not fully implemented. Returning None.")
        return None
        
        # Example implementation with OpenAI (uncomment and add API key):
        """
        import openai
        
        prompt = f'''
        Evaluate the quality of the predicted answer compared to the ground truth.
        
        Question: {question}
        Ground Truth Answer: {ground_truth}
        Predicted Answer: {prediction}
        
        Rate the predicted answer on a scale of 0 to 1, where:
        - 1.0 = Perfect match, semantically equivalent
        - 0.7-0.9 = Good answer, captures main points
        - 0.4-0.6 = Partial answer, some correct information
        - 0.0-0.3 = Poor answer, mostly incorrect
        
        Respond with ONLY a number between 0 and 1.
        '''
        
        response = openai.ChatCompletion.create(
            model=self.llm_judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return None
        """

    def batch_evaluate(self, predictions, ground_truths, future_chapters_list, questions=None):
        """
        Evaluate multiple predictions at once.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            future_chapters_list: List of future chapter lists
            questions: Optional list of questions
            
        Returns:
            List of evaluation dictionaries
        """
        results = []
        
        for i, (pred, truth, future) in enumerate(zip(predictions, ground_truths, future_chapters_list)):
            q = questions[i] if questions else None
            result = self.evaluate(pred, truth, future, question=q)
            results.append(result)
        
        return results

    def compute_aggregate_metrics(self, results):
        """
        Compute aggregate metrics from a list of evaluation results.
        
        Args:
            results: List of evaluation dictionaries
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        total = len(results)
        
        avg_bert_score = sum(r["bert_score"] for r in results) / total
        answer_accuracy = sum(1 for r in results if r["answer_equivalent"]) / total
        spoiler_rate = sum(1 for r in results if r["spoiler_violation"]) / total
        avg_spoiler_score = sum(r["spoiler_score"] for r in results) / total
        
        aggregate = {
            "total_questions": total,
            "avg_bert_score": avg_bert_score,
            "answer_accuracy": answer_accuracy,
            "spoiler_rate": spoiler_rate,
            "spoiler_free_rate": 1 - spoiler_rate,
            "avg_spoiler_score": avg_spoiler_score,
        }
        
        # Add LLM judge metrics if available
        llm_scores = [r["llm_judge_score"] for r in results if r["llm_judge_score"] is not None]
        if llm_scores:
            aggregate["avg_llm_judge_score"] = sum(llm_scores) / len(llm_scores)
            aggregate["llm_judge_count"] = len(llm_scores)
        
        return aggregate
