from sentence_transformers import SentenceTransformer, util
from typing import Optional

class BookEvaluator:
    """
    Evaluator for per-chapter BookQA with answer equivalence and spoiler detection.
    
    Uses:
    1. BERT-based semantic similarity for answer equivalence
    2. Semantic similarity for spoiler detection
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.5,
                 spoiler_threshold: float = 0.6):
        """
        Initialize evaluator.
        
        Args:
            similarity_threshold: Threshold for answer equivalence (0-1)
            spoiler_threshold: Threshold for spoiler detection (0-1) - higher = less sensitive
        """
        print("\nLoading BERT model for semantic similarity...\n")

        #Using a model optimized for semantic similarity
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        self.spoiler_threshold = spoiler_threshold

    def evaluate(self, 
                 prediction, 
                 ground_truth, 
                 future_chapters):
        """
        Evaluate answer quality and check for spoilers.
        
        Args:
            prediction: Model-generated answer
            ground_truth: True answer
            future_chapters: List of chapter texts that should NOT be revealed
            
        Returns:
            dict with metrics:
                - bert_score: Semantic similarity between prediction and ground truth
                - answer_equivalent: Boolean, whether answers are semantically equivalent
                - spoiler_violation: Boolean, whether prediction contains future info
                - spoiler_score: Max similarity with future chapters
        """
        
        #1. BERT-based Answer Equivalence
        bert_score = self._compute_bert_similarity(prediction, 
                                                   ground_truth)
        answer_equivalent = bert_score >= self.similarity_threshold
        
        #2. Spoiler Detection (semantic similarity with future chapters)
        spoiler_score, spoiler_violation = self._check_spoilers(prediction, 
                                                                future_chapters)
        
        return {
            "bert_score": bert_score,
            "answer_equivalent": answer_equivalent,
            "spoiler_violation": spoiler_violation,
            "spoiler_score": spoiler_score,
        }

    def _compute_bert_similarity(self, 
                                 text1, 
                                 text2):
        """
        Compute semantic similarity using BERT embeddings.
        Returns cosine similarity score (0-1).
        """
        if not text1 or not text2:
            return 0.0
        
        #Encoding texts
        emb1 = self.bert_model.encode(text1, 
                                      convert_to_tensor=True)
        emb2 = self.bert_model.encode(text2, 
                                      convert_to_tensor=True)
        
        #Computing cosine similarity
        similarity = util.cos_sim(emb1, 
                                  emb2).item()
        
        return max(0.0, min(1.0, similarity))  #Clamping to [0, 1]

    def _check_spoilers(self, 
                        prediction, 
                        future_chapters):
        """
        Check if prediction contains information from future chapters.
        
        Returns:
            spoiler_score: Maximum similarity with any future chapter
            spoiler_violation: Boolean indicating if threshold exceeded
        """
        if not future_chapters or not prediction:
            return 0.0, False
        
        #Encoding prediction once
        pred_emb = self.bert_model.encode(prediction, 
                                          convert_to_tensor=True)
        
        max_similarity = 0.0
        
        #Checking similarity with each future chapter
        for chapter in future_chapters:
            if not chapter or len(chapter.strip()) < 50:
                continue
            
            #For long chapters, sample chunks to avoid memory issues
            chapter_text = chapter[:5000]  #First 5000 chars
            
            chapter_emb = self.bert_model.encode(chapter_text, 
                                                 convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, 
                                      chapter_emb).item()
            
            max_similarity = max(max_similarity, 
                                 similarity)
        
        spoiler_violation = max_similarity > self.spoiler_threshold
        
        return max_similarity, spoiler_violation

    def compute_aggregate_metrics(self, 
                                  results):
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
        
        return aggregate
