from rouge_score import rouge_scorer

class BookEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate(self, prediction, ground_truth, future_chapters):
        """Calculating accuracy and checks for future spoilers."""
        #1. Standard Accuracy
        rouge_l = self.scorer.score(prediction, ground_truth)['rougeL'].fmeasure
        
        #2. Spoiler Check
        spoiler_violation = False
        for f_chapter in future_chapters:
            overlap = self.scorer.score(prediction, f_chapter)['rougeL'].fmeasure
            if overlap > 0.3: #Threshold for suspected spoiler
                spoiler_violation = True
                break
                
        return {
            "rougeL": rouge_l,
            "spoiler_violation": spoiler_violation
        }