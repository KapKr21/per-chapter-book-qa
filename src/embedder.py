from sentence_transformers import SentenceTransformer
import torch

class BookEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        #Using a lightweight model for fast local embedding
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_chapters(self, chapters):
        """Converts a list of chapter strings into numerical vectors."""
        print(f"Embedding {len(chapters)} chapters...")
        return self.model.encode(chapters, show_progress_bar=True)

    def embed_query(self, query):
        """Converts a single question into a vector."""
        return self.model.encode([query])