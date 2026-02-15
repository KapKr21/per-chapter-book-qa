from sentence_transformers import SentenceTransformer

class BookEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device="cpu"):
        # Force CPU so it doesn't steal VRAM from the LLM
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_chapters(self, chapters):
        print(f"Embedding {len(chapters)} chapters...")
        return self.model.encode(chapters, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode([query])