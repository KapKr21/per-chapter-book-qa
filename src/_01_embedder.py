from sentence_transformers import SentenceTransformer

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from transformers.utils import logging
logging.set_verbosity_error()

class BookEmbedder:
    def __init__(self, 
                 model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 device="cpu"):

        self.device = device
        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def embed_chapters(self, chapters):
        print(f"Embedding {len(chapters)} chapters...")
        
        return self.model.encode(chapters, 
                                 show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode([query])