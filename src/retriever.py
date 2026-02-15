import faiss
import numpy as np

class ChapterRestrictedRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chapter_map = [] #To keep track of which vector belongs to which chapter

    def build_index(self, chapters):
        """
        Indexes chapters while keeping track of chapter IDs.
        """
        embeddings = self.embedder.embed_chapters(chapters)
        dimension = embeddings.shape[1]
        
        #Using a simple Flat index because book-scale data is small for FAISS
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        #Storing metadata: which index belongs to which chapter
        self.chapter_map = list(range(len(chapters)))

    def retrieve_safe_context(self, question, max_allowed_chapter_idx, top_k=3):
        """
        The core novelty: Retrieve ONLY from chapters <= max_allowed_chapter_idx.
        """
        query_vec = self.embedder.embed_query(question)
        
        #Searching for more results than we need so we can filter out spoilers
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), self.index.ntotal)
        
        safe_results = []
        for idx in indices[0]:
            chapter_id = self.chapter_map[idx]
            
            #SPOILER PROTECTION: Hard boundary check
            if chapter_id <= max_allowed_chapter_idx:
                safe_results.append(chapter_id)
            
            if len(safe_results) >= top_k:
                break
                
        return safe_results