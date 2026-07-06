import faiss
from sentence_transformers import SentenceTransformer
from configuration.config import EMBEDDING_MODEL
from configuration.logger import get_logger
from langc

logger = get_logger("faiss-process")

class FaissSearch:
    
    def __init__(self, embedding_model=EMBEDDING_MODEL):
        
        try:
            if not embedding_model:
                raise ValueError("embedding model must required")
        
            self.embed_model = SentenceTransformer(embedding_model)
            self.d = self.embed_model.get_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.d)
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in Faiss initialization: {e}")
            raise
    