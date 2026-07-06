import faiss
from sentence_transformers import SentenceTransformer
from configuration.config import EMBEDDING_MODEL
from configuration.logger import get_logger
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np

logger = get_logger("faiss-process")

class FaissSearch:
    
    def __init__(self,documents, embedding_model=EMBEDDING_MODEL ):
        
        try:
            if not embedding_model:
                raise ValueError("embedding model must required")
    
            self.hf_embed_model = HuggingFaceEmbeddings(model_name=embedding_model)
            dim = len(
                    self.hf_embed_model.embed_query("hello"))
            self.index = faiss.IndexFlatIP(dim)
            logger.info("faiss index and huggingface embedding model have created")
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in Faiss initialization: {e}")
            raise
        
    
    def build_index(self, documents):
        
        try:
            if not documents:
                raise ValueError("Documents are empty or none")
            
            texts = [doc.page_content for doc in documents]
            embeddings = self.hf_embed_model.embed_documents(texts)
            
            embeddings = np.array(
                embeddings,
                dtype= np.float32
            )
            
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in build faiss index: {e}")
            raise
        
    