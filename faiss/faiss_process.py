import faiss
from sentence_transformers import SentenceTransformer
from configuration.config import EMBEDDING_MODEL
from configuration.logger import get_logger
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np

logger = get_logger("faiss-process")

class FaissSearch:
    
    def __init__(self, documents,embedding_model=EMBEDDING_MODEL ):
        
        try:
            if not embedding_model:
                raise ValueError("embedding model must required")
            
            if not documents:
                    raise ValueError("Documents are empty or none")
                
            self.documents = documents
    
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
        
    
    def build_index(self):
        
        try:
        
            texts = [doc.page_content for doc in self.documents]
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
    
    def faiss_search(self, query):
        
        try:
            if not query:
                raise ValueError("Query is empty or none")
            
            faiss_query = "give the calories and energy amount with units of ".join(query)
            
            query_embeddings = self.hf_embed_model.embed_query(faiss_query)
            
            query_embeddings = np.array(
                query_embeddings,
                dtype=np.float32
            )
            
            faiss.normalize_L2(query_embeddings)
            distance, indices = self.index.search(query_embeddings,k=3)
            
            if distance[0][0] < 0.75:
                return "No confident match found"
            
            
            return [
                {
                    "result": self.documents[idx].page_content,
                    "score": float(distance[0][i])
                }
                
                for i, idx in enumerate(indices[0])
            ]
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in faiss search: {e}")
            raise
        
    