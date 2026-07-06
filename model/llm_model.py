from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from configuration.config import LLM_MODEL
from configuration.logger import get_logger
import torch

logger = get_logger("LLM-model")

class LLMModel:
    
    def __init__(self,llm_model=LLM_MODEL):
        
        try:
            
            if not llm_model:
                raise ValueError("LLM model is empty or none")
            
            device = 0 if torch.cuda.is_available() else -1
            llm = HuggingFacePipeline.from_model_id(
                model_id=llm_model,
                task="text-generation",
                device= device,
                model_kwargs={
                    "temperature": 0.6,
                    "max_new_tokens": 256
                }
            )
            
            logger.info("llm is created")
            self.chat_llm = ChatHuggingFace(llm=llm)
            logger.info("chat llm is created")
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in LLM initialization: {e}")
            raise
    
    