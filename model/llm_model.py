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
            
            device = 0 if
            pipeline = HuggingFacePipeline.from_model_id(
                model_id=llm_model,
                task="text-generation",
                device= 0
            )
        
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in LLM initialization: {e}")
            raise
    
    