from typing import Optional

from configuration.config import IMAGE_MODEL
from transformers import AutoImageProcessor, AutoModelForImageTextToText
import torch
from configuration.logger import get_logger

logger = get_logger("image-model")
class ImageModel:
    
    def __init__(self, model_name=IMAGE_MODEL):
        
        try:
            if not model_name:
                raise ValueError("Image Model name cannot be empty")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"device is inititated to {self.device}")
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name
            ).to(self.device)
            logger.info(f"{model_name} has loaded")
            
            self.processor = AutoImageProcessor.from_pretrained(
                model_name
            )
            logger.info(f"{model_name} processor has loaded")

        except ValueError as e:
            print(f"Value error: {e}")
            raise
    
        except Exception as e:
            print(f"Error in logger initialization: {e}")
            raise
        
    def image_input(self, 
                    image_path:Optional[str], image_url:Optional[str]):
        
        
        