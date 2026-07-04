from typing import Optional

from configuration.config import IMAGE_MODEL
from transformers import AutoImageProcessor, AutoModelForImageTextToText
import torch

class ImageModel:
    
    def __init__(self, model_name=IMAGE_MODEL):
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name
        ).to(self.device)
        
        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )
    
    def image_input(self, 
                    image_path:Optional[str], image_url:Optional[str]):
        
        
        