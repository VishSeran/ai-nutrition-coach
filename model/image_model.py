from typing import Optional
from PIL import Image
from configuration.config import IMAGE_MODEL,system_prompt
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from configuration.logger import get_logger
import requests
import io

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
            
            self.processor = AutoProcessor.from_pretrained(
                model_name
            )
            self.processor.tokenizer.pad_token_id = None
            logger.info(f"{model_name} processor has loaded")

        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in Image Model initialization: {e}")
            raise
        
    def get_image(self, 
                    image_path:Optional[str], image_url:Optional[str]):
        
        image = None
        
        try:
            
            if image_url:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                logger.info("image got from url")
            
            elif image_path:
                image = Image.open(image_path).convert("RGB")
                logger.info("image got from local path")
            
            else:
                raise ValueError("At least image path or url needed")
            
            return image
                
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
        
        except requests.RequestException as e:
            logger.error(f"Error in url fetching:{e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in image input: {e}")
            raise
        
    def get_response_from_model(self, image):
        
        try:
            
            if not image:
                raise ValueError("Image cannot be empty or none")
            
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type" : "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image",
                            "image": image
                        }
                    ]
                }
            ]
            logger.info("message has crafted")
            
            inputs = self.processor.apply_chat_template(
                message,
                add_generation_prompt = True,
                tokenize = True,
                return_dict = True,
                return_tensors = "pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("model input has crafted")
            
            generated_ids = self.model.generate(
                **inputs,
                do_sample = False,
                max_new_tokens = 256
            )
            
            if generated_ids is None:
                logger.info("Response fetch failed")
                
            generated_ids = generated_ids[:,inputs["input_ids"].shape[1]:]
            
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            logger.info("response has fetched")
            return response
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in get response: {e}")
            raise
        