import pandas as pd
from configuration.config import DATASET_PATH
from configuration.logger import get_logger

logger = get_logger("document-loader")

class DocumentLoader:
    
    def __init__(self, dataset_name=DATASET_PATH):
        
        try:
            if not dataset_name:
                raise ValueError("dataset name is empty or none")
            self.df = pd.read_csv(dataset_name)
            logger.info("Dataset has imported successfull")
            self.documents = []
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in document loader: {e}")
            raise
        