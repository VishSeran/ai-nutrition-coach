import pandas as pd
from configuration.config import DATASET_PATH
from configuration.logger import get_logger
from langchain_core.documents import Document

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
        
    def format_documents(self):
        
        try:
            for _, row in self.df.iterrows():
                
                text = f"""
                        Food: {row['FoodItem']}
                        Category: {row["FoodCategory"]}
                        Calories: {row["Cals_per100grams"]} cal per 100g
                        Energy: {row["KJ_per100grams"]} KJ per 100g""".strip()
                
                self.documents.append(Document(page_content=text))
                
            logger.info("Documents are formatted")
            return self.documents
        
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
    
        except Exception as e:
            logger.error(f"Error in format documents: {e}")
            raise    