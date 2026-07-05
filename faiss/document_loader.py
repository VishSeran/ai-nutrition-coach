import pandas as pd
from configuration.config import DATASET_PATH

class DocumentLoader:
    
    def __init__(self, dataset_name=DATASET_PATH):
        
        self.df = pd.read_csv(dataset_name)
        