IMAGE_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-base-en"
DATASET_PATH = "dataset/calories.csv"

system_prompt = """ You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide the food content and estimated weight of each food item in the following format in a single line:
        1. **Food Item**: Estimate weight in grams."""
