from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from configuration.config import LLM_MODEL
from configuration.logger import get_logger
import torch

logger = get_logger("LLM-model")
class LLMModel:
    def __init__(self, llm_model=LLM_MODEL):

        try:
            if not llm_model:
                raise ValueError("LLM model is empty or none")
            
            self.chain_llm = None

            device = 0 if torch.cuda.is_available() else -1
            pipeline = HuggingFacePipeline.from_model_id(
                model_id=llm_model,
                task="text-generation",
                device=device,
                model_kwargs={"temperature": 0.6, "max_new_tokens": 256},
            )

            logger.info("llm is created")
            self.chat_llm = ChatHuggingFace(llm=pipeline)
            logger.info("chat llm is created")
            
            self.build_llm_chain()

        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise

        except Exception as e:
            logger.error(f"Error in LLM initialization: {e}")
            raise

    def build_llm_chain(self):

        try:
            template = """
                You are an expert nutrition coach.

                Food information:
                {content}

                Generate a friendly response including:

                1. Meal description
                2. Estimated nutritional profile
                3. Positive health impacts
                4. Possible concerns
                5. Suggestions for healthier alternatives
                6. A short recommendation

                Keep the response concise and easy to understand.
                """
                
            prompt_template = PromptTemplate(
                template=template,
                input_variables=["content"]
            )
            
            logger.info("Prompt template is created")
            
            self.chain_llm = prompt_template | self.chat_llm | StrOutputParser()
            logger.info("llm chain is created")
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise

        except Exception as e:
            logger.error(f"Error in llm chain: {e}")
            raise

    def get_response_from_llm(self, query):

        try:
            if not query:
                raise ValueError("Query cannot be empty or none")

            logger.info(f"Generating response for query length: {len(query)}")
            response = self.chain_llm.invoke({
                "content": query
            })
            
            if response is None or response.strip() == "":
                raise ValueError("Empty LLM response")
            
            logger.info("Response is retrieved")
            return response
        
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise

        except Exception as e:
            logger.error(f"Error in get response from llm: {e}")
            raise
