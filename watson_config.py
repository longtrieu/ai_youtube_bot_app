from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

def setup_credentials():
  """Set up IBM Watson credentials and configuration."""
  model_id = "meta-llama/llama-3-2-3b-instruct"
  credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
  client = APIClient(credentials)
  project_id = "skills-network"
  return model_id, credentials, client, project_id

def define_parameters():
  """Define parameters for WatsonX model."""
  return {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 900,
  }

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
  """Initialize WatsonX LLM with specified configuration."""
  return WatsonxLLM(
    model_id=model_id,
    url=credentials.get("url"),
    project_id=project_id,
    params=parameters
  )

def setup_embedding_model(credentials, project_id):
  """Set up WatsonX embedding model."""
  return WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    project_id=project_id
  )
