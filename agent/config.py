"""
Agent configuration - shared defaults for local tests & standalone scripts.

NOTE: rag_agent.py does NOT import this file.  It reads configuration at
load time from mlflow.models.ModelConfig() (values passed via model_config
in notebook 04).  Notebooks also get catalog/schema/model via DAB job
parameters (dbutils.widgets.get), NOT from this file.

Edit these values to match your Databricks workspace.
"""

# Unity Catalog (define first - used by other constants)
CATALOG = "main"
SCHEMA = "corporate_affairs"
MODEL_NAME = "corporate_affairs_chatbot"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Foundation Model endpoint for the agent's LLM
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"

# LLM Judge endpoint for evaluation (notebook 05)
JUDGE_LLM_ENDPOINT_NAME = "databricks-claude-opus-4-6"

# Embedding model for Vector Search
EMBEDDING_ENDPOINT_NAME = "databricks-gte-large-en"

# Vector Search
VECTOR_SEARCH_ENDPOINT = "corporate_affairs_vs_endpoint"
VECTOR_SEARCH_INDEX = f"{CATALOG}.{SCHEMA}.corporate_docs_index"

# MLflow Prompt Registry (3-level UC name required)
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.rag_prompt"
PROMPT_ALIAS = "production"
