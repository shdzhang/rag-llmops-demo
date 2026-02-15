"""
Agent configuration - shared defaults for local tests & standalone scripts.

NOTE: rag_agent.py does NOT import this file.  It reads configuration at
load time from mlflow.models.ModelConfig() (values passed via model_config
in notebook 04).  Notebooks also get values via DAB job parameters
(dbutils.widgets.get), NOT from this file.

All default values here mirror those in databricks.yml so local scripts
and tests behave identically to deployed runs.  When changing a value,
update databricks.yml (the single source of truth) and keep this file
in sync.
"""

# ---------------------------------------------------------------------------
# Unity Catalog
# ---------------------------------------------------------------------------
CATALOG = "shidong_catalog"
SCHEMA = "corp_affairs"
MODEL_NAME = "corp_chatbot"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# ---------------------------------------------------------------------------
# Foundation Model endpoints  (must match databricks.yml variables)
# ---------------------------------------------------------------------------
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
JUDGE_LLM_ENDPOINT_NAME = "databricks-claude-opus-4-1"
EMBEDDING_ENDPOINT_NAME = "databricks-gte-large-en"

# ---------------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------------
VECTOR_SEARCH_ENDPOINT = "corp_vs_endpoint"
VECTOR_SEARCH_INDEX = f"{CATALOG}.{SCHEMA}.docs_index"

# ---------------------------------------------------------------------------
# MLflow Prompt Registry (3-level UC name required)
# ---------------------------------------------------------------------------
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.rag_prompt"
PROMPT_ALIAS = "production"
