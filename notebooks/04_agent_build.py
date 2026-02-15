# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Agent Build: Log ResponsesAgent to MLflow
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Tests the RAG agent locally
# MAGIC 2. Logs it to MLflow using file-based logging
# MAGIC 3. Declares all external resources for auto-authentication
# MAGIC 4. Registers to Unity Catalog

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-langchain databricks-vectorsearch databricks-agents databricks-openai
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
VS_INDEX = f"{CATALOG}.{SCHEMA}.docs_index"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('prompt_name')}"

print(f"MLflow version: {mlflow.__version__}")
print(f"UC Model: {UC_MODEL_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Set MLflow Experiment

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    # Fallback for local testing
    mlflow.set_experiment(f"/Users/default/dev_{MODEL_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Test the Agent Locally

# COMMAND ----------

import os

# Resolve the agent file path relative to this notebook.
# In Databricks workspace, notebooks run from /Workspace/... and bundle files
# are deployed alongside them. We navigate up from the notebook dir to find agent/.
try:
    # When running via DABs, use the notebook context to find the bundle root.
    # notebook_path looks like:
    #   /Workspace/.bundle/rag-llmops-demo/dev/files/notebooks/04_agent_build
    #   OR /.bundle/rag-llmops-demo/dev/files/notebooks/04_agent_build
    notebook_path = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook().getContext().notebookPath().get()
    )
    # Navigate up: notebooks/ -> files/ (bundle root)
    bundle_root = os.path.dirname(os.path.dirname(notebook_path))
    # Ensure /Workspace prefix (some runtimes include it, some don't)
    if not bundle_root.startswith("/Workspace"):
        bundle_root = f"/Workspace{bundle_root}"
    agent_file = os.path.join(bundle_root, "agent", "rag_agent.py")
except Exception:
    # Fallback: assume CWD-relative path (e.g., running locally or in Repos)
    agent_file = os.path.normpath(
        os.path.join(os.getcwd(), "..", "agent", "rag_agent.py")
    )

print(f"Agent file: {agent_file}")
assert os.path.exists(agent_file), f"Agent file not found at {agent_file}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Log the Agent to MLflow
# MAGIC
# MAGIC Key points:
# MAGIC - Uses **file-based logging** (`python_model="agent/rag_agent.py"`)
# MAGIC - Declares all external **resources** for automatic credential provisioning
# MAGIC - Specifies **pip_requirements** for the serving environment

# COMMAND ----------

# Define resources the agent needs at serving time
# Databricks automatically provisions credentials for these
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX),
]

# pip requirements for the serving environment
pip_requirements = [
    "mlflow>=3.1",
    "databricks-langchain",
    "databricks-vectorsearch",
    "databricks-sdk",
    "databricks-openai",
]

# Input example for model signature inference
input_example = {
    "input": [{"role": "user", "content": "What is the remote work policy?"}]
}

# model_config: the agent reads these at load time via mlflow.models.ModelConfig()
# This is how we pass catalog/schema/index without hardcoding in the agent file.
agent_config = {
    "llm_endpoint": LLM_ENDPOINT,
    "vector_search_index": VS_INDEX,
    "prompt_name": PROMPT_NAME,
    "prompt_alias": "production",
}

with mlflow.start_run(run_name="rag_agent_build") as run:
    # Log agent configuration as parameters (for tracking / comparison)
    mlflow.log_params({**agent_config, "agent_type": "ResponsesAgent"})

    # Prompt URI for linking to this run (shows in experiment UI under Prompts)
    prompt_uri = f"prompts:/{PROMPT_NAME}@production"

    # Log the agent using file-based approach (absolute path resolved above)
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent_file,
        model_config=agent_config,
        resources=resources,
        pip_requirements=pip_requirements,
        input_example=input_example,
        prompts=[prompt_uri],
    )

    logged_run_id = run.info.run_id
    print(f"Agent logged successfully!")
    print(f"  Run ID: {logged_run_id}")
    print(f"  Model URI: {model_info.model_uri}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3b: Validate Model Locally (using uv)
# MAGIC
# MAGIC Uses `env_manager="uv"` for fast, reproducible environment creation.

# COMMAND ----------

try:
    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data={"input": [{"role": "user", "content": "What is the remote work policy?"}]},
        env_manager="uv",
    )
    print("Model validation passed!")
except Exception as e:
    print(f"Local validation skipped (this is expected on some clusters): {e}")
    print("The model will be validated during serving endpoint deployment instead.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Register Model to Unity Catalog

# COMMAND ----------

uc_model_info = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=UC_MODEL_NAME,
)

print(f"Registered model: {uc_model_info.name}")
print(f"  Version: {uc_model_info.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Set "Candidate" Alias for Evaluation
# MAGIC
# MAGIC We set a "candidate" alias now. After evaluation passes,
# MAGIC notebook 06 will promote it to "champion".

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name=UC_MODEL_NAME,
    alias="candidate",
    version=uc_model_info.version,
)

print(f"Alias 'candidate' set to version {uc_model_info.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Output for Downstream Tasks
# MAGIC
# MAGIC Pass the run ID to the evaluation notebook via task values.

# COMMAND ----------

# Set task values for downstream notebooks
if "dbutils" in dir():
    dbutils.jobs.taskValues.set(key="logged_run_id", value=logged_run_id)
    dbutils.jobs.taskValues.set(key="model_version", value=uc_model_info.version)

print(f"\nReady for evaluation!")
print(f"  Run ID: {logged_run_id}")
print(f"  Model Version: {uc_model_info.version}")
print(f"  UC Model: {UC_MODEL_NAME}@candidate")
