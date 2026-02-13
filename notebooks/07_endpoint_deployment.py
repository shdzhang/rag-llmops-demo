# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Agent Deployment with OBO (On-Behalf-Of Users)
# MAGIC
# MAGIC This notebook deploys the champion model using `databricks.agents.deploy()`:
# MAGIC 1. Retrieves the champion model version from UC
# MAGIC 2. Deploys with `agents.deploy()` (enables Review App, AI Gateway, Inference Tables)
# MAGIC 3. Tests the deployed endpoint
# MAGIC
# MAGIC ## On-Behalf-Of (OBO) Authentication
# MAGIC
# MAGIC When deployed via `agents.deploy()`, the agent automatically gets:
# MAGIC - **AI Gateway integration** for request/response logging
# MAGIC - **Inference tables** for monitoring
# MAGIC - **Review App** for human feedback collection
# MAGIC
# MAGIC For full OBO (user-level data isolation), deploy as a **Databricks App**:
# MAGIC - The App passes the user's OAuth token via `x-forwarded-access-token`
# MAGIC - The agent uses `get_user_workspace_client()` to authenticate as the user
# MAGIC - Vector Search queries respect the user's Unity Catalog permissions
# MAGIC
# MAGIC See `docs/obo_setup_guide.md` for the Databricks App setup.

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-agents
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow import MlflowClient
from databricks import agents

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

print(f"Deploying: {UC_MODEL_NAME}")
print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Get Champion Model Version

# COMMAND ----------

champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
champion_version = champion.version

print(f"Champion version: {champion_version}")
print(f"  Run ID: {champion.run_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Deploy with agents.deploy()
# MAGIC
# MAGIC `agents.deploy()` provides:
# MAGIC - Model Serving endpoint with AI Gateway
# MAGIC - Automatic inference table logging
# MAGIC - Review App for human feedback
# MAGIC - Automatic scaling

# COMMAND ----------

# agents.deploy() is idempotent: if the endpoint already exists it updates the
# model version; if not it creates a new endpoint.  This makes the notebook
# safe to re-run multiple times.
deployment = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=champion_version,
    tags={"environment": "dev", "source": "llmops-demo"},
)

print(f"Deployment initiated (create or update)!")
print(f"  Endpoint name: {deployment.endpoint_name}")
print(f"  Query endpoint: {deployment.query_endpoint}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Wait for Endpoint to be Ready
# MAGIC
# MAGIC Deployment takes ~15 minutes. The endpoint status can be monitored
# MAGIC in the Databricks UI or via SDK.

# COMMAND ----------
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoint_name = deployment.endpoint_name
print(f"Waiting for endpoint '{endpoint_name}' to be ready...")

for i in range(60):  # 30 minutes max
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state

        if str(state.ready) == "READY" or "READY" in str(state.ready):
            print(f"\nEndpoint is READY! (took ~{i * 30}s)")
            break
        elif "FAILED" in str(state.config_update or ""):
            print(f"\nEndpoint deployment FAILED!")
            print(f"  Error: {state}")
            break
        else:
            print(f"  [{i * 30}s] State: {state.ready}, Config: {state.config_update}")
    except Exception as e:
        print(f"  [{i * 30}s] Waiting... ({e})")

    time.sleep(30)
else:
    print("Timeout waiting for endpoint. Check Databricks UI for status.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Test the Deployed Endpoint

# COMMAND ----------

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

try:
    response = w.serving_endpoints.query(
        name=endpoint_name,
        messages=[
            ChatMessage(
                role=ChatMessageRole.USER,
                content="What is the company's remote work policy?",
            )
        ],
        max_tokens=500,
    )

    print("Test query successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Test query failed: {e}")
    print("The endpoint may still be starting up. Try again in a few minutes.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Verify Inference Table & Review App

# COMMAND ----------

print(f"""
Deployment Summary:
==================
Endpoint:     {endpoint_name}
Model:        {UC_MODEL_NAME} v{champion_version}
Review App:   https://<workspace-url>/ml/review/{endpoint_name}

Inference Table (auto-created by AI Gateway):
  Catalog:    {CATALOG}
  Schema:     {SCHEMA}
  Table:      {endpoint_name.replace('-', '_')}_payload

Key inference table columns:
  - request_time: Timestamp of the request
  - request: The input payload (JSON)
  - response: The output payload (JSON)
  - execution_duration_ms: End-to-end latency
  - status_code: HTTP status code
  - trace: Full MLflow trace (when tracing is enabled)

Next Steps:
  1. Share the Review App URL with stakeholders for feedback
  2. Monitor via notebook 09_monitoring_dashboard.py
  3. For OBO deployment, see docs/obo_setup_guide.md
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## (Optional) OBO Deployment via Databricks Apps
# MAGIC
# MAGIC For full On-Behalf-Of user authentication, deploy as a Databricks App:
# MAGIC
# MAGIC ```yaml
# MAGIC # app.yml
# MAGIC command:
# MAGIC   - uvicorn
# MAGIC   - app:app
# MAGIC   - --host=0.0.0.0
# MAGIC   - --port=8000
# MAGIC
# MAGIC permissions:
# MAGIC   - permission: CAN_USE
# MAGIC     group_name: users
# MAGIC
# MAGIC env:
# MAGIC   - name: MODEL_SERVING_ENDPOINT
# MAGIC     value: "{endpoint_name}"
# MAGIC ```
# MAGIC
# MAGIC The app receives the user's OAuth token and uses
# MAGIC `databricks.sdk.config.Config(credentials_strategy="x-forwarded-access-token")`
# MAGIC to authenticate API calls as the end user.
