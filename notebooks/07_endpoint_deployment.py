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
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print(f"Deploying: {UC_MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Get Champion Model Version

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
    champion_version = champion.version
    print(f"Champion version: {champion_version}")
    print(f"  Run ID: {champion.run_id}")
except Exception:
    # Fall back to candidate if champion hasn't been set yet
    print("No 'champion' alias found - falling back to 'candidate'")
    champion = client.get_model_version_by_alias(UC_MODEL_NAME, "candidate")
    champion_version = champion.version
    print(f"Candidate version: {champion_version}")
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

# agents.deploy() creates or updates the serving endpoint.
# It may fail if:
#   - The endpoint is currently updating from a previous deploy (retry after wait)
#   - The exact same model version is already deployed (treat as success)
import time
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()
# agents.deploy() truncates endpoint names to 63 characters
endpoint_name = f"agents_{UC_MODEL_NAME}".replace(".", "-")[:63]

# If the endpoint exists and is updating, wait for it to finish first
try:
    _ep = _w.serving_endpoints.get(endpoint_name)
    _cfg = str(_ep.state.config_update) if _ep.state and _ep.state.config_update else ""
    if "IN_PROGRESS" in _cfg or "NOT_READY" in str(_ep.state.ready):
        print(f"Endpoint '{endpoint_name}' is currently updating - waiting for it to finish...")
        for _wi in range(40):  # up to 20 min
            time.sleep(30)
            _ep = _w.serving_endpoints.get(endpoint_name)
            _cfg = str(_ep.state.config_update) if _ep.state and _ep.state.config_update else "NONE"
            _rdy = str(_ep.state.ready)
            if "READY" in _rdy and "IN_PROGRESS" not in _cfg and "NOT_READY" not in _cfg:
                print(f"  Previous update finished (waited ~{(_wi+1)*30}s)")
                break
            print(f"  [{(_wi+1)*30}s] ready={_rdy}, config_update={_cfg}")
        else:
            print("  Warning: timed out waiting for previous update")
except Exception:
    pass  # Endpoint doesn't exist yet - that's fine

try:
    deployment = agents.deploy(
        model_name=UC_MODEL_NAME,
        model_version=champion_version,
        tags={"source": "llmops-demo"},
    )
    endpoint_name = deployment.endpoint_name
    print(f"Deployment initiated (create or update)!")
    print(f"  Endpoint name: {endpoint_name}")
    print(f"  Query endpoint: {deployment.query_endpoint}")
except ValueError as e:
    if "already serves" in str(e):
        print(f"Endpoint already serves {UC_MODEL_NAME} v{champion_version} - no update needed.")
        print(f"  Endpoint name: {endpoint_name}")
    elif "currently updating" in str(e):
        print(f"Endpoint is still updating - will wait for it in the next cell.")
        print(f"  Endpoint name: {endpoint_name}")
    else:
        raise

# Pass endpoint name to downstream notebooks (08, 09)
if "dbutils" in dir():
    dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)

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

print(f"Waiting for endpoint '{endpoint_name}' to be ready...")
print("  (endpoint must be READY with no pending config updates)")

for i in range(60):  # 30 minutes max
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state
        ready = str(state.ready)
        config = str(state.config_update) if state.config_update else "NONE"

        is_ready = "READY" in ready
        is_updating = "IN_PROGRESS" in config or "NOT_READY" in config

        if is_ready and not is_updating:
            print(f"\nEndpoint is READY! (took ~{i * 30}s)")
            break
        elif "FAILED" in config:
            print(f"\nEndpoint deployment FAILED!")
            print(f"  Error: {state}")
            raise Exception(f"Endpoint deployment failed: {state}")
        else:
            print(f"  [{i * 30}s] ready={ready}, config_update={config}")
    except Exception as e:
        if "FAILED" in str(e):
            raise
        print(f"  [{i * 30}s] Waiting... ({e})")

    time.sleep(30)
else:
    print("Timeout waiting for endpoint. Check Databricks UI for status.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Test the Deployed Endpoint

# COMMAND ----------

# The agent is a ResponsesAgent (Responses API format).
# Use SDK's api_client which handles all auth methods (PAT, Azure AAD, OAuth).

try:
    result = w.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint_name}/invocations",
        body={"input": [{"role": "user", "content": "What is the company's remote work policy?"}]},
    )

    # Extract answer from Responses API output format
    answer = ""
    for item in result.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    answer += block.get("text", "")

    print("Test query successful!")
    print(f"Response: {answer[:500]}")
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
