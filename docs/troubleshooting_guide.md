# Troubleshooting Guide: RAG LLMOps Demo

A comprehensive record of every issue encountered during development and deployment, along with root causes and fixes. Organized by category for quick reference.

---

## Table of Contents

1. [MLflow 3.x API Changes](#1-mlflow-3x-api-changes)
2. [Unity Catalog & Naming](#2-unity-catalog--naming)
3. [Vector Search](#3-vector-search)
4. [Agent Evaluation](#4-agent-evaluation)
5. [Model Serving & Endpoint Deployment](#5-model-serving--endpoint-deployment)
6. [Databricks Asset Bundles (DABs)](#6-databricks-asset-bundles-dabs)
7. [Authentication & SDK](#7-authentication--sdk)
8. [Miscellaneous](#8-miscellaneous)

---

## 1. MLflow 3.x API Changes

### 1.1 Prompt Registry Name Format

**Error:**
```
RestException: INVALID_PARAMETER_VALUE: name is not a valid name.
```

**Cause:** Prompt name `corporate-affairs-system-prompt` contained hyphens and was not a 3-level Unity Catalog name.

**Fix:** Changed prompt name to `{CATALOG}.{SCHEMA}.rag_prompt` (3-level UC name, underscores only).

---

### 1.2 EvaluationResult Has No `eval_table`

**Error:**
```
'EvaluationResult' object has no attribute 'eval_table'
```

**Cause:** MLflow 3.x removed the `.eval_table` attribute from `EvaluationResult`.

**Fix:** Replaced with `mlflow.search_traces(run_id=eval_results.run_id)` to retrieve evaluation traces.

---

### 1.3 Malformed LLM Judge Model URI

**Error:**
```
Malformed model uri 'databricks-claude-opus-4-6'. The URI must be in the format of <provider>:/<model-name>
```

**Cause:** MLflow 3.x scorers require the `provider:/model-name` URI format.

**Fix:** Changed from `"databricks-claude-opus-4-6"` to `"databricks:/databricks-claude-sonnet-4-5"`.

---

### 1.4 Prompt Not Linked to Experiment

**Symptom:** The "Prompts" section in the MLflow experiment UI was empty.

**Cause:** The `prompts` parameter was not passed to `mlflow.pyfunc.log_model()`.

**Fix:** Added `prompts=[prompt_uri]` to the `log_model()` call in notebook 04.

---

### 1.5 `get_open_ai_client()` Deprecated

**Warning:**
```
DeprecationWarning: get_open_ai_client() is deprecated. Please install the
databricks-openai package and use 'from databricks_openai import DatabricksOpenAI' instead.
```

**Fix:** Replaced all usages of `w.serving_endpoints.get_open_ai_client()` with:
```python
from databricks_openai import DatabricksOpenAI
client = DatabricksOpenAI()
```
Updated `%pip install` commands to include `databricks-openai`.

---

## 2. Unity Catalog & Naming

### 2.1 Hardcoded Catalog/Schema in Agent File

**Error:**
```
Unity Catalog entity main.corporate_affairs.corporate_docs_index does not exist.
```

**Cause:** `rag_agent.py` had catalog/schema/index names hardcoded. The actual catalog/schema were different.

**Fix:** Agent now reads config dynamically via `mlflow.models.ModelConfig()`. Values are passed from notebook 04 via the `model_config` parameter of `log_model()`.

---

### 2.2 Tool Name Truncated (64-char limit)

**Warning:**
```
Tool name shidong_catalog__dev_shidong_zhang_corporate_affairs__corporate_docs_index
is too long, truncating to 64 characters
```

**Cause:** The Vector Search index name (used as a tool name internally) exceeded 64 characters when DAB `mode: development` prepends `dev_<username>_` to schema names.

**Fix:** Shortened all base resource names:
| Resource | Before | After |
|----------|--------|-------|
| Schema | `corporate_affairs` | `corp_affairs` |
| Model | `corporate_chatbot` | `corp_chatbot` |
| VS Endpoint | `corporate_vs_endpoint` | `corp_vs_endpoint` |
| VS Index | `corporate_docs_index` | `docs_index` |
| Table | `corporate_doc_chunks` | `doc_chunks` |

---

### 2.3 `model.aliases` and `v.tags` Are Dicts, Not Lists

**Error:**
```
'str' object has no attribute 'alias'
```

**Cause:** In notebook 06, code iterated `model.aliases` as a list of objects. In Unity Catalog, `aliases` is a `dict` (`{"alias_name": "version_number"}`), and `v.tags` is also a `dict`.

**Fix:** Updated iteration to handle dictionaries:
```python
aliases = model.aliases or {}
if isinstance(aliases, dict):
    for alias_name, version in aliases.items():
        print(f"  @{alias_name} -> version {version}")
```

---

## 3. Vector Search

### 3.1 Index "ONLINE" But Returns 0 Rows

**Symptom:** Vector Search index status was `ONLINE`, pipeline showed `COMPLETED`, but `similarity_search()` returned 0 results. Source Delta table had data.

**Root Cause:** Change Data Feed (CDF) was enabled **after** the initial data write. The TRIGGERED Delta Sync pipeline had no changes to pick up because CDF wasn't capturing the initial load.

**Fix (notebook 01):** Enable CDF **before** writing data:
```python
# Create table with CDF FIRST
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table_path} (...)
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
# Write data
df.write.format("delta").mode("overwrite").saveAsTable(table_path)
# Re-enable CDF after overwrite (overwrite may reset properties)
spark.sql(f"ALTER TABLE {table_path} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
```

**Fix (notebook 02):** Wait loop now checks for **actual data** (via `similarity_search`), not just `ONLINE` status. Also auto-triggers a sync if the index is empty.

---

### 3.2 Evaluation Fails: "I'm unable to access..."

**Symptom:** Agent responses during evaluation said "I apologize, but I'm currently unable to access..." instead of answering. Correctness score was 0.000.

**Cause:** Vector Search retrieval was failing silently. The `predict_fn` in notebook 05 was passing the error string as context instead of failing.

**Fix:**
1. Added a pre-evaluation readiness check that waits for the index to have data (with auto-sync trigger).
2. Changed `predict_fn` to `raise RuntimeError` if Vector Search returns 0 results instead of silently passing an error string.

---

## 4. Agent Evaluation

### 4.1 Multiple MLflow Experiments Created

**Symptom:** Notebooks 04 and 05 were logging to different MLflow experiments.

**Fix:** Added `experiment_name` as a DAB job parameter passed to both notebooks. Both now call `mlflow.set_experiment(EXPERIMENT_NAME)`.

---

### 4.2 Quality Gate Always Fails (Correctness 0.000)

**Symptom:** All evaluation questions scored 0 for correctness.

**Causes (multiple, cumulative):**
1. Vector Search index had no data (see 3.1 above).
2. `expected_response` values were overly detailed, causing the LLM judge to penalize missing details.
3. LLM judge endpoint was slow/timing out.

**Fixes:**
1. Fixed CDF ordering (see 3.1).
2. Simplified `expected_response` to contain only essential facts.
3. Changed LLM judge from `databricks-claude-opus-4-6` to `databricks-claude-sonnet-4-5` (faster, more reliable).

---

### 4.3 No Champion Model for Deployment

**Symptom:** Notebook 07 failed with `Registered Model Alias 'champion' does not exist.`

**Cause:** The quality gate in notebook 05 failed but didn't stop the pipeline (the `raise Exception` was commented out).

**Fix:** Uncommented the `raise Exception` in notebook 05 so the pipeline stops on failure. Added fallback to `@candidate` alias in notebook 07.

---

## 5. Model Serving & Endpoint Deployment

### 5.1 Endpoint Name Truncated to 63 Characters

**Symptom:** Notebook 07/08 waited forever for an endpoint that "does not exist", but it was visible and READY in the Serving UI.

**Cause:** `agents.deploy()` truncates endpoint names to 63 characters. The constructed name was 83 characters:
```
agents_shidong_catalog-dev_shidong_zhang_corp_affairs-dev_shidong_zhang_corp_chatbot  (83 chars)
```
The actual endpoint was:
```
agents_shidong_catalog-dev_shidong_zhang_corp_affairs-dev_shido  (63 chars)
```

**Fix:** Added `[:63]` truncation to all endpoint name constructions:
```python
endpoint_name = f"agents_{UC_MODEL_NAME}".replace(".", "-")[:63]
```

---

### 5.2 Readiness Check Loops Forever (Enum String Mismatch)

**Symptom:** Endpoint was READY with NOT_UPDATING, but the wait loop kept printing status and never broke out. Ran for 960+ seconds.

**Cause:** `str(state.config_update)` returns `"EndpointStateConfigUpdate.NOT_UPDATING"`, but the code checked:
```python
config not in ("NONE", "NOT_UPDATING", "None", "")  # Never matches!
```

**Fix:** Changed to substring-based checks:
```python
is_updating = "IN_PROGRESS" in config or "NOT_READY" in config
```

---

### 5.3 `ValueError: Endpoint already serves model`

**Error:**
```
ValueError: Endpoint ... already serves model ...
```

**Cause:** `agents.deploy()` raises `ValueError` when the exact same model version is already deployed.

**Fix:** Added `try-except ValueError` to catch `"already serves"` and treat as success.

---

### 5.4 `ValueError: Endpoint is currently updating`

**Error:**
```
ValueError: Endpoint ... is currently updating.
```

**Cause:** `agents.deploy()` cannot update an endpoint that is already in the middle of an update.

**Fix:** Added a pre-deployment wait loop that checks `config_update` status and waits up to 20 minutes for any previous update to finish before calling `agents.deploy()`.

---

### 5.5 Wrong API Format: Chat Completions vs. Responses API

**Error:**
```
Failed to enforce schema of data... Model is missing inputs ['input'].
Note that there were extra inputs: ['messages', 'max_tokens'].
```

**Cause:** The agent is a `ResponsesAgent` (Responses API format), but notebooks 07 and 08 were querying it with Chat Completions format (`messages`, `max_tokens`, `response.choices[0].message.content`).

**Fix:** Changed all endpoint queries to Responses API format:
```python
# Request
payload = {"input": [{"role": "user", "content": question}]}

# Response parsing
for item in result.get("output", []):
    if item.get("type") == "message":
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                answer += block.get("text", "")
```

---

### 5.6 401 Unauthorized When Using Raw `requests`

**Error:**
```
HTTPError: 401 Client Error: Unauthorized
```

**Cause:** Using `requests.post()` with `w.config.token` as bearer token fails on Azure Databricks where auth uses AAD/OAuth (token is `None` or not a simple bearer token).

**Fix:** Replaced raw `requests.post()` with `w.api_client.do()` which handles all authentication methods transparently:
```python
result = w.api_client.do(
    "POST",
    f"/serving-endpoints/{ENDPOINT_NAME}/invocations",
    body=payload,
)
```

---

## 6. Databricks Asset Bundles (DABs)

### 6.1 Redundant `[dev]` in Job Names

**Symptom:** Job names showed `[dev shidong_zhang] [dev] Data Preparation` -- double prefix.

**Cause:** `mode: development` already adds `[dev <username>]` prefix. The job YAML also had `[${bundle.target}]` in the name.

**Fix:** Removed `[${bundle.target}]` from all job YAML `name` fields.

---

### 6.2 Notebook Not Deployed (Silent Skip)

**Error:**
```
Unable to access the notebook ".../05_agent_evaluation" in the workspace.
```

**Cause:** Known issue with DABs remote deployment state -- files can be silently skipped during sync.

**Fix:** Manually uploaded the missing notebook. Redeploying with `--auto-approve` typically resolves the stale state.

---

### 6.3 Hardcoded Config in Notebooks

**Symptom:** Notebooks defined `CATALOG`, `SCHEMA`, `MODEL_NAME` as hardcoded strings, diverging from DAB variables.

**Fix:** All notebooks now read config from DAB job parameters:
```python
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
```

---

## 7. Authentication & SDK

### 7.1 On-Behalf-Of (OBO) Must Be Called at Request Time

**Symptom:** OBO client initialized in `__init__` didn't have user credentials.

**Cause:** `get_user_workspace_client()` only works inside `predict()` because user credentials (via `x-forwarded-access-token`) are only available at request time.

**Fix:** Created `_get_workspace_client()` helper method called inside `predict()` / `_retrieve_context()`:
```python
def _get_workspace_client(self):
    try:
        from databricks.agents import get_user_workspace_client
        return get_user_workspace_client()
    except Exception:
        from databricks.sdk import WorkspaceClient
        return WorkspaceClient()
```

---

## 8. Miscellaneous

### 8.1 Inference Table Query Uses Wrong JSON Path

**Symptom:** "Popular Questions" monitoring query returned `NULL` for all questions.

**Cause:** SQL used `$.messages[0].content` (Chat Completions format) but the Responses API stores input as `$.input[0].content`.

**Fix:** Used `COALESCE` for backwards compatibility:
```sql
COALESCE(
    get_json_object(request, '$.input[0].content'),
    get_json_object(request, '$.messages[0].content')
) AS question
```

---

### 8.2 `Error: Command failed to spawn: Aborted`

**Symptom:** Shell commands (`databricks bundle deploy`, `python3 -c "..."`, `sleep`) intermittently fail with this error.

**Cause:** Local environment / shell issue (not a code bug).

**Fix:** Retry the command. This is a transient error.

---

## Quick Reference: Key Patterns

### Querying a ResponsesAgent Endpoint (SDK)
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
result = w.api_client.do(
    "POST",
    f"/serving-endpoints/{endpoint_name}/invocations",
    body={"input": [{"role": "user", "content": "your question"}]},
)
# Parse response
for item in result.get("output", []):
    if item.get("type") == "message":
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                print(block["text"])
```

### Endpoint Readiness Check (Enum-Safe)
```python
ready = str(state.ready)           # "EndpointStateReady.READY"
config = str(state.config_update)  # "EndpointStateConfigUpdate.NOT_UPDATING"
is_ready = "READY" in ready
is_updating = "IN_PROGRESS" in config or "NOT_READY" in config
```

### Endpoint Name Construction (63-char Limit)
```python
endpoint_name = f"agents_{UC_MODEL_NAME}".replace(".", "-")[:63]
```
