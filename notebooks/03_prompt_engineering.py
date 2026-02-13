# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Prompt Engineering with MLflow Prompt Registry
# MAGIC
# MAGIC This notebook demonstrates how to use **MLflow Prompt Registry** to:
# MAGIC 1. Register versioned prompt templates
# MAGIC 2. Test prompts against sample queries
# MAGIC 3. Set production aliases for deployment
# MAGIC 4. Attach model configuration to prompts

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk openai
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow.entities.model_registry import PromptModelConfig

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.rag_prompt"

print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Register the Initial Prompt (v1 - Basic)

# COMMAND ----------

system_prompt_v1 = """\
You are a corporate affairs assistant. Answer employee questions based on the provided context.

Context: {{context}}

Question: {{question}}
"""

prompt_v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=system_prompt_v1,
    commit_message="v1: Basic corporate affairs prompt",
)

print(f"Registered prompt '{prompt_v1.name}' version {prompt_v1.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Register an Improved Prompt (v2 - Enhanced)

# COMMAND ----------

system_prompt_v2 = """\
You are a knowledgeable corporate affairs assistant helping employees navigate \
company policies, procedures, and general corporate information.

## Instructions
- Answer the employee's question based ONLY on the provided context documents.
- If the context does not contain enough information, clearly state that and suggest \
who to contact (e.g., HR, Legal, Finance).
- Cite specific documents or sections when possible.
- Use a professional but approachable tone.
- For policy questions, always note the effective date if available.

## Context from Corporate Documents
{{context}}

## Employee Question
{{question}}
"""

prompt_v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=system_prompt_v2,
    commit_message="v2: Enhanced with structured instructions, citations, and fallback guidance",
)

print(f"Registered prompt '{prompt_v2.name}' version {prompt_v2.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Attach Model Configuration
# MAGIC
# MAGIC Store recommended LLM parameters alongside the prompt version.

# COMMAND ----------

model_config = PromptModelConfig(
    model_name="databricks-claude-sonnet-4-5",
    temperature=0.1,
    max_tokens=1000,
)

mlflow.genai.set_prompt_model_config(
    name=PROMPT_NAME,
    version=prompt_v2.version,
    model_config=model_config,
)

print(f"Model config attached to version {prompt_v2.version}")

# Verify
loaded = mlflow.genai.load_prompt(PROMPT_NAME, version=prompt_v2.version)
print(f"Model config: {loaded.model_config}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Test the Prompt

# COMMAND ----------
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

# Sample test
sample_context = """
Document: Employee Handbook - Remote Work Policy (Effective: Jan 2025)
Section 3.2: Employees in eligible roles may work remotely up to 3 days per week.
Remote work arrangements must be approved by the employee's direct manager.
Equipment stipend of $500 is available for home office setup.
"""

sample_question = "How many days can I work from home?"

# Load and format the prompt
prompt = mlflow.genai.load_prompt(PROMPT_NAME, version=prompt_v2.version)
formatted = prompt.format(context=sample_context, question=sample_question)

response = openai_client.chat.completions.create(
    model=prompt.model_config["model_name"],
    messages=[{"role": "user", "content": formatted}],
    temperature=prompt.model_config["temperature"],
    max_tokens=prompt.model_config["max_tokens"],
)

print("Response:")
print(response.choices[0].message.content)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Set Production Alias
# MAGIC
# MAGIC Once satisfied with the prompt quality, set the `production` alias
# MAGIC so the deployed agent picks it up automatically.

# COMMAND ----------

mlflow.genai.set_prompt_alias(
    name=PROMPT_NAME,
    alias="production",
    version=prompt_v2.version,
)

print(f"Alias 'production' now points to version {prompt_v2.version}")

# Verify alias-based loading (this is what the agent uses)
prod_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"Loaded via @production alias: version {prod_prompt.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | API | Purpose |
# MAGIC |------|-----|---------|
# MAGIC | Register prompt | `mlflow.genai.register_prompt()` | Create versioned prompt template |
# MAGIC | Attach config | `mlflow.genai.set_prompt_model_config()` | Store LLM parameters with prompt |
# MAGIC | Load prompt | `mlflow.genai.load_prompt()` | Retrieve specific version or alias |
# MAGIC | Format | `prompt.format(...)` | Fill template variables |
# MAGIC | Set alias | `mlflow.genai.set_prompt_alias()` | Point alias to specific version |
# MAGIC
# MAGIC The deployed agent uses `mlflow.genai.load_prompt("prompts:/<name>@production")`
# MAGIC to dynamically load the current production prompt without redeployment.
