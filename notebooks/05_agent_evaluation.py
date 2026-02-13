# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Agent Evaluation with MLflow GenAI
# MAGIC
# MAGIC This notebook evaluates the RAG agent using **MLflow GenAI Evaluate**:
# MAGIC 1. Creates an evaluation dataset
# MAGIC 2. Runs the agent against test cases
# MAGIC 3. Applies built-in and custom LLM-as-judge scorers
# MAGIC 4. Enforces quality gates for promotion

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-agents databricks-vectorsearch openai pandas
# MAGIC %restart_python

# COMMAND ----------
import mlflow
import pandas as pd

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Use the same experiment as notebook 04 so all runs are grouped together
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

# Model endpoints
AGENT_LLM = "databricks-claude-sonnet-4-5"       # Agent uses Sonnet 4/5
JUDGE_LLM = "databricks:/databricks-claude-opus-4-6"  # LLM Judge uses Opus 4/6 (databricks:/ prefix required)

# Quality gates (metric names match scorer output keys)
QUALITY_THRESHOLDS = {
    "correctness": 0.7,         # 70% of responses must be correct
    "professional_tone": 0.8,   # 80% must have professional tone
}

print(f"MLflow version: {mlflow.__version__}")
print(f"Evaluating: {UC_MODEL_NAME}@candidate")
print(f"Agent LLM: {AGENT_LLM}")
print(f"Judge LLM: {JUDGE_LLM}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Prepare Evaluation Dataset
# MAGIC
# MAGIC Uses `mlflow.genai.datasets` to create a tracked evaluation dataset.

# COMMAND ----------

# Define test cases with expected outputs (ground truth)
eval_data = [
    {
        "inputs": {"question": "What is the remote work policy?"},
        "expectations": {
            "expected_response": (
                "Employees in eligible roles may work remotely up to 3 days per week. "
                "Remote work arrangements must be approved by the direct manager."
            ),
        },
    },
    {
        "inputs": {"question": "How much is the equipment stipend for home office?"},
        "expectations": {
            "expected_response": (
                "The company provides a $500 equipment stipend for home office setup."
            ),
        },
    },
    {
        "inputs": {"question": "What is the parental leave policy?"},
        "expectations": {
            "expected_response": (
                "The company offers 16 weeks of paid parental leave for primary caregivers "
                "and 8 weeks for secondary caregivers."
            ),
        },
    },
    {
        "inputs": {"question": "How do I submit an expense report?"},
        "expectations": {
            "expected_response": (
                "Expense reports should be submitted through the Concur system within "
                "30 days of the expense. Receipts are required for expenses over $25."
            ),
        },
    },
    {
        "inputs": {"question": "What are the company holidays for 2025?"},
        "expectations": {
            "expected_response": (
                "The company observes standard US federal holidays plus the week between "
                "Christmas and New Year's Day as additional company holidays."
            ),
        },
    },
]

eval_df = pd.DataFrame(eval_data)
print(f"Evaluation dataset: {len(eval_df)} test cases")
eval_df.head()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Define the Predict Function
# MAGIC
# MAGIC Wraps the logged model so `mlflow.genai.evaluate()` can call it.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

# Initialize clients once (shared across all predict_fn calls)
_w = WorkspaceClient()
_openai_client = _w.serving_endpoints.get_open_ai_client()
_vsc = VectorSearchClient()

VS_INDEX = f"{CATALOG}.{SCHEMA}.corporate_docs_index"
VS_ENDPOINT = "corporate_affairs_vs_endpoint"

_vs_index = _vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)


def predict_fn(question: str) -> str:
    """
    Predict function for evaluation - runs the full RAG pipeline.

    The parameter name ('question') must match the key in eval_data['inputs'].
    mlflow.genai.evaluate() calls this as predict_fn(question="...").

    Steps:
    1. Retrieve relevant docs from Vector Search
    2. Load prompt from Prompt Registry
    3. Format prompt with retrieved context + question
    4. Call the LLM
    """
    # Step 1: Retrieve context from Vector Search
    try:
        results = _vs_index.similarity_search(
            query_text=question,
            columns=["content", "source_file"],
            num_results=3,
        )
        docs = results.get("result", {}).get("data_array", [])
        context = "\n\n".join(
            f"[Source: {row[1]}]\n{row[0]}" for row in docs
        )
    except Exception as e:
        context = f"(retrieval error: {e})"

    # Step 2-3: Load and format prompt from Prompt Registry
    prompt = mlflow.genai.load_prompt(f"prompts:/{CATALOG}.{SCHEMA}.rag_prompt@production")
    formatted = prompt.format(context=context, question=question)

    # Step 4: Call the LLM
    response = _openai_client.chat.completions.create(
        model=AGENT_LLM,
        messages=[{"role": "user", "content": formatted}],
        temperature=0.1,
        max_tokens=500,
    )

    return response.choices[0].message.content


# Quick test
test_response = predict_fn("What is the remote work policy?")
print(f"Test response: {test_response[:200]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Define Scorers
# MAGIC
# MAGIC Using MLflow's built-in scorers and custom judges.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, Guidelines

# Built-in scorers - all using Opus 4/6 as the LLM judge
scorers = [
    # Checks if the response is correct given the expected answer
    Correctness(model=JUDGE_LLM),
    # Custom guideline scorer for corporate tone
    Guidelines(
        name="professional_tone",
        model=JUDGE_LLM,
        guidelines=(
            "The response should maintain a professional and helpful tone. "
            "It should not be overly casual, use slang, or be dismissive. "
            "It should be clear and actionable."
        ),
    ),
    Guidelines(
        name="source_citation",
        model=JUDGE_LLM,
        guidelines=(
            "When the response references specific policies or documents, "
            "it should cite the source (e.g., document name, section number). "
            "If the information is not available, the response should clearly "
            "state that and suggest who to contact."
        ),
    ),
]

print(f"Configured {len(scorers)} scorers (judge model: {JUDGE_LLM})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Run Evaluation
# MAGIC
# MAGIC `mlflow.genai.evaluate()` orchestrates:
# MAGIC - Running the predict function on each test case
# MAGIC - Applying all scorers
# MAGIC - Logging results to MLflow experiment

# COMMAND ----------

eval_results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=predict_fn,
    scorers=scorers,
)

print("Evaluation complete!")
print(f"\nMetrics:")
for metric, value in eval_results.metrics.items():
    print(f"  {metric}: {value}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Analyze Results

# COMMAND ----------

# Display per-row results (MLflow 3.x: use search_traces instead of eval_table)
results_df = mlflow.search_traces(run_id=eval_results.run_id)
display(results_df) if "display" in dir() else print(results_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Quality Gate Check
# MAGIC
# MAGIC Verify the agent meets minimum quality thresholds before promotion.

# COMMAND ----------

metrics = eval_results.metrics

# Check quality gates
gate_results = {}
all_passed = True

for metric_name, threshold in QUALITY_THRESHOLDS.items():
    # Find matching metric (metrics may have prefixes like 'correctness/mean')
    matching_metrics = {k: v for k, v in metrics.items() if metric_name in k.lower() and "mean" in k.lower()}

    if matching_metrics:
        metric_key, value = next(iter(matching_metrics.items()))
        passed = value >= threshold
        gate_results[metric_name] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  {metric_name}: {value:.3f} (threshold: {threshold}) [{status}]")
    else:
        print(f"  {metric_name}: metric not found in results (available: {list(metrics.keys())})")

print(f"\nOverall Quality Gate: {'PASSED' if all_passed else 'FAILED'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Conditional Promotion
# MAGIC
# MAGIC If quality gates pass, promote candidate to champion.

# COMMAND ----------

if all_passed:
    client = mlflow.MlflowClient()

    # Get candidate version
    candidate = client.get_model_version_by_alias(UC_MODEL_NAME, "candidate")
    version = candidate.version

    # Promote to champion
    client.set_registered_model_alias(
        name=UC_MODEL_NAME,
        alias="champion",
        version=version,
    )

    print(f"Model version {version} promoted to 'champion'!")
    print(f"  {UC_MODEL_NAME}@champion -> version {version}")

    # Set task value for deployment notebook
    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="promoted_version", value=version)
        dbutils.jobs.taskValues.set(key="quality_gate_passed", value=True)
else:
    print("Quality gates FAILED - model NOT promoted.")
    print("Review the evaluation results above and improve the agent.")

    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="quality_gate_passed", value=False)

    # Optionally fail the notebook to stop the pipeline
    # raise Exception("Quality gate check failed")
