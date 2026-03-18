# RAG LLMOps Demo

End-to-end LLMOps demonstration for a RAG chatbot agent on Databricks, showcasing the complete lifecycle from data preparation through production monitoring.

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Agent Framework** | MLflow 3.x `ResponsesAgent` |
| **Prompt Management** | MLflow Prompt Registry with versioning and aliases |
| **Retrieval** | Databricks Vector Search (managed embeddings + similarity search) |
| **Evaluation** | `mlflow.genai.evaluate()` with built-in scorers |
| **Deployment** | `databricks.agents.deploy()` with AI Gateway |
| **OBO Auth** | `CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS` + optional Databricks Apps |
| **Monitoring** | MLflow External Monitor + AI Gateway Inference Tables |
| **Orchestration** | Databricks Asset Bundles (DABs) |

## Project Structure

```
rag-llmops-demo/
├── agent/                          # Agent source code (logged to MLflow)
│   └── rag_agent.py                # ResponsesAgent with Prompt Registry + VS
├── notebooks/                      # Databricks notebooks (run sequentially)
│   ├── 01_data_ingestion.py        # Ingest documents to Delta
│   ├── 02_vector_index_creation.py # Create Vector Search index
│   ├── 03_prompt_engineering.py    # Register prompts in MLflow Prompt Registry
│   ├── 04_agent_build.py           # Log ResponsesAgent to MLflow + UC
│   ├── 05_agent_evaluation.py      # Evaluate with mlflow.genai.evaluate()
│   ├── 06_model_registration.py    # Model governance (aliases, tags)
│   ├── 07_endpoint_deployment.py   # Deploy with agents.deploy() + OBO
│   ├── 08_inference_testing.py     # Endpoint testing & latency benchmarks
│   └── 09_monitoring_dashboard.py  # Production monitoring queries
├── resources/                      # DAB resource definitions
│   ├── model_artifacts.yml         # UC model, experiment, schema
│   ├── data_preparation.job.yml    # Data pipeline job
│   ├── build_evaluate.job.yml      # Build + evaluate pipeline job
│   ├── deploy.job.yml              # Deployment job
│   ├── monitoring.job.yml          # Scheduled monitoring job
│   └── end_to_end.job.yml          # Orchestrator job (runs all stages)
├── docs/                           # Documentation
│   ├── obo_setup_guide.md          # OBO setup instructions
│   └── troubleshooting_guide.md    # All issues encountered and fixes
├── databricks.yml                  # DAB bundle configuration
├── pyproject.toml                  # Python dependencies (uv, MLflow 3.x)
└── README.md                       # This file
```

## Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks CLI configured (`databricks auth login`)
- Python 3.10+

### 1. Configure

Edit `databricks.yml` to set your workspace URL and update the variables (catalog, schema, model name, endpoints) to match your environment.

### 2. Deploy Infrastructure

```bash
databricks bundle deploy -t dev
```

### 3. Run the Full Pipeline (Single Trigger)

```bash
# Runs all stages: data prep -> build/evaluate -> deploy -> monitor
databricks bundle run end_to_end -t dev
```

Or run individual stages:

```bash
databricks bundle run data_preparation -t dev
databricks bundle run build_evaluate -t dev
databricks bundle run deploy_agent -t dev
databricks bundle run monitoring -t dev
```

## LLMOps Lifecycle

```
Data Prep -> Prompt Registry -> Agent Build -> Evaluate -> Deploy -> Test -> Monitor
  (01, 02)      (03)              (04)         (05, 06)    (07)     (08)    (09)
```

1. **Data Preparation**: Ingest docs, create Vector Search index
2. **Prompt Engineering**: Register versioned prompts, set `@production` alias
3. **Agent Build**: Log ResponsesAgent with resources to MLflow + UC
4. **Evaluation**: Run `mlflow.genai.evaluate()`, enforce quality gates
5. **Deployment**: `agents.deploy()` with AI Gateway + Review App
6. **Monitoring**: MLflow External Monitor (automated judges) + inference table analytics

## Technologies

- **MLflow 3.x**: ResponsesAgent, Prompt Registry, GenAI Evaluate, Tracing
- **Databricks Foundation Models**: Claude Sonnet 4.5 (RAG), Claude Opus 4.1 (evaluation judge) via AI Gateway
- **Databricks Vector Search**: Managed embedding + retrieval
- **Unity Catalog**: Model registry, governance, permissions
- **Databricks Asset Bundles**: Infrastructure as Code

## Key Design Decisions

1. **ResponsesAgent over ChatAgent**: MLflow 3.x's Responses API interface (uses `input`/`output`, not `messages`/`choices`)
2. **File-based logging**: Agent code is logged as a file, not pickled - more portable and debuggable
3. **Prompt Registry over local files**: Version control, aliases, and hot-reload without redeployment
4. **agents.deploy() over manual endpoint creation**: Automatic AI Gateway, inference tables, and review app
5. **UC aliases over stages**: `@candidate` / `@champion` instead of deprecated stage transitions
6. **Built-in scorers over custom evaluation**: MLflow-native evaluation with full tracking integration
