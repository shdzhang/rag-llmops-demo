# Architecture Overview

## LLMOps Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data Prep  │───>│   Prompt    │───>│ Agent Build │───>│  Evaluate   │───>│   Deploy    │───>│   Monitor   │
│              │    │  Registry   │    │  & Log      │    │  & Promote  │    │  & Test     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
  Notebooks:         Notebook:          Notebook:          Notebooks:         Notebooks:         Notebook:
  01, 02             03                 04                 05, 06             07, 08             09
```

## Component Details

### 1. Data Preparation (01, 02)
- Ingest documents into Delta tables
- Chunk and create embeddings
- Build Vector Search index

### 2. Prompt Engineering (03)
- Register prompts in **MLflow Prompt Registry**
- Version prompts with commit messages
- Attach model configuration (temperature, max_tokens)
- Set `@production` alias for deployment

### 3. Agent Build (04)
- **ResponsesAgent** (MLflow 3.x) - modern agent interface
- File-based logging with `mlflow.pyfunc.log_model(python_model="agent/rag_agent.py")`
- Declare resources: `DatabricksServingEndpoint`, `DatabricksVectorSearchIndex`
- Register to Unity Catalog with `@candidate` alias

### 4. Evaluation (05)
- **mlflow.genai.evaluate()** with built-in scorers
- Scorers: `Correctness`, `Guidelines` (professional tone, source citation)
- Quality gates: Minimum thresholds for each metric
- Auto-promote to `@champion` on pass

### 5. Deployment & Testing (07, 08)
- **databricks.agents.deploy()** for managed deployment
- Enables AI Gateway, Inference Tables, Review App
- OBO support via Databricks Apps (optional)
- Endpoint testing: basic queries, edge cases, latency benchmarks

### 6. Monitoring (09)
- AI Gateway Inference Tables for request/response logging
- Latency percentiles (P50, P95, P99)
- Token usage and cost estimation
- Alerting via Databricks SQL

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | MLflow 3.x ResponsesAgent |
| LLM | Databricks Foundation Models (Claude Sonnet 4.5) |
| Retrieval | Databricks Vector Search |
| Prompt Management | MLflow Prompt Registry |
| Evaluation | MLflow GenAI Evaluate |
| Model Registry | Unity Catalog |
| Deployment | databricks.agents.deploy() |
| Monitoring | AI Gateway Inference Tables |
| Orchestration | Databricks Asset Bundles (DABs) |

## Key Design Decisions

1. **ResponsesAgent over ChatAgent**: MLflow 3.x's latest interface with OpenAI-compatible I/O
2. **File-based logging**: Agent code is logged as a file, not pickled - more portable and debuggable
3. **Prompt Registry over local files**: Version control, aliases, and hot-reload without redeployment
4. **agents.deploy() over manual endpoint creation**: Automatic AI Gateway, inference tables, and review app
5. **UC aliases over stages**: `@candidate` / `@champion` instead of deprecated stage transitions
6. **Built-in scorers over custom evaluation**: MLflow-native evaluation with full tracking integration
