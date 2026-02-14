# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Production Monitoring Dashboard
# MAGIC
# MAGIC This notebook monitors the deployed agent using **AI Gateway Inference Tables**:
# MAGIC 1. Query volume and trends
# MAGIC 2. Latency analysis
# MAGIC 3. Error rate monitoring
# MAGIC 4. Token usage and cost estimation
# MAGIC 5. Quality monitoring via MLflow traces
# MAGIC
# MAGIC The inference table is automatically populated by `agents.deploy()`.

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk pandas
# MAGIC %restart_python

# COMMAND ----------
import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")

# Endpoint name using agents.deploy() convention:
# agents_{catalog}-{schema}-{model} (dots->hyphens, underscores kept)
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
ENDPOINT_NAME = f"agents_{UC_MODEL_NAME}".replace(".", "-")

# Inference table created by AI Gateway: <catalog>.<schema>.`<endpoint_with_underscores>_payload`
_endpoint_table_name = ENDPOINT_NAME.replace("-", "_")
INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.`{_endpoint_table_name}_payload`"

print(f"Monitoring endpoint: {ENDPOINT_NAME}")
print(f"Inference table: {INFERENCE_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Check: Does the Inference Table Exist?

# COMMAND ----------

try:
    row_count = spark.sql(f"SELECT count(*) AS cnt FROM {INFERENCE_TABLE}").first()["cnt"]
    print(f"Inference table has {row_count} rows")
except Exception as e:
    print(f"Inference table not found or not yet populated: {e}")
    print("Deploy the agent (notebook 07) and send some queries first.")
    dbutils.notebook.exit("Inference table not ready") if "dbutils" in dir() else None

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Query Volume Trends
# MAGIC
# MAGIC Track daily request volumes to understand usage patterns.

# COMMAND ----------

query_volume_df = spark.sql(f"""
    SELECT
      date_trunc('day', request_time) AS day,
      count(*) AS total_requests,
      count(CASE WHEN status_code = 200 THEN 1 END) AS successful,
      count(CASE WHEN status_code != 200 THEN 1 END) AS failed,
      round(count(CASE WHEN status_code = 200 THEN 1 END) * 100.0 / count(*), 1) AS success_rate_pct
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 30)
    GROUP BY 1
    ORDER BY 1 DESC
""")

display(query_volume_df) if "display" in dir() else query_volume_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Latency Analysis
# MAGIC
# MAGIC Monitor P50, P95, P99 latencies to catch performance degradation.

# COMMAND ----------

latency_df = spark.sql(f"""
    SELECT
      date_trunc('day', request_time) AS day,
      count(*) AS requests,
      round(avg(execution_duration_ms), 0) AS avg_latency_ms,
      round(percentile(execution_duration_ms, 0.5), 0) AS p50_ms,
      round(percentile(execution_duration_ms, 0.95), 0) AS p95_ms,
      round(percentile(execution_duration_ms, 0.99), 0) AS p99_ms
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 30)
      AND status_code = 200
    GROUP BY 1
    ORDER BY 1 DESC
""")

display(latency_df) if "display" in dir() else latency_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Error Analysis

# COMMAND ----------

error_df = spark.sql(f"""
    SELECT
      status_code,
      count(*) AS count,
      round(count(*) * 100.0 / sum(count(*)) OVER(), 1) AS pct
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 7)
    GROUP BY 1
    ORDER BY 2 DESC
""")

display(error_df) if "display" in dir() else error_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Token Usage & Cost Estimation
# MAGIC
# MAGIC Extract token counts from the response payload for cost tracking.

# COMMAND ----------

token_df = spark.sql(f"""
    SELECT
      date_trunc('day', request_time) AS day,
      count(*) AS requests,
      sum(get_json_object(response, '$.usage.prompt_tokens')) AS total_input_tokens,
      sum(get_json_object(response, '$.usage.completion_tokens')) AS total_output_tokens,
      sum(get_json_object(response, '$.usage.total_tokens')) AS total_tokens
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 30)
      AND status_code = 200
    GROUP BY 1
    ORDER BY 1 DESC
""")

display(token_df) if "display" in dir() else token_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Popular Questions Analysis
# MAGIC
# MAGIC Understand what employees are asking about most frequently.

# COMMAND ----------

questions_df = spark.sql(f"""
    SELECT
      get_json_object(request, '$.messages[0].content') AS question,
      count(*) AS frequency
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 7)
      AND status_code = 200
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 20
""")

display(questions_df) if "display" in dir() else questions_df.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. MLflow Trace Quality Monitoring
# MAGIC
# MAGIC When tracing is enabled (via `mlflow.openai.autolog()`), the inference table
# MAGIC includes a `trace` column with full MLflow traces. Use the MLflow Tracking
# MAGIC Server to analyze trace quality.

# COMMAND ----------

# Query recent traces programmatically
w = WorkspaceClient()

print("""
MLflow Trace Monitoring
=======================

Traces are automatically captured by mlflow.openai.autolog() and stored
alongside each request in the inference table.

To analyze traces:
1. Open MLflow UI -> Traces tab
2. Filter by endpoint name or date range
3. Inspect individual traces for:
   - Retrieval quality (were relevant docs found?)
   - LLM response quality
   - Latency breakdown (retrieval vs. generation)

For programmatic access:
  mlflow.search_traces(experiment_ids=[...])
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Alerting Configuration
# MAGIC
# MAGIC Set up Databricks SQL Alerts for proactive monitoring.

# COMMAND ----------

print(f"""
Recommended Alerts (create in Databricks SQL):
===============================================

1. Error Rate Alert:
   - Trigger: Error rate > 5% over 1 hour window
   - SQL: SELECT count(CASE WHEN status_code != 200 THEN 1 END) * 100.0 / count(*)
          FROM {INFERENCE_TABLE} WHERE request_time >= date_sub(current_timestamp(), INTERVAL 1 HOUR)
   - Action: Notify #oncall-channel

2. Latency Alert:
   - Trigger: P95 latency > 10 seconds
   - SQL: SELECT percentile(execution_duration_ms, 0.95)
          FROM {INFERENCE_TABLE} WHERE request_time >= date_sub(current_timestamp(), INTERVAL 1 HOUR)
   - Action: Notify engineering team

3. Volume Anomaly Alert:
   - Trigger: Hourly volume drops >50% vs. same hour yesterday
   - Action: Investigate potential outage

4. Cost Alert:
   - Trigger: Daily token usage exceeds budget threshold
   - Action: Notify finance + engineering
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Lakehouse Monitor (Optional)
# MAGIC
# MAGIC Create a Lakehouse Monitor on the inference table for automated
# MAGIC drift detection and quality monitoring.

# COMMAND ----------

# Uncomment to create a Lakehouse Monitor
# from databricks.sdk.service.catalog import MonitorTimeSeries
#
# w.quality_monitors.create(
#     table_name=INFERENCE_TABLE.replace("`", ""),
#     output_schema_name=f"{CATALOG}.{SCHEMA}",
#     assets_dir=f"/Workspace/Users/{w.current_user.me().user_name}/monitors/{ENDPOINT_NAME}",
#     time_series=MonitorTimeSeries(
#         timestamp_col="request_time",
#         granularities=["1 hour", "1 day"],
#     ),
# )
#
# print("Lakehouse Monitor created!")
# print("View drift reports in the Databricks UI -> Quality tab")
