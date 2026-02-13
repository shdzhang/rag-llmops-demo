# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Vector Search Index Creation
# MAGIC
# MAGIC Create a Databricks Vector Search index over the document chunks table.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Create or get a Vector Search endpoint
# MAGIC 2. Create a Delta Sync index with managed embeddings
# MAGIC 3. Wait for the index to be ready
# MAGIC 4. Test retrieval

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import time

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.corporate_docs_index"
VS_ENDPOINT_NAME = "corporate_affairs_vs_endpoint"
EMBEDDING_MODEL = "databricks-gte-large-en"

print(f"Source table: {SOURCE_TABLE}")
print(f"Index name: {INDEX_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Create Vector Search Endpoint (if needed)

# COMMAND ----------

vsc = VectorSearchClient()

# Check if endpoint exists
try:
    endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists (status: {endpoint.get('endpoint_status', {}).get('state', 'unknown')})")
except Exception:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'...")
    vsc.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD",
    )
    print("Endpoint creation initiated. This may take a few minutes.")

    # Wait for endpoint
    for i in range(60):
        try:
            endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
            state = endpoint.get("endpoint_status", {}).get("state", "unknown")
            if state == "ONLINE":
                print(f"Endpoint is ONLINE!")
                break
            print(f"  [{i * 30}s] Endpoint state: {state}")
        except Exception:
            pass
        time.sleep(30)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Enable Change Data Feed on Source Table

# COMMAND ----------

spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Enabled Change Data Feed on {SOURCE_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Create Delta Sync Index with Managed Embeddings

# COMMAND ----------

try:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=INDEX_NAME)
    print(f"Index '{INDEX_NAME}' already exists")
except Exception:
    print(f"Creating index '{INDEX_NAME}'...")
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        primary_key="chunk_id",
        pipeline_type="TRIGGERED",
        embedding_source_column="content",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
        columns_to_sync=["content", "source_file", "department", "chunk_index"],
    )
    print("Index creation initiated. This may take 5-15 minutes.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Wait for Index to be Ready

# COMMAND ----------

print("Waiting for index to be ready...")
for i in range(60):
    try:
        index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=INDEX_NAME)
        status = index.describe().get("status", {})
        state = status.get("ready", False)
        detailed_state = status.get("detailed_state", "UNKNOWN")

        if state:
            print(f"\nIndex is READY!")
            break

        print(f"  [{i * 30}s] State: {detailed_state}")
    except Exception as e:
        print(f"  [{i * 30}s] Waiting... ({e})")

    time.sleep(30)
else:
    print("Timeout waiting for index. Check Databricks UI for status.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Test Retrieval

# COMMAND ----------

# Query the index
results = index.similarity_search(
    query_text="What is the remote work policy?",
    columns=["content", "source_file", "department"],
    num_results=3,
)

print("Search Results for 'What is the remote work policy?':\n")
for i, row in enumerate(results.get("result", {}).get("data_array", []), 1):
    print(f"--- Result {i} ---")
    print(f"Source: {row[1]}")
    print(f"Department: {row[2]}")
    print(f"Content: {row[0][:200]}...")
    print()

# COMMAND ----------

print(f"""
Vector Search Index Ready!
==========================
Endpoint: {VS_ENDPOINT_NAME}
Index: {INDEX_NAME}
Embedding Model: {EMBEDDING_MODEL}

Next: Run notebook 03_prompt_engineering.py
""")
