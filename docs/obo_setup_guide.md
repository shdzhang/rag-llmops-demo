# On-Behalf-Of (OBO) Setup Guide

## What is OBO?

On-Behalf-Of (OBO) authentication allows your deployed agent to make API calls
**as the end user** rather than as the service principal. This enables:

- **Per-user data isolation**: Vector Search queries respect the user's Unity Catalog permissions
- **Audit trail**: All actions are logged under the user's identity
- **Compliance**: Row-level security and column masking apply per user

## Architecture

```
End User -> Databricks App (OBO proxy) -> Agent Endpoint
                |                              |
                |-- User's OAuth token ------->|
                |                              |-- Vector Search (as user)
                |                              |-- LLM endpoint (as service)
```

## Two Deployment Modes

### Mode 1: Standard (agents.deploy) - No OBO

This is the default mode used in notebook `07_endpoint_deployment.py`.

- The agent runs as the **service principal**
- All users share the same permissions
- Simpler to set up, good for most use cases

```python
from databricks import agents
agents.deploy(model_name="main.corp.chatbot", model_version="1")
```

### Mode 2: Full OBO via Databricks Apps

For user-level data isolation, deploy as a **Databricks App** that proxies
requests to the Model Serving endpoint.

#### Step 1: Create the App

```python
# app.py - FastAPI app with OBO support
from fastapi import FastAPI, Request
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    # Get the user's token from the forwarded header
    user_token = request.headers.get("x-forwarded-access-token")

    if user_token:
        # Create a workspace client authenticated as the user
        config = Config(
            host=os.environ["DATABRICKS_HOST"],
            token=user_token,
        )
        user_client = WorkspaceClient(config=config)
    else:
        # Fallback to service principal
        user_client = WorkspaceClient()

    # Query the agent endpoint on behalf of the user
    body = await request.json()
    response = user_client.serving_endpoints.query(
        name=os.environ["MODEL_SERVING_ENDPOINT"],
        messages=body["messages"],
    )

    return response.as_dict()
```

#### Step 2: Configure app.yml

```yaml
command:
  - uvicorn
  - app:app
  - --host=0.0.0.0
  - --port=8000

permissions:
  - permission: CAN_USE
    group_name: users

env:
  - name: DATABRICKS_HOST
    value: "${workspace.host}"
  - name: MODEL_SERVING_ENDPOINT
    value: "agents_<catalog>-<schema>-<model>"  # Replace with your actual endpoint name
```

#### Step 3: Deploy with DAB

Add to `databricks.yml`:

```yaml
resources:
  apps:
    chatbot_app:
      name: "corp-chatbot"
      source_code_path: ./app
      config:
        command:
          - uvicorn
          - app:app
          - --host=0.0.0.0
          - --port=8000
      permissions:
        - user_name: users
          level: CAN_USE
```

#### Step 4: Verify OBO

```python
# Test that user identity is passed through
import requests

response = requests.post(
    "https://<app-url>/chat",
    json={"messages": [{"role": "user", "content": "Who am I?"}]},
    headers={"Authorization": f"Bearer {user_token}"},
)
```

## Security Considerations

1. **Token Validation**: Always validate the `x-forwarded-access-token` before using it
2. **Scopes**: The OBO token has the same permissions as the user
3. **Audit Logging**: All API calls made via OBO are logged under the user's identity
4. **Token Expiry**: OBO tokens have the same expiry as the user's session token

## When to Use OBO

| Scenario | Recommended Mode |
|----------|-----------------|
| All users see the same data | Standard (agents.deploy) |
| Different users have different permissions | OBO via Databricks App |
| Regulatory compliance requires per-user audit | OBO via Databricks App |
| Quick POC / demo | Standard (agents.deploy) |
