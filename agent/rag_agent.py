"""
RAG Agent - Corporate Affairs Chatbot using ResponsesAgent (MLflow 3.x)

This is the main agent file that gets logged to MLflow and deployed.
It uses:
- ResponsesAgent for the agent interface (MLflow 3.x)
- Databricks Vector Search for retrieval
- MLflow Prompt Registry for versioned prompt management
- ModelConfig for parameterised configuration (no hardcoded catalog/schema)
- On-Behalf-Of (OBO) user authorization for per-user UC permissions
"""

import uuid
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from typing import Generator

# ---------------------------------------------------------------------------
# Configuration - loaded from model_config (set at log_model time)
# ---------------------------------------------------------------------------
config = mlflow.models.ModelConfig()

LLM_ENDPOINT_NAME = config.get("llm_endpoint")
VECTOR_SEARCH_INDEX = config.get("vector_search_index")
PROMPT_NAME = config.get("prompt_name")
PROMPT_ALIAS = config.get("prompt_alias")

# Enable automatic tracing for OpenAI-compatible calls
mlflow.openai.autolog()


class CorporateAffairsAgent(ResponsesAgent):
    """
    RAG agent for corporate affairs Q&A.

    Uses Databricks Vector Search for retrieval and MLflow Prompt Registry
    for versioned system prompts.

    Supports On-Behalf-Of (OBO) user authorization: when deployed via
    agents.deploy() or Databricks Apps, Vector Search queries run with
    the calling user's UC permissions. Falls back to the service principal
    during local development and testing.
    """

    def __init__(self):
        super().__init__()

        from databricks_openai import DatabricksOpenAI

        # LLM client is shared across requests (stateless, uses SP creds)
        self.openai_client = DatabricksOpenAI()

    # ------------------------------------------------------------------
    # OBO: obtain a workspace client scoped to the calling user
    # ------------------------------------------------------------------
    def _get_workspace_client(self):
        """
        Return a WorkspaceClient scoped to the calling user (OBO).

        Must be called inside predict(), NOT in __init__(), because
        user credentials are only available at request time via the
        x-forwarded-access-token header.

        Falls back to the service-principal WorkspaceClient for local
        development, testing, and non-OBO serving deployments.
        """
        try:
            from databricks.agents import get_user_workspace_client
            return get_user_workspace_client()
        except Exception:
            from databricks.sdk import WorkspaceClient
            return WorkspaceClient()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    @mlflow.trace
    def _retrieve_context(self, question: str) -> str:
        """Retrieve relevant documents from Vector Search using OBO client."""
        from databricks.vector_search.client import VectorSearchClient

        try:
            vsc = VectorSearchClient(
                workspace_client=self._get_workspace_client(),
                disable_notice=True,
            )
            index = vsc.get_index(index_name=VECTOR_SEARCH_INDEX)
            results = index.similarity_search(
                query_text=question,
                columns=["content", "source_file", "department"],
                num_results=5,
            )
            docs = results.get("result", {}).get("data_array", [])
            if not docs:
                return "No relevant documents found."
            return "\n\n".join(
                f"[Source: {row[1]} | Dept: {row[2]}]\n{row[0]}" for row in docs
            )
        except Exception as e:
            return f"Error retrieving documents: {e}"

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _load_and_format_prompt(self, context: str, question: str) -> str:
        """
        Load the prompt from MLflow Prompt Registry and fill in variables.

        The prompt template contains {{context}} and {{question}} placeholders.
        By loading at query time with a short alias TTL (60s default),
        prompt updates propagate without redeploying the agent.
        """
        try:
            prompt = mlflow.genai.load_prompt(
                f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}"
            )
            return prompt.format(context=context, question=question)
        except Exception:
            # Fallback prompt if registry unavailable
            return (
                "You are a helpful corporate affairs assistant. "
                "Answer the employee's question based ONLY on the provided context. "
                "If you don't know, say so. Cite your sources.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Handle a user request using RAG.

        Steps:
        1. Retrieve relevant documents from Vector Search (OBO)
        2. Load and format prompt from MLflow Prompt Registry
        3. Generate response with Claude via Foundation Model API
        4. Return formatted response with source citations
        """
        # Extract user message
        user_message = ""
        for msg in request.input:
            if msg.role == "user":
                user_message = msg.content

        # Step 1: Retrieve context (uses OBO client for UC permissions)
        context = self._retrieve_context(user_message)

        # Step 2: Load prompt from Prompt Registry and fill in variables
        formatted_prompt = self._load_and_format_prompt(
            context=context, question=user_message
        )

        # Step 3: Build messages for the LLM
        messages = [
            {"role": "user", "content": formatted_prompt},
        ]

        # Step 4: Call LLM
        response = self.openai_client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        # Return using ResponsesAgent helper method
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=answer, id=f"msg_{uuid.uuid4().hex[:8]}"
                )
            ]
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming version - delegates to non-streaming for simplicity."""
        result = self.predict(request)
        for item in result.output:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=item,
            )


# ---------------------------------------------------------------------------
# Export for MLflow model logging
# ---------------------------------------------------------------------------
AGENT = CorporateAffairsAgent()
mlflow.models.set_model(AGENT)
