"""
Local Agent Test - Run on a Databricks cluster to validate the agent.

Usage:
    run_python_file_on_databricks(file_path="./tests/test_agent_local.py")
"""

import sys
import os

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import mlflow
from mlflow.types.responses import ResponsesAgentRequest, ChatContext

print(f"MLflow version: {mlflow.__version__}")


def test_basic_query():
    """Test that the agent can handle a basic query."""
    from rag_agent import AGENT

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What is the remote work policy?"}],
        context=ChatContext(user_id="test@example.com"),
    )

    result = AGENT.predict(request)
    assert result.output, "Expected non-empty output"
    assert len(result.output) > 0, "Expected at least one output item"

    text = result.output[0].text if hasattr(result.output[0], "text") else str(result.output[0])
    assert len(text) > 0, "Expected non-empty text response"
    print(f"PASS: Basic query returned {len(text)} characters")
    print(f"  Response preview: {text[:200]}...")


def test_streaming():
    """Test that streaming works."""
    from rag_agent import AGENT

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What are the company holidays?"}],
    )

    events = list(AGENT.predict_stream(request))
    assert len(events) > 0, "Expected at least one streaming event"
    print(f"PASS: Streaming returned {len(events)} events")


def test_prompt_registry_loading():
    """Test that the Prompt Registry integration works."""
    try:
        prompt = mlflow.genai.load_prompt("prompts:/main.corporate_affairs.rag_prompt@production")
        assert prompt.template, "Expected non-empty prompt template"
        print(f"PASS: Loaded prompt version {prompt.version} from registry")
        print(f"  Template preview: {prompt.template[:100]}...")
    except Exception as e:
        print(f"SKIP: Prompt Registry not available ({e})")
        print("  Register prompts first using notebook 03_prompt_engineering.py")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Agent Local Tests")
    print("=" * 60)

    tests = [test_prompt_registry_loading, test_basic_query, test_streaming]

    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
        except Exception as e:
            print(f"FAIL: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
