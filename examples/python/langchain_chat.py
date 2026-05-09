"""
LangChain chat example using AX Engine.

Requires:
    pip install langchain-core
    ax-engine-server running on http://127.0.0.1:8080
"""

from langchain_core.messages import HumanMessage, SystemMessage
from ax_engine.langchain import AXEngineChatModel

chat = AXEngineChatModel(
    base_url="http://127.0.0.1:8080",
    max_tokens=256,
    temperature=0.7,
)

# Blocking call
response = chat.invoke(
    [
        SystemMessage(content="You are AX Engine."),
        HumanMessage(content="Say hello in one sentence."),
    ]
)
print(response.content)

# Streaming
print("\n--- streaming ---")
for chunk in chat.stream([HumanMessage(content="Count from 1 to 5.")]):
    print(chunk.content, end="", flush=True)
print()
