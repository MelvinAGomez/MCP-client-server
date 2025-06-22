import os
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

async def main():
    # Initialize MCP client with tool definitions
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # Get GROQ API Key from environment
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("Missing GROQ_API_KEY in environment variables.")
    os.environ["GROQ_API_KEY"] = groq_key  # Optional if needed by the SDK

    # Load tools
    tools = await client.get_tools()

    # Instantiate model and agent
    model = ChatGroq(model="qwen-qwq-32b")
    agent = create_react_agent(model, tools)

    # Run a math query
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    # print("Math response:", math_response['messages'][-1]['content'])
    print("Math response:", math_response["messages"][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
