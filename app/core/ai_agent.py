from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.config.settings import settings


def get_response_from_ai_agents(
    llm_id: str,
    query,
    allow_search: bool,
    system_prompt: str
) -> str:
    """
    Main function to get responses from multiple AI agents (LLM + Tavily search).

    Args:
        llm_id (str): The name or ID of the LLM to use.
        query (str | list[str]): User query or list of queries.
        allow_search (bool): Whether to enable Tavily web search.
        system_prompt (str): The system or context prompt for the agent.

    Returns:
        str: The generated AI response.
    """

    # ----------------------------
    # 1. Initialize the language model
    # ----------------------------
    llm = ChatGroq(model=llm_id)

    # ----------------------------
    # 2. Configure tools
    # ----------------------------
    tools = []
    if allow_search:
        # ✅ Properly configured TavilySearch tool
        # "search_depth" must be "basic" or "advanced"
        # "max_results" controls how many results are fetched
        tavily_tool = TavilySearch(max_results=5, search_depth="advanced")
        tools.append(tavily_tool)

    # ----------------------------
    # 3. Create the ReAct-style agent with LangGraph
    # ----------------------------
    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    # ----------------------------
    # 4. Build conversation state
    # ----------------------------
    if isinstance(query, list):
        user_messages = [HumanMessage(content=q) for q in query]
    else:
        user_messages = [HumanMessage(content=query)]

    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            *user_messages
        ]
    }

    # ----------------------------
    # 5. Invoke the agent
    # ----------------------------
    try:
        response = agent.invoke(state)
    except Exception as e:
        return f"Error while invoking agent: {e}"

    # ----------------------------
    # 6. Extract and return AI responses
    # ----------------------------
    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    if not ai_messages:
        return "No response generated."

    # Return only the latest AI message
    return ai_messages[-1]
