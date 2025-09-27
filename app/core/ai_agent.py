from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.config.settings import settings

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    # Initialize LLM
    llm = ChatGroq(model=llm_id)

    # Add Tavily tool only if allowed
    tools = [TavilySearch(max_results=2)] if allow_search else []

    # Create the agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    # Ensure query is always wrapped as HumanMessage
    if isinstance(query, list):
        user_messages = [HumanMessage(content=q) for q in query]
    else:
        user_messages = [HumanMessage(content=query)]

    # Build conversation state with system + user messages
    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            *user_messages
        ]
    }

    # Run the agent
    response = agent.invoke(state)

    # Extract AI messages
    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    return ai_messages[-1] if ai_messages else "No response generated."
