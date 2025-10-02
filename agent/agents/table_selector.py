"""
Table Selector Agent
Uses tools to find relevant database tables for the query
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from agent.tools.database_tools import database_tools


# Global LLM instance - will be set by main agent
_llm: ChatGoogleGenerativeAI = None


def set_table_selector_llm(api_key: str):
    """Initialize the LLM for the table selector agent"""
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0
    )


def table_selector_agent(state: AgentState) -> AgentState:
    """
    Table Selector Agent - Finds relevant tables using database inspection tools.

    This agent autonomously uses tools to:
    1. Get all tables or search by keyword
    2. Examine table schemas
    3. Select the minimal set needed

    Uses LangGraph ToolNode pattern - tools are executed by the graph,
    not inside this function.

    Args:
        state: Current agent state

    Returns:
        Updated state with LLM response (may contain tool_calls)
    """
    user_query = state["user_query"]
    intent = state.get("intent", user_query)
    entities = state.get("entities", [])
    messages = state["messages"]

    # Check if this is first invocation or continuation after tool execution
    # If messages contain tool results, we're continuing
    # Otherwise, we're starting fresh

    # Build prompt for first call
    system_prompt = f"""You are a table selection specialist for a SQL database.

User wants: {intent}
Original query: {user_query}
Entities mentioned: {entities}

Your task:
1. Use get_all_tables() to see all available tables
2. Use search_tables_by_keyword() to find tables related to keywords in the query
3. Use get_table_schema() to examine table structures
4. Select the MINIMAL set of tables needed to answer the query

Available tools:
- get_all_tables(): Get list of all tables
- get_table_schema(table_name): Get schema for a specific table
- search_tables_by_keyword(keyword): Search for tables by keyword

Think step by step and use tools to find the right tables. Once you've identified the tables, respond with a final summary listing the selected tables."""

    # Check if we need to add system message (first call only)
    has_system_msg = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system_msg:
        messages = [SystemMessage(content=system_prompt)] + list(messages)

    # Bind tools to LLM
    llm_with_tools = _llm.bind_tools(database_tools)

    # Invoke LLM - it will decide whether to use tools or respond
    response = llm_with_tools.invoke(messages)

    # Extract selected tables from tool results in messages
    selected_tables = []
    table_schemas = {}

    for msg in messages:
        # Look for tool results from get_table_schema
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # Try to parse table schema results
            if '"table":' in msg.content:
                # This is a tool result - try to extract table info
                import json
                try:
                    result = json.loads(msg.content) if msg.content.startswith('{') else eval(msg.content)
                    if isinstance(result, dict) and "table" in result:
                        table_name = result["table"]
                        table_schemas[table_name] = result
                        if table_name not in selected_tables:
                            selected_tables.append(table_name)
                except:
                    pass

    return {
        **state,
        "messages": [response],
        "selected_tables": selected_tables,
        "table_schemas": table_schemas
    }


def should_continue_table_selector(state: AgentState) -> str:
    """
    Determine whether to continue to tools or finish.

    Returns:
        "tools" if agent wants to use tools, "supervisor" if done
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done - go back to supervisor
    return "supervisor"


# Export both agent and routing function
__all__ = ["table_selector_agent", "should_continue_table_selector", "database_tools"]
