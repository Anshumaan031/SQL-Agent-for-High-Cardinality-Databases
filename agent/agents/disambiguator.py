"""
Value Disambiguator Agent
Uses tools to resolve fuzzy/ambiguous values
"""
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from agent.tools.disambiguator_tools import disambiguator_tools


# Global LLM instance - will be set by main agent
_llm: ChatGoogleGenerativeAI = None


def set_disambiguator_llm(api_key: str):
    """Initialize the LLM for the disambiguator agent"""
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )


def disambiguator_agent(state: AgentState) -> AgentState:
    """
    Value Disambiguator Agent - Resolves fuzzy/ambiguous values.

    This agent:
    1. Identifies which entities need disambiguation
    2. Determines which table/column they likely belong to
    3. Uses vector_search or fuzzy_match to find best matches
    4. Returns disambiguated values

    Uses LangGraph ToolNode pattern - tools are executed by the graph.

    Args:
        state: Current agent state

    Returns:
        Updated state with LLM response (may contain tool_calls)
    """
    entities = state.get("entities", [])
    selected_tables = state.get("selected_tables", [])
    table_schemas = state.get("table_schemas", {})
    messages = state["messages"]

    # If no entities, skip disambiguation
    if not entities:
        from langchain_core.messages import AIMessage
        return {
            **state,
            "messages": [AIMessage(
                content="[DISAMBIGUATOR] No entities to disambiguate",
                name="disambiguator"
            )],
            "disambiguated_values": {}
        }

    # Build system prompt
    schema_summary = "\n".join([
        f"- {table}: {schema.get('columns', [])}"
        for table, schema in table_schemas.items()
    ])

    system_prompt = f"""You are a value disambiguation specialist.

Entities to resolve: {entities}
Available tables and columns:
{schema_summary}

Your task:
1. For each entity, determine which table and column it likely belongs to
2. Use vector_search_values(query, table, column, top_k=5) to find semantically similar values
3. Use get_column_values(table, column, limit=100) if you need to see what values exist
4. Select the best match with highest similarity score (ideally > 0.7)

Available tools:
- get_column_values(table, column, limit): Get sample values from a column
- vector_search_values(query, table, column, top_k): Search for similar values using embeddings
- fuzzy_match_values(query, candidates): Fuzzy string matching against candidates

For each entity, find its correct database value. When done, provide a summary of mappings found."""

    # Check if we need to add system message (first call only)
    has_system_msg = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system_msg:
        messages = [SystemMessage(content=system_prompt)] + list(messages)

    # Bind tools to LLM
    llm_with_tools = _llm.bind_tools(disambiguator_tools)

    # Invoke LLM
    response = llm_with_tools.invoke(messages)

    # Extract disambiguated values from tool results
    disambiguated_values = {}

    for msg in messages:
        # Look for vector_search_values results
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if 'similarity_score' in msg.content:
                # This is likely a vector search result
                import json
                try:
                    result = json.loads(msg.content) if msg.content.startswith('[') else eval(msg.content)
                    if isinstance(result, list) and result:
                        best_match = result[0]
                        if isinstance(best_match, dict) and 'value' in best_match:
                            similarity = best_match.get('similarity_score', 0)
                            if similarity > 0.7:
                                # Find which entity this was for by looking at previous messages
                                # This is a simplified extraction - in practice you'd track better
                                disambiguated_values[best_match['value']] = best_match['value']
                except:
                    pass

    return {
        **state,
        "messages": [response],
        "disambiguated_values": disambiguated_values
    }


def should_continue_disambiguator(state: AgentState) -> str:
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


# Export agent, routing function, and tools
__all__ = ["disambiguator_agent", "should_continue_disambiguator", "disambiguator_tools"]
