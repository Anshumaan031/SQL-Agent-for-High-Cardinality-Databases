"""
SQL Generator Agent
Writes and tests SQL queries using SQL tools
"""
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from agent.tools.sql_tools import sql_tools


# Global LLM instance - will be set by main agent
_llm: ChatGoogleGenerativeAI = None


def set_sql_generator_llm(api_key: str):
    """Initialize the LLM for the SQL generator agent"""
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )


def sql_generator_agent(state: AgentState) -> AgentState:
    """
    SQL Generator Agent - Writes and tests SQL queries.

    This agent:
    1. Drafts a SQL query based on intent and tables
    2. Validates syntax using validate_sql_syntax
    3. Checks execution plan with explain_query_plan
    4. Executes the query with execute_sql_query
    5. Returns results or error

    Uses LangGraph ToolNode pattern - tools are executed by the graph.

    Args:
        state: Current agent state

    Returns:
        Updated state with LLM response (may contain tool_calls)
    """
    user_query = state["user_query"]
    intent = state.get("intent", user_query)
    selected_tables = state.get("selected_tables", [])
    table_schemas = state.get("table_schemas", {})
    disambiguated_values = state.get("disambiguated_values", {})
    messages = state["messages"]

    # Format table schemas for prompt
    schema_text = ""
    for table, schema in table_schemas.items():
        schema_text += f"\nTable: {table}\n"
        schema_text += f"  Columns: {', '.join(schema.get('columns', []))}\n"
        if schema.get('foreign_keys'):
            schema_text += f"  Foreign Keys: {', '.join(schema['foreign_keys'])}\n"

    # Format disambiguated values
    disambiguation_text = ""
    if disambiguated_values:
        disambiguation_text = "\n\nIMPORTANT - Use these exact values:\n"
        for original, actual in disambiguated_values.items():
            disambiguation_text += f"  - Instead of '{original}', use '{actual}'\n"

    system_prompt = f"""You are a SQL writing and testing specialist.

User wants: {intent}
Original query: "{user_query}"

Database Schema:
{schema_text}
{disambiguation_text}

Your task:
1. Write a SQL query to answer the user's question
2. Use validate_sql_syntax(sql) to check if it's valid
3. Use execute_sql_query(sql) to run it and get results
4. If it works, you're done! If not, revise and try again

Available tools:
- validate_sql_syntax(sql): Check if SQL is valid
- explain_query_plan(sql): See how the query will execute
- execute_sql_query(sql): Run the query and get results

Rules:
- Use SQLite syntax
- Use the EXACT disambiguated values provided
- Use JOIN clauses when querying multiple tables
- Return only necessary columns
- Keep queries efficient

Start by writing the SQL query, then test it."""

    # Check if we need to add system message (first call only)
    has_system_msg = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system_msg:
        messages = [SystemMessage(content=system_prompt)] + list(messages)

    # Bind tools to LLM
    llm_with_tools = _llm.bind_tools(sql_tools)

    # Invoke LLM
    response = llm_with_tools.invoke(messages)

    # Extract results from tool messages
    results = None
    error = None
    candidate_sql = None

    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # Look for execute_sql_query results
            if '"success"' in msg.content:
                import json
                try:
                    result = json.loads(msg.content) if msg.content.startswith('{') else eval(msg.content)
                    if isinstance(result, dict):
                        if result.get('success'):
                            results = result.get('results')
                            error = None
                        else:
                            error = result.get('error')
                except:
                    pass

    # Try to extract SQL from response content
    if hasattr(response, 'content'):
        import re
        # Convert content to string if it's a list
        content = response.content
        if isinstance(content, list):
            content = ' '.join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)

        sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            candidate_sql = sql_match.group(1).strip()
        else:
            select_match = re.search(r'(SELECT\s+.*?;?)\s*$', content, re.DOTALL | re.IGNORECASE)
            if select_match:
                candidate_sql = select_match.group(1).strip()

    return {
        **state,
        "messages": [response],
        "candidate_sql": candidate_sql or state.get("candidate_sql"),
        "results": results,
        "error": error
    }


def should_continue_sql_generator(state: AgentState) -> str:
    """
    Determine whether to continue to tools, go to debugger, or finish.

    Returns:
        "tools" if agent wants to use tools
        "supervisor" if done (either success or failed)
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done - go back to supervisor
    # Supervisor will decide if we need debugger based on error state
    return "supervisor"


# Export agent, routing function, and tools
__all__ = ["sql_generator_agent", "should_continue_sql_generator", "sql_tools"]
