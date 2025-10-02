"""
Supervisor Agent
Decides which specialist agent to invoke next
"""
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState


# Global LLM instance - will be set by main agent
_llm: ChatGoogleGenerativeAI = None


def set_supervisor_llm(api_key: str):
    """Initialize the LLM for the supervisor agent"""
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor Agent - Routes to appropriate specialist agent.

    Analyzes current state and decides which agent should act next.
    Makes intelligent routing decisions based on:
    - What has been completed
    - What errors occurred
    - Whether the task is done

    Args:
        state: Current agent state

    Returns:
        Updated state with next_agent set
    """
    user_query = state["user_query"]
    intent = state.get("intent")
    selected_tables = state.get("selected_tables", [])
    disambiguated_values = state.get("disambiguated_values", {})
    candidate_sql = state.get("candidate_sql")
    results = state.get("results")
    error = state.get("error")
    sql_attempts = state.get("sql_attempts", [])

    # Build current state summary
    state_summary = f"""Current Progress:
- User Query: "{user_query}"
- Intent Parsed: {'Yes' if intent else 'No'} {f'({intent})' if intent else ''}
- Tables Selected: {'Yes' if selected_tables else 'No'} {f'({len(selected_tables)} tables)' if selected_tables else ''}
- Values Disambiguated: {'Yes' if disambiguated_values else 'No'} {f'({len(disambiguated_values)} values)' if disambiguated_values else ''}
- SQL Generated: {'Yes' if candidate_sql else 'No'}
- Query Executed: {'Yes' if results or error else 'No'}
- Results: {'Success' if results else 'Failed' if error else 'Not yet'}
- SQL Attempts: {len(sql_attempts)}
- Error: {error if error else 'None'}"""

    prompt = f"""You are a supervisor managing SQL query generation agents.

{state_summary}

Available Specialist Agents:
1. PLANNER - Analyzes query and extracts intent/entities (call this first!)
2. TABLE_SELECTOR - Finds relevant database tables
3. DISAMBIGUATOR - Resolves fuzzy values like "ACDC" -> "AC/DC"
4. SQL_GENERATOR - Writes and tests SQL queries
5. SQL_DEBUGGER - Fixes broken queries (only when there's an error)
6. FINISH - Task completed successfully

Decision Rules:
- Always start with PLANNER if intent not parsed
- After PLANNER, go to TABLE_SELECTOR
- After TABLE_SELECTOR, go to DISAMBIGUATOR (even if no entities, it will skip itself)
- After DISAMBIGUATOR, go to SQL_GENERATOR
- If SQL_GENERATOR returns error, go to SQL_DEBUGGER
- After SQL_DEBUGGER fixes query, go back to SQL_GENERATOR to retry
- If results are successful, go to FINISH
- If SQL attempts > 3, go to FINISH (give up)

Which agent should act next? Respond with ONLY the agent name (one of: PLANNER, TABLE_SELECTOR, DISAMBIGUATOR, SQL_GENERATOR, SQL_DEBUGGER, FINISH)."""

    response = _llm.invoke([{"role": "user", "content": prompt}])

    # Extract agent name from response - handle list content
    content = response.content
    if isinstance(content, list):
        content = ' '.join(str(item) for item in content)
    elif not isinstance(content, str):
        content = str(content)
    next_agent = content.strip().upper()

    # Validate agent name
    valid_agents = ["PLANNER", "TABLE_SELECTOR", "DISAMBIGUATOR", "SQL_GENERATOR", "SQL_DEBUGGER", "FINISH"]
    if next_agent not in valid_agents:
        # Try to find a valid agent name in the response
        for agent in valid_agents:
            if agent in next_agent:
                next_agent = agent
                break
        else:
            # Default fallback logic
            if not intent:
                next_agent = "PLANNER"
            elif not selected_tables:
                next_agent = "TABLE_SELECTOR"
            elif results:
                next_agent = "FINISH"
            else:
                next_agent = "SQL_GENERATOR"

    # Create supervisor message
    agent_message = AIMessage(
        content=f"[SUPERVISOR] Routing to: {next_agent}",
        name="supervisor"
    )

    return {
        **state,
        "messages": [agent_message],
        "next_agent": next_agent
    }
