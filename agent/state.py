"""
Agent State Definition
Shared state structure for all agents in the LangGraph system
"""
from typing import TypedDict, List, Dict, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state across all agents in the multi-agent system.
    Uses LangGraph message pattern for conversation tracking.
    """

    # Message history - uses add_messages for proper accumulation
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Original query
    user_query: str

    # Query understanding
    intent: Optional[str]
    entities: List[str]

    # Table selection
    selected_tables: List[str]
    table_schemas: Dict[str, Dict]

    # Value disambiguation
    disambiguated_values: Dict[str, str]

    # SQL generation and execution
    candidate_sql: Optional[str]
    sql_attempts: List[Dict]
    final_sql: Optional[str]
    results: Optional[List]
    error: Optional[str]

    # Routing control
    next_agent: str


def create_initial_state(user_query: str) -> AgentState:
    """
    Create initial state for a new query.

    Args:
        user_query: User's natural language question

    Returns:
        Initial AgentState with all fields initialized
    """
    return {
        "messages": [],
        "user_query": user_query,
        "intent": None,
        "entities": [],
        "selected_tables": [],
        "table_schemas": {},
        "disambiguated_values": {},
        "candidate_sql": None,
        "sql_attempts": [],
        "final_sql": None,
        "results": None,
        "error": None,
        "next_agent": ""
    }
