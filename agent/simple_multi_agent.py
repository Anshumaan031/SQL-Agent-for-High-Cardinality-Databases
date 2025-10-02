"""
Simplified Multi-Agent SQL System
Reduces API calls by using a fixed workflow instead of supervisor routing
"""
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from agent.state import AgentState, create_initial_state
from agent.agents.planner import planner_agent, set_planner_llm
from agent.agents.table_selector import (
    table_selector_agent,
    set_table_selector_llm,
    should_continue_table_selector,
    database_tools
)
from agent.agents.disambiguator import (
    disambiguator_agent,
    set_disambiguator_llm,
    should_continue_disambiguator,
    disambiguator_tools
)
from agent.agents.sql_generator import (
    sql_generator_agent,
    set_sql_generator_llm,
    should_continue_sql_generator,
    sql_tools
)
from agent.tools.database_tools import set_db_path
from agent.tools.disambiguator_tools import set_disambiguator_config
from agent.tools.sql_tools import set_sql_db_path


class SimpleMultiAgentSQL:
    """
    Simplified Multi-Agent SQL System with fixed workflow.

    Uses a LINEAR workflow instead of supervisor routing:
    planner → table_selector → disambiguator → sql_generator → END

    This reduces API calls from ~10-15 to ~4-6 per query.
    """

    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.api_key = api_key

        # Initialize agents
        set_planner_llm(api_key)
        set_table_selector_llm(api_key)
        set_disambiguator_llm(api_key)
        set_sql_generator_llm(api_key)

        # Initialize tools
        set_db_path(db_path)
        set_disambiguator_config(db_path, api_key)
        set_sql_db_path(db_path)

        # Build graph
        self.app = self._build_graph()

    def _build_graph(self):
        """Build a linear workflow with ToolNodes"""
        workflow = StateGraph(AgentState)

        # Add agent nodes
        workflow.add_node("planner", planner_agent)
        workflow.add_node("table_selector", table_selector_agent)
        workflow.add_node("disambiguator", disambiguator_agent)
        workflow.add_node("sql_generator", sql_generator_agent)

        # Add tool nodes
        workflow.add_node("table_tools", ToolNode(database_tools))
        workflow.add_node("disamb_tools", ToolNode(disambiguator_tools))
        workflow.add_node("sql_tools", ToolNode(sql_tools))

        # Linear workflow (no supervisor!)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "table_selector")

        # Table selector with tools
        workflow.add_conditional_edges(
            "table_selector",
            should_continue_table_selector,
            {
                "tools": "table_tools",
                "supervisor": "disambiguator"  # Go directly to next agent
            }
        )
        workflow.add_edge("table_tools", "table_selector")

        # Disambiguator with tools
        workflow.add_conditional_edges(
            "disambiguator",
            should_continue_disambiguator,
            {
                "tools": "disamb_tools",
                "supervisor": "sql_generator"  # Go directly to next agent
            }
        )
        workflow.add_edge("disamb_tools", "disambiguator")

        # SQL generator with tools
        workflow.add_conditional_edges(
            "sql_generator",
            should_continue_sql_generator,
            {
                "tools": "sql_tools",
                "supervisor": END  # Finish when done
            }
        )
        workflow.add_edge("sql_tools", "sql_generator")

        return workflow.compile()

    def query(self, user_query: str, verbose: bool = False) -> dict:
        """Execute query through simplified workflow"""
        initial_state = create_initial_state(user_query)
        initial_state["messages"] = [HumanMessage(content=user_query, name="user")]

        if verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {user_query}")
            print(f"{'='*60}\n")

        final_state = self.app.invoke(initial_state)

        if verbose:
            print("\n--- Agent Execution Trace ---")
            for i, msg in enumerate(final_state["messages"], 1):
                if hasattr(msg, 'name') and msg.name:
                    print(f"{i}. [{msg.name.upper()}] {msg.content[:100]}...")
                else:
                    content_preview = str(msg)[:100] if not hasattr(msg, 'content') else msg.content[:100]
                    print(f"{i}. {content_preview}...")

        result = {
            "success": final_state.get("results") is not None,
            "sql": final_state.get("candidate_sql"),
            "results": final_state.get("results"),
            "error": final_state.get("error"),
            "intent": final_state.get("intent"),
            "selected_tables": final_state.get("selected_tables"),
            "disambiguated_values": final_state.get("disambiguated_values"),
            "messages": final_state.get("messages", [])
        }

        if verbose:
            print("\n" + "="*60)
            if result["success"]:
                print("✓ Query Succeeded")
                print(f"SQL: {result['sql']}")
                if result["results"]:
                    print(f"Rows: {result['results']['row_count']}")
            else:
                print("✗ Query Failed")
                print(f"Error: {result['error']}")
            print("="*60 + "\n")

        return result

    def format_results(self, results: dict) -> str:
        """Format query results for display"""
        if not results["success"]:
            return f"Query failed: {results['error']}"

        if not results["results"] or not results["results"]["rows"]:
            return "No results found."

        columns = results["results"]["columns"]
        rows = results["results"]["rows"]

        output = " | ".join(columns) + "\n"
        output += "-" * len(output) + "\n"

        for row in rows:
            output += " | ".join(str(val) for val in row) + "\n"

        return output
