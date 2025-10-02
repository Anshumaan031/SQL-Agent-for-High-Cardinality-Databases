"""
Multi-Agent SQL System
Main graph orchestrator that connects all agents using LangGraph
"""
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from agent.state import AgentState, create_initial_state
from agent.agents.supervisor import supervisor_agent, set_supervisor_llm
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
from agent.agents.sql_debugger import sql_debugger_agent, set_sql_debugger_llm
from agent.tools.database_tools import set_db_path
from agent.tools.disambiguator_tools import set_disambiguator_config
from agent.tools.sql_tools import set_sql_db_path


class MultiAgentSQL:
    """
    Multi-Agent SQL Query System using LangGraph.

    Architecture:
    - Supervisor agent routes between specialist agents
    - Each specialist agent has its own ToolNode for executing tools
    - Agents communicate via shared state
    - Dynamic routing based on current state

    Graph Structure:
        supervisor → planner → supervisor
                  → table_selector ⇄ table_tools → supervisor
                  → disambiguator ⇄ disamb_tools → supervisor
                  → sql_generator ⇄ sql_tools → supervisor
                  → sql_debugger → supervisor
    """

    def __init__(self, db_path: str, api_key: str):
        """
        Initialize the multi-agent system.

        Args:
            db_path: Path to SQLite database
            api_key: Google API key for Gemini
        """
        self.db_path = db_path
        self.api_key = api_key

        # Initialize all agents
        self._initialize_agents()

        # Initialize tools
        self._initialize_tools()

        # Build the graph
        self.app = self._build_graph()

    def _initialize_agents(self):
        """Initialize all agent LLMs"""
        set_supervisor_llm(self.api_key)
        set_planner_llm(self.api_key)
        set_table_selector_llm(self.api_key)
        set_disambiguator_llm(self.api_key)
        set_sql_generator_llm(self.api_key)
        set_sql_debugger_llm(self.api_key)

    def _initialize_tools(self):
        """Initialize all tools with database configuration"""
        set_db_path(self.db_path)
        set_disambiguator_config(self.db_path, self.api_key)
        set_sql_db_path(self.db_path)

    def _build_graph(self):
        """
        Build the LangGraph workflow with ToolNodes.

        Each agent that uses tools has:
        1. An agent node
        2. A ToolNode for executing its tools
        3. Conditional edge to route to tools or supervisor
        4. Edge from ToolNode back to agent
        """
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add agent nodes
        workflow.add_node("supervisor", supervisor_agent)
        workflow.add_node("planner", planner_agent)
        workflow.add_node("table_selector", table_selector_agent)
        workflow.add_node("disambiguator", disambiguator_agent)
        workflow.add_node("sql_generator", sql_generator_agent)
        workflow.add_node("sql_debugger", sql_debugger_agent)

        # Add ToolNodes - one for each agent that uses tools
        workflow.add_node("table_tools", ToolNode(database_tools))
        workflow.add_node("disamb_tools", ToolNode(disambiguator_tools))
        workflow.add_node("sql_tools", ToolNode(sql_tools))

        # === Entry Point ===
        workflow.add_edge(START, "supervisor")

        # === Supervisor Routing ===
        def route_from_supervisor(state: AgentState) -> str:
            """Route based on supervisor's decision"""
            next_agent = state.get("next_agent", "").upper()

            if next_agent == "FINISH":
                return END
            elif next_agent == "PLANNER":
                return "planner"
            elif next_agent == "TABLE_SELECTOR":
                return "table_selector"
            elif next_agent == "DISAMBIGUATOR":
                return "disambiguator"
            elif next_agent == "SQL_GENERATOR":
                return "sql_generator"
            elif next_agent == "SQL_DEBUGGER":
                return "sql_debugger"
            else:
                # Default: stay in supervisor (shouldn't happen)
                return "supervisor"

        workflow.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "planner": "planner",
                "table_selector": "table_selector",
                "disambiguator": "disambiguator",
                "sql_generator": "sql_generator",
                "sql_debugger": "sql_debugger",
                "supervisor": "supervisor",
                END: END
            }
        )

        # === Planner (no tools) ===
        workflow.add_edge("planner", "supervisor")

        # === Table Selector (with tools) ===
        workflow.add_conditional_edges(
            "table_selector",
            should_continue_table_selector,
            {
                "tools": "table_tools",
                "supervisor": "supervisor"
            }
        )
        # Tool node loops back to agent
        workflow.add_edge("table_tools", "table_selector")

        # === Disambiguator (with tools) ===
        workflow.add_conditional_edges(
            "disambiguator",
            should_continue_disambiguator,
            {
                "tools": "disamb_tools",
                "supervisor": "supervisor"
            }
        )
        # Tool node loops back to agent
        workflow.add_edge("disamb_tools", "disambiguator")

        # === SQL Generator (with tools) ===
        workflow.add_conditional_edges(
            "sql_generator",
            should_continue_sql_generator,
            {
                "tools": "sql_tools",
                "supervisor": "supervisor"
            }
        )
        # Tool node loops back to agent
        workflow.add_edge("sql_tools", "sql_generator")

        # === SQL Debugger (no tools) ===
        workflow.add_edge("sql_debugger", "supervisor")

        # Compile the graph
        return workflow.compile()

    def query(self, user_query: str, verbose: bool = False) -> dict:
        """
        Execute a query through the multi-agent system.

        Args:
            user_query: User's natural language question
            verbose: Print agent interactions

        Returns:
            Dictionary with results and execution trace
        """
        # Create initial state
        initial_state = create_initial_state(user_query)

        # Add user message
        initial_state["messages"] = [
            HumanMessage(content=user_query, name="user")
        ]

        # Run the graph
        if verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {user_query}")
            print(f"{'='*60}\n")

        final_state = self.app.invoke(initial_state)

        # Print agent trace if verbose
        if verbose:
            print("\n--- Agent Execution Trace ---")
            for i, msg in enumerate(final_state["messages"], 1):
                if hasattr(msg, 'name') and msg.name:
                    print(f"{i}. [{msg.name.upper()}] {msg.content[:100]}...")
                else:
                    content_preview = str(msg)[:100] if not hasattr(msg, 'content') else msg.content[:100]
                    print(f"{i}. {content_preview}...")

        # Format results
        result = {
            "success": final_state.get("results") is not None,
            "sql": final_state.get("candidate_sql"),
            "results": final_state.get("results"),
            "error": final_state.get("error"),
            "intent": final_state.get("intent"),
            "selected_tables": final_state.get("selected_tables"),
            "disambiguated_values": final_state.get("disambiguated_values"),
            "sql_attempts": len(final_state.get("sql_attempts", [])),
            "messages": final_state.get("messages", [])
        }

        if verbose:
            print("\n" + "="*60)
            if result["success"]:
                print("✓ Query Succeeded")
                print(f"SQL: {result['sql']}")
                print(f"Rows: {result['results']['row_count']}")
            else:
                print("✗ Query Failed")
                print(f"SQL: {result['sql']}")
                print(f"Error: {result['error']}")
            print("="*60 + "\n")

        return result

    def format_results(self, results: dict) -> str:
        """
        Format query results for display.

        Args:
            results: Results dictionary from query()

        Returns:
            Formatted string
        """
        if not results["success"]:
            return f"Query failed: {results['error']}"

        if not results["results"] or not results["results"]["rows"]:
            return "No results found."

        # Create table
        columns = results["results"]["columns"]
        rows = results["results"]["rows"]

        # Header
        output = " | ".join(columns) + "\n"
        output += "-" * len(output) + "\n"

        # Rows
        for row in rows:
            output += " | ".join(str(val) for val in row) + "\n"

        return output
