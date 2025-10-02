# Multi-Agent SQL System with LangGraph

A true agentic architecture for the SQL Agent using LangGraph, featuring autonomous agents with tools, dynamic routing, and a supervisor orchestrator.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [State Definition](#state-definition)
- [Agents](#agents)
  - [1. Supervisor Agent](#1-supervisor-agent-orchestrator)
  - [2. Planner Agent](#2-planner-agent-intent-parser)
  - [3. Table Selector Agent](#3-table-selector-agent-with-tools)
  - [4. Value Disambiguator Agent](#4-value-disambiguator-agent-with-tools)
  - [5. SQL Generator Agent](#5-sql-generator-agent-with-tools)
  - [6. SQL Debugger Agent](#6-sql-debugger-agent)
- [Graph Construction](#graph-construction)
- [Execution](#execution)
- [Comparison: Pipeline vs Agentic System](#comparison-pipeline-vs-agentic-system)
- [Key Advantages](#key-advantages)

---

## Architecture Overview

```
                    ┌─────────────────┐
                    │  Supervisor     │ (decides which agent to call)
                    │     Agent       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │   Table    │  │   Value    │  │    SQL     │
     │  Selector  │  │Disambiguator│  │  Generator │
     │   Agent    │  │   Agent    │  │   Agent    │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
     [Tool Nodes]    [Tool Nodes]    [Tool Nodes]
```

**Flow:**
1. **Supervisor** analyzes current state and routes to appropriate specialist agent
2. **Specialist agents** use tools autonomously to complete their tasks
3. Agents return to **Supervisor** who decides the next step
4. Process continues until query is successfully answered or max attempts reached

---

## State Definition

```python
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # Conversation history
    user_query: str
    next_agent: str  # Which agent to call next

    # Shared working memory
    intent: Optional[str]
    entities: List[str]
    selected_tables: List[str]
    table_schemas: Dict
    disambiguated_values: Dict[str, str]
    candidate_sql: Optional[str]
    sql_attempts: List[Dict]  # Track all attempts
    final_sql: Optional[str]
    results: Optional[List]
    error: Optional[str]

    # Agent communication
    agent_scratchpad: str  # For reasoning traces
```

**Key Features:**
- `messages`: Accumulates all agent communications (using `operator.add`)
- `agent_scratchpad`: Tracks reasoning across agents
- `sql_attempts`: Maintains history for self-reflection
- All state is immutable - each agent returns updated state

---

## Agents

### 1. Supervisor Agent (Orchestrator)

**Role:** Decides which specialist agent to invoke next based on current state.

```python
def supervisor_agent(state: AgentState) -> AgentState:
    """
    Main orchestrator - decides which specialist agent to invoke next
    Uses an LLM to make routing decisions
    """

    supervisor_prompt = f"""You are a supervisor managing SQL query agents.

Current state:
- User query: {state['user_query']}
- Intent parsed: {state.get('intent', 'Not yet')}
- Tables selected: {state.get('selected_tables', 'Not yet')}
- Values disambiguated: {state.get('disambiguated_values', 'Not yet')}
- SQL generated: {state.get('candidate_sql', 'Not yet')}
- Results: {state.get('results', 'Not yet')}
- Error: {state.get('error', 'None')}

Available agents:
1. PLANNER - Analyzes query and plans approach
2. TABLE_SELECTOR - Finds relevant tables
3. VALUE_DISAMBIGUATOR - Resolves fuzzy values
4. SQL_GENERATOR - Writes SQL queries
5. SQL_DEBUGGER - Fixes broken queries
6. FINISH - Query completed successfully

Which agent should act next? Respond with ONLY the agent name."""

    response = llm.invoke([
        {"role": "system", "content": supervisor_prompt}
    ])

    next_agent = response.content.strip()

    return {
        **state,
        "next_agent": next_agent,
        "messages": state["messages"] + [
            {"role": "supervisor", "content": f"Routing to: {next_agent}"}
        ]
    }
```

**Capabilities:**
- ✓ Dynamic routing based on state
- ✓ Can skip unnecessary steps
- ✓ Can loop back to previous agents
- ✓ Decides when task is complete

---

### 2. Planner Agent (Intent Parser)

**Role:** Analyzes user query to extract intent, entities, and query complexity.

```python
def planner_agent(state: AgentState) -> AgentState:
    """
    Autonomous agent that analyzes query and plans approach
    """

    planner_prompt = f"""You are a query planning agent.

User query: {state['user_query']}

Tasks:
1. Extract the user's intent
2. Identify entities/values mentioned
3. Determine if this query needs:
   - Multiple tables (JOINs)
   - Aggregations
   - Filtering
   - Sorting

Respond in JSON format:
{{
    "intent": "...",
    "entities": [...],
    "complexity": "simple|moderate|complex",
    "requires_joins": true/false,
    "reasoning": "..."
}}"""

    response = llm.invoke([{"role": "user", "content": planner_prompt}])
    parsed = json.loads(response.content)

    return {
        **state,
        "intent": parsed["intent"],
        "entities": parsed["entities"],
        "agent_scratchpad": state["agent_scratchpad"] + f"\n[PLANNER] {parsed['reasoning']}",
        "messages": state["messages"] + [
            {"role": "planner", "content": f"Intent: {parsed['intent']}, Entities: {parsed['entities']}"}
        ]
    }
```

**Output:**
- Intent description
- List of entities
- Query complexity assessment
- Reasoning trace

---

### 3. Table Selector Agent with Tools

**Role:** Identifies relevant tables using database inspection tools.

**Tools Available:**

```python
from langchain.tools import tool

@tool
def get_all_tables() -> List[str]:
    """Get list of all tables in database"""
    return database_inspector.get_table_names()

@tool
def get_table_schema(table_name: str) -> Dict:
    """Get schema for a specific table"""
    return database_inspector.get_table_schema(table_name)

@tool
def search_tables_by_keyword(keyword: str) -> List[str]:
    """Search for tables containing keyword in name or description"""
    return database_inspector.search_tables(keyword)
```

**Agent Implementation:**

```python
def table_selector_agent(state: AgentState) -> AgentState:
    """
    Agent that uses tools to find relevant tables
    Has access to: get_all_tables, get_table_schema, search_tables_by_keyword
    """

    agent_prompt = f"""You are a table selection specialist.

User wants: {state['intent']}
Entities: {state['entities']}

Use your tools to:
1. Search for relevant tables
2. Examine their schemas
3. Select the minimal set needed

You have access to these tools:
- get_all_tables()
- get_table_schema(table_name)
- search_tables_by_keyword(keyword)

Think step by step and use tools as needed."""

    # Create agent with tools
    tools = [get_all_tables, get_table_schema, search_tables_by_keyword]
    agent = create_react_agent(llm, tools, agent_prompt)

    # Agent reasons and uses tools autonomously
    result = agent.invoke({
        "input": f"Find tables for: {state['intent']}",
        "chat_history": state["messages"]
    })

    selected_tables = extract_tables_from_agent_output(result)

    return {
        **state,
        "selected_tables": selected_tables,
        "agent_scratchpad": state["agent_scratchpad"] + f"\n[TABLE_SELECTOR] {result['reasoning']}",
        "messages": state["messages"] + [
            {"role": "table_selector", "content": f"Selected: {selected_tables}"}
        ]
    }
```

**Process:**
1. Agent analyzes intent and entities
2. Uses `search_tables_by_keyword` to find candidate tables
3. Calls `get_table_schema` to inspect relevant tables
4. Selects minimal set needed for query
5. Returns to supervisor

---

### 4. Value Disambiguator Agent with Tools

**Role:** Resolves fuzzy/ambiguous values using vector search and fuzzy matching.

**Tools Available:**

```python
@tool
def get_column_values(table: str, column: str, limit: int = 100) -> List[str]:
    """Get sample values from a column"""
    return database.query(f"SELECT DISTINCT {column} FROM {table} LIMIT {limit}")

@tool
def vector_search(query: str, table: str, column: str, top_k: int = 5) -> List[tuple]:
    """Search for similar values using embeddings"""
    return vector_db.search(query, f"{table}.{column}", top_k)

@tool
def fuzzy_match(query: str, candidates: List[str]) -> List[tuple]:
    """Fuzzy string matching"""
    return fuzzy_matcher.match(query, candidates)
```

**Agent Implementation:**

```python
def value_disambiguator_agent(state: AgentState) -> AgentState:
    """
    Agent that resolves fuzzy/ambiguous values
    Tools: get_column_values, vector_search, fuzzy_match
    """

    if not state['entities']:
        return {**state, "disambiguated_values": {}}

    agent_prompt = f"""You are a value disambiguation specialist.

Entities to resolve: {state['entities']}
Available tables: {state['selected_tables']}

For each entity:
1. Determine which table/column it likely belongs to
2. Use vector_search or fuzzy_match to find best matches
3. Return the disambiguated values

Tools available:
- get_column_values(table, column, limit)
- vector_search(query, table, column, top_k)
- fuzzy_match(query, candidates)"""

    tools = [get_column_values, vector_search, fuzzy_match]
    agent = create_react_agent(llm, tools, agent_prompt)

    result = agent.invoke({
        "input": f"Disambiguate: {state['entities']}",
        "tables": state['selected_tables']
    })

    return {
        **state,
        "disambiguated_values": result['disambiguated'],
        "messages": state["messages"] + [
            {"role": "disambiguator", "content": f"Resolved: {result['disambiguated']}"}
        ]
    }
```

**Process:**
1. For each entity, determines likely column
2. Uses `vector_search` for semantic matching
3. Falls back to `fuzzy_match` if needed
4. Returns mapping: `{"ACDC": "AC/DC", "SF": "San Francisco"}`

---

### 5. SQL Generator Agent with Tools

**Role:** Writes SQL queries and validates/tests them.

**Tools Available:**

```python
@tool
def validate_sql_syntax(sql: str) -> Dict:
    """Check if SQL syntax is valid"""
    return sql_validator.validate(sql)

@tool
def explain_query(sql: str) -> str:
    """Get query execution plan"""
    return database.execute(f"EXPLAIN QUERY PLAN {sql}")

@tool
def execute_sql(sql: str) -> Dict:
    """Execute SQL and return results"""
    try:
        results = database.execute(sql)
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Agent Implementation:**

```python
def sql_generator_agent(state: AgentState) -> AgentState:
    """
    Agent that writes and tests SQL queries
    Tools: validate_sql_syntax, explain_query, execute_sql
    """

    agent_prompt = f"""You are a SQL writing specialist.

Task: {state['intent']}
Tables: {state['selected_tables']}
Table schemas: {state.get('table_schemas', {})}
Values to use: {state.get('disambiguated_values', {})}

Generate SQL that:
1. Uses the exact disambiguated values
2. Joins tables correctly
3. Answers the user's question

You can:
- validate_sql_syntax(sql) - Check syntax
- explain_query(sql) - See execution plan
- execute_sql(sql) - Test the query

Write SQL, validate it, and execute it."""

    tools = [validate_sql_syntax, explain_query, execute_sql]
    agent = create_react_agent(llm, tools, agent_prompt)

    result = agent.invoke({"input": state['intent']})

    return {
        **state,
        "candidate_sql": result['sql'],
        "results": result.get('results'),
        "error": result.get('error'),
        "sql_attempts": state['sql_attempts'] + [result],
        "messages": state["messages"] + [
            {"role": "sql_generator", "content": f"Generated: {result['sql']}"}
        ]
    }
```

**Process:**
1. Drafts SQL query
2. Calls `validate_sql_syntax` to check correctness
3. Uses `explain_query` to review execution plan
4. Executes with `execute_sql`
5. Returns results or error

---

### 6. SQL Debugger Agent

**Role:** Analyzes failed queries and generates fixes.

```python
def sql_debugger_agent(state: AgentState) -> AgentState:
    """
    Agent that analyzes and fixes SQL errors
    Uses reflection and past attempts
    """

    debugger_prompt = f"""You are a SQL debugging specialist.

Failed SQL: {state['candidate_sql']}
Error: {state['error']}

Previous attempts:
{json.dumps(state['sql_attempts'], indent=2)}

Available schemas:
{state['table_schemas']}

Analyze the error and generate a fixed version.
Consider:
- Syntax errors
- Missing columns
- Incorrect joins
- Type mismatches

Respond with fixed SQL and explanation."""

    response = llm.invoke([{"role": "user", "content": debugger_prompt}])

    # Extract fixed SQL
    fixed_sql = extract_sql_from_response(response.content)

    # Test it
    success, result = database.execute_safe(fixed_sql)

    return {
        **state,
        "candidate_sql": fixed_sql,
        "results": result if success else None,
        "error": None if success else result,
        "sql_attempts": state['sql_attempts'] + [{
            "sql": fixed_sql,
            "success": success,
            "result": result
        }],
        "messages": state["messages"] + [
            {"role": "debugger", "content": f"Fixed attempt: {fixed_sql}"}
        ]
    }
```

**Capabilities:**
- Analyzes error messages
- Reviews past attempts to avoid repeating mistakes
- Self-reflects on why previous queries failed
- Generates improved versions

---

## Graph Construction

```python
from langgraph.graph import StateGraph, END

# Build graph
workflow = StateGraph(AgentState)

# Add agent nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("table_selector", table_selector_agent)
workflow.add_node("disambiguator", value_disambiguator_agent)
workflow.add_node("sql_generator", sql_generator_agent)
workflow.add_node("sql_debugger", sql_debugger_agent)

# Routing function
def route_to_agent(state: AgentState) -> str:
    """Route based on supervisor's decision"""
    next_agent = state.get("next_agent", "").upper()

    if next_agent == "FINISH":
        return END
    elif next_agent == "PLANNER":
        return "planner"
    elif next_agent == "TABLE_SELECTOR":
        return "table_selector"
    elif next_agent == "VALUE_DISAMBIGUATOR":
        return "disambiguator"
    elif next_agent == "SQL_GENERATOR":
        return "sql_generator"
    elif next_agent == "SQL_DEBUGGER":
        return "sql_debugger"
    else:
        return "supervisor"  # Default back to supervisor

# Set entry point
workflow.set_entry_point("supervisor")

# All nodes route back to supervisor (supervisor decides next step)
workflow.add_conditional_edges("supervisor", route_to_agent)
workflow.add_edge("planner", "supervisor")
workflow.add_edge("table_selector", "supervisor")
workflow.add_edge("disambiguator", "supervisor")
workflow.add_edge("sql_generator", "supervisor")
workflow.add_edge("sql_debugger", "supervisor")

# Compile
app = workflow.compile()
```

**Flow Diagram:**

```
START → supervisor → [route_to_agent] → specialist_agent → supervisor → ...
                                                                  ↓
                                                            [FINISH] → END
```

---

## Execution

```python
# Initialize state
initial_state = {
    "messages": [],
    "user_query": "Show me tracks by ACDC",
    "next_agent": "",
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
    "agent_scratchpad": ""
}

# Run graph
result = app.invoke(initial_state)

# Access results
print(f"Final SQL: {result['final_sql']}")
print(f"Results: {result['results']}")
print(f"\nAgent conversation:")
for msg in result['messages']:
    print(f"  {msg['role']}: {msg['content']}")
```

**Example Execution Trace:**

```
supervisor: Routing to: PLANNER
planner: Intent: Find tracks by artist, Entities: ['ACDC']
supervisor: Routing to: TABLE_SELECTOR
table_selector: Selected: ['Artist', 'Album', 'Track']
supervisor: Routing to: VALUE_DISAMBIGUATOR
disambiguator: Resolved: {'ACDC': 'AC/DC'}
supervisor: Routing to: SQL_GENERATOR
sql_generator: Generated: SELECT Track.Name FROM Track JOIN ...
supervisor: Routing to: FINISH
```

---

## Comparison: Pipeline vs Agentic System

| Aspect | Pipeline (Current) | Agentic System (LangGraph) |
|--------|-------------------|---------------------------|
| **Control Flow** | Fixed sequence: intent → tables → disambiguate → SQL | Dynamic: Supervisor decides next step |
| **Agents** | None (just functions) | 6 autonomous agents with reasoning |
| **Tools** | Direct function calls | Agents use tools via ReAct pattern |
| **Flexibility** | Same path every time | Can skip steps, loop, backtrack |
| **Error Handling** | Hard-coded retries (max 2) | Debugger agent reasons about errors |
| **Reasoning** | Embedded in code logic | Agent scratchpad + chain-of-thought |
| **Collaboration** | Sequential function calls | Agents communicate via shared state |
| **Observability** | Limited to logs | Full message history + reasoning traces |
| **Adaptability** | Must modify code | Supervisor adapts to new scenarios |
| **Debugging** | Step through code | Inspect agent messages and decisions |

---

## Key Advantages

### 1. **Autonomous Decision Making**
- Supervisor can skip unnecessary steps (e.g., no disambiguation needed for simple queries)
- Agents decide which tools to use and in what order
- System adapts to query complexity

### 2. **Self-Correction**
- SQL Debugger reviews past attempts and learns from mistakes
- Can loop multiple times with different strategies
- Agents reflect on failures before retrying

### 3. **Tool-Based Interaction**
- Agents don't hardcode database access
- Tools are modular and testable
- Easy to add new tools without changing agent logic

### 4. **Observability**
- Full trace of agent decisions in `messages`
- Reasoning captured in `agent_scratchpad`
- Can replay execution to debug

### 5. **Extensibility**
- Add new agents (e.g., Query Optimizer, Result Validator)
- Supervisor automatically incorporates them
- No changes to existing agents needed

### 6. **Human-in-the-Loop**
- Easy to add approval nodes (e.g., before SQL execution)
- Can pause execution and resume later
- Supports checkpointing

### 7. **Parallel Execution** (Future Enhancement)
```python
# Can run independent agents in parallel
workflow.add_conditional_edges(
    "supervisor",
    route_multiple_agents,
    {
        "parallel": ["table_selector", "schema_analyzer"],
        "sequential": "planner"
    }
)
```

### 8. **Persistent Memory**
- State persists across agent calls
- Can implement conversation history
- Support multi-turn queries

---

## Implementation Checklist

- [ ] Define `AgentState` TypedDict
- [ ] Implement Supervisor Agent with routing logic
- [ ] Create Planner Agent for intent parsing
- [ ] Build Table Selector Agent + tools
- [ ] Build Value Disambiguator Agent + tools
- [ ] Build SQL Generator Agent + tools
- [ ] Build SQL Debugger Agent
- [ ] Construct LangGraph workflow
- [ ] Add conditional routing
- [ ] Test with sample queries
- [ ] Add observability/logging
- [ ] Implement checkpointing (optional)
- [ ] Add human-in-the-loop approval (optional)

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Multi-Agent Supervisor Pattern](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [ReAct Pattern](https://react-lm.github.io/)
- [Current Pipeline Implementation](agent.py)
