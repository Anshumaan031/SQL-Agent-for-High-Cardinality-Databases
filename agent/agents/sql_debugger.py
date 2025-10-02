"""
SQL Debugger Agent
Analyzes failed queries and generates fixes
"""
import json
import re
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState


# Global LLM instance - will be set by main agent
_llm: ChatGoogleGenerativeAI = None


def set_sql_debugger_llm(api_key: str):
    """Initialize the LLM for the SQL debugger agent"""
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )


def sql_debugger_agent(state: AgentState) -> AgentState:
    """
    SQL Debugger Agent - Analyzes and fixes failed SQL queries.

    This agent:
    1. Reviews the failed SQL query and error message
    2. Analyzes past attempts to avoid repeating mistakes
    3. Generates a fixed version of the query
    4. Explains the fix

    Args:
        state: Current agent state

    Returns:
        Updated state with fixed candidate_sql
    """
    candidate_sql = state.get("candidate_sql")
    error = state.get("error")
    table_schemas = state.get("table_schemas", {})
    sql_attempts = state.get("sql_attempts", [])
    intent = state.get("intent", state["user_query"])

    # Format table schemas
    schema_text = ""
    for table, schema in table_schemas.items():
        schema_text += f"\nTable: {table}\n"
        schema_text += f"  Columns: {', '.join(schema.get('columns', []))}\n"
        if schema.get('foreign_keys'):
            schema_text += f"  Foreign Keys: {', '.join(schema['foreign_keys'])}\n"

    # Format past attempts
    attempts_text = ""
    if sql_attempts:
        attempts_text = "\n\nPrevious attempts (learn from these mistakes):\n"
        for i, attempt in enumerate(sql_attempts[-3:], 1):  # Last 3 attempts
            attempts_text += f"\nAttempt {i}:\n"
            attempts_text += f"  SQL: {attempt.get('sql', 'N/A')}\n"
            result = attempt.get('result', {})
            if result.get('success'):
                attempts_text += f"  Result: Success\n"
            else:
                attempts_text += f"  Error: {result.get('error', 'Unknown')}\n"

    # Classify error type
    error_type = _classify_error(error) if error else "unknown"

    prompt = f"""You are a SQL debugging specialist.

User wants: {intent}

Failed SQL Query:
{candidate_sql}

Error Type: {error_type}
Error Message: {error}

Database Schema:
{schema_text}
{attempts_text}

Your task:
Analyze the error and generate a FIXED version of the SQL query.

Common fixes:
- syntax: Check for missing keywords, incorrect punctuation, wrong operators
- missing_table: Verify table name exists in schema
- missing_column: Check column names match schema exactly
- other: Review logic and ensure query makes sense

Respond in JSON format:
{{
    "fixed_sql": "corrected SQL query here",
    "explanation": "brief explanation of what was wrong and how you fixed it"
}}"""

    response = _llm.invoke([{"role": "user", "content": prompt}])

    # Parse response - handle list content
    try:
        content = response.content
        if isinstance(content, list):
            content = ' '.join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        content = content.strip()

        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            fixed_sql = parsed.get("fixed_sql", candidate_sql)
            explanation = parsed.get("explanation", "No explanation provided")
        else:
            # Fallback: try to extract SQL
            fixed_sql = _extract_sql(content) or candidate_sql
            explanation = "Could not parse structured response"

    except Exception as e:
        # If parsing fails, use original SQL
        fixed_sql = candidate_sql
        explanation = f"Parsing error: {str(e)}"

    # Create agent message
    agent_message = AIMessage(
        content=f"[SQL_DEBUGGER] Fixed query | Explanation: {explanation}",
        name="sql_debugger"
    )

    return {
        **state,
        "messages": [agent_message],
        "candidate_sql": fixed_sql,
        "error": None  # Clear error so generator can retry
    }


def _classify_error(error: str) -> str:
    """
    Classify SQL error type.

    Returns:
        Error category: 'syntax', 'missing_table', 'missing_column', 'performance', 'other'
    """
    if not error:
        return "unknown"

    error_lower = error.lower()

    if "syntax error" in error_lower or "near" in error_lower:
        return "syntax"
    elif "no such table" in error_lower:
        return "missing_table"
    elif "no such column" in error_lower:
        return "missing_column"
    elif "timeout" in error_lower or "too many" in error_lower:
        return "performance"
    else:
        return "other"


def _extract_sql(text: str) -> str:
    """
    Extract SQL query from text.
    """
    # Try to find SQL code block
    sql_block = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if sql_block:
        return sql_block.group(1).strip()

    sql_block = re.search(r'```\s*(SELECT.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if sql_block:
        return sql_block.group(1).strip()

    # Try to find SELECT statement
    select_match = re.search(r'(SELECT\s+.*?;?)', text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()

    return None
