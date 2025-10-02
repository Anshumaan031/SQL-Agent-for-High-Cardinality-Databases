"""
SQL Tools for SQL Generator Agent
Tools for validating, explaining, and executing SQL queries
"""
import sqlite3
from typing import Dict, Tuple
from langchain_core.tools import tool


# Global database path - will be set by the main agent
_DB_PATH: str = ""


def set_sql_db_path(db_path: str):
    """Set the database path for SQL tools"""
    global _DB_PATH
    _DB_PATH = db_path


@tool
def validate_sql_syntax(sql: str) -> Dict:
    """
    Check if SQL syntax is valid without executing the query.
    Uses EXPLAIN to validate the query.

    Args:
        sql: SQL query to validate

    Returns:
        Dictionary with 'valid' (bool) and optional 'error' message
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    try:
        # Use EXPLAIN to validate without executing
        cursor.execute(f"EXPLAIN {sql}")
        conn.close()
        return {"valid": True, "message": "SQL syntax is valid"}
    except sqlite3.Error as e:
        conn.close()
        return {"valid": False, "error": str(e)}


@tool
def explain_query_plan(sql: str) -> str:
    """
    Get the query execution plan to understand how SQLite will execute the query.
    Useful for checking if the query will be efficient.

    Args:
        sql: SQL query to analyze

    Returns:
        Execution plan as a string, or error message
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        plan = cursor.fetchall()
        conn.close()

        # Format the plan
        plan_str = "Query Execution Plan:\n"
        for row in plan:
            plan_str += f"  {' '.join(str(x) for x in row)}\n"

        return plan_str
    except sqlite3.Error as e:
        conn.close()
        return f"Error getting query plan: {str(e)}"


@tool
def execute_sql_query(sql: str) -> Dict:
    """
    Execute a SQL query and return the results.
    This is the main tool for running SQL queries.

    Args:
        sql: SQL query to execute

    Returns:
        Dictionary with 'success' (bool), and either 'results' or 'error'
        Results include 'columns' and 'rows' lists
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()

        # Get column names
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        conn.close()
        return {
            "success": True,
            "results": {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            }
        }
    except sqlite3.Error as e:
        conn.close()
        return {
            "success": False,
            "error": str(e),
            "error_type": _classify_error(str(e))
        }


def _classify_error(error: str) -> str:
    """
    Classify SQL error type.
    Helper function (not a tool).

    Returns:
        Error category: 'syntax', 'missing_table', 'missing_column', 'performance', 'other'
    """
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


# List of all SQL tools for easy import
sql_tools = [
    validate_sql_syntax,
    explain_query_plan,
    execute_sql_query
]
