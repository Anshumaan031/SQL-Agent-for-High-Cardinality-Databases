"""
Database Tools for Table Selector Agent
Tools for inspecting database schema and finding relevant tables
"""
import sqlite3
from typing import List, Dict
from langchain_core.tools import tool


# Global database path - will be set by the main agent
_DB_PATH: str = ""


def set_db_path(db_path: str):
    """Set the database path for all tools"""
    global _DB_PATH
    _DB_PATH = db_path


@tool
def get_all_tables() -> List[str]:
    """
    Get list of all tables in the database.

    Returns:
        List of table names
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


@tool
def get_table_schema(table_name: str) -> Dict:
    """
    Get detailed schema for a specific table including columns and foreign keys.

    Args:
        table_name: Name of the table to inspect

    Returns:
        Dictionary with columns and foreign_keys information
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    try:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()

        # Format schema
        col_info = [f"{col[1]} ({col[2]})" for col in columns]
        fk_info = [f"{fk[3]} -> {fk[2]}({fk[4]})" for fk in foreign_keys]

        result = {
            "table": table_name,
            "columns": col_info,
            "foreign_keys": fk_info
        }

    except Exception as e:
        result = {"error": f"Could not get schema for {table_name}: {str(e)}"}

    conn.close()
    return result


@tool
def search_tables_by_keyword(keyword: str) -> List[str]:
    """
    Search for tables whose names contain the given keyword.

    Args:
        keyword: Keyword to search for in table names (case-insensitive)

    Returns:
        List of matching table names
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Case-insensitive search
    keyword_lower = keyword.lower()
    matching_tables = [t for t in all_tables if keyword_lower in t.lower()]

    return matching_tables


# List of all database tools for easy import
database_tools = [
    get_all_tables,
    get_table_schema,
    search_tables_by_keyword
]
