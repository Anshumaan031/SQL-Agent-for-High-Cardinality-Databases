"""
Query Generator - Builds SQL queries progressively
"""
import sqlite3
from typing import Dict, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


class QueryGenerator:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )

    def generate_query(
        self,
        user_query: str,
        table_schemas: str,
        disambiguated_values: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate SQL query from user intent

        Args:
            user_query: Original user question
            table_schemas: Schema information for relevant tables
            disambiguated_values: Dict of {original_value: disambiguated_value}

        Returns:
            SQL query string
        """
        # Build context about disambiguated values
        disambiguation_context = ""
        if disambiguated_values:
            disambiguation_context = "\n\nDisambiguated Values:\n"
            for original, actual in disambiguated_values.items():
                disambiguation_context += f"- User said '{original}' but the actual value is '{actual}'\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Generate a valid SQLite query based on the user's question.

Rules:
- Return ONLY the SQL query, no explanations
- Use proper SQLite syntax
- Use JOIN clauses when querying multiple tables
- Use appropriate WHERE clauses for filtering
- Use aggregate functions (COUNT, SUM, AVG) when appropriate
- ALWAYS use the disambiguated values provided instead of the user's original input
- Ensure column and table names are properly quoted if needed
- Keep queries optimized and efficient"""),
            ("user", """Database Schema:
{schema}
{disambiguation}

User Question: {query}

SQL Query:""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "schema": table_schemas,
            "disambiguation": disambiguation_context,
            "query": user_query
        })

        # Clean up the SQL query
        sql = response.content.strip()

        # Remove markdown code blocks if present
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]

        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()

    def validate_query(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query syntax

        Returns:
            (is_valid, error_message)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Use EXPLAIN to validate without executing
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return (True, None)
        except sqlite3.Error as e:
            conn.close()
            return (False, str(e))

    def execute_query(self, sql: str) -> Tuple[bool, any]:
        """
        Execute SQL query and return results

        Returns:
            (success, results_or_error)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(sql)
            results = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            conn.close()
            return (True, {"columns": columns, "rows": results})
        except sqlite3.Error as e:
            conn.close()
            return (False, str(e))

    def classify_error(self, error: str) -> str:
        """
        Classify SQL error type

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

    def fix_query(self, sql: str, error: str, table_schemas: str) -> str:
        """
        Attempt to fix a broken SQL query

        Args:
            sql: The broken SQL query
            error: The error message
            table_schemas: Available table schemas

        Returns:
            Fixed SQL query
        """
        error_type = self.classify_error(error)

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a SQL expert fixing a broken query.

Error Type: {error_type}
Error Message: {error}

Rules:
- Return ONLY the fixed SQL query
- No explanations or markdown
- Fix the specific error mentioned
- Ensure the query is valid SQLite syntax"""),
            ("user", """Database Schema:
{schema}

Broken Query:
{sql}

Fixed Query:""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "schema": table_schemas,
            "sql": sql
        })

        # Clean up the SQL query
        fixed_sql = response.content.strip()

        # Remove markdown code blocks if present
        if fixed_sql.startswith("```sql"):
            fixed_sql = fixed_sql[6:]
        elif fixed_sql.startswith("```"):
            fixed_sql = fixed_sql[3:]

        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3]

        return fixed_sql.strip()

    def format_results(self, results: Dict) -> str:
        """Format query results for display"""
        if not results['rows']:
            return "No results found."

        # Create header
        output = " | ".join(results['columns']) + "\n"
        output += "-" * len(output) + "\n"

        # Add rows
        for row in results['rows']:
            output += " | ".join(str(val) for val in row) + "\n"

        return output