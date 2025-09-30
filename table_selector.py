"""
Table Selector - Identifies relevant tables from user queries
"""
import sqlite3
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


class TableSelector:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        self.schema_info = self._load_schema()

    def _load_schema(self) -> Dict[str, str]:
        """Load table schemas from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        schema_info = {}
        for table in tables:
            # Get column info
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            foreign_keys = cursor.fetchall()

            # Format schema
            col_info = [f"{col[1]} ({col[2]})" for col in columns]
            fk_info = [f"{fk[3]} -> {fk[2]}({fk[4]})" for fk in foreign_keys]

            schema_info[table] = {
                "columns": col_info,
                "foreign_keys": fk_info
            }

        conn.close()
        return schema_info

    def select_tables(self, user_query: str) -> List[str]:
        """Select relevant tables based on user query"""
        # Format schema for LLM
        schema_text = self._format_schema()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database expert. Given a user query and database schema,
identify the MINIMAL set of tables needed to answer the query.

Rules:
- Only include tables that are DIRECTLY relevant
- Include junction tables if needed for relationships
- Return ONLY table names, one per line
- No explanations or extra text"""),
            ("user", """Database Schema:
{schema}

User Query: {query}

Relevant tables:""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "schema": schema_text,
            "query": user_query
        })

        # Parse table names from response
        tables = [line.strip() for line in response.content.strip().split('\n') if line.strip()]

        # Validate tables exist
        valid_tables = [t for t in tables if t in self.schema_info]

        return valid_tables

    def _format_schema(self) -> str:
        """Format schema information for LLM"""
        lines = []
        for table, info in self.schema_info.items():
            lines.append(f"\nTable: {table}")
            lines.append(f"  Columns: {', '.join(info['columns'])}")
            if info['foreign_keys']:
                lines.append(f"  Foreign Keys: {', '.join(info['foreign_keys'])}")
        return '\n'.join(lines)

    def get_table_schemas(self, tables: List[str]) -> str:
        """Get detailed schema for specific tables"""
        lines = []
        for table in tables:
            if table in self.schema_info:
                info = self.schema_info[table]
                lines.append(f"\nTable: {table}")
                lines.append(f"  Columns: {', '.join(info['columns'])}")
                if info['foreign_keys']:
                    lines.append(f"  Foreign Keys: {', '.join(info['foreign_keys'])}")
        return '\n'.join(lines)