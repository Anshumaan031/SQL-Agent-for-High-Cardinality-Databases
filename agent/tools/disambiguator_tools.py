"""
Disambiguator Tools for Value Disambiguation Agent
Tools for resolving fuzzy/ambiguous values using vector search
"""
import sqlite3
from typing import List, Dict, Tuple
from langchain_core.tools import tool
from difflib import SequenceMatcher
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Global variables - will be set by the main agent
_DB_PATH: str = ""
_API_KEY: str = ""
_chroma_client = None
_embeddings = None


def set_disambiguator_config(db_path: str, api_key: str):
    """Set configuration for disambiguator tools"""
    global _DB_PATH, _API_KEY, _chroma_client, _embeddings
    _DB_PATH = db_path
    _API_KEY = api_key
    _chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    _embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )


@tool
def get_column_values(table: str, column: str, limit: int = 100) -> List[str]:
    """
    Get distinct values from a specific column in a table.

    Args:
        table: Table name
        column: Column name
        limit: Maximum number of values to return (default: 100)

    Returns:
        List of distinct values from the column
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit};")
        values = [str(row[0]) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        values = [f"Error: {str(e)}"]
    finally:
        conn.close()

    return values


@tool
def vector_search_values(query: str, table: str, column: str, top_k: int = 5) -> List[Dict]:
    """
    Search for similar values in a column using vector similarity (embeddings).
    This is useful for fuzzy matching like "ACDC" -> "AC/DC".

    Args:
        query: The value to search for
        table: Table name
        column: Column name
        top_k: Number of top matches to return (default: 5)

    Returns:
        List of dictionaries with 'value' and 'similarity_score' keys
    """
    try:
        # Build collection name
        collection_name = f"{table}_{column}".replace(" ", "_").lower()

        # Try to get existing collection or create new one
        try:
            collection = _chroma_client.get_collection(collection_name)
        except:
            # Create new collection
            collection = _chroma_client.create_collection(
                name=collection_name,
                metadata={"table": table, "column": column}
            )

            # Get values and add to collection
            conn = sqlite3.connect(_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 10000;")
            values = [str(row[0]) for row in cursor.fetchall()]
            conn.close()

            if values:
                # Generate embeddings
                embeddings_list = _embeddings.embed_documents(values)

                # Add to collection
                collection.add(
                    documents=values,
                    embeddings=embeddings_list,
                    ids=[f"{table}_{column}_{i}" for i in range(len(values))]
                )

        # Generate embedding for query
        query_embedding = _embeddings.embed_query(query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        # Format results
        matches = []
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            matches.append({
                "value": doc,
                "similarity_score": round(similarity, 3)
            })

        return matches

    except Exception as e:
        return [{"error": f"Vector search failed: {str(e)}"}]


@tool
def fuzzy_match_values(query: str, candidates: List[str]) -> List[Dict]:
    """
    Perform fuzzy string matching against a list of candidate values.
    Uses SequenceMatcher for basic string similarity.

    Args:
        query: The value to match
        candidates: List of candidate values to match against

    Returns:
        List of dictionaries with 'value' and 'similarity_score' keys, sorted by score
    """
    matches = []

    for candidate in candidates:
        # Calculate similarity ratio
        ratio = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
        matches.append({
            "value": candidate,
            "similarity_score": round(ratio, 3)
        })

    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)

    return matches


# List of all disambiguator tools for easy import
disambiguator_tools = [
    get_column_values,
    vector_search_values,
    fuzzy_match_values
]
