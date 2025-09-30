"""
Value Disambiguator - Matches fuzzy values using vector search
"""
import sqlite3
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class ValueDisambiguator:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        self.collections = {}

    def _get_column_values(self, table: str, column: str) -> List[str]:
        """Get distinct values from a table column"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 10000;")
            values = [str(row[0]) for row in cursor.fetchall()]
        except sqlite3.Error:
            values = []
        finally:
            conn.close()

        return values

    def _build_collection(self, table: str, column: str, show_progress: bool = False) -> chromadb.Collection:
        """Build vector collection for a table column"""
        collection_name = f"{table}_{column}".replace(" ", "_").lower()

        # Check if collection exists
        try:
            collection = self.client.get_collection(collection_name)
            if show_progress:
                print(f"  ✓ Using cached collection for {table}.{column}")
            return collection
        except:
            pass

        if show_progress:
            print(f"  → Building embeddings for {table}.{column}...", end=" ", flush=True)

        # Create new collection
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"table": table, "column": column}
        )

        # Get values and add to collection
        values = self._get_column_values(table, column)

        if values:
            if show_progress:
                print(f"(fetched {len(values)} values)", end=" ", flush=True)

            # Generate embeddings using Gemini
            embeddings_list = self.embeddings.embed_documents(values)

            if show_progress:
                print("(embeddings generated)", end=" ", flush=True)

            # Add to collection
            collection.add(
                documents=values,
                embeddings=embeddings_list,
                ids=[f"{table}_{column}_{i}" for i in range(len(values))]
            )

            if show_progress:
                print("✓")

        self.collections[collection_name] = collection
        return collection

    def disambiguate(self, user_value: str, table: str, column: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find the most similar values in the database

        Args:
            user_value: The fuzzy input from user
            table: Table name
            column: Column name
            top_k: Number of results to return

        Returns:
            List of (value, similarity_score) tuples
        """
        collection = self._build_collection(table, column)

        # Generate embedding for user value
        query_embedding = self.embeddings.embed_query(user_value)

        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        # Calculate similarity scores (ChromaDB returns distances, convert to similarity)
        matches = []
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            matches.append((doc, similarity))

        return matches

    def find_best_match(self, user_value: str, table: str, column: str, threshold: float = 0.7) -> str:
        """
        Find the best matching value above threshold

        Args:
            user_value: The fuzzy input from user
            table: Table name
            column: Column name
            threshold: Minimum similarity threshold

        Returns:
            Best matching value or original value if no good match
        """
        matches = self.disambiguate(user_value, table, column, top_k=1)

        if matches and matches[0][1] >= threshold:
            return matches[0][0]

        return user_value

    def identify_high_cardinality_columns(self, tables: List[str]) -> Dict[str, List[str]]:
        """
        Identify columns that might benefit from disambiguation

        Returns:
            Dict of {table: [columns]} for high-cardinality text columns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        high_card_columns = {}

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()

            text_columns = []
            for col in columns:
                col_name = col[1]
                col_type = col[2].upper()

                # Check if text column
                if any(t in col_type for t in ['VARCHAR', 'NVARCHAR', 'TEXT', 'CHAR']):
                    # Check cardinality
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table};")
                    distinct_count = cursor.fetchone()[0]

                    cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    total_count = cursor.fetchone()[0]

                    # High cardinality if >10 distinct values and >50% unique
                    if distinct_count > 10 and (distinct_count / max(total_count, 1)) > 0.5:
                        text_columns.append(col_name)

            if text_columns:
                high_card_columns[table] = text_columns

        conn.close()
        return high_card_columns

    def clear_collections(self):
        """Clear all vector collections"""
        for collection_name in list(self.collections.keys()):
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
        self.collections = {}