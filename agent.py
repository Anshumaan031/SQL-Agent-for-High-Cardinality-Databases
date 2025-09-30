"""
SQL Agent - Main orchestration logic
"""
import re
from typing import Dict, Optional
from table_selector import TableSelector
from value_disambiguator import ValueDisambiguator
from query_generator import QueryGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


class SQLAgent:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.api_key = api_key

        # Initialize components
        self.table_selector = TableSelector(db_path, api_key)
        self.value_disambiguator = ValueDisambiguator(db_path, api_key)
        self.query_generator = QueryGenerator(db_path, api_key)

        # LLM for intent parsing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )

        self.max_retries = 2

    def parse_intent(self, user_query: str) -> Dict:
        """
        Parse user query to extract intent and potential values

        Returns:
            Dict with {intent, entities}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent parser for SQL queries. Extract:
1. The user's intent (what they want to know)
2. Any specific values or entities mentioned (names, locations, etc.)

Return in this format:
INTENT: <brief description>
ENTITIES: <comma-separated list of specific values mentioned>

If no specific entities, write "ENTITIES: none"""),
            ("user", "{query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": user_query})

        # Parse response
        content = response.content.strip()
        intent_match = re.search(r'INTENT:\s*(.+?)(?:\n|$)', content)
        entities_match = re.search(r'ENTITIES:\s*(.+?)(?:\n|$)', content)

        intent = intent_match.group(1).strip() if intent_match else user_query
        entities_str = entities_match.group(1).strip() if entities_match else "none"

        entities = []
        if entities_str.lower() != "none":
            entities = [e.strip() for e in entities_str.split(',')]

        return {
            "intent": intent,
            "entities": entities
        }

    def disambiguate_values(self, entities: list, tables: list, verbose: bool = False) -> Dict[str, str]:
        """
        Disambiguate entity values against database

        Returns:
            Dict of {original_value: disambiguated_value}
        """
        disambiguated = {}

        if not entities:
            return disambiguated

        # Identify high-cardinality columns in selected tables
        high_card_columns = self.value_disambiguator.identify_high_cardinality_columns(tables)

        # Try to match each entity
        for entity in entities:
            best_match = None
            best_score = 0

            # Search across all high-cardinality columns
            for table, columns in high_card_columns.items():
                for column in columns:
                    if verbose:
                        print(f"    Checking {table}.{column} for '{entity}'...")
                    matches = self.value_disambiguator.disambiguate(entity, table, column, top_k=1)

                    if matches and matches[0][1] > best_score:
                        best_match = matches[0][0]
                        best_score = matches[0][1]

            # Only use disambiguation if confidence is high enough
            if best_match and best_score > 0.7:
                disambiguated[entity] = best_match
                if verbose:
                    print(f"    ✓ Matched '{entity}' → '{best_match}' (score: {best_score:.2f})")

        return disambiguated

    def query(self, user_query: str, verbose: bool = False) -> Dict:
        """
        Main query method - orchestrates the entire pipeline

        Args:
            user_query: Natural language question
            verbose: Print intermediate steps

        Returns:
            Dict with {success, sql, results, error, steps}
        """
        steps = []

        try:
            # Step 1: Parse intent
            if verbose:
                print("\n[1] Parsing intent...")
            intent_info = self.parse_intent(user_query)
            steps.append(f"Intent: {intent_info['intent']}")
            steps.append(f"Entities: {intent_info['entities']}")

            # Step 2: Select relevant tables
            if verbose:
                print("[2] Selecting relevant tables...")
            tables = self.table_selector.select_tables(user_query)
            steps.append(f"Selected tables: {tables}")

            if not tables:
                return {
                    "success": False,
                    "sql": None,
                    "results": None,
                    "error": "Could not identify relevant tables",
                    "steps": steps
                }

            # Step 3: Disambiguate values
            if verbose:
                print("[3] Disambiguating values...")
            disambiguated = self.disambiguate_values(intent_info['entities'], tables, verbose=verbose)
            if disambiguated:
                steps.append(f"Disambiguated: {disambiguated}")

            # Step 4: Get table schemas
            table_schemas = self.table_selector.get_table_schemas(tables)

            # Step 5: Generate query
            if verbose:
                print("[4] Generating SQL query...")
            sql = self.query_generator.generate_query(
                user_query,
                table_schemas,
                disambiguated if disambiguated else None
            )
            steps.append(f"Generated SQL: {sql}")

            # Step 6: Execute query with retry logic
            if verbose:
                print("[5] Executing query...")

            retry_count = 0
            while retry_count <= self.max_retries:
                success, result = self.query_generator.execute_query(sql)

                if success:
                    steps.append("Query executed successfully")
                    return {
                        "success": True,
                        "sql": sql,
                        "results": result,
                        "error": None,
                        "steps": steps
                    }
                else:
                    # Query failed - try to fix
                    error = result
                    error_type = self.query_generator.classify_error(error)
                    steps.append(f"Error ({error_type}): {error}")

                    if retry_count < self.max_retries:
                        if verbose:
                            print(f"[6] Fixing query (attempt {retry_count + 1})...")

                        sql = self.query_generator.fix_query(sql, error, table_schemas)
                        steps.append(f"Fixed SQL: {sql}")
                        retry_count += 1
                    else:
                        return {
                            "success": False,
                            "sql": sql,
                            "results": None,
                            "error": error,
                            "steps": steps
                        }

        except Exception as e:
            steps.append(f"Exception: {str(e)}")
            return {
                "success": False,
                "sql": None,
                "results": None,
                "error": str(e),
                "steps": steps
            }

    def format_response(self, result: Dict) -> str:
        """Format agent response for display"""
        output = []

        if result['success']:
            output.append("✓ Query succeeded\n")
            output.append(f"SQL: {result['sql']}\n")
            output.append("\nResults:")
            output.append(self.query_generator.format_results(result['results']))
        else:
            output.append("✗ Query failed\n")
            if result['sql']:
                output.append(f"SQL: {result['sql']}\n")
            output.append(f"Error: {result['error']}")

        output.append("\n--- Execution Steps ---")
        for step in result['steps']:
            output.append(f"  {step}")

        return "\n".join(output)