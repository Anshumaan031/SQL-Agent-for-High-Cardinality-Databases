"""
Test Script for Multi-Agent SQL System
Demonstrates how to use the LangGraph-based SQL agent
"""
import os
from dotenv import load_dotenv
from agent.multi_agent_sql import MultiAgentSQL


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configuration
    db_path = "Chinook.db"  # Update with your database path
    api_key = os.getenv("GOOGLE_API_KEY")  # Or hardcode for testing

    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return

    # Initialize the multi-agent system
    print("Initializing Multi-Agent SQL System...")
    agent = MultiAgentSQL(db_path, api_key)
    print("✓ System initialized\n")

    # Test queries
    test_queries = [
        "Show me tracks by AC/DC",
        # "What are the top 5 genres by total sales?",
        # "List all customers from USA",
        # "Find the most expensive album",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"Test Query {i}: {query}")
        print(f"{'#'*70}")

        # Execute query with verbose output
        result = agent.query(query, verbose=True)

        # Display formatted results
        if result["success"]:
            print("\n--- Results ---")
            print(agent.format_results(result))
        else:
            print(f"\n--- Error ---")
            print(f"Failed to execute query: {result['error']}")

        print(f"\n--- Summary ---")
        print(f"Intent: {result['intent']}")
        print(f"Tables: {result['selected_tables']}")
        print(f"Disambiguated: {result['disambiguated_values']}")
        print(f"SQL Attempts: {result['sql_attempts']}")


def interactive_mode():
    """
    Interactive mode - ask questions to the database
    """
    # Load environment variables from .env file
    load_dotenv()

    db_path = "Chinook.db"
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return

    agent = MultiAgentSQL(db_path, api_key)

    print("\n" + "="*70)
    print("Multi-Agent SQL System - Interactive Mode")
    print("="*70)
    print("Ask questions about your database in natural language.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_query = input("\nYour question: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_query:
                continue

            # Execute query
            result = agent.query(user_query, verbose=True)

            # Display results
            if result["success"]:
                print("\n" + agent.format_results(result))
            else:
                print(f"\nError: {result['error']}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
