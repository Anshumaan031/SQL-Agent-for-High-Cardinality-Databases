"""
Multi-Agent SQL System - Quick Start
Run this script to test the LangGraph-based agentic SQL system

NOTE: Uses SimpleMultiAgentSQL to reduce API calls and avoid rate limits
"""
import os
from dotenv import load_dotenv
from agent import SimpleMultiAgentSQL  # Uses linear workflow (fewer API calls)


def main():
    """Run example queries through the multi-agent system"""

    # Load environment variables from .env file
    load_dotenv()

    # Configuration
    db_path = "Chinook.db"
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: Please set GOOGLE_API_KEY environment variable")
        print("Example: export GOOGLE_API_KEY='your-api-key'")
        return

    print("="*70)
    print("Multi-Agent SQL System - Simplified (Low API Usage)")
    print("="*70)
    print(f"\nDatabase: {db_path}")
    print("Initializing agents...")

    # Initialize the simplified multi-agent system
    # Uses linear workflow instead of supervisor routing
    # Reduces API calls from ~10-15 to ~4-6 per query
    agent = SimpleMultiAgentSQL(db_path, api_key)
    print("✓ System ready!\n")
    print("ℹ️  Using simplified workflow to avoid rate limits")

    # Example queries
    queries = [
        #"Show me tracks by AC/DC",
        #"What are the top 5 genres by number of tracks?",
        'Find customers who have spent more than the average customer'
        # "List customers from USA",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}\n")

        # Execute query
        result = agent.query(query, verbose=True)

        # Display results
        if result["success"]:
            print("\n📊 Results:")
            print(agent.format_results(result))
            print(f"\nℹ️  SQL: {result['sql']}")
        else:
            print(f"\n❌ Error: {result['error']}")

        input("\n Press Enter to continue...")


if __name__ == "__main__":
    main()
