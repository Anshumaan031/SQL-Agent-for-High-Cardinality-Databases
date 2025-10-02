"""
Test script for SQL Agent
"""
import os
from dotenv import load_dotenv
from agent import SQLAgent


def test_queries():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set it in .env file")
        return

    # Initialize agent
    db_path = "Chinook.db"
    agent = SQLAgent(db_path, api_key)

    # Test queries
    test_cases = [
        # "Show me all tracks by AC/DC",
        # "How many customers are from USA?",
        # "What are the top 5 best selling tracks?",
        "Find invoices from customers in SF",  # Test disambiguation (SF -> San Francisco)
    ]

    print("Testing SQL Agent")
    print("=" * 70)

    for i, query in enumerate(test_cases, 1):
        print(f"\n\nTest {i}: {query}")
        print("-" * 70)

        result = agent.query(query, verbose=True)

        print("\n" + agent.format_response(result))


if __name__ == "__main__":
    test_queries()