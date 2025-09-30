"""
SQL Agent - Main entry point
"""
import os
from dotenv import load_dotenv
from agent import SQLAgent


def main():
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

    print("SQL Agent for High-Cardinality Databases")
    print("=" * 50)
    print("Database: Chinook.db")
    print("Type 'exit' to quit\n")

    # Interactive loop
    while True:
        try:
            user_query = input("\nYour question: ").strip()

            if user_query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            if not user_query:
                continue

            # Process query
            result = agent.query(user_query, verbose=True)

            # Display results
            print("\n" + "=" * 50)
            print(agent.format_response(result))
            print("=" * 50)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
