"""
Setup script - Pre-builds vector embeddings for high-cardinality columns
"""
import os
import sys
from dotenv import load_dotenv
from value_disambiguator import ValueDisambiguator
from table_selector import TableSelector


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set it in .env file or with: set GOOGLE_API_KEY=your_key_here (Windows)")
        return

    db_path = "Chinook.db"

    print("SQL Agent Setup - Pre-building Vector Embeddings")
    print("=" * 70)

    # Initialize components
    print("\n[1/3] Initializing components...")
    table_selector = TableSelector(db_path, api_key)
    disambiguator = ValueDisambiguator(db_path, api_key)

    # Get all tables
    print("\n[2/3] Analyzing database schema...")
    all_tables = list(table_selector.schema_info.keys())
    print(f"Found {len(all_tables)} tables: {', '.join(all_tables)}")

    # Identify high-cardinality columns
    print("\n[3/3] Identifying high-cardinality columns...")
    high_card_columns = disambiguator.identify_high_cardinality_columns(all_tables)

    if not high_card_columns:
        print("\n✓ No high-cardinality columns found. No embeddings needed.")
        return

    # Display what will be processed
    print(f"\nFound {sum(len(cols) for cols in high_card_columns.values())} high-cardinality columns:")
    total_columns = 0
    for table, columns in high_card_columns.items():
        print(f"  • {table}: {', '.join(columns)}")
        total_columns += len(columns)

    # Build embeddings
    print(f"\n{'=' * 70}")
    print("Building vector embeddings...")
    print(f"{'=' * 70}\n")

    built_count = 0
    for table, columns in high_card_columns.items():
        for column in columns:
            built_count += 1
            print(f"[{built_count}/{total_columns}]", end=" ")

            try:
                # This will build and cache the collection
                collection = disambiguator._build_collection(table, column, show_progress=True)
            except Exception as e:
                print(f"  ✗ Error building {table}.{column}: {str(e)}")

    print(f"\n{'=' * 70}")
    print(f"✓ Setup complete! Built embeddings for {built_count} columns.")
    print(f"{'=' * 70}")
    print("\nYou can now run:")
    print("  • python main.py       - Interactive mode")
    print("  • python test_agent.py - Run tests")


if __name__ == "__main__":
    main()