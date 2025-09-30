"""
Check ChromaDB collections - View what embeddings have been created
"""
import os
from dotenv import load_dotenv
from value_disambiguator import ValueDisambiguator


def main():
    # Load environment variables
    load_dotenv()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set in .env file")
        return

    db_path = "Chinook.db"
    disambiguator = ValueDisambiguator(db_path, api_key)

    print("ChromaDB Collections Status")
    print("=" * 70)

    # Get all collections
    try:
        all_collections = disambiguator.client.list_collections()

        if not all_collections:
            print("\n⚠ No collections found. Run 'python setup.py' first.")
            return

        print(f"\nFound {len(all_collections)} collection(s):\n")

        for collection in all_collections:
            metadata = collection.metadata
            count = collection.count()

            table = metadata.get('table', 'unknown')
            column = metadata.get('column', 'unknown')

            print(f"  ✓ {table}.{column}")
            print(f"    - Collection name: {collection.name}")
            print(f"    - Vectors stored: {count}")

            # Show sample values
            if count > 0:
                sample = collection.get(limit=5)
                if sample and sample['documents']:
                    print(f"    - Sample values: {', '.join(sample['documents'][:3])}...")
            print()

        print("=" * 70)
        print(f"Total vectors: {sum(c.count() for c in all_collections)}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()