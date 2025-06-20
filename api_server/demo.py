import os
import asyncio
import argparse
import logging
import fireducks.pandas as pd
from typing import List

from src.hipporag.HippoRAG import HippoRAG

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_rag_instance() -> HippoRAG:
    """
    Initializes and returns an instance of the HippoRAG class
    with configuration loaded from environment variables.
    """
    
    logging.info("Initializing HippoRAG instance...")
    return HippoRAG(
        save_dir=os.getenv('RAG_SAVE_DIR'),
        llm_name=os.getenv('LLM_MODEL_NAME', 'deepseek-chat'),
        llm_base_url=os.getenv('LLM_BASE_URL', 'https://api.deepseek.com'),
        llm_api_key=os.getenv('LLM_API_KEY'),
        embedding_name=os.getenv('EMBEDDING_MODEL_NAME', 'deepvk/USER-bge-m3'),
        embedding_base_url=os.getenv('EMBEDDING_BASE_URL', 'http://127.0.0.1:8888'),
        embedding_api_key=os.getenv('EMBEDDING_API_KEY', "sk-your_custom_key")
    )

def load_queries_from_file(filepath: str) -> List[str]:
    """
    Loads a list of questions from a CSV file.
    Expects the CSV to have a column named 'Question'.
    
    :param filepath: Path to the CSV file.
    :return: A list of questions as strings.
    """
    try:
        logging.info(f"Loading queries from '{filepath}'...")
        df = pd.read_csv(filepath)
        if 'Question' not in df.columns:
            raise ValueError("CSV file must contain a 'Question' column.")
        return df['Question'].dropna().tolist()
    except FileNotFoundError:
        logging.error(f"Error: The file '{filepath}' was not found.")
        exit(1)
    except Exception as e:
        logging.error(f"Failed to read or parse the queries file: {e}")
        exit(1)

def print_results(queries: List[str], rag_results: list):
    """
    Formats and prints the results from the RAG query in a readable way.
    
    :param queries: The original list of questions.
    :param rag_results: The nested list of result objects from HippoRAG.
    """
    if not rag_results:
        logging.warning("Received no results from the RAG query.")
        return

    results_for_queries = rag_results[0]
    
    print("\n" + "="*80)
    print(" " * 30 + "QUERY RESULTS")
    print("="*80 + "\n")

    for i, (query, result_obj) in enumerate(zip(queries, results_for_queries)):
        print(f"--- Query {i+1}/{len(queries)} ---")
        print(f"❓ Question: {query}")
        print("-" * 20)
        print(f"💬 Short Answer: {getattr(result_obj, 'short_answer', 'N/A')}")
        print(f"📖 Full Answer: {getattr(result_obj, 'full_answer', 'N/A')}")
        print(f"📚 Sources: {getattr(result_obj, 'docs', 'N/A')}")
        print("\n" + "="*80 + "\n")


async def main():
    """
    Main execution function for the demonstration script.
    It parses command-line arguments, initializes the RAG system,
    indexes documents, and runs queries against them.
    """
    parser = argparse.ArgumentParser(
        description="A demonstration script for the HippoRAG system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--documents",
        required=True,
        nargs='+',
        help="One or more paths to the document files (e.g., .csv) to be indexed."
    )
    parser.add_argument(
        "--queries-file",
        required=True,
        help="Path to a CSV file containing a 'Question' column with queries to run."
    )
    args = parser.parse_args()

    # --- Setup and Execution ---
    hipporag = setup_rag_instance()
    queries = load_queries_from_file(args.queries_file)
    
    # 1. Indexing
    logging.info(f"Starting indexing process for: {args.documents}")
    await hipporag.index(docs=args.documents)
    logging.info("Indexing complete.")

    # 2. Question Answering
    logging.info(f"Starting Q&A process for {len(queries)} queries...")
    rag_results = await hipporag.rag_qa(queries=queries)
    logging.info("Q&A process complete.")

    # 3. Print results
    print_results(queries, rag_results)


if __name__ == "__main__":
    asyncio.run(main())
