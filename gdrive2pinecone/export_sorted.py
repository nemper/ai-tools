from os import getenv
from pinecone import Pinecone
from numpy import zeros as np_zeros
import logging

logging.basicConfig(level=logging.INFO)

def get_env_vars(*args):
    return {arg: getenv(arg) for arg in args}

env_vars = get_env_vars('PINECONE_API_KEY', 'PINECONE_HOST', 'NAMESPACE')

PINECONE_API_KEY = env_vars['PINECONE_API_KEY']
PINECONE_HOST = env_vars['PINECONE_HOST']
NAMESPACE = env_vars['NAMESPACE']
PLACEHOLDER_VECTOR = np_zeros(3072).tolist()


def connect_to_pinecone():
    """
    Connects to the Pinecone index using the API key and host from environment variables.
    
    Returns:
        pinecone_client.Index: The connected Pinecone index.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
    return pinecone_client.Index(host=PINECONE_HOST)

def fetch_data_from_pinecone(index, namespace, date_filter, top_k=100):
    """
    Fetches data from Pinecone based on the specified namespace and date filter.
    """
    # Define the filter based on the data type of 'date'
    if isinstance(date_filter, int) or isinstance(date_filter, str):
        filter_condition = {"date": {"$eq": date_filter}}
    else:
        raise TypeError("date_filter must be an integer or a string.")
    
    try:
        # Perform the query
        results = index.query(
            vector=PLACEHOLDER_VECTOR, 
            top_k=top_k, 
            filter=filter_condition, 
            namespace=namespace,
            include_metadata=True
        )
        
        logging.info(f"Number of matches found: {len(results.get('matches', []))}")
        
    except Exception as e:
        logging.error(f"An error occurred during the query: {e}")
        return []
    
    sorting_id = "..."
    extracted_data = [
        {sorting_id: entry["metadata"][sorting_id], "context": entry["metadata"]["context"]}
        for entry in results.get("matches", [])
        if "metadata" in entry and sorting_id in entry["metadata"] and "context" in entry["metadata"]
    ]
    extracted_data = sorted(extracted_data, key=lambda x: x[sorting_id])
    
    logging.info(f"Total entries extracted and sorted: {len(extracted_data)}")
    
    return extracted_data

def write_to_file(data, filename="pinecone_out.txt"):
    """
    Writes the 'context' from each data entry to a text file, separated by double newlines.
    """
    try:
        with open(filename, "w", encoding="utf-8-sig") as file:
            file_content = "\n\n".join(entry["context"] for entry in data)
            file.write(file_content)
        logging.info(f"Data successfully written to {filename}")
    except Exception as e:
        logging.error(f"An error occurred while writing to the file: {e}")

def main():
    index = connect_to_pinecone()
    
    namespace = NAMESPACE
    date_filter = "..."  # Define appropriately based on your schema

    data = fetch_data_from_pinecone(index, namespace, date_filter, top_k=100)
    
    if data:
        write_to_file(data)
    else:
        logging.info("No data fetched. Please check your filters and try again.")

if __name__ == "__main__":
    main()
