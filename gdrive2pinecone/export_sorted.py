from os import getenv
from pinecone import Pinecone
import numpy as np

def connect_to_pinecone():
    """
    Connects to the Pinecone index using the API key and host from environment variables.
    
    Returns:
        pinecone_client.Index: The connected Pinecone index.
    """
    # Retrieve API key from environment variables
    pinecone_api_key = getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    
    # Define Pinecone host
    pinecone_host = getenv('PINECONE_HOST')
    
    # Initialize Pinecone client
    pinecone_client = Pinecone(api_key=pinecone_api_key, host=pinecone_host)
    
    # Connect to the index
    return pinecone_client.Index(host=pinecone_host)

def fetch_data_from_pinecone(index, namespace, date_filter, top_k=100):
    """
    Fetches data from Pinecone based on the specified namespace and date filter.
    
    Args:
        index (pinecone_client.Index): The connected Pinecone index.
        namespace (str): The namespace to query.
        date_filter (int or str): The date value to filter by.
        top_k (int): The number of top results to fetch.
    
    Returns:
        list: A list of dictionaries containing 'chunk' and 'context' from each matched entry.
    """
    # Placeholder vector (length 3072, all zeros)
    placeholder_vector = np.zeros(3072).tolist()
    
    # Define the filter based on the data type of 'date'
    # Adjust the type (int or str) based on your metadata schema
    if isinstance(date_filter, int):
        filter_condition = {"date": {"$eq": date_filter}}
    elif isinstance(date_filter, str):
        filter_condition = {"date": {"$eq": date_filter}}
    else:
        raise TypeError("date_filter must be an integer or a string.")
    
    try:
        # Perform the query
        results = index.query(
            vector=placeholder_vector, 
            top_k=top_k, 
            filter=filter_condition, 
            namespace=namespace,
            include_metadata=True  # Ensure metadata is included in the results
        )
        
        print(f"Number of matches found: {len(results.get('matches', []))}")
        
    except Exception as e:
        print(f"An error occurred during the query: {e}")
        return []
    
    # Extract 'chunk' and 'context' from each match
    extracted_data = []
    for entry in results.get("matches", []):
        metadata = entry.get("metadata", {})
        if "chunk" in metadata and "context" in metadata:
            extracted_data.append({
                "chunk": metadata["chunk"],
                "context": metadata["context"]
            })
        else:
            print(f"Entry missing 'chunk' or 'context': {metadata}")
    
    # Sort the data by 'chunk' to maintain order
    extracted_data = sorted(extracted_data, key=lambda x: x["chunk"])
    
    print(f"Total entries extracted and sorted: {len(extracted_data)}")
    
    return extracted_data

def write_to_file(data, filename="out.txt"):
    """
    Writes the 'context' from each data entry to a text file, separated by double newlines.
    
    Args:
        data (list): A list of dictionaries containing 'chunk' and 'context'.
        filename (str): The name of the output file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            for entry in data:
                file.write(entry["context"] + "\n\n")
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def main():
    """
    Main function to execute the data extraction and writing process.
    """
    # Connect to Pinecone
    index = connect_to_pinecone()
    
    # Define the namespace
    namespace = getenv('NAMESPACE')
    date_filter = '' # depends on what format you use
    
    # Fetch data from Pinecone
    data = fetch_data_from_pinecone(index, namespace, date_filter, top_k=100)
    
    if data:
        # Write the fetched data to out.txt
        write_to_file(data, filename="out.txt")
    else:
        print("No data fetched. Please check your filters and try again.")

if __name__ == "__main__":
    main()
