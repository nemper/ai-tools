import os
import json
from time import sleep
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor

# Set up OpenAI API client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Set up Pinecone client and index
api_key = os.environ.get('PINECONE_API_KEY')
host = os.getenv("PINECONE_HOST")  # Replace with your actual host

pinecone = Pinecone(api_key=api_key, host=host)
index = pinecone.Index(host=host)

# Function to process each JSON file
def process_json_file(json_file, json_dir, index, namespace, embed_model="text-embedding-3-large", batch_size=100):
    err_log = ""
    json_path = os.path.join(json_dir, json_file)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists
    texts = []
    metadatas = []
    ids = []

    # Process each JSON object in the data
    for idx, item in enumerate(data):
        text = item.get('context', '')
        if not text.strip():
            continue  # Skip empty texts

        texts.append(text)
        metadata = {key: value for key, value in item.items()}

        metadatas.append(metadata)
        # Generate a unique ID for each entry
        ids.append(f"{os.path.splitext(json_file)[0]}_{metadata.get('page', idx)}")

    # Process embeddings in batches
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Upserting {json_file}"):
        # Get batch data
        i_end = min(len(texts), i + batch_size)
        batch_texts = texts[i:i_end]
        batch_metadatas = metadatas[i:i_end]
        batch_ids = ids[i:i_end]

        # Create embeddings
        try:
            # Note: OpenAI API now returns an object, and we need to access 'data' attribute
            response = client.embeddings.create(input=batch_texts, model=embed_model)
            embeddings = [record.embedding for record in response.data]
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i_end} in {json_file}: {e}")
            err_log += f"Error generating embeddings for batch {i}-{i_end} in {json_file}: {e}\n"
            sleep(5)
            continue  # Skip this batch

        # Prepare data for upsert
        vectors = []
        for j in range(len(embeddings)):
            vector = {
                'id': batch_ids[j],
                'values': embeddings[j],
                'metadata': batch_metadatas[j]
            }
            vectors.append(vector)

        # Upsert vectors to Pinecone
        try:
            index.upsert(vectors=vectors, namespace=namespace)
        except Exception as e:
            print(f"Error upserting batch {i}-{i_end} in {json_file}: {e}")
            err_log += f"Error upserting batch {i}-{i_end} in {json_file}: {e}\n"
            sleep(5)
            continue  # Skip this batch

    print(f"Upserted data from {json_file}")

    # Return any errors to be logged later
    return err_log

# Function to do embeddings concurrently
def do_embeddings(json_dir, index, namespace):
    # List all JSON files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # Track errors in processing
    err_log = ""

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=12) as executor:
        # Submit the jobs and process them concurrently
        futures = [executor.submit(process_json_file, json_file, json_dir, index, namespace) for json_file in json_files]

        # Collect the results (error logs) from the threads
        for future in tqdm(futures, desc="Processing all files"):
            err_log += future.result()  # Append error logs from each thread

    # Save error log if any
    if err_log:
        with open("err_log.txt", "w", encoding="utf-8") as file:
            file.write(err_log)
        print("Errors occurred during upsert. Check err_log.txt for details.")

    print("Data upserted to Pinecone successfully.")

if __name__ == '__main__':
    do_embeddings(json_dir='./gdrive_jsons', index=index, namespace='denty-komercijalista')
