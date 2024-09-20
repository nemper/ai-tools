import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Function to check if the text is in English (you should implement this)
def check_if_english(text: str) -> bool:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
          messages=[
            {"role": "system", "content": """You are a helpful assistant that does the following:
             - Determines if a provided text is in English or not.
             - If it isn't: return False. If it is in English you remove unintelligible parts from it, and return the edited text.
             - If the entirety of the text is unintelligible (language can't be defined), return False.

             Do NOT edit the text in any other way."""},
            {"role": "user", "content": "6 186 7 3/8 Kabeldurchfhrung Cable opening Passage de cble Pasacable Holzbalken wood studs poutre en bois madero 1260 49 5/8 410 16 1/8 347 13 5/8 53 2 1/8 162 6 3/8 2300 90 1/2 Deckenhhe Ceiling height Hauteur Altura del techo min"},
            {"role": "assistant", "content": "False"},
            {"role": "user", "content": "p~=a~=p=de 59 87 685 D 3352 D 3352.105.01.12.02 Dear customer You would like to have and will have many years of satisfaction with your p~ X-ray unit. Safety and reliability are necessary to ensure this."},
            {"role": "assistant", "content": "Dear customer You would like to have and will have many years of satisfaction with your p~ X-ray unit. Safety and reliability are necessary to ensure this."},
            {"role": "user", "content": "61 25 665 D3437 D3437.076.01.15.02 06.2012 15"},
            {"role": "assistant", "content": "False"},
            {"role": "user", "content": f"{text}"},
        ]
    )

    return response.choices[0].message.content.strip()


# Function to process each JSON file
def process_json_file(file_path):
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out non-English entries
    filtered_data = [entry for entry in data if check_if_english(entry['text']) not in [False, "False"]]

    # Save the filtered data back to the same file (or a different one if preferred)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    return file_path

# Function to handle processing all JSON files concurrently
def process_json_files_concurrently(json_folder, max_workers=12):
    # Find all JSON files in the folder
    json_files = list(Path(json_folder).glob("*.json"))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each file
        future_to_file = {executor.submit(process_json_file, file): file for file in json_files}
        
        # Process the results as they complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                print(f"Processed file: {result}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    # Folder containing your JSON files
    json_folder = "json_output"
    
    # Process JSON files concurrently using 12 cores
    process_json_files_concurrently(json_folder, max_workers=12)
