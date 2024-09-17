import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from openai import OpenAI
from langdetect import detect, LangDetectException
import re
import string
import concurrent.futures

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to convert a PDF page to an image
def convert_page_to_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# Function to perform OCR on an image
def ocr_image(image):
    return pytesseract.image_to_string(image)

# Function to extract text from a page (uses OCR if necessary)
def extract_text_from_page(page):
    text = page.get_text()
    if text.strip():
        return text
    else:
        # No text found, perform OCR
        page_image = convert_page_to_image(page)
        ocr_text = ocr_image(page_image)
        return ocr_text

# Function to clean up extracted text
def clean_text(text):
    # Remove non-printable characters
    text = ''.join(filter(lambda x: x in string.printable, text))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to check if text is in English
def is_text_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except LangDetectException:
        # If detection fails, assume it's not English
        return False

# Function to check if text is meaningful
def is_text_meaningful(text):
    # Check if text is empty after cleaning
    if not text.strip():
        return False

    # Calculate the proportion of alphabetic characters
    num_alpha = sum(c.isalpha() for c in text)
    num_chars = len(text)
    if num_chars == 0:
        return False
    alpha_ratio = num_alpha / num_chars

    # If the ratio of alphabetic characters is low, consider text not meaningful
    if alpha_ratio < 0.1:
        return False

    # If text length is too short, consider it not meaningful
    if len(text) < 30:
        return False

    return True

# Function to extract the device name using OpenAI API
def get_device_name(text):
    print("Extracting device name using OpenAI API...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a helpful assistant that analyzes given text to extract device name. You do not change the form of the name, only extract it as it is.
                Be careful not to alter the device name in any way; avoid adding any additional information or context to the extracted name. Do not use company name by mistake, e.g. "Sirona" is not a device name.
                If there is no device name in the text, please respond with the closest alternative that you can find in the provided text.
                Given the following text, extract the device name:"""
            },
            {
                "role": "user",
                "content": f"{text}"
            }
        ],
        temperature=0.0
    )
    device_name = response.choices[0].message.content.strip()
    return device_name


def process_single_pdf(pdf_file, pdf_dir, output_dir):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]
    json_output = os.path.join(output_dir, pdf_name + '.json')
    print(f"\nProcessing {pdf_file}...")

    # Open the PDF
    document = fitz.open(pdf_path)

    # For the first page, extract text and get device name
    first_page = document.load_page(0)
    first_page_text = extract_text_from_page(first_page)
    first_page_text_cleaned = clean_text(first_page_text)

    # Check if the first page text is meaningful
    if not is_text_meaningful(first_page_text_cleaned):
        print(f"Skipping {pdf_file} as first page text is not meaningful.")
        return

    # Check if the first page text is in English
    # if not is_text_english(first_page_text_cleaned):
    #    print(f"Skipping {pdf_file} as it is not in English.")
    #    continue  # Skip this document

    device_name = get_device_name(first_page_text_cleaned)
    print(f"Extracted device name: {device_name}")

    # Process each page
    results = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = extract_text_from_page(page)
        page_text_cleaned = clean_text(page_text)

        # Check if the page text is meaningful
        if not is_text_meaningful(page_text_cleaned):
            print(f"Skipping page {page_num + 1} (text not meaningful).")
            continue

        # Check if the page text is in English
        if not is_text_english(page_text_cleaned):
            print(f"Skipping page {page_num + 1} (not in English).")
            continue

        page_dict = {
            'date': 20240917,
            'page': page_num + 1,
            'text': page_text_cleaned,
            'device': device_name,
            'source': pdf_name + ".pdf",
            'url': ''
        }
        results.append(page_dict)

    if results:
        # Save the results to a JSON file
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {json_output}")
    else:
        print(f"No valid pages found in {pdf_file}.")


# Main function to process PDFs
def process_pdfs(pdf_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [
        f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf_file, pdf_dir, output_dir): pdf_file
            for pdf_file in pdf_files
        }
        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{pdf_file} generated an exception: {exc}")
            else:
                print(f"{pdf_file} processed successfully.")



if __name__ == '__main__':
    process_pdfs('./out', './json_output')