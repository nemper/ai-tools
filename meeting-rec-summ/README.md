# Code Summarization Repository

This repository contains a Streamlit application that facilitates code summarization by utilizing OpenAI's language models. The application aids users in summarizing code and generating human-readable explanations for it. Below are explanations for the main functions and features of the application:

## Getting Started

To start using this code summarization application, follow these steps:

1. **Clone the Repository:** Clone this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```

2. **Navigate to the Directory:** Access the repository directory by running:

   ```bash
   cd your-repo
   ```

3. **Install Dependencies:** Install the necessary dependencies by executing:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:** Configure the required environment variables as follows:

   - `OPENAI_API_KEY`: This variable should store your OpenAI API key.

5. **Run the Application:** Launch the Streamlit application by running the following command:

   ```bash
   streamlit run your_app.py
   ```

## Usage

Once you've set up the application and launched it locally, you can use it to summarize code and generate explanations. Here's a breakdown of how to utilize the application:

### 1. Code Upload

- **Upload a Code File**: Begin by uploading a code file in one of the supported formats: `.txt`, `.pdf`, or `.docx`.

### 2. Model Selection

- **Choose a Language Model**: Select your preferred language model from the available options, which include GPT-4 8K and GPT-3.5 Turbo 16K.

### 3. Custom Prompts

- **Optional Prompt Upload**: If desired, you can upload a starting prompt and a final prompt for the summarization process. These prompts can be customized to provide specific instructions to the summarization model.

### 4. Summarization

- **Initiate Summarization**: Click the "Submit" button to initiate the summarization process. The application will automatically split the code into smaller chunks if necessary and summarize each segment based on your provided prompts.

### 5. Download Summaries

- **Download Generated Summaries**: Once the summarization is complete, you can download the generated summary as a text file, PDF, or DOCX file for your convenience.

### 6. User Feedback

- **Provide Feedback**: The application also allows you to provide feedback on the summarization results through the feedback interface.

## Features and Functions

Here's a more detailed explanation of the key functions and features of the application:

### Code Summarization

- The application's primary function is to summarize code and produce comprehensible explanations.

- Users can choose between two language models: GPT-4 8K and GPT-3.5 Turbo 16K.

- Large code files are automatically segmented into smaller chunks for efficient processing.

- Custom prompts can be tailored to guide the summarization process according to specific requirements.

- Users have the option to download the generated summaries in various formats.

### Additional Functions

In addition to code summarization, the application includes two additional functions:

1. **Fix Names**: This function allows users to correct misspelled names in a transcript. Users can upload a text file for correction and provide the corrected names.

2. **Transcript**: Users can convert an MP3 audio file to text by selecting the desired language. The resulting transcript can be downloaded for further use.

## Further Information

- This application relies on OpenAI's language models, so ensure you have a valid API key set up.

- Users can fine-tune the code summarization process by adjusting the prompts and instructions according to their needs.

- The application's versatility extends beyond code summarization, making it a valuable tool for various text-related tasks.

- For any issues or suggestions for improvement, users are encouraged to contribute to the repository or report any problems encountered.

Experience the convenience of code summarization with this application and explore its other useful features!
