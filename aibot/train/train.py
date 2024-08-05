
import os
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from ai71 import AI71
from sentence_transformers import SentenceTransformer
from langchain_core.language_models import LLM
from pydantic import Field
from langchain.embeddings.base import Embeddings

# Set API key for AI71
AI71_API_KEY = "ai71-api-0b484342-254e-4b5e-9d9e-0a6ca1c43f4f"

if AI71_API_KEY is None:
    raise ValueError("API key not found. Please check your .env file.")
else:
    print("API key successfully loaded.")

# Initialize AI71 client
client = AI71(AI71_API_KEY)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

############ TEXT LOADERS ############
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text

###############################################
# Directory path
train_directory = r'D:\PROGRAMING\CodeBluepdf\basic-no-web-app\train_files'

# Read documents from the directory
text = read_documents_from_directory(train_directory)

# Split text into chunks
char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
text_chunks = char_text_splitter.split_text(text)

# Create embeddings using SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

embedding_instance = SentenceTransformerEmbeddings(model)
docsearch = FAISS.from_texts(text_chunks, embedding_instance)

# Define a concrete subclass of LLM for AI71
class ConcreteAI71LLM(LLM):
    client: AI71 = Field(...)

    def __init__(self, client: AI71):
        super().__init__(client=client)

    def _call(self, prompt: str, **kwargs):
        try:
            # Ensure the stop parameter is a list or provide a default empty list
            stop = kwargs.pop('stop', [])
            if not isinstance(stop, list):
                stop = []

            # Use the completions.create method from AI71
            response = self.client.completions.create(
                model="tiiuae/falcon-40B",  # Replace with the correct model name
                prompt=prompt,
                stop=stop,
                **kwargs  # Pass other arguments as needed
            )

            # Adjust based on how the API returns the response
            return response['choices'][0]['text'] if 'choices' in response and len(response['choices']) > 0 else str(response)
        except Exception as e:
            print(f"Error calling AI71 API: {e}")
            return f"Error: {str(e)}"

    @property
    def _llm_type(self):
        return "AI71"

# Initialize LLM and QA chain
llm = ConcreteAI71LLM(client)
chain = load_qa_chain(llm, chain_type="stuff")

##################################################
# Ask a question
query = "which is the most anxious nation in the world?"

# Perform similarity search
docs = docsearch.similarity_search(query)

# Generate and print the response
try:
    response = chain.invoke({"input_documents": docs, "question": query})

    # Extract the relevant part of the response starting from 'Falcon'
    output_text = response['output_text']
    start_index = output_text.find("text=")
    if start_index != -1:
        relevant_output = output_text[start_index+6:].strip()
    else:
        relevant_output = output_text.strip()  # Fallback if 'Falcon:' is not found

    # Remove the last part after "\nUser:"
    end_index = relevant_output.find("\\nUser:")
    if end_index != -1:
        relevant_output = relevant_output[:end_index].strip()
    relevant_output=relevant_output.replace('\\n','\n')
    print(" ")
    print(query)
    print(relevant_output)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the AI71 API documentation for the correct method to generate responses.")