# my_pdf_processor.py

#from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from transformers import pipeline
from ai71 import AI71
from sentence_transformers import SentenceTransformer
from langchain_core.language_models import LLM
from pydantic import Field

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

# Initialize summarization pipeline
summarizer = pipeline("summarization")

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def process_pdf_query(pdf_path, query, mode='question'):
    text = read_pdf(pdf_path)

    # Split into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = char_text_splitter.split_text(text)

    # Debugging information
    print(f"Number of text chunks: {len(text_chunks)}")

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
                stop = kwargs.pop('stop', [])
                if not isinstance(stop, list):
                    stop = []

                response = self.client.completions.create(
                    model="tiiuae/falcon-40B",
                    prompt=prompt,
                    stop=stop,
                    **kwargs
                )

                return response['choices'][0]['text'] if 'choices' in response and len(response['choices']) > 0 else str(response)
            except Exception as e:
                print(f"Error calling AI71 API: {e}")
                return f"Error: {str(e)}"

        @property
        def _llm_type(self):
            return "AI71"

    # Initialize LLM
    llm = ConcreteAI71LLM(client)

    if mode == 'question':
        # Perform similarity search and generate a response
        docs = docsearch.similarity_search(query)
        if not docs:
            return "No relevant documents found for the query."

        context = " ".join([doc.page_content for doc in docs])
        prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
        response = llm._call(prompt, temperature=0.3, max_tokens=100)  # Adjust parameters as needed
        
        # Extract the relevant part of the response starting from 'text='
        start_index = response.find("text=")
        if (start_index != -1):
            relevant_response = response[start_index+6:].strip()
        else:
            relevant_response = response.strip()  # Fallback if 'text=' is not found

            # Remove the last part after "\nUser:"
            end_index = relevant_response.find("\\nUser:")
            if end_index != -1:
                relevant_response = relevant_response[:end_index].strip()

            return relevant_response


    elif mode == 'summarize':
        if not text_chunks:
            return "No text available to summarize."

        # Summarize each chunk separately
        summaries = []
        max_chunk_length = 1024  # Ensure this is within the model's limits
        for chunk in text_chunks:
            if len(chunk) > max_chunk_length:
                # Further split chunks if needed
                chunk_splitter = CharacterTextSplitter(separator="\n", chunk_size=max_chunk_length, chunk_overlap=200, length_function=len)
                sub_chunks = chunk_splitter.split_text(chunk)
                for sub_chunk in sub_chunks:
                    try:
                        summary = summarizer(sub_chunk, max_length=200, min_length=50, do_sample=False)
                        if summary:
                            summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Error during summarization: {e}")
                        summaries.append("Summary could not be generated.")
            else:
                try:
                    summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
                    if summary:
                        summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    print(f"Error during summarization: {e}")
                    summaries.append("Summary could not be generated.")

        return " ".join(summaries)

    elif mode == 'tutor':
        pdf_text = text  # Save the PDF text to use in tutor mode
        print("PDF content successfully loaded. The AI will now ask questions based on the content.")
        print("Type 'exit' to end the session.")

        # Generate questions based on the PDF content
        questions_prompt = f"Based on the following content, generate questions:\n\n{pdf_text}\n\nQuestions:"
        questions_response = llm._call(questions_prompt, temperature=0.3, max_tokens=100)

        questions = questions_response.split("\n")
        questions = [q.strip() for q in questions if q.strip()]

        for question in questions:
            print(f"\nQuestion: {question}")
            user_answer = input("Your answer: ")

            feedback_prompt = f"Here is the content:\n{pdf_text}\n\nThe question was: {question}\n\nThe user's answer: {user_answer}\n\nProvide feedback on the user's answer:"
            feedback_response = llm._call(feedback_prompt, temperature=0.3, max_tokens=100)
            
            print(f"Feedback: {feedback_response}")

    else:
        return "Invalid mode. Choose 'question', 'summarize', or 'interactive'."
