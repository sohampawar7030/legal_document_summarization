import os
from dotenv import load_dotenv
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Load environment variables from .env file
load_dotenv()

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def create_vector_store(text, embeddings_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Creates a FAISS vector store from the input text."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    return FAISS.from_texts(texts, embeddings)

def create_qa_pipeline(vector_store, llm_model="EleutherAI/gpt-neo-2.7B"):
    """Creates a Retrieval-based Question-Answering pipeline."""
    
    # Get the Hugging Face API token from the environment variable
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if huggingfacehub_api_token is None:
        raise ValueError("HuggingFace Hub API token is missing! Please set the 'HUGGINGFACEHUB_API_TOKEN' in your .env file.")
    
    retriever = vector_store.as_retriever()

    # Initialize Hugging Face LLM with the API token
    llm = HuggingFaceHub(
        repo_id=llm_model,  # specify the repo_id (e.g., gpt-neo-2.7B)
        huggingfacehub_api_token=huggingfacehub_api_token, 
        task="text-generation"  # specify the task (e.g., text-generation for language models)
    )
    
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

def process_pdf_and_answer(pdf_path):
    """Processes the PDF and returns answers to the text inside."""
    
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Create a FAISS vector store
    vector_store = create_vector_store(text)

    # Create a QA pipeline
    qa_pipeline = create_qa_pipeline(vector_store)

    # Answer the question
    # Since you no longer need to ask a question manually, just extract some context
    answer = qa_pipeline.run("Extract key information from the PDF.")  # Modify to get a summary or key data
    return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Pipeline for PDF analysis")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = args.pdf

    # Process the PDF and get results
    answer = process_pdf_and_answer(pdf_path)
    print(f"Answer: {answer}")
