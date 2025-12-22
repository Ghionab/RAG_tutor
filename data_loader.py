import os
from google import genai  # Corrected Import
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize the Client
# It will automatically look for GOOGLE_API_KEY in your .env
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

EMBED_MODEL = "text-embedding-004"

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # New SDK method: client.models.embed_content
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config={'task_type': 'RETRIEVAL_DOCUMENT'}
    )
    
    # Each item in response.embeddings has a .values attribute containing the floats
    return [item.values for item in response.embeddings]