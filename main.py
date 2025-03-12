from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import os
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv
from db import add_to_collection, retrieve_from_collection
from io import BytesIO
import camelot
import json
import requests
import traceback

# Load environment variables
load_dotenv()
os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="PDF RAG Backend")

# Pydantic models for request/response validation
class URLRequest(BaseModel):
    urls: List[str]

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

# Directus configuration (no token for now with public access)
DIRECTUS_URL = "http://localhost:8055"  # Your Directus instance URL
HEADERS = {
    "Content-Type": "application/json"  # No Authorization header for public access
}

# Helper functions (adapted from your code)
def download_pdf_from_url(url: str) -> BytesIO:
    """Download a PDF from a URL and return it as a BytesIO object."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        if 'application/pdf' not in response.headers.get('Content-Type', ''):
            raise HTTPException(status_code=400, detail=f"URL {url} does not point to a PDF.")
        pdf_bytes = BytesIO(response.content)
        pdf_bytes.name = url.split('/')[-1] or "downloaded_pdf.pdf"
        return pdf_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF from {url}: {str(e)}")

def get_pdf_text(pdf_files: List[BytesIO]) -> List[Document]:
    """Extract text and tables from PDFs."""
    documents = []
    for pdf in pdf_files:
        temp_path = f"temp_{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())

        # Extract text with PyPDF2
        text_data = {"text": []}
        try:
            pdf_reader = PdfReader(temp_path)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text().strip()
                if page_text:
                    text_data["text"].append({"page": page_num + 1, "content": page_text})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting text from {pdf.name}: {str(e)}")

        # Extract tables with camelot
        table_data = {"tables": []}
        try:
            tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
            for i, table in enumerate(tables):
                if table.df.size > 0:
                    table_data["tables"].append({
                        "table_id": f"Table {i + 1}",
                        "page": table.parsing_report['page'],
                        "data": table.df.replace({None: ""}).to_dict(orient='records')
                    })
            if not table_data["tables"]:
                tables = camelot.read_pdf(temp_path, pages='all', flavor='stream')
                for i, table in enumerate(tables):
                    if table.df.size > 0:
                        table_data["tables"].append({
                            "table_id": f"Table {i + 1}",
                            "page": table.parsing_report['page'],
                            "data": table.df.replace({None: ""}).to_dict(orient='records')
                        })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting tables from {pdf.name}: {str(e)}")

        combined_data = {
            "source": pdf.name,
            "text": text_data["text"],
            "tables": table_data["tables"]
        }
        json_content = json.dumps(combined_data, ensure_ascii=False, indent=2)

        doc = Document(page_content=json_content, metadata={"filename": pdf.name})
        documents.append(doc)
        os.remove(temp_path)

    return documents

def get_text_chunks(documents: List[Document]) -> List[Document]:
    """Chunk documents and add to ChromaDB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250, separators=["\n\n", "\n", ","])
    all_chunks = []

    for doc in documents:
        try:
            data = json.loads(doc.page_content)
            text_entries = data.get("text", [])
            table_entries = data.get("tables", [])

            # Process text
            if text_entries:
                text_content = "\n".join(entry.get("content", "") for entry in text_entries)
                if text_content:
                    text_chunks = text_splitter.create_documents(
                        texts=[text_content],
                        metadatas=[{"source": doc.metadata["filename"], "type": "text"}]
                    )
                    chunk_texts = [chunk.page_content for chunk in text_chunks]
                    add_to_collection(chunk_texts, doc.metadata['filename'])
                    all_chunks.extend(text_chunks)

            # Process tables
            for table in table_entries:
                table_json = json.dumps(table, ensure_ascii=False)
                table_chunk = Document(
                    page_content=table_json,
                    metadata={"source": doc.metadata["filename"], "type": "table"}
                )
                add_to_collection([table_json], doc.metadata['filename'])
                all_chunks.append(table_chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {doc.metadata['filename']}: {str(e)}")

    return all_chunks

def format_docs(docs, sources):
    return "\n\n".join(f"Source: {source}\n{doc_text}" for doc_text, source in zip(docs, sources))

def save_to_directus(query: str, response: str):
    """Save conversation history to Directus."""
    payload = {
        "query": query,
        "response": response
    }
    print(f"Sending payload: {payload}")  # Debug payload
    try:
        response = requests.post(f"{DIRECTUS_URL}/items/conversation_history", json=payload, headers=HEADERS)
        response.raise_for_status()
        print(f"Directus response: {response.text}")  # Debug response
    except requests.exceptions.RequestException as e:
        print(f"Directus error: {str(e)}")  # Debug error
        print(f"Response text: {getattr(e.response, 'text', 'No response text')}")
        raise HTTPException(status_code=500, detail=f"Error saving to Directus: {str(e)}")
    

def get_conversation_history():
    """Retrieve conversation history from Directus, sorted by latest ID."""
    try:
        response = requests.get(f"{DIRECTUS_URL}/items/conversation_history?sort=-id&limit=4", headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json().get("data", [])
        print('/n')
        print("abcdefg")
        print('/n')
        print(f"Raw API response: {response.text}")  # Log the full response
        print(f"Retrieved history (order): {[entry['id'] for entry in data]}")  # Log IDs in order
        return data
    except requests.exceptions.RequestException as e:
        print("Warning: Could not connect to Directus. Proceeding without history.")
        return []

def process_query(user_question: str) -> str:
    """Process a user query with conversation history and return LLM response."""
    # Retrieve relevant documents
    docs = retrieve_from_collection(user_question, top_k=8)
    if not docs:
        return "No relevant information found in the provided PDFs."

    sources = [doc['metadata'].get('filename', 'Unknown') for doc in docs]
    doc_texts = [doc['document'] for doc in docs]
    context = format_docs(doc_texts, sources)

    # Retrieve conversation history
    history = get_conversation_history()
    history_str = "Previous Conversation History:\n" + "\n".join(
        f"User: {entry['query']}\nAssistant: {entry['response']}" for entry in history
    ) if history else "No previous history."

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set.")

    # Prepare LLM prompt with history
    model = ChatOpenAI(model="gpt-4", temperature=0.3)
    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant that helps users understand PDF documents. 
            The following information comes from these PDF files: {', '.join(set(sources))}
            **Conversation Memory:** Use this history for context:
            {history_str}
            **IMPORTANT: Response Format**  
            - If you find relevant information: **"Sources: [list of PDF filenames, comma-separated]"**  
            - If you don't find relevant information: **"No relevant information found in the provided PDFs."**  
            - After stating the sources, provide a detailed and structured answer.
            **Rules:**  
            - Use **only** the provided context and history.  
            - If not available: **"Answer is not available in the context."**  
            - Perform calculations if needed.  
            - Ask for clarification if unclear."""
        },
        {"role": "user", "content": f"Question: {user_question}\n\nInformation from PDFs:\n{context}"}
    ]
    response = model.invoke(messages).content

    # Save the new query-response pair to Directus
    save_to_directus(user_question, response)

    return response

# API Endpoints
@app.post("/upload-pdf")
async def upload_pdf(files: List[UploadFile]):
    """Upload PDF files and process them."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    pdf_bytes = [BytesIO(await file.read()) for file in files]
    for i, pdf in enumerate(pdf_bytes):
        pdf.name = files[i].filename
    
    try:
        documents = get_pdf_text(pdf_bytes)
        get_text_chunks(documents)
        return {"message": f"Processed {len(documents)} PDFs successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/process-url")
async def process_url(request: URLRequest):
    """Process PDFs from URLs."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")
    
    pdf_bytes = []
    for url in request.urls:
        pdf = download_pdf_from_url(url)
        if pdf:
            pdf_bytes.append(pdf)
    
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="No valid PDFs downloaded from URLs.")
    
    try:
        documents = get_pdf_text(pdf_bytes)
        get_text_chunks(documents)
        return {"message": f"Processed {len(documents)} PDFs from URLs successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URLs: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a user question based on stored PDFs with conversation history."""
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    try:
        response = process_query(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)