# working with user conversation history display 20/03

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
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import requests
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.getenv("OPENAI_API_KEY")

# FastAPI App Initialization and Middleware
# Initialize FastAPI app
app = FastAPI(title="PDF RAG Backend")

# cors configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Match React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.options("/query")
async def options_query():
    return {"message": "CORS preflight handled"}

# Authentication with Directus

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        response = requests.get(
            "http://directus:8055/users/me",  # Use 'directus' if in Docker network
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_data = response.json().get("data")
        user_data["access_token"] = token
        return user_data
    except Exception as e:
        logger.error(f"Error with token: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Pydantic models for request/response validation
class URLRequest(BaseModel):
    urls: List[str]

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

# Directus configuration 
DIRECTUS_URL = "http://directus:8055"  # Directus instance URL

# Helper functions extract text, handle url, chunking and source name
def download_pdf_from_url(url: str) -> BytesIO:
    """Download a PDF from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        pdf = BytesIO(response.content)
        pdf.name = url.split("/")[-1]
        return pdf
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Error downloading PDF from {url}: {str(e)}")

def get_pdf_text(pdf_files: List[BytesIO]) -> List[Document]:
    """Extract text and tables from PDFs."""
    documents = []
    for pdf in pdf_files:
        temp_path = f"temp_{pdf.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(pdf.read())
        except Exception as e:
            logger.error(f"Error writing temporary file {temp_path}: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error writing temporary file: {str(e)}")

        # Extract text with PyPDF2
        text_data = {"text": []}
        try:
            pdf_reader = PdfReader(temp_path)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text().strip()
                if page_text:
                    text_data["text"].append({"page": page_num + 1, "content": page_text})
        except Exception as e:
            logger.error(f"Error extracting text from {pdf.name}: {str(e)}\n{traceback.format_exc()}")
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
            # if not table_data["tables"]:
            #     tables = camelot.read_pdf(temp_path, pages='all', flavor='stream')
            #     for i, table in enumerate(tables):
            #         if table.df.size > 0:
            #             table_data["tables"].append({
            #                 "table_id": f"Table {i + 1}",
            #                 "page": table.parsing_report['page'],
            #                 "data": table.df.replace({None: ""}).to_dict(orient='records')
            #             })
        except Exception as e:
            logger.warning(f"Failed to extract tables from {pdf.name}, proceeding without tables: {str(e)}\n{traceback.format_exc()}")

        combined_data = {
            "source": pdf.name,
            "text": text_data["text"],
            "tables": table_data["tables"]
        }
        json_content = json.dumps(combined_data, ensure_ascii=False, indent=2)

        doc = Document(page_content=json_content, metadata={"filename": pdf.name})
        documents.append(doc)

        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}\n{traceback.format_exc()}")

    return documents

def get_text_chunks(documents: List[Document], user_id: str) -> List[Document]:
    """Chunk documents and add to ChromaDB with user_id."""
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
                    add_to_collection(chunk_texts, doc.metadata['filename'], user_id)
                    all_chunks.extend(text_chunks)

            # Process tables
            for table in table_entries:
                table_json = json.dumps(table, ensure_ascii=False)
                table_chunk = Document(
                    page_content=table_json,
                    metadata={"source": doc.metadata["filename"], "type": "table"}
                )
                add_to_collection([table_json], doc.metadata['filename'], user_id)
                all_chunks.append(table_chunk)
        except Exception as e:
            logger.error(f"Error processing {doc.metadata['filename']}: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error processing {doc.metadata['filename']}: {str(e)}")

    return all_chunks

def format_docs(docs, sources):
    return "\n\n".join(f"Source: {source}\n{doc_text}" for doc_text, source in zip(docs, sources))

# Saving the query and response to Directus.
def save_to_directus(query: str, response: str, user_id: str, conversation_id: int = None):
    """Save conversation history to Directus with an incremental conversation_id and user_id."""
    # Ensure response is a string
    if not isinstance(response, str):
        print(f"Error: Response is not a string, received: {response}")
        response = str(response)  # Convert to string as a fallback

    # Determine the next conversation_id
    if conversation_id is None:
        try:
            response_get = requests.get(f"{DIRECTUS_URL}/items/conversation_history?sort=-conversation_id&limit=1", headers=HEADERS, timeout=5)
            response_get.raise_for_status()
            data = response_get.json().get("data", [])
            last_conversation_id = max((entry["conversation_id"] or 0) for entry in data)  # Handle null values
            conversation_id = last_conversation_id + 1
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error fetching conversation ID: {str(e)}")
            conversation_id = 1  # Start with 1 if no history or error

    payload = {
        "query": query,
        "response": response,
        "conversation_id": conversation_id,
        "user_created": user_id  # Associate the conversation with the user
    }
    print(f"Sending payload: {payload}")  # Debug payload
    try:
        response_post = requests.post(f"{DIRECTUS_URL}/items/conversation_history", json=payload, headers=HEADERS)
        response_post.raise_for_status()
        print(f"Directus response: {response_post.text}")  # Debug response
    except requests.exceptions.RequestException as e:
        print(f"Directus error: {str(e)}")  # Debug error
        print(f"Response text: {getattr(e.response, 'text', 'No response text')}")
        raise HTTPException(status_code=500, detail=f"Error saving to Directus: {str(e)}")
    
# Retrieves the last 5 conversation entries from Directus.
def get_conversation_history(user_id: str):
    """Retrieve the last 5 conversations for a specific user from Directus, sorted by latest conversation_id."""
    try:
        response = requests.get(
            f"{DIRECTUS_URL}/items/conversation_history",
            params={
                "filter[user_created][_eq]": user_id,  # Filter by user
                "sort": "-conversation_id",  # Newest first
                "limit": 5,  # Last 5 conversations
            },
            headers=HEADERS,
            timeout=5
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        print(f"Raw API response for user {user_id}: {response.text}")  # Log the full response
        print(f"Retrieved history for user {user_id} (order): {[entry['conversation_id'] for entry in data]}")  # Log conversation_ids in order
        return data
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to Directus for user {user_id}. Proceeding without history: {str(e)}")
        return []

def get_user_conversation_history(user_id: str, limit: int = 50, offset: int = 0):
    try:
        response = requests.get(
            f"{DIRECTUS_URL}/items/conversation_history",
            params={
                "filter[user_created][_eq]": user_id,
                "sort": "conversation_id",  # Oldest first
                "limit": limit,
                "offset": offset,
            },
            headers=HEADERS,
            timeout=5
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        print(f"Retrieved user history for {user_id}: {[entry['conversation_id'] for entry in data]}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation history: {str(e)}")
    

def process_query(user_question: str, user_id: str) -> str:
    """Process a user query with conversation history and return LLM response."""
    # Retrieve relevant documents
    docs = retrieve_from_collection(user_question,user_id, top_k=8)
    
    # Prepare context and sources
    if not docs:
        context = "No PDF content available."
        sources = ["None"]
    else:
        sources = [doc['metadata'].get('filename', 'Unknown') for doc in docs]
        doc_texts = [doc['document'] for doc in docs]
        context = format_docs(doc_texts, sources)

    # Retrieve conversation history for the specific user
    history = get_conversation_history(user_id)
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
            "content": f"""You are a friendly AI assistant that helps users understand PDF documents and engage in general conversation. 
            The following information comes from these PDF files: {', '.join(set(sources))}
            **Conversation Memory:** Use this history for context:
            {history_str}
            **Behavior:**
            - If the user greets you (e.g., says "hello", "hi", "hey"), respond with a friendly greeting like "Hi, how can I assist you?" or "Hello! How can I help you today?"
            - If the user asks a question and relevant PDF content is available, respond with: **"Sources: [list of PDF filenames, comma-separated]"** followed by a detailed and structured answer based on the PDF content.
            - If the user asks a question but no relevant PDF content is available, respond with: **"No relevant information found in the provided PDFs."** and offer to help with something else (e.g., "Can I assist you with something else?").
            **Rules:**  
            - Use the provided context and history to answer questions about PDFs.  
            - If the answer is not available in the context, say: **"Answer is not available in the context."**  
            - Perform calculations if needed.  
            - Ask for clarification if the user's query is unclear.
            - Be friendly and conversational in your responses."""
        },
        {"role": "user", "content": f"Question: {user_question}\n\nInformation from PDFs:\n{context}"}
    ]
    response = model.invoke(messages).content

    # Save the new query-response pair to Directus
    try:
        save_to_directus(user_question, response, user_id)
    except Exception as e:
        print(f"Failed to save to Directus: {str(e)}")
    return response

# API Endpoints
@app.post("/upload-pdf")
async def upload_pdf(files: List[UploadFile], user: dict = Depends(get_current_user)):
    """Upload PDF files, process them, and save metadata to Directus."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token.")

    pdf_bytes = [BytesIO(await file.read()) for file in files]
    for i, pdf in enumerate(pdf_bytes):
        pdf.name = files[i].filename
    
    try:
        documents = get_pdf_text(pdf_bytes)
        get_text_chunks(documents, user_id)
        # for doc in documents:
        #     save_pdf_metadata_to_directus(doc.metadata['filename'], user_id)
        return {"message": f"Processed {len(documents)} PDFs successfully."}
    except Exception as e:
        logger.error(f"Error in /upload-pdf endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")
    
@app.post("/process-url")
async def process_url(request: URLRequest, user: dict = Depends(get_current_user)):
    """Process PDFs from URLs."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")
    
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token.")

    pdf_bytes = []
    for url in request.urls:
        pdf = download_pdf_from_url(url)
        if pdf:
            pdf_bytes.append(pdf)
    
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="No valid PDFs downloaded from URLs.")
    
    try:
        documents = get_pdf_text(pdf_bytes)
        get_text_chunks(documents, user_id)  # Pass user_id
        return {"message": f"Processed {len(documents)} PDFs from URLs successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URLs: {str(e)}")

# Update the /query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user: dict = Depends(get_current_user)):
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided.")
    try:
        global HEADERS  # Avoid redefining globally; pass token per request
        HEADERS = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {user.get('access_token', 'default-token')}"  # Use token from Directus auth
        }
        print(f"Setting HEADERS in /query: {HEADERS}")
        user_id = user.get("id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in token.")
        response = process_query(request.question, user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
@app.get("/conversation-history")
async def fetch_conversation_history(limit: int = 50, offset: int = 0,user: dict = Depends(get_current_user)):
    """Fetch the conversation history for the authenticated user."""
    try:
        global HEADERS
        HEADERS = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {user.get('access_token', 'default-token')}"
        }
        user_id = user.get("id")
        print("got the userid now passing to---------------------------------------------------------------------------")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in token.")
        history = get_user_conversation_history(user_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversation history: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)