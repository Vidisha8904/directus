# db.py
import chromadb
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB client
client = chromadb.HttpClient(host="chroma", port=8000)

# Create or get the collection
collection = client.get_or_create_collection("deploy", metadata={"hnsw:space": "cosine", "embedding_dimension": 768})

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="/root/.cache/huggingface/models--nomic-ai--nomic-embed-text-v1/snapshots/720244025c1a7e15661a174c63cce63c8218e52b",
    model_kwargs={'trust_remote_code': True}
)

def add_to_collection(text_chunks, filename, user_id):
    """Add text chunks and their embeddings to the ChromaDB collection with user_id."""
    try:
        # Generate embeddings
        embeddings = embedding_model.embed_documents(text_chunks)

        # Prepare metadata with the filename and user_id
        metadatas = [{"filename": filename, "user_id": user_id} for _ in range(len(text_chunks))]

        # Add data to ChromaDB collection
        collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(text_chunks))],
            documents=text_chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Added {len(text_chunks)} chunks from {filename} by user {user_id} to the collection.")
    except Exception as e:
        logger.error(f"Error adding chunks to collection: {str(e)}\n{traceback.format_exc()}")
        raise

def retrieve_from_collection(query, user_id, top_k=8):
    """Retrieve the most relevant text chunks from ChromaDB based on the query, filtered by user_id."""
    try:
        filename_filter = None
        if "from " in query.lower() and ".pdf" in query.lower():
            query_parts = query.split(".pdf")
            from_parts = query_parts[0].split("from ")
            if len(from_parts) > 1:
                filename_filter = from_parts[-1].strip() + ".pdf"
                query = query_parts[1].strip().lstrip(',').strip()
                logger.info(f"Extracted filename filter: {filename_filter}")

        query_embedding = embedding_model.embed_query(query)

        where_filter = {"user_id": user_id}
        if filename_filter:
            where_filter["filename"] = filename_filter

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
            where=where_filter
        )

        if results['documents'] and results['metadatas']:
            return [
                {'document': doc, 'metadata': meta}
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ]
        return []
    except Exception as e:
        logger.error(f"Error retrieving from collection: {str(e)}\n{traceback.format_exc()}")
        return []