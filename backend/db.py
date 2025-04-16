# # db.py
# import chromadb
# import uuid
# from langchain_huggingface import HuggingFaceEmbeddings
# import logging
# import traceback

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize ChromaDB client
# client = chromadb.HttpClient(host="3.1.84.212", port=8000)

# # Create or get the collection
# collection = client.get_or_create_collection("deploy", metadata={"hnsw:space": "cosine", "embedding_dimension": 768})

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="/root/.cache/huggingface/models--nomic-ai--nomic-embed-text-v1/snapshots/720244025c1a7e15661a174c63cce63c8218e52b",
#     model_kwargs={'trust_remote_code': True}
# )

# def add_to_collection(text_chunks, filename, user_id):
#     """Add text chunks and their embeddings to the ChromaDB collection with user_id."""
#     try:
#         # Generate embeddings
#         embeddings = embedding_model.embed_documents(text_chunks)

#         # Prepare metadata with the filename and user_id
#         metadatas = [{"filename": filename, "user_id": user_id} for _ in range(len(text_chunks))]

#         # Add data to ChromaDB collection
#         collection.add(
#             ids=[str(uuid.uuid4()) for _ in range(len(text_chunks))],
#             documents=text_chunks,
#             embeddings=embeddings,
#             metadatas=metadatas
#         )

#         logger.info(f"Added {len(text_chunks)} chunks from {filename} by user {user_id} to the collection.")
#     except Exception as e:
#         logger.error(f"Error adding chunks to collection: {str(e)}\n{traceback.format_exc()}")
#         raise

# def retrieve_from_collection(query, user_id, top_k=8):
#     """Retrieve the most relevant text chunks from ChromaDB based on the query, filtered by user_id."""
#     try:
#         filename_filter = None
#         logger.info(f"Querying ChromaDB with query: {query}, user_id: {user_id}, top_k: {top_k}")
#         if "from " in query.lower() and ".pdf" in query.lower():
#             query_parts = query.split(".pdf")
#             from_parts = query_parts[0].split("from ")
#             if len(from_parts) > 1:
#                 filename_filter = from_parts[-1].strip() + ".pdf"
#                 query = query_parts[1].strip().lstrip(',').strip()
#                 logger.info(f"Extracted filename filter: {filename_filter}")

#         query_embedding = embedding_model.embed_query(query)

#         where_filter = {"user_id": user_id}
#         logger.info(f"Using filter: {where_filter}")
#         if filename_filter:
#             where_filter["filename"] = filename_filter

#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#             include=['documents', 'metadatas', 'distances'],
#             where=where_filter
#         )
#         logger.info(f"Query results: {results}")
#         if results['documents'] and results['metadatas']:
#             return [
#                 {'document': doc, 'metadata': meta}
#                 for doc, meta in zip(results['documents'][0], results['metadatas'][0])
#             ]
#         logger.info("No documents found.")
#         return []
#     except Exception as e:
#         logger.error(f"Error retrieving from collection: {str(e)}\n{traceback.format_exc()}")
#         raise

import chromadb
import uuid
import logging
import traceback
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings  # OpenAI embeddings
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB client
client = chromadb.HttpClient(host="chroma-vidisha-lb-1390783373.ap-southeast-1.elb.amazonaws.com", port=8000)
# client = chromadb.HttpClient(host="chroma", port=8000)

# Create or get the collection

# collection = client.get_or_create_collection("deploy3", metadata={"hnsw:space": "cosine", "embedding_dimension": 1536})  # OpenAI embeddings have 1536 dimensions
# collection = client.get_collection(name="test-1")
collection= client.get_collection(name='nomic')

# Load OpenAI embedding model
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",  # Change to "text-embedding-ada-002" if needed
#     openai_api_key= OPENAI_API_KEY  # Replace with your actual API key
# # )
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="/root/.cache/huggingface/models--nomic-ai--nomic-embed-text-v1/snapshots/720244025c1a7e15661a174c63cce63c8218e52b",
        model_kwargs={'trust_remote_code': True}
    )
except Exception as e:
    logger.error(f"Error loading model {str(e)}\n{traceback.format_exc()}")
    raise


def add_to_collection(text_chunks, filename, user_id):
    """Add text chunks and their embeddings to the ChromaDB collection with user_id."""
    try:
        # Generate embeddings using OpenAI
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
    logger.info(f"Querying ChromaDB with query: {query}, user_id: {user_id}, top_k: {top_k}")
    try:
        filename_filter = None
        logger.info(f"Querying ChromaDB with query: {query}, user_id: {user_id}, top_k: {top_k}")

        if "from " in query.lower() and ".pdf" in query.lower():
            query_parts = query.split(".pdf")
            from_parts = query_parts[0].split("from ")
            if len(from_parts) > 1:
                filename_filter = from_parts[-1].strip() + ".pdf"
                query = query_parts[1].strip().lstrip(',').strip()
                logger.info(f"Extracted filename filter: {filename_filter}")

        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)

        where_filter = {"user_id": user_id}
        logger.info(f"Using filter: {where_filter}")

        if filename_filter:
            where_filter["filename"] = filename_filter

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
            where=where_filter
        )

        logger.info(f"Query results: {results}")

        if results['documents'] and results['metadatas']:
            return [
                {'document': doc, 'metadata': meta}
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ]

        logger.info("No documents found.")
        return []
    except Exception as e:
        logger.error(f"Error retrieving from collection: {str(e)}\n{traceback.format_exc()}")
        raise

def overwriting_chromadb(user_id: str, filename: str):
    """for overwriting"""
    """Delete all chunks associated with a specific filename and user_id from ChromaDB."""
    try:
        # Query to find all chunks with matching user_id and filename
        where_filter = {
            "$and": [
                {"user_id": user_id},
                {"filename": filename}
            ]
        }
        results = collection.get(
            where=where_filter,
            include=["metadatas"]  # We only need metadata to get the IDs
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} existing chunks for {filename} for user {user_id}")
        else:
            logger.info(f"No existing chunks found for {filename} for user {user_id}")
    except Exception as e:
        logger.error(f"Error deleting chunks for {filename}: {str(e)}\n{traceback.format_exc()}")
        raise

def delete_pdf_from_collection(user_id: str, filename: str):
    """Delete all chunks, embeddings, and metadata for a specific PDF from ChromaDB."""
    try:
        where_filter = {
            "$and": [
                {"user_id": user_id},
                {"filename": filename}
            ]
        }
        results = collection.get(
            where=where_filter,
            include=["metadatas"]
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for {filename} for user {user_id} from ChromaDB")
        else:
            logger.info(f"No chunks found to delete for {filename} for user {user_id}")
    except Exception as e:
        logger.error(f"Error deleting PDF {filename} from ChromaDB: {str(e)}\n{traceback.format_exc()}")
        raise