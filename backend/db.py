# db.py
import chromadb
import uuid
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize ChromaDB client
client = chromadb.HttpClient(host="chroma", port=8000)

# Create or get the collection
# collection = client.get_or_create_collection("allminilm-test-1")
# client.delete_collection("allminilm-test-1")
collection = client.get_or_create_collection("deploy", metadata={"hnsw:space": "cosine", "embedding_dimension": 768})
# Load embedding model
embedding_model= HuggingFaceEmbeddings(
        model_name="/root/.cache/huggingface/models--nomic-ai--nomic-embed-text-v1/snapshots/720244025c1a7e15661a174c63cce63c8218e52b",
        model_kwargs={'trust_remote_code': True}
    )


def add_to_collection(text_chunks, filename):
    """Add text chunks and their embeddings to the ChromaDB collection."""
    try:
        # remove_existing_file_entries(filename)  # Ensure duplicates are not added

        # Generate embeddings
        embeddings = embedding_model.embed_documents(text_chunks)  # Fix: Use embed_documents()

        # Prepare metadata with the filename
        metadatas = [{"filename": filename} for _ in range(len(text_chunks))]

        # Add data to ChromaDB collection
        collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(text_chunks))],  # Unique IDs for each chunk
            documents=text_chunks,  # Searchable text content
            embeddings=embeddings,  # Fix: embeddings is already a list, no need to convert
            metadatas=metadatas  # Metadata with filename
        )

        print(f"Added {len(text_chunks)} chunks from {filename} to the collection.")
    except Exception as e:
        print(f"Error adding chunks to collection: {str(e)}")
        raise

def retrieve_from_collection(query, top_k=8):
    """Retrieve the most relevant text chunks from ChromaDB based on the query."""
    try:
        # Check if the query contains a filename filter
        filename_filter = None
        if "from " in query.lower() and ".pdf" in query.lower():
            # Extract the filename from queries like "from QUIZ 2017-18.pdf, tell me..."
            query_parts = query.split(".pdf")
            from_parts = query_parts[0].split("from ")
            print("working till here..............................")
            if len(from_parts) > 1:
                filename_filter = from_parts[-1].strip() + ".pdf"
                # Remove the filename part from the query
                query = query_parts[1].strip().lstrip(',').strip()
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(filename_filter)
        # Convert query to embedding
        query_embedding = embedding_model.embed_query(query)

        # Prepare filter if a filename was specified
        where_filter = None
        if filename_filter:
            where_filter = {"filename": filename_filter}
        
        # Query ChromaDB using the computed embedding with optional filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
            where=where_filter  # Apply filter if a filename was specified
        )

        # print(123, results)        

        # Format results
        if results['documents'] and results['metadatas']:
            return [
                {'document': doc, 'metadata': meta}
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ]
        return []
        
    except Exception as e:
        print(f"Error retrieving from collection: {str(e)}")
        return []
