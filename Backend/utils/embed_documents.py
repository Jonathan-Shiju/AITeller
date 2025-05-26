from langchain_ollama import OllamaEmbeddings
import chromadb
import uuid
import os

def embed_and_save(docs, ollama_model, save_path):
    """
    Embeds documents and saves the ChromaDB vectorstore to disk.

    :param docs: List of strings (documents).
    :param ollama_model: Name of the Ollama model to use for embeddings.
    :param save_path: Path to save the ChromaDB collection.
    """
    embeddings = OllamaEmbeddings(model=ollama_model)

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=save_path)

    # Create collection with unique name
    collection_name = f"documents_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name)

    # Generate embeddings and add to collection
    ids = [f"doc_{i}" for i in range(len(docs))]
    doc_embeddings = [embeddings.embed_query(doc) for doc in docs]

    collection.add(
        embeddings=doc_embeddings,
        documents=docs,
        ids=ids
    )

    # Save collection name for later retrieval
    with open(os.path.join(save_path, "collection_name.txt"), "w") as f:
        f.write(collection_name)

    return collection_name

def load_vectorstore(save_path, ollama_model):
    """
    Load a previously saved ChromaDB vectorstore.

    :param save_path: Path where the ChromaDB collection is saved.
    :param ollama_model: Name of the Ollama model to use for embeddings.
    :return: ChromaDB collection and embeddings model.
    """
    # Read collection name
    with open(os.path.join(save_path, "collection_name.txt"), "r") as f:
        collection_name = f.read().strip()

    # Load persistent client
    client = chromadb.PersistentClient(path=save_path)
    collection = client.get_collection(name=collection_name)

    embeddings = OllamaEmbeddings(model=ollama_model)

    return collection, embeddings

# Example usage:
# docs = ["Document 1 text.", "Document 2 text."]
# collection_name = embed_and_save(docs, ollama_model="llama2", save_path="./chroma_db")
# collection, embeddings = load_vectorstore(save_path="./chroma_db", ollama_model="llama2")
