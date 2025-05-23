from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

def embed_and_save(docs, ollama_model, save_path):
    """
    Embeds documents and saves the FAISS vectorstore to disk.

    :param docs: List of strings (documents).
    :param ollama_model: Name of the Ollama model to use for embeddings.
    :param save_path: Path to save the FAISS index.
    """
    embeddings = OllamaEmbeddings(model=ollama_model)
    vectorstore = FAISS.from_texts(docs, embedding=embeddings)
    vectorstore.save_local(save_path)

# Example usage:
# docs = ["Document 1 text.", "Document 2 text."]
# embed_and_save(docs, ollama_model="llama2", save_path="./faiss_index")
