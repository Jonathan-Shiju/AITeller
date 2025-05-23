from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

def embed_and_save_dummy_doc(doc_path, ollama_model, save_path):
    with open(doc_path, "r") as f:
        doc = f.read()
    embeddings = OllamaEmbeddings(model=ollama_model)
    vectorstore = FAISS.from_texts([doc], embedding=embeddings)
    vectorstore.save_local(save_path)

if __name__ == "__main__":
    embed_and_save_dummy_doc(
        doc_path="./dummy_rag_doc.txt",
        ollama_model="llama2",
        save_path="./faiss_dummy_rag"
    )
