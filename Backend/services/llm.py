from langchain_ollama import OllamaLLM, OllamaEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate
from Backend.tools.dummy import get_bank_account_info
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from Backend.utils.embed_documents import load_vectorstore
import os

class ChromaVectorStore:
    """Vector store using ChromaDB"""
    def __init__(self, texts, embeddings_model, collection_name=None):
        self.texts = texts
        self.embeddings_model = embeddings_model

        # Initialize ChromaDB client
        self.client = chromadb.Client()

        # Use unique collection name if not provided
        if collection_name is None:
            collection_name = f"documents_{uuid.uuid4().hex[:8]}"

        # Delete existing collection if it exists, then create new one
        try:
            self.client.delete_collection(name=collection_name)
        except:
            pass

        self.collection = self.client.create_collection(name=collection_name)

        # Add documents to collection
        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = [embeddings_model.embed_query(text) for text in texts]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids
        )

    def similarity_search(self, query, k=4):
        query_embedding = self.embeddings_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results['documents'][0] if results['documents'] else []

class PersistentChromaVectorStore:
    """Persistent vector store using ChromaDB"""
    def __init__(self, texts=None, embeddings_model=None, collection_name=None, persistent_path=None):
        self.embeddings_model = embeddings_model

        if persistent_path and os.path.exists(persistent_path):
            # Load existing persistent store
            self.client = chromadb.PersistentClient(path=persistent_path)
            if not collection_name:
                # Read collection name from file
                with open(os.path.join(persistent_path, "collection_name.txt"), "r") as f:
                    collection_name = f.read().strip()
            self.collection = self.client.get_collection(name=collection_name)

        else:
            # Create new persistent store
            if persistent_path:
                os.makedirs(persistent_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persistent_path)
            else:
                self.client = chromadb.Client()

            if collection_name is None:
                collection_name = f"documents_{uuid.uuid4().hex[:8]}"

            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass

            self.collection = self.client.create_collection(name=collection_name)

            # Add documents if provided
            if texts:
                ids = [f"doc_{i}" for i in range(len(texts))]
                embeddings = [embeddings_model.embed_query(text) for text in texts]

                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    ids=ids
                )

            # Save collection name if persistent
            if persistent_path:
                with open(os.path.join(persistent_path, "collection_name.txt"), "w") as f:
                    f.write(collection_name)

    def similarity_search(self, query, k=4):
        query_embedding = self.embeddings_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results['documents'][0] if results['documents'] else []

def simple_rag_retriever(vectorstore, llm):
    """Simple RAG implementation with ChromaDB"""
    def rag_func(query):
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(query, k=3)

        # Combine documents into context
        context = "\n".join(docs)

        # Create prompt with context
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

        # Get response from LLM
        response = llm.invoke(prompt)
        return response

    return rag_func

def advanced_rag_retriever(vectorstore, llm):
    """Advanced RAG implementation with better retrieval"""
    def rag_func(query):
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(query, k=5)

        # Re-rank documents by relevance (simple scoring)
        scored_docs = []
        for doc in docs:
            # Simple keyword matching score
            score = sum(1 for word in query.lower().split() if word in doc.lower())
            scored_docs.append((doc, score))

        # Sort by score and take top 3
        top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:3]
        context = "\n".join([doc[0] for doc in top_docs])

        # Enhanced prompt with instructions
        prompt = f"""You are a helpful assistant. Use the provided context to answer the question accurately.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Be concise but complete

Answer:"""

        response = llm.invoke(prompt)
        return response

    return rag_func

def multi_query_rag_retriever(vectorstore, llm):
    """Multi-query RAG that generates multiple search queries"""
    def rag_func(query):
        # Generate alternative queries
        query_gen_prompt = f"""Generate 2 alternative ways to ask this question:
Original: {query}

Alternative 1:
Alternative 2:"""

        alternatives = llm.invoke(query_gen_prompt)
        queries = [query] + [alt.strip() for alt in alternatives.split('\n') if alt.strip()]

        # Retrieve docs for all queries
        all_docs = set()
        for q in queries[:3]:  # Limit to avoid too many calls
            docs = vectorstore.similarity_search(q, k=3)
            all_docs.update(docs)

        context = "\n".join(list(all_docs)[:5])  # Limit context size

        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

        response = llm.invoke(prompt)
        return response

    return rag_func

def chunked_rag_setup(texts, embeddings_model, persistent_path=None):
    """Better document chunking for RAG with optional persistence"""
    # Split texts into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    # Split all texts
    chunks = []
    for text in texts:
        splits = text_splitter.split_text(text)
        chunks.extend(splits)

    # Create vector store with chunks (persistent if path provided)
    if persistent_path:
        vectorstore = PersistentChromaVectorStore(
            texts=chunks,
            embeddings_model=embeddings_model,
            persistent_path=persistent_path
        )
    else:
        vectorstore = ChromaVectorStore(chunks, embeddings_model)

    return vectorstore

def load_persistent_rag(persistent_path, embeddings_model):
    """Load a persistent RAG vectorstore"""
    return PersistentChromaVectorStore(
        embeddings_model=embeddings_model,
        persistent_path=persistent_path
    )

def init_agent(
    ollama_model: str,
    rag_docs: list = None,
    tools: list = None,
    persistent_rag_path: str = None,
):
    """
    Initializes a LangChain agent with Ollama, RAG, and tool access.
    """
    # Initialize Ollama LLM
    llm = OllamaLLM(model=ollama_model)

    # Setup RAG if documents are provided OR persistent path exists
    if rag_docs or (persistent_rag_path and os.path.exists(persistent_rag_path)):
        embeddings = OllamaEmbeddings(model=ollama_model)

        if persistent_rag_path and os.path.exists(persistent_rag_path):
            # Load from persistent storage
            vectorstore = load_persistent_rag(persistent_rag_path, embeddings)
        else:
            # Create new with chunked setup (with persistence if path provided)
            vectorstore = chunked_rag_setup(rag_docs, embeddings, persistent_rag_path)

        # Choose retrieval method (uncomment the one you prefer)
        rag_retriever = advanced_rag_retriever(vectorstore, llm)
        # rag_retriever = multi_query_rag_retriever(vectorstore, llm)
        # rag_retriever = simple_rag_retriever(vectorstore, llm)  # Original

        rag_tool = Tool(
            name="RAG_Retriever",
            func=rag_retriever,
            description="Useful for answering questions based on provided documents."
        )
        if tools is None:
            tools = []
        tools.append(rag_tool)

    # Add bank info tool
    bank_tool = Tool(
        name="Bank_Account_Lookup",
        func=get_bank_account_info,
        description="Retrieve bank account details by account number or name."
    )
    if tools is None:
        tools = []
    tools.append(bank_tool)

    # Get a react prompt from hub
    prompt = hub.pull("hwchase17/react")

    # Create agent
    agent = create_react_agent(llm, tools or [], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools or [], verbose=True)

    return agent_executor

class ConversationManager:
    """
    Manages conversation history and agent interaction for few-shot prompting.
    """
    def __init__(self, ollama_model, rag_docs=None, tools=None, agent_type="zero-shot-react-description", persistent_rag_path=None):
        self.history = []
        self.agent = init_agent(
            ollama_model=ollama_model,
            rag_docs=rag_docs,
            tools=tools,
            persistent_rag_path=persistent_rag_path
        )

    def conversational_agent(self, prompt: str, use_history: bool = True):
        """
        Handles a prompt, letting the agent decide if the prompt is complete enough to respond.
        The agent will use tools/RAG if needed, and manages few-shot context.

        :param prompt: User's prompt.
        :param use_history: Whether to include previous prompts for few-shot context.
        :return: Agent's response or None if agent decides not to respond.
        """
        if use_history and self.history:
            # Combine previous prompts and current prompt for few-shot context
            conversation = "\n".join(self.history + [prompt])
        else:
            conversation = prompt

        # Add instructions for the agent to evaluate if the prompt is complete
        evaluation_prompt = (
            f"{conversation}\n\n"
            "Instructions: First determine if the above prompt is complete enough to respond to. "
            "If the prompt seems incomplete, unclear, or requires additional context that isn't available, "
            "respond with '<<INCOMPLETE_PROMPT>>' and nothing else. Otherwise, respond normally,"
            "If the quesion seems complete, but for requires additional context, "
            "please ask for the same.\n\n"
            "Respond with the most relevant information or use tools/RAG if necessary."
           )

        response = self.agent.run(evaluation_prompt)

        # Only save prompt to history if we're responding to it
        if response.strip() != "<<INCOMPLETE_PROMPT>>":
            self.history.append(prompt)
            return response
        else:
            return None

# Global conversation manager instance
conversation_manager = None

def initialize_conversation_manager(ollama_model="llama2", rag_docs=None, tools=None, persistent_rag_path=None):
    """
    Initialize the global conversation manager.
    """
    global conversation_manager
    conversation_manager = ConversationManager(
        ollama_model=ollama_model,
        rag_docs=rag_docs,
        tools=tools,
        persistent_rag_path=persistent_rag_path
    )

def generate_reply(prompt):
    """
    Generate a reply using the conversation manager.
    This function is used by the Twilio VOIP service.

    :param prompt: User's input prompt
    :return: Generated response text
    """
    global conversation_manager

    # Initialize conversation manager if not already done
    if conversation_manager is None:
        initialize_conversation_manager()

    response = conversation_manager.conversational_agent(prompt)

    # If the agent thinks the prompt is incomplete, return a default response
    if response is None:
        return "I didn't quite understand that. Could you please provide more details?"

    return response

# Example usage:
# cm = ConversationManager(ollama_model="llama2")
# print(cm.conversational_agent("What is the balance for Alice Johnson?"))
# print(cm.conversational_agent("And what about Bob Smith?"))
# Example usage:
# agent = init_agent(
#     ollama_model="llama2",
#     rag_docs=["LangChain is a framework for developing applications powered by language models."],
#     tools=[Tool(name="Calculator", func=lambda x: eval(x), description="Performs calculations.")]
# )
