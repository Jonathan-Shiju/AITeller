from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from Backend.tools.dummy import get_bank_account_info

def init_agent(
    ollama_model: str,
    rag_docs: list = None,
    tools: list = None,
    agent_type: str = "zero-shot-react-description"
):
    """
    Initializes a LangChain agent with Ollama, RAG, and tool access.

    :param ollama_model: Name of the Ollama model to use.
    :param rag_docs: List of documents for retrieval-augmented generation (RAG).
    :param tools: List of Tool objects for agent tool access.
    :param agent_type: Type of agent to initialize.
    :return: Initialized agent.
    """
    # Initialize Ollama LLM
    llm = Ollama(model=ollama_model)

    # Setup RAG if documents are provided
    retriever = None
    if rag_docs:
        embeddings = OllamaEmbeddings(model=ollama_model)
        vectorstore = FAISS.from_texts(rag_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        rag_tool = Tool(
            name="RAG Retriever",
            func=qa_chain.run,
            description="Useful for answering questions based on provided documents."
        )
        if tools is None:
            tools = []
        tools.append(rag_tool)

    # Add bank info tool
    bank_tool = Tool(
        name="Bank Account Lookup",
        func=get_bank_account_info,
        description="Retrieve bank account details by account number or name."
    )
    if tools is None:
        tools = []
    tools.append(bank_tool)

    # Initialize agent with tools
    agent = initialize_agent(
        tools or [],
        llm,
        agent=agent_type,
        verbose=True
    )
    return agent

class ConversationManager:
    """
    Manages conversation history and agent interaction for few-shot prompting.
    """
    def __init__(self, ollama_model, rag_docs=None, tools=None, agent_type="zero-shot-react-description"):
        self.history = []
        self.agent = init_agent(
            ollama_model=ollama_model,
            rag_docs=rag_docs,
            tools=tools,
            agent_type=agent_type
        )

    def conversational_agent(self, prompt: str, use_history: bool = True):
        """
        Handles a prompt, uses tools/RAG if needed, and manages few-shot context.
        :param prompt: User's prompt.
        :param use_history: Whether to include previous prompts for few-shot context.
        :return: Agent's response.
        """
        if use_history and self.history:
            # Combine previous prompts and current prompt for few-shot context
            conversation = "\n".join(self.history + [prompt])
        else:
            conversation = prompt

        response = self.agent.run(conversation)
        self.history.append(prompt)
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