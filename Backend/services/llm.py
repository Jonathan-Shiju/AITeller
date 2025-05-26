from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_faiss import FAISS
from langchain.chains.retrieval_qa import RetrievalQA
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.prompts import PromptTemplate
from Backend.tools.dummy import get_bank_account_info

def init_agent(
    ollama_model: str,
    rag_docs: list = None,
    tools: list = None,
):
    """
    Initializes a LangChain agent with Ollama, RAG, and tool access.
    """
    # Initialize Ollama LLM
    llm = OllamaLLM(model=ollama_model)

    # Setup RAG if documents are provided
    if rag_docs:
        embeddings = OllamaEmbeddings(model=ollama_model)
        vectorstore = FAISS.from_texts(rag_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        rag_tool = Tool(
            name="RAG_Retriever",
            func=qa_chain.run,
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

def initialize_conversation_manager(ollama_model="llama2", rag_docs=None, tools=None):
    """
    Initialize the global conversation manager.
    """
    global conversation_manager
    conversation_manager = ConversationManager(
        ollama_model=ollama_model,
        rag_docs=rag_docs,
        tools=tools
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
