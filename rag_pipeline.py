import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
conversation_chain = None

def clean_response(text: str) -> str:
    # Remove Markdown bold **text**
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove markdown headers ###
    text = re.sub(r'###\s*', '', text)
    # Replace <br> and <br> with newline
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Replace \n with actual newlines
    text = text.replace('\\n', '\n')
    # Remove extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def initialize_rag():
    global conversation_chain

    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("api key not set in .env file")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="vector_database",
        embedding_function=embedding
    )

    retriever = vector_store.as_retriever()

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.4,
        api_key=api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )


def ask_question(question: str):
    q = question.lower().strip()

    # Handle greetings
    if q in ["hi", "hello", "hey"]:
        return "Hello! How can I help you with medicines today?"

    # Handle empty input
    if not q:
        return "Please ask a question."
    if q in ["thanks" , "thank you"]:
        return "You're welcome!"
    result = conversation_chain.invoke({"question": question})
    return clean_response(result["answer"])
