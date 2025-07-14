from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
import os
def load_docs(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()  # e.g., '.pdf'
    loader_class = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader
    }.get(ext)
    return loader_class(file_path).load()


def embedding_model():

    return HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")


def chat_model():
    return ChatGroq(
        model="gemma2-9b-it",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )


class Messages:
    def __init__(self):
        self.messages = []

    def human_message(self,message:str):
        self.add_message(HumanMessage(content=message))

    def ai_message(self,message:str):
        self.add_message(AIMessage(content=message))

    def add_message(self, message: AIMessage|HumanMessage):
        self.messages.append(message)

    def get_messages(self,*args):
        return self.messages[-5:]

    def get_openai_format(self):
        return convert_to_openai_messages(self.messages)
