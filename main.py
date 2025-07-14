from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from MyFun import *
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import streamlit as st
import tempfile
import os
load_dotenv()

st.title("Interactive Document Chatbot")
uploaded_file = st.file_uploader("Upload a pdf", type=["pdf","docx","txt"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.session_state.vectorstore = InMemoryVectorStore(embedding_model())
    docs = load_docs(file_path = temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    st.session_state.vectorstore.add_documents(docs)


if "messages" not in st.session_state:
    st.session_state.messages = Messages()


llm = chat_model()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant for question-answering tasks.
        Use ONLY the following retrieved context to answer the question.
        If you don't know the answer from the provided context, just say that you don't know.
        Your answer should be concise and helpful.
        context:
        {context}."""),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])




# Display chat messages from history on app rerun
for message in st.session_state.messages.get_openai_format():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if "vectorstore" in st.session_state:
    question = st.chat_input("Say something")
    if question:


        chain = ( prompt
                | llm
                | StrOutputParser()
        )

        ai_response = chain.invoke({"question": question,"history": st.session_state.messages.get_messages(),"context":"\n\n".join(
                        doc.page_content for doc in
                        st.session_state.vectorstore.as_retriever(k=3).invoke(question)
                    )})
        st.session_state.messages.human_message(question)
        st.session_state.messages.ai_message(ai_response)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            st.markdown(ai_response)
            with st.expander("View Sources"):
                docs = st.session_state.vectorstore.as_retriever(k=3).invoke(question)

                for doc in docs:
                    st.markdown(doc.page_content.strip())
                    st.markdown("---")


