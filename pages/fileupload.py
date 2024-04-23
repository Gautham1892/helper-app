import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import json
from streamlit_lottie import st_lottie


os.environ["GOOGLE_API_KEY"] = "AIzaSyB8kszduNytZVO_u3oXsnxOvDjTZSNQLuo"

load_dotenv()
google_api_key = os.getenv("AIzaSyB8kszduNytZVO_u3oXsnxOvDjTZSNQLuo")
genai.configure(api_key=google_api_key)


def load_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

config = load_config()

@st.cache_data()
def lottie_local(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_pdf_text_upload(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Your task is to answer questions (<Question>...</Question>) related to disaster data (<Context>...</Context>).
    Follow these guidelines:
    
    - If the question can be answered using the given context, provide a clear and concise answer, referencing relevant details from the context.
    - If the question asks for the definition or meaning of a term related to disasters, provide an appropriate explanation.
    - If the question cannot be answered using the given context alone, state that you don't have enough information.
    - If the question is unrelated to disasters, state it's outside the scope of the provided context.
    
    <Context> \n {context}\n </Context>
    <Question> \n{question}\n</Question>

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, temperature=0.3,
                                   safety_settings={
                                       HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                       HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                       HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                       HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                                   })

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question, k=3)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Disaster Data GPT", page_icon=":fire:")
    st.header("Disaster Data GPT :fire_engine:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm Disaster Data GPT. Ask me anything related to disaster data.")]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        anim = lottie_local('Animation - 1713708278778.json')
        st_lottie(anim,
                  speed=1,
                  reverse=False,
                  loop=True,
                  height=130,
                  width=250,
                  quality="high",
                  key=None)
        
        option = st.radio(" ", ["Upload Disaster Data"])
        if option == "Select from Options":
            pass
        elif option == "Upload Disaster Data":
            pdf_docs = st.file_uploader("Upload your Disaster Data PDFs and Click Submit",
                                        accept_multiple_files=True, type='pdf')
            if st.button("Submit"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text_upload(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("Completed")
        
    user_query = st.chat_input("Ask any questions about disaster data", key="user_input")

    if user_query:
        if st.session_state.vector_store is not None:
            response_text = user_input(user_query, st.session_state.vector_store)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response_text))
        else:
            st.warning("Please upload or select disaster data first.")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

if __name__ == "__main__":
    main()
