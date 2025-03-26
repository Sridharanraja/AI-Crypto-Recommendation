import streamlit as st
import os
import pickle
import faiss
from groq import Groq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # ✅ Fix: Use LangChain Groq Wrapper

API = "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"

st.title("Crypto Trade Advisor")

# ✅ Use LangChain's Groq wrapper to initialize the LLM
llm = ChatGroq(api_key=API, model_name="llama3-8b-8192")

urls = ["https://coinmarketcap.com/currencies/dogecoin/","https://coinmarketcap.com/currencies/ethereum/","https://coinmarketcap.com/currencies/shiba-inu/","https://coinmarketcap.com/currencies/bitcoin/"]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Use Hugging Face Embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"})

# Local FAISS storage path
faiss_index_path = "faiss_index.pkl"

# Load or create FAISS index
if os.path.exists(faiss_index_path):
    with open(faiss_index_path, "rb") as f:
        vectorstore = pickle.load(f)
    st.success("Loaded FAISS index from local storage.")
else:
    if not docs:
        st.error("No data found in the documents. Check if the URLs are accessible.")
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(faiss_index_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("FAISS index created and saved locally.")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

query = st.chat_input("Ask about cryptocurrency trading: ")

system_prompt = (
    "You are a cryptocurrency trading expert. "
    "Use the latest market data from reliable sources to provide "
    "accurate trading advice. If you don't have live data, indicate that "
    "you cannot provide a real-time response. Keep your answer concise "
    "and focus on whether it's the right time to buy or sell based on "
    "market trends."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)  # ✅ Fix: Pass `ChatGroq`
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])
