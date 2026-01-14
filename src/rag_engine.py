import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def load_rag_chain():
    load_dotenv()
    hf_token = os.getenv('token')

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="vector_store/chroma_db", embedding_function=embeddings)


    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )

    llm = ChatHuggingFace(llm=llm_endpoint)

    template = """You are a financial analyst for CrediTrust. Use the context to answer.
    Context: {context}
    Question: {question}
    Answer:"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )