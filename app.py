import streamlit as st
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate 
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever 

# .env 로드
load_dotenv()

st.set_page_config(
  page_title="Hybrid RAG Chatbot", 
  page_icon="💬",
  layout="wide",
  initial_sidebar_state="expanded" 
)
st.title("💬 Hybrid RAG Chatbot")

# 1. 무료 임베딩 모델 설정
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

embeddings = load_embeddings()

# 2. 파일 업로드
uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])

if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 문서 로드 및 분할
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. 하이브리드 리트리버 구성
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    st.success("Document analysis complete! You can start the conversation now.")

    # LLM 설정
    llm = ChatOpenAI(
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    # ---------------------------------------------------------
    # 2. 사이드바
    # ---------------------------------------------------------
    if "rel_map" not in st.session_state:
        st.session_state.rel_map = None

    with st.sidebar:
        st.header("Graph RAG Preview")
        st.write("Visualize the relationships between characters in the document.")
        
        if st.button("📊 Start Relationship Analysis"):
            with st.spinner("Analyzing relationships..."):
                rel_docs = ensemble_retriever.invoke("Relationships between characters and key events")
                rel_context = "\n".join([d.page_content for d in rel_docs])
                
                rel_prompt = f"""
                Summarize the relationships between people and organizations based on the content below in a [table].
                Format: [Entity A | Relationship | Entity B | Details]
                - Answer in English only.
                
                Content:
                {rel_context}
                """
                rel_response = llm.invoke(rel_prompt)
                st.session_state.rel_map = rel_response.content
        
        # ✨ 수정 포인트: Expander 없이 결과가 있으면 바로 표시
        if st.session_state.rel_map:
            st.divider()
            st.subheader("📍 Analysis Results")
            st.markdown(st.session_state.rel_map)

    # ---------------------------------------------------------
    # 3. 채팅 루프
    # ---------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        template = """You are an expert in document analysis.
        Strictly follow the guidelines below and answer based on the [Context].
        
        [Guidelines]
        1. Answer in English only.
        2. Use the content from [Context] as much as possible. If there is no direct answer, explain using relevant clues.
        3. If there is absolutely no relevant content, answer "I cannot find relevant information in the document."
        
        [Context]: {context}
        
        Question: {question}
        Answer:"""

        prompt_template = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # 멀티 쿼리 리트리버 유지
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever, 
            llm=llm
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=advanced_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Reading the document carefully and generating an answer..."):
                response = qa_chain.invoke(prompt)
                answer = response['result']
                source_documents = response.get('source_documents', [])

            st.markdown(answer)

            if source_documents:
                with st.expander("🔍 Check References"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**[Source {i+1}]**")
                        st.write(doc.page_content)
                        st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})

    try:
        os.remove(uploaded_file.name)
    except:
        pass