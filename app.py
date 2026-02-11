import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # 무료 임베딩
from langchain.chains import RetrievalQA

# .env 로드
load_dotenv()

st.set_page_config(page_title="RAG Chatbot Study", page_icon="📚")
st.title("📚 0원 RAG 챗봇 (OpenRouter)")

# 1. 무료 임베딩 모델 설정 (처음 실행 시 모델 다운로드로 인해 시간이 조금 걸릴 수 있음)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask") # 한국어 성능이 좋은 모델

embeddings = load_embeddings()

# 2. 파일 업로드
uploaded_file = st.file_uploader("문서를 업로드하세요", type=['pdf', 'docx', 'txt'])

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. 벡터 저장소 생성 (Chroma)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    st.success("문서 분석 완료!")

    # 4. 채팅 루프
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 5. OpenRouter 무료 모델 연결
        llm = ChatOpenAI(
            model_name="meta-llama/llama-3.3-70b-instruct:free", # OpenRouter 무료 모델
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)
            answer = response['result']
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    os.remove(uploaded_file.name)