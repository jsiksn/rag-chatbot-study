import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever # 멀티 쿼리 추가
from langchain.prompts import PromptTemplate # 프롬프트 템플릿 추가

# .env 로드
load_dotenv()

st.set_page_config(page_title="RAG Chatbot Study", page_icon="📚")
st.title("📚 Advanced RAG 챗봇")

# 1. 무료 임베딩 모델 설정
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

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
    # 인물 정보를 위해 청크 사이즈를 약간 키우고 오버랩을 늘렸습니다.
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

    st.success("문서 분석 완료! 이제 대화를 시작하세요.")

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

        # 5. OpenRouter 모델 및 프롬프트 설정
        llm = ChatOpenAI(
            model_name="meta-llama/llama-3.3-70b-instruct:free",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0 # 창의성 0: 지침 준수 극대화
        )

        # ✨ 엄격한 한글 전용 프롬프트 정의
        template = """당신은 한국어 문서 분석 전문가입니다. 
아래 지침을 반드시 엄수하여 [Context]를 바탕으로 답변하세요.

[지침]
1. 반드시 한국어로만 답변하세요.
2. **절대로 한자(漢字)를 쓰지 마세요.** 모든 단어는 한글로만 표기하세요.
3. [Context]의 내용을 최대한 활용하되, 직접적인 답이 없다면 관련 단서라도 찾아 설명하세요.
4. 정말 관련 내용이 없다면 "문서에서 관련 내용을 찾을 수 없습니다"라고 답하세요.

[Context]: {context}

질문: {question}
답변:"""

        prompt_template = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # 6. 멀티 쿼리 리트리버 (검색 성능 강화)
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever, 
            llm=llm
        )

        # 7. QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=advanced_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template} # 프롬프트 주입
        )
        
        with st.chat_message("assistant"):
            # 🌀 로딩 스피너 추가 (UX 개선)
            with st.spinner("문서를 꼼꼼히 읽고 답변을 생성하고 있습니다..."):
                response = qa_chain.invoke(prompt)
                answer = response['result']
                source_documents = response.get('source_documents', [])

            st.markdown(answer)

            if source_documents:
                with st.expander("🔍 참고 문헌 확인하기"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**[Source {i+1}]**")
                        st.write(doc.page_content)
                        if doc.metadata:
                            metadata_text = f"📄 출처: {doc.metadata.get('source', '알 수 없음')}"
                            if 'page' in doc.metadata:
                                metadata_text += f" (Page: {doc.metadata['page'] + 1})"
                            st.caption(metadata_text)
                        st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})

    # 파일 삭제
    os.remove(uploaded_file.name)