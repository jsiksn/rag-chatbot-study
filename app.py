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

    # ---------------------------------------------------------
    # 1. LLM 설정을 채팅 루프 밖으로 이동 (분석 버튼에서도 써야 하니까요)
    # ---------------------------------------------------------
    llm = ChatOpenAI(
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    # ---------------------------------------------------------
    # 2. 사이드바 및 Expander(접이식 메뉴) 적용
    # ---------------------------------------------------------
    if "rel_map" not in st.session_state:
        st.session_state.rel_map = None

    with st.sidebar:
        st.header("🗺️ 그래프 RAG 맛보기")
        st.write("문서의 인물 관계를 한눈에 파악하세요.")
        
        if st.button("📊 인물 관계도 분석 시작"):
            with st.spinner("관계를 분석 중입니다..."):
                # 하이브리드 리트리버로 핵심 맥락 추출
                rel_docs = ensemble_retriever.invoke("인물들 사이의 관계와 주요 사건")
                rel_context = "\n".join([d.page_content for d in rel_docs])
                
                rel_prompt = f"""
                아래 내용을 바탕으로 인물 및 조직 간의 관계를 [표]로 요약해줘.
                형식: [대상 A | 관계 | 대상 B | 상세 설명]
                - 반드시 한국어로만 답변할 것.
                - 절대로 한자(漢字)를 사용하지 말 것.
                
                내용:
                {rel_context}
                """
                rel_response = llm.invoke(rel_prompt)
                st.session_state.rel_map = rel_response.content
        
        # ✨ 2번 옵션: 결과가 있으면 접이식 메뉴로 표시
        if st.session_state.rel_map:
            st.divider()
            # expanded=True로 설정하면 분석 직후에 자동으로 펼쳐집니다.
            with st.expander("📍 인물 관계도 상세보기", expanded=True):
                st.markdown(st.session_state.rel_map)

    # ---------------------------------------------------------
    # 3. 채팅 루프 (기존 코드에서 LLM 설정 부분만 제외하면 됩니다)
    # ---------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # (LLM 설정 부분은 위로 옮겼으므로 여기서는 생략)
        
        # ✨ 프롬프트 템플릿 정의 (기존과 동일)
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

        # 6. 멀티 쿼리 리트리버
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
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        with st.chat_message("assistant"):
            with st.spinner("문서를 꼼꼼히 읽고 답변을 생성하고 있습니다..."):
                response = qa_chain.invoke(prompt)
                answer = response['result']
                source_documents = response.get('source_documents', [])

            st.markdown(answer)
            # ... (참고 문헌 출력 로직 동일) ...

            st.session_state.messages.append({"role": "assistant", "content": answer})

    # 파일 삭제 로직 유지
    try:
        os.remove(uploaded_file.name)
    except:
        pass