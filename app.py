import streamlit as st
import os
import requests

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate 
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever 
from typing import List, Optional, Any

class CustomOSSLLM(BaseChatModel):
    endpoint: str
    model_name: str

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages],
            "stream": False
        }
        
        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        result_json = response.json()

        # ✨ 서버 응답 구조(message -> content)에 맞게 추출
        if "message" in result_json:
            content = result_json["message"]["content"]
        elif "choices" in result_json: # 혹시 OpenAI 표준으로 올 경우를 대비
            content = result_json["choices"][0]["message"]["content"]
        else:
            content = "응답 형식을 해석할 수 없습니다."
            
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "custom_oss_llm"

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

    # 3. 문서 로드 및 분할
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 4. 하이브리드 리트리버 구성
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    st.success("Document analysis complete! You can start the conversation now.")

    # 5. LLM 설정
    # [OpenRouter]
    # llm = ChatOpenAI(
    #     model_name="meta-llama/llama-3.3-70b-instruct:free",
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url="https://openrouter.ai/api/v1",
    #     temperature=0
    # )
    # [Custom LLM]
    llm = CustomOSSLLM(
        endpoint=os.getenv("OSS_BASE_URL"), 
        model_name=os.getenv("OSS_MODEL_NAME", "gpt-oss:20b")
    )

    # ---------------------------------------------------------
    # 사이드바
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
                Summarize the relationships in a Markdown Table.
                For every key term, use the format: 'Original Term (Korean Translation)'.

                [Format Example]
                | Subject A | Relationship | Subject B | Description |
                |:---:|:---:|:---:|:---:|
                | Entité (개체) | Étudier (연구하다) | Système (시스템) | Description in Korean (한글 설명) |

                [Task]
                Analyze the following content and output ONLY the table.
                1. Use 'Original Term from text (Korean Translation)' for all entities.
                2. If the original term is already in Korean, just write it in Korean.
                3. Copy the original word exactly as it appears in the [Content].

                [Content to Analyze]
                {rel_context}
                """
                rel_response = llm.invoke(rel_prompt)
                st.session_state.rel_map = rel_response.content
        
        if st.session_state.rel_map:
            st.divider()
            st.subheader("📍 Analysis Results")
            st.markdown(st.session_state.rel_map)

    # ---------------------------------------------------------
    # 채팅 루프
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
        1. **Language Matching**: Answer in the SAME language as the Question. 
           (If the question is in Korean, answer in Korean. If English, answer in English.)
        2. **Korean Constraint**: If answering in Korean, strictly avoid using Chinese characters (Hanja).
        3. **Context Usage**: Use the content from [Context] as much as possible. If there is no direct answer, explain using relevant clues.
        4. **Handling Uncertainty**: If there is absolutely no relevant information, answer:
           - (In Korean): "문서에서 관련 정보를 찾을 수 없습니다."
           - (In English): "I cannot find relevant information in the document."
        
        [Context]: {context}
        
        Question: {question}
        Answer:"""

        prompt_template = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

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