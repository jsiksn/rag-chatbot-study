# 🤖 Hybrid RAG Chatbot with Multi-Query

### (하이브리드 RAG & 멀티 쿼리 지능형 챗봇)

This project is a high-performance RAG chatbot that automatically matches the user's language and provides accurate, context-aware responses. It supports both Custom OSS LLM and OpenRouter models.
이 프로젝트는 사용자의 언어에 맞춰 자동으로 답변하고 정확한 정보를 제공하는 고성능 RAG 챗봇입니다. Custom OSS LLM과 OpenRouter 모델을 모두 지원하여 유연한 서버 구성이 가능합니다.

---

## 🌟 Key Features (주요 기능)

| Feature (기능)            | Description (설명)                                                                                                                              |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hybrid Search**         | Combines Vector and BM25 search for maximum accuracy. <br> 벡터와 BM25 검색을 결합해 검색 정확도를 극대화했습니다.                              |
| **Language Matching**     | Automatically detects the input language and responds in the same language. <br> 사용자의 질문 언어를 감지하여 원문과 동일한 언어로 답변합니다. |
| **Dual LLM Support**      | Supports both Custom OSS LLM and OpenRouter (Llama 3.3). <br> Custom OSS LLM과 OpenRouter 모델 중 선택하여 사용할 수 있습니다.                  |
| **Relationship Analysis** | Visualizes entities and relationships in a table format. <br> 문서 내 주요 인물과 개체 간의 관계를 사이드바에서 분석합니다.                     |
| **Resource Mgmt**         | Real-time RAM cleanup using `gc` and `del`. <br> **`gc` 및 `del`**을 통해 사용 후 메모리를 즉시 최적화합니다.                                   |

---

## 🛠 Tech Stack (기술 스택)

- **Framework:** LangChain (`community`, `core`, `openai`)
- **Frontend:** Streamlit
- **LLM:** - **Custom OSS LLM** (Default)
  - **OpenRouter Llama 3.3** (Option - Commented in code)
- **Vector DB:** ChromaDB (**In-memory mode**)
- **Embeddings:** HuggingFace (`ko-sroberta-multitask`)

---

## 🚀 Getting Started (시작하기)

### 1. Requirements (사전 준비)

Create a `.env` file in the root directory.
루트 폴더에 `.env` 파일을 생성하고 필요한 키를 설정합니다.

**File: .env**

```env

# For OpenRouter (Optional)
OPENAI_API_KEY=your_openrouter_key

# For Custom OSS LLM
OSS_MODEL_NAME=your_model_name
OSS_BASE_URL=your_endpoint_url
```

### 2. Installation (설치)

```bash
# 1. Create Virtual Environment (가상환경 생성)
python3 -m venv .venv  # Mac/Linux
# python -m venv .venv  # Windows

# 2. Activate Virtual Environment (가상환경 활성화)
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate  # Windows

# 3. Install Dependencies (의존성 설치)
pip install --upgrade pip
pip install streamlit langchain langchain-openai langchain-community \
chromadb sentence-transformers pypdf docx2txt rank_bm25 python-dotenv requests
```

### 3. Run (실행)

```bash
streamlit run app.py
```

---

## 💡 How It Works (작동 원리)

1. **Language Intelligence (지능형 다국어 대응):**
   - The system automatically detects the language of the user's question and strictly responds in the same language (e.g., Korean to Korean, English to English).
   - 사용자의 질문 언어를 자동으로 감지하여, 질문과 동일한 언어로 답변을 생성합니다 (한국어 질문에는 한국어, 영어 질문에는 영어로 대응).

2. **Flexible Model Switching (유연한 모델 전환):**
   - Supports both `CustomOSSLLM` for private API endpoints and `OpenRouter` for cloud-based models (Llama 3.3). Users can easily switch between them by toggling comments in the code.
   - 프라이빗 API 연동을 위한 `CustomOSSLLM` 클래스와 클라우드 기반의 `OpenRouter`를 모두 지원하며, 코드 내 주석 처리를 통해 필요에 따라 모델을 즉시 교체할 수 있습니다.

3. **In-Memory & Memory Optimization (메모리 우선 처리 및 최적화):**
   - To prevent SQLite "readonly database" errors, the vector store is kept in-memory.
   - Uses `gc.collect()` and `del` to explicitly clear RAM and delete temporary files whenever a document is replaced or removed.
   - SQLite 파일 잠금 에러를 방지하기 위해 벡터 저장소를 RAM에 유지하며, 파일이 변경되거나 삭제될 때마다 `gc.collect()`와 `del`을 사용해 메모리 자원을 즉시 최적화합니다.

4. **Smart Session & UI Management (지능형 세션 및 UI 관리):**
   - Implements conditional `st.rerun()` logic to prevent infinite loading loops while ensuring all UI remnants (sidebar analysis, chat history) are cleared on file changes.
   - 새로운 파일 업로드 시 이전 채팅 기록과 사이드바 잔상을 깨끗이 지우되, 조건부 `st.rerun()` 로직을 적용하여 무한 로딩 없는 안정적인 UI 전환을 보장합니다.

5. **Hybrid Retrieval Pipeline (하이브리드 검색 파이프라인):**
   - Combines semantic search (Vector) and keyword search (BM25) with a balanced weighting to provide the most relevant context to the LLM.
   - 의미(Vector)와 키워드(BM25) 검색 결과를 적절한 가중치로 결합하여, 질문에 대해 가장 정확한 컨텍스트를 추출해냅니다.
