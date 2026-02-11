# 🤖 Hybrid RAG Chatbot with Multi-Query

### (하이브리드 RAG & 멀티 쿼리 지능형 챗봇)

This project is a high-performance, cost-efficient RAG (Retrieval-Augmented Generation) chatbot designed to accurately retrieve information from documents (PDF, DOCX, TXT) and provide clean Korean responses.

이 프로젝트는 문서(PDF, DOCX, TXT)에서 정확한 정보를 찾아내고, 한자 없이 깔끔한 한국어 답변을 제공하는 고성능·비용 효율적 RAG 챗봇입니다.

---

## 🌟 Key Features (주요 기능)

| Feature (기능)            | Description (설명)                                                                                                                                                           |
| :------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hybrid Search**         | Combines **Vector (Semantic)** and **BM25 (Keyword)** search for maximum accuracy. <br> 벡터(의미)와 BM25(키워드) 검색을 결합해 검색 정확도를 극대화했습니다.                |
| **Multi-Query Retrieval** | Automatically expands a single user query into multiple variations to find hidden info. <br> 사용자의 질문을 AI가 여러 개로 확장해 문서 구석구석의 정보를 찾아냅니다.        |
| **Korean-Only Prompt**    | Strictly optimized for Korean, preventing unnecessary Hanja (Chinese characters). <br> 한국어 답변에 최적화된 프롬프트를 사용하여 불필요한 한자 노출을 방지합니다.           |
| **Zero-Cost Stack**       | Uses **OpenRouter (Llama 3.3 Free)** and **HuggingFace Embeddings** for a "0 won" setup. <br> OpenRouter 무료 모델과 허깅페이스 임베딩을 사용해 비용 부담 없이 구축했습니다. |

---

## 🛠 Tech Stack (기술 스택)

- **Framework:** LangChain
- **Frontend:** Streamlit
- **LLM:** Llama 3.3 70B (via OpenRouter)
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace (\`ko-sroberta-multitask\`)

---

## 🚀 Getting Started (시작하기)

### 1. Requirements (사전 준비)

Create a \`.env\` file in the root directory and add your OpenRouter API key.

**File: .env**
\`\`\`env
OPENAI_API_KEY=your_openrouter_api_key_here
\`\`\`

### 2. Installation (설치)

\`\`\`bash
pip install streamlit langchain langchain-openai chromadb sentence-transformers pypdf docx2txt rank_bm25 python-dotenv
\`\`\`

### 3. Run (실행)

\`\`\`bash
streamlit run app.py
\`\`\`

---

## 💡 How It Works (작동 원리)

1. **Ingestion (데이터 주입):**
   - Documents are split into chunks (900 chars) with 200-char overlap.
   - 문서를 900자 단위로 자르고 200자씩 겹치게 하여 문맥을 보존합니다.

2. **Multi-Query (질문 확장):**
   - LLM expands the user's question into 3-5 variations.
   - AI가 질문을 여러 개로 늘려 더 많은 정보를 찾아냅니다.

3. **Hybrid Retrieval (하이브리드 검색):**
   - Searches by "meaning" and "keywords" with a 6:4 weight.
   - 의미와 키워드 두 가지 방식으로 동시에 검색합니다.

4. **Generation (답변 생성):**
   - Provides clean Korean answers, strictly following "No Hanja" rules.
   - 한자 없이 깔끔한 한국어 답변을 생성합니다.
