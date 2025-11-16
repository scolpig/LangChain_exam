# 📄 RAG 기반 가상 면접 질문 생성 챗봇 (Streamlit + LangChain + FAISS)

이 프로젝트는 사용자가 업로드한 **PDF/TXT 포트폴리오 문서**를 분석하여  
해당 문서 기반의 **기술 면접 질문**을 생성해주는 **RAG(Retrieval-Augmented Generation)** 기반 챗봇입니다.

OpenAI Embeddings + FAISS 벡터스토어 + LangChain + Streamlit 로 구성되며,  
문서를 벡터화 후 저장하여 다음 실행 때 재사용합니다.

---

## 🚀 주요 기능

- PDF/TXT 문서 업로드 후 자동 텍스트 추출  
- RecursiveCharacterTextSplitter 로 문서 청크 분할  
- OpenAIEmbeddings 로 문서 임베딩 생성  
- FAISS 벡터스토어에 인덱스 저장 및 로드  
- LangChain RAG 파이프라인으로 문서 기반 질문 생성  
- Streamlit UI 제공  

---

## 📁 프로젝트 구조

.<br>
├── app.py # Streamlit 실행 메인 파일 <br>
├── .env # OPENAI_API_KEY 저장<br>
├── faiss_index/ # 벡터스토어 저장 폴더 (실행 후 생성됨)<br>
└── README.md<br>

.env 파일 생성:

OPENAI_API_KEY=your_api_key_here

▶️ 실행 방법

Streamlit 앱 실행:

streamlit run virtual_interview.py

브라우저에서 자동 실행됩니다.

🧠 사용 방법
1. 문서 업로드

PDF 또는 TXT 파일 업로드

문서 분할 → 임베딩 생성 → 벡터스토어 구축 자동 수행

2. 벡터스토어 자동 로드

faiss_index/ 폴더가 있으면 자동 로드

문서 재업로드 불필요

3. 질문 생성

버튼 클릭 → 문서 기반 기술 면접 질문 자동 생성

🔍 RAG 파이프라인 구조
User Question
        │
        ▼
Retriever (FAISS)
        │
        ▼
Context + Prompt 구성
        │
        ▼
ChatOpenAI (LLM)
        │
        ▼
면접 질문 생성

📌 주요 코드 설명
문서 로드 & 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)

벡터스토어 생성
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(docs, embeddings)
vectordb.save_local("faiss_index")

벡터스토어 로드
FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

RAG 체인 구성
rag_chain = (
    {
        "context": retriever,
        "question": RunnableLambda(lambda x: x["question"])
    }
    | prompt
    | llm
)
