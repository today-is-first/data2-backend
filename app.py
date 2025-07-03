import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 환경변수 불러오기
load_dotenv()

# API 키 및 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # 예: "us-east-1"
INDEX_NAME = "aircon-rag"

# Pinecone 클라이언트 설정
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# 임베딩 모델과 벡터스토어 설정
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding,
    text_key="context"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM (GPT-4 또는 gpt-3.5 등)
chat_model = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# RAG QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    return_source_documents=True,
)

# FastAPI 서버 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 바디 스키마 정의
class MessageRequest(BaseModel):
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    result = qa_chain.invoke({"query": req.messages[-1].content })
    return {
        "reply": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
