# embed_upload.py
import os
import json
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import time

# .env 불러오기
load_dotenv()

# 환경 변수
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "aircon-rag"


# OpenAI 클라이언트
openai = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone 초기화
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

if INDEX_NAME in pc.list_indexes().names():
    print(f"🧹 Deleting existing index: {INDEX_NAME}")
    pc.delete_index(INDEX_NAME)
    time.sleep(5)

# create new index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(INDEX_NAME)




# JSON 데이터 로드
with open("qa_dataset_full.json", encoding="utf-8") as f:
    data = json.load(f)

# 벡터 업로드
for i, item in enumerate(data):
    content = f"Q: {item['question']}\nA: {item['answer']}"

    # 최신 OpenAI 방식으로 임베딩 생성
    embedding_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=content
    )
    embedding = embedding_response.data[0].embedding

    # Pinecone에 업로드
    index.upsert([
        (str(uuid.uuid4()), embedding, {"context": content})
    ])

    if i % 10 == 0:
        print(f"✅ Uploaded {i+1}/{len(data)}")

print("🎉 모든 데이터 업로드 완료!")
