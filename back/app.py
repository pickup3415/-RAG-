import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

load_dotenv()

# upstage models
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "futsal-400"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4) 
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    print(req)
    qa = RetrievalQA.from_chain_type(llm=chat_upstage,
                                     chain_type="stuff",
                                     retriever=pinecone_retriever,
                                     return_source_documents=True)
    
    # Define the prompt based on the type of futsal-related queries
    futsal_prompt = (
        """
        당신은 풋살 규칙에 대한 전문가이자 심판입니다. 사용자 요청에 따라 풋살 규칙을 설명하거나 심판 역할을 수행하세요.
        가능한 한 명확하고 간결하게 한국어로 응답하십시오. 아래는 예시입니다.

        - 풋살 규칙 설명 예시:
        질문: 풋살 경기 시간은 얼마나 되나요?
        답변: 풋살 경기는 전반과 후반 각각 20분으로 진행됩니다. 필요 시 전후반 10분의 휴식 시간이 주어질 수 있습니다.

        - 심판 역할 예시:
        상황: 골키퍼가 페널티 구역 밖에서 공을 손으로 잡았습니다. 이는 규칙 위반인가요?
        답변: 네, 이는 규칙 위반입니다. 골키퍼는 페널티 구역 안에서만 손으로 공을 다룰 수 있습니다. 페널티 구역 밖에서 공을 손으로 다룰 경우 상대팀에게 프리킥이 주어집니다.

        사용자 요청: {message}
        """
    ).format(message=req.message)

    result = qa(req.message)
    
    return {
        "reply": result['result'],
        "sources": [doc.metadata for doc in result['source_documents']]
    }

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
