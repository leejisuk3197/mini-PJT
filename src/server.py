"""POST /query 요청을 받아 에이전트 결과를 돌려주는 FastAPI 서버 모듈."""

from fastapi import FastAPI
from pydantic import BaseModel

from agent import build_agent, run_query

app = FastAPI(title="웨더핏 에이전트")
agent = build_agent()


class QueryRequest(BaseModel):
    """/query 요청 본문. question 필드 하나만 읽는다."""

    question: str


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """질문을 받아 answer, contexts, trace 세 키로 답을 돌려준다."""
    return run_query(agent, request.question)
