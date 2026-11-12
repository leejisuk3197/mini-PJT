"""POST /query 요청을 받아 에이전트 결과를 돌려주는 FastAPI 서버 모듈."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import build_agent, run_query

WEB_PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"

app = FastAPI(title="웨더핏 에이전트")
agent = build_agent()


@app.get("/")
def index() -> FileResponse:
    """질문을 넣어 /query 결과를 확인하는 테스트 화면을 돌려준다."""
    return FileResponse(WEB_PAGE)


class QueryRequest(BaseModel):
    """/query 요청 본문. question 필드 하나만 읽는다."""

    question: str


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """질문을 받아 answer, contexts, trace 세 키로 답을 돌려준다."""
    return run_query(agent, request.question)
