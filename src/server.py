"""POST /query 요청을 받아 에이전트 결과를 돌려주는 FastAPI 서버 모듈."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import regions
from agent import build_agent, run_query
from tools import current_today, get_weather

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WEB_PAGE = WEB_DIR / "index.html"

app = FastAPI(title="웨더핏 에이전트")
agent = build_agent()


@app.get("/")
def index() -> FileResponse:
    """질문을 넣어 /query 결과를 확인하는 테스트 화면을 돌려준다."""
    return FileResponse(WEB_PAGE)


@app.get("/regions.js")
def regions_script() -> FileResponse:
    """브라우저에서 가장 가까운 시·군·구를 계산하는 스크립트를 돌려준다."""
    return FileResponse(WEB_DIR / "regions.js", media_type="text/javascript")


@app.get("/regions")
def region_list() -> dict:
    """화면용 지역 목록(시/도 → 시·군·구, 대표 좌표)과 서버 기준 오늘 날짜(한국 시간)를 돌려준다."""
    return {"today": current_today().isoformat(), **regions.regions_payload()}


@app.get("/weather")
def weather(sido: str, sigungu: str | None = None, date: str = "오늘") -> dict:
    """화면에서 고른 지역·날짜의 하루 날씨와 필수 준비물을 돌려준다 (AI 모델을 부르지 않는다)."""
    place = regions.lookup(sido, sigungu)
    if place is None:
        return {"error": "region_not_found", "region": regions.full_name(sido, sigungu),
                "message": "국내 시·군·구만 지원합니다. 화면의 지역 변경에서 골라 주세요."}
    return get_weather.invoke({"region": place["name"], "date_text": date})


class Location(BaseModel):
    """화면에서 고른 위치. 정확한 좌표가 아니라 시/도·시·군·구 이름만 받는다."""

    sido: str
    sigungu: str | None = None


class QueryRequest(BaseModel):
    """/query 요청 본문. question 은 필수, location·date 는 화면에서 고른 값(선택)이다."""

    question: str
    location: Location | None = None
    date: str | None = None


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """질문을 받아 answer, contexts, trace 세 키로 답을 돌려준다."""
    location = request.location.model_dump() if request.location else None
    return run_query(agent, request.question, location=location, date=request.date)
