"""메인 에이전트 그래프를 정의하는 모듈."""

import ast
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools import calc_feels_like, get_weather, required_items, search_outfit_docs

load_dotenv()

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_CARD_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
ITEMS_PATH = Path(__file__).resolve().parent.parent / "data" / "items.json"
ITEM_CATALOG: dict = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
ITEM_ID_BY_NAME = {v["name"]: k for k, v in ITEM_CATALOG.items()}
ItemId = Literal[tuple(ITEM_CATALOG)]

SYSTEM_PROMPT = """당신은 사내 직원의 출근 옷차림을 도와주는 '웨더핏 에이전트'입니다.
- 항상 정중한 존댓말로 답합니다.
- 질문에서 도시와 날짜를 찾아 get_weather 도구로 날씨를 먼저 조회합니다. 날짜는 '오늘', '내일', '모레' 또는 YYYY-MM-DD로 넘깁니다.
- 여러 날을 물으면 날짜마다, 여러 도시를 물으면 도시마다 get_weather를 따로 호출하고, 답변도 날짜별로 나눠 씁니다.
- 체감온도는 calc_feels_like 도구에 최저기온과 풍속을 넣어 계산하고, 계산 근거(기온·풍속)를 함께 밝힙니다.
- get_weather가 error를 돌려주면 날씨 수치를 쓰지 말고 지원 도시나 가능한 날짜를 안내합니다.
- 조회한 기온·강수확률·강수 유형을 근거로 출근 옷차림을 추천합니다. 조회하지 않은 날씨 수치는 지어내지 않습니다.
- 옷차림을 추천하기 전에 search_outfit_docs로 관련 가이드 문서를 검색하고, 문서 내용에 근거해 추천합니다. 날씨(기온·강수)와 개인 상황(외근, 추위 탐 등)을 검색어에 담습니다.
- 답변 끝에 근거로 쓴 문서의 doc_id를 '근거: guide_rain, guide_temp_bands'처럼 밝힙니다. 검색 결과에 없는 doc_id는 쓰지 않습니다.
- 질문에 담긴 개인 상황(추위를 탄다, 외근이 있다 등)을 옷차림에 반영합니다.
- 필수 준비물 줄('☂ 필수 준비물: ...')은 시스템이 자동으로 붙이므로 직접 쓰지 않습니다."""

REQUIRED_LINE_PREFIX = "☂ 필수 준비물: "
REQUIRED_LINE_PATTERN = re.compile(r"^[\s☂🧤🧣*#>-]*필수 준비물\s*[:：].*$", re.MULTILINE)

CARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 출근 옷차림 답변을 화면용 코디 카드로 정리합니다. 답변에 나온 내용만 쓰고 새 옷을 지어내지 않습니다.\n"
     "- title: 오늘의 코디 제목 (15자 이내)\n"
     "- description: 코디 설명 한두 문장\n"
     "- tags: 날씨·상황을 나타내는 짧은 태그 2~4개 (예: #비, #체감3℃, #추위탐)\n"
     "- items: 답변에서 추천한 품목을 사전의 item_id로 고르고, 추천 이유를 한 문장으로 씁니다. 사전에 없는 품목은 뺍니다."),
    ("human", "질문: {question}\n\n답변:\n{answer}"),
])


class ItemPick(BaseModel):
    """코디 카드에 넣을 아이템 하나 (모델이 고르는 부분)."""

    item_id: ItemId = Field(description="아이템 사전의 id")
    reason: str = Field(description="이 아이템을 추천하는 이유 한 문장")


class OutfitCard(BaseModel):
    """화면에 보여 줄 오늘의 코디 카드 (모델의 구조화 출력)."""

    title: str = Field(description="오늘의 코디 제목, 15자 이내")
    description: str = Field(description="코디 설명 한두 문장")
    tags: list[str] = Field(description="날씨·상황 태그 2~4개")
    items: list[ItemPick] = Field(description="추천 아이템 목록")


def build_model(model_id: str | None = None) -> ChatBedrockConverse:
    """모델 ID(없으면 환경변수 BEDROCK_MODEL_ID)와 리전으로 Bedrock 채팅 모델을 만든다."""
    return ChatBedrockConverse(
        model=model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=0,
    )


def build_card_chain():
    """답변을 코디 카드로 정리하는 LCEL 구조화 출력 체인을 만든다 (프롬프트 | 모델.with_structured_output)."""
    model = build_model(os.getenv("CARD_MODEL_ID", DEFAULT_CARD_MODEL_ID))
    return CARD_PROMPT | model.with_structured_output(OutfitCard)


def build_agent():
    """모델과 도메인 도구, 필수품 보장 미들웨어를 묶어 에이전트 그래프를 만든다."""
    return create_agent(
        model=build_model(),
        tools=[get_weather, calc_feels_like, search_outfit_docs],
        system_prompt=SYSTEM_PROMPT,
        middleware=[RequiredItemsMiddleware()],
    )


def parse_tool_output(content) -> dict:
    """도구 메시지의 문자열 내용을 딕셔너리로 바꾼다. 바꿀 수 없으면 원문을 담아 돌려준다."""
    if isinstance(content, dict):
        return content
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(content)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError, TypeError):
            continue
    return {"raw": content}


def collect_tool_trace(messages: list) -> list[dict]:
    """메시지 목록에서 도구 호출과 그 결과를 짝지어 trace 항목으로 만든다."""
    calls = {}
    trace = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                calls[call["id"]] = call
        elif isinstance(msg, ToolMessage):
            call = calls.get(msg.tool_call_id, {})
            trace.append({
                "step": f"tool:{call.get('name', msg.name)}",
                "input": call.get("args", {}),
                "output": parse_tool_output(msg.content),
            })
    return trace


WEEKDAYS = "월화수목금토일"


def decide_required_items(trace: list[dict]) -> dict[str, list[str]]:
    """get_weather 결과마다 규칙 함수를 적용해 날짜별 필수 준비물을 정한다.

    같은 날짜에 도시가 여럿이면 그 날짜 안에서만 중복 없이 합친다. 날짜끼리는 섞지 않는다.
    """
    by_date: dict[str, list[str]] = {}
    for step in trace:
        weather = step["output"]
        if step["step"] != "tool:get_weather" or "error" in weather or "date" not in weather:
            continue
        items = by_date.setdefault(weather["date"], [])
        for item in required_items(weather.get("precip_type", "none"), weather.get("precip_prob", 0)):
            if item not in items:
                items.append(item)
    return dict(sorted(by_date.items()))


def format_date_label(iso_date: str) -> str:
    """ISO 날짜를 '12/16(화)' 형태로 바꾼다."""
    d = date.fromisoformat(iso_date)
    return f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})"


def build_required_line(items_by_date: dict[str, list[str]]) -> str | None:
    """날짜별 필수 준비물로 '☂ 필수 준비물' 줄을 만든다. 필수품이 하나도 없으면 None.

    날짜가 하나면 품목만, 여럿이면 필수품이 있는 날짜마다 날짜를 붙여 ' / '로 잇는다.
    """
    days_with_items = {d: items for d, items in items_by_date.items() if items}
    if not days_with_items:
        return None
    if len(items_by_date) == 1:
        return REQUIRED_LINE_PREFIX + ", ".join(next(iter(days_with_items.values())))
    parts = [f"{format_date_label(d)} {', '.join(items)}" for d, items in days_with_items.items()]
    return REQUIRED_LINE_PREFIX + " / ".join(parts)


def collect_contexts(trace: list[dict]) -> list[dict]:
    """search_outfit_docs 결과에서 근거 청크를 모아 중복 없이 {doc_id, text} 목록으로 만든다."""
    contexts, seen = [], set()
    for step in trace:
        if step["step"] != "tool:search_outfit_docs":
            continue
        for hit in step["output"].get("results", []):
            if hit["text"] not in seen:
                seen.add(hit["text"])
                contexts.append({"doc_id": hit["doc_id"], "text": hit["text"]})
    return contexts


def apply_required_line(text: str, items_by_date: dict[str, list[str]]) -> str:
    """모델이 쓴 필수 준비물 줄을 모두 지우고, 규칙으로 정한 줄을 답변 끝에 다시 붙인다."""
    body = REQUIRED_LINE_PATTERN.sub("", text).strip()
    line = build_required_line(items_by_date)
    return f"{body}\n\n{line}" if line else body


class RequiredItemsMiddleware(AgentMiddleware):
    """마지막 답변의 필수 준비물 줄을 규칙 함수 결과로 강제하는 after_model 미들웨어."""

    def after_model(self, state, runtime):
        """도구 호출이 없는 최종 답변일 때만, 같은 id의 메시지로 바꿔 끼운다."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        items_by_date = decide_required_items(collect_tool_trace(state["messages"]))
        fixed = apply_required_line(last.text, items_by_date)
        if fixed == last.text:
            return None
        return {"messages": [AIMessage(content=fixed, id=last.id)]}


def finalize_items(card: OutfitCard | None, required_names: list[str]) -> list[dict]:
    """모델이 고른 아이템에 이름과 필수 여부를 코드로 채우고, 빠진 필수품을 앞에 넣는다."""
    picks = {}
    for pick in (card.items if card else []):
        if pick.item_id in ITEM_CATALOG and pick.item_id not in picks:
            picks[pick.item_id] = pick.reason
    items = []
    for name in required_names:
        item_id = ITEM_ID_BY_NAME[name]
        items.append({"item_id": item_id, "name": name, "required": True,
                      "reason": picks.pop(item_id, "비·눈 규칙에 따른 필수 준비물입니다.")})
    for item_id, reason in picks.items():
        items.append({"item_id": item_id, "name": ITEM_CATALOG[item_id]["name"], "required": False, "reason": reason})
    return items


_card_chain = None


def make_outfit_card(question: str, answer: str) -> OutfitCard:
    """답변을 구조화 출력 체인에 넣어 코디 카드를 만든다. 체인은 한 번만 만든다."""
    global _card_chain
    if _card_chain is None:
        _card_chain = build_card_chain()
    return _card_chain.invoke({"question": question, "answer": answer})


def run_query(agent, question: str) -> dict:
    """질문 하나를 에이전트로 처리해 answer, contexts, trace와 화면용 items, outfit_summary를 돌려준다."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    trace = collect_tool_trace(messages)
    items_by_date = decide_required_items(trace)
    trace.append({"step": "rule:required_items", "input": "get_weather 결과", "output": items_by_date})

    answer = messages[-1].text.strip()
    trace.append({"step": "middleware:required_line", "input": items_by_date,
                  "output": build_required_line(items_by_date)})

    required_names = list(dict.fromkeys(n for names in items_by_date.values() for n in names))
    card = None
    if items_by_date:
        try:
            card = make_outfit_card(question, answer)
            trace.append({"step": "llm:outfit_card", "input": "answer", "output": card.model_dump()})
        except Exception as e:  # 카드가 실패해도 답변과 필수품은 그대로 돌려준다
            trace.append({"step": "llm:outfit_card", "input": "answer", "output": {"error": str(e)[:200]}})
    items = finalize_items(card, required_names)
    trace.append({"step": "rule:finalize_items", "input": required_names, "output": items})

    outfit_summary = card.model_dump(include={"title", "description", "tags"}) if card else None
    return {"answer": answer, "contexts": collect_contexts(trace), "trace": trace,
            "items": items, "outfit_summary": outfit_summary}
