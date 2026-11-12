"""메인 에이전트 그래프를 정의하는 모듈."""

import ast
import json
import os
from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, ToolMessage

from tools import calc_feels_like, get_weather, required_items

load_dotenv()

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

SYSTEM_PROMPT = """당신은 사내 직원의 출근 옷차림을 도와주는 '웨더핏 에이전트'입니다.
- 항상 정중한 존댓말로 답합니다.
- 질문에서 도시와 날짜를 찾아 get_weather 도구로 날씨를 먼저 조회합니다. 날짜는 '오늘', '내일', '모레' 또는 YYYY-MM-DD로 넘깁니다.
- 여러 날을 물으면 날짜마다, 여러 도시를 물으면 도시마다 get_weather를 따로 호출하고, 답변도 날짜별로 나눠 씁니다.
- 체감온도는 calc_feels_like 도구에 최저기온과 풍속을 넣어 계산하고, 계산 근거(기온·풍속)를 함께 밝힙니다.
- get_weather가 error를 돌려주면 날씨 수치를 쓰지 말고 지원 도시나 가능한 날짜를 안내합니다.
- 조회한 기온·강수확률·강수 유형을 근거로 출근 옷차림을 추천합니다. 조회하지 않은 날씨 수치는 지어내지 않습니다.
- 질문에 담긴 개인 상황(추위를 탄다, 외근이 있다 등)을 옷차림에 반영합니다.
- 필수 준비물 줄('☂ 필수 준비물: ...')은 시스템이 자동으로 붙이므로 직접 쓰지 않습니다."""

REQUIRED_LINE_PREFIX = "☂ 필수 준비물: "


def build_model() -> ChatBedrockConverse:
    """환경변수의 모델 ID와 리전으로 Bedrock 채팅 모델을 만든다."""
    return ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=0,
    )


def build_agent():
    """모델과 도메인 도구를 묶어 에이전트 그래프를 만든다."""
    return create_agent(
        model=build_model(),
        tools=[get_weather, calc_feels_like],
        system_prompt=SYSTEM_PROMPT,
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


def run_query(agent, question: str) -> dict:
    """질문 하나를 에이전트로 처리해 answer, contexts, trace를 돌려준다."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    trace = collect_tool_trace(messages)
    items_by_date = decide_required_items(trace)
    trace.append({"step": "rule:required_items", "input": "get_weather 결과", "output": items_by_date})

    answer = messages[-1].text.strip()
    required_line = build_required_line(items_by_date)
    if required_line:
        answer = f"{answer}\n\n{required_line}"

    return {"answer": answer, "contexts": [], "trace": trace}
