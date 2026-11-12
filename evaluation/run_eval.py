"""test_queries.csv 인-아웃 세트를 돌려 결정적 검사와 LLM-as-Judge로 채점하는 평가 실행기.

사용 예:
    python evaluation/run_eval.py --round 1            # 끝나지 않은 케이스만 이어서 실행
    python evaluation/run_eval.py --round 1 --only 8 16
    python evaluation/run_eval.py --round 1 --rerun    # 처음부터 다시
    python evaluation/run_eval.py --round 1 --url http://127.0.0.1:8000/query   # 서버로 보내기

결과는 evaluation/round{N}_results.json 에 케이스마다 저장된다. 모델 한도 등으로 중간에 멈춰도
다음 실행에서 남은 케이스부터 이어서 돈다.
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from langchain_aws import ChatBedrockConverse  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

load_dotenv(ROOT / ".env")

CSV_PATH = ROOT / "evaluation" / "test_queries.csv"
DEFAULT_JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REQUIRED_PREFIX = "☂ 필수 준비물: "

# 케이스별 기대 필수 준비물 줄 (CSV note와 data/weather_mock.json의 강수 규칙으로 정함)
# 값이 None이면 필수 준비물 줄이 없어야 한다. 문자열이면 줄 전체가 정확히 같아야 한다.
EXPECTED_REQUIRED_LINE = {
    "1": "☂ 필수 준비물: 우산",
    "2": "☂ 필수 준비물: 장갑, 목도리",
    "3": "☂ 필수 준비물: 우산, 장갑, 목도리",
    "4": None,
    "5": None,
    "6": None,
    "7": "☂ 필수 준비물: 우산",
    "8": "☂ 필수 준비물: 12/15(화) 장갑, 목도리 / 12/16(수) 우산, 장갑, 목도리",
    "9": None,
    "10": None,
    "11": None,
    "12": None,
    "13": None,
    "14": "☂ 필수 준비물: 우산",
    "15": None,
    "16": "☂ 필수 준비물: 우산",
    "17": None,
    "18": None,
    "19": "☂ 필수 준비물: 우산",
    "20": "☂ 필수 준비물: 장갑, 목도리",
}

# 날씨 수치가 없어야 하는 안내 케이스
GUIDANCE_CASES = {"9", "10", "11", "12", "18"}
WEATHER_NUMBER_PATTERN = re.compile(r"-?\d+(\.\d+)?\s*(℃|°C|도)|\d+\s*%")


class JudgeResult(BaseModel):
    """LLM-as-Judge 채점 결과."""

    score: int = Field(ge=1, le=5, description="1~5점. 기대 특성을 모두 지키고 금지 사항이 없으면 5")
    is_correct: bool = Field(description="사용자에게 그대로 내보내도 되는 올바른 답인지")
    forbidden_violations: list[str] = Field(description="답변이 어긴 금지 항목. 없으면 빈 목록")
    missing_traits: list[str] = Field(description="답변에 빠진 기대 특성. 없으면 빈 목록")
    reason: str = Field(description="판정 이유 한두 문장")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 사내 출근 옷차림 에이전트의 답변을 채점하는 평가자입니다. "
     "기대 특성(expected_traits)과 금지 항목(forbidden)은 세미콜론(;)으로 구분돼 있습니다. "
     "각 기대 특성이 답변에 들어 있는지, 각 금지 항목을 어겼는지 하나씩 확인합니다. "
     "답변 끝의 '☂ 필수 준비물:' 줄은 시스템이 붙인 필수 준비물 표시입니다. "
     "참고 날씨 데이터에 없는 수치를 답변이 쓰면 지어낸 것으로 봅니다. "
     "점수 기준: 5=모든 기대 특성 충족·금지 위반 없음, 4=사소한 누락 하나, 3=중요한 누락, 2=금지 위반 또는 큰 오류, 1=질문과 무관하거나 위험."),
    ("human",
     "[질문]\n{question}\n\n[기대 특성]\n{expected_traits}\n\n[금지 항목]\n{forbidden}\n\n"
     "[참고 날씨 데이터(에이전트가 조회한 결과)]\n{weather}\n\n[에이전트 답변]\n{answer}"),
])


def load_cases() -> list[dict]:
    """test_queries.csv를 읽어 케이스 목록을 돌려준다."""
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def call_agent_in_process():
    """같은 프로세스에서 에이전트를 만들어, 질문을 받아 응답 dict를 돌려주는 함수를 만든다."""
    from agent import build_agent, run_query
    agent = build_agent()
    return lambda question: run_query(agent, question)


def call_agent_over_http(url: str):
    """POST /query 서버로 질문을 보내 응답 dict를 돌려주는 함수를 만든다."""
    def call(question: str) -> dict:
        body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            return json.loads(res.read().decode("utf-8"))
    return call


def called_tools(response: dict) -> list[str]:
    """trace에서 호출된 도구 이름을 순서대로 뽑는다."""
    return [s["step"][len("tool:"):] for s in response.get("trace", []) if s["step"].startswith("tool:")]


def required_line_of(answer: str) -> str | None:
    """답변에서 '☂ 필수 준비물:' 줄을 찾는다. 없으면 None."""
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip().startswith(REQUIRED_PREFIX)]
    return lines[-1] if lines else None


def deterministic_checks(case: dict, response: dict) -> dict:
    """도구 호출, 필수 준비물 줄, 안내 케이스의 수치 노출을 코드로 검사한다."""
    cid, answer = case["id"], response.get("answer", "")
    checks = {}

    expected_tools = [t for t in case["expected_tools"].split(";") if t.strip()]
    tools = called_tools(response)
    missing = [t for t in expected_tools if t not in tools]
    checks["expected_tools"] = {"pass": not missing, "expected": expected_tools, "called": tools, "missing": missing}

    expected_line = EXPECTED_REQUIRED_LINE.get(cid)
    actual_line = required_line_of(answer)
    checks["required_line"] = {"pass": actual_line == expected_line, "expected": expected_line, "actual": actual_line}

    if cid in GUIDANCE_CASES:
        numbers = [m.group(0) for m in WEATHER_NUMBER_PATTERN.finditer(answer)]
        checks["no_weather_numbers"] = {"pass": not numbers, "found": numbers}

    return {"pass": all(c["pass"] for c in checks.values()), "checks": checks}


_judge_chain = None


def judge(case: dict, response: dict) -> JudgeResult:
    """Haiku 구조화 출력으로 답변을 채점한다. 체인은 한 번만 만든다."""
    global _judge_chain
    if _judge_chain is None:
        model = ChatBedrockConverse(
            model=os.getenv("JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL_ID),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            temperature=0,
        )
        _judge_chain = JUDGE_PROMPT | model.with_structured_output(JudgeResult)
    weather = [s["output"] for s in response.get("trace", []) if s["step"] == "tool:get_weather"]
    return _judge_chain.invoke({
        "question": case["input"],
        "expected_traits": case["expected_traits"],
        "forbidden": case["forbidden"],
        "weather": json.dumps(weather, ensure_ascii=False) if weather else "(조회 없음)",
        "answer": response.get("answer", ""),
    })


def evaluate_case(case: dict, call) -> dict:
    """케이스 하나를 실행하고 결정적 검사와 Judge 결과를 합쳐 통과 여부를 정한다."""
    response = call(case["input"])
    det = deterministic_checks(case, response)
    jr = judge(case, response)
    passed = det["pass"] and jr.score >= 4 and jr.is_correct and not jr.forbidden_violations
    return {
        "id": case["id"], "category": case["category"], "input": case["input"],
        "pass": passed, "deterministic": det, "judge": jr.model_dump(), "response": response,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_results(path: Path) -> dict:
    """저장된 결과 파일을 읽는다. 없으면 빈 구조를 돌려준다."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"meta": {}, "cases": {}}


def save_results(path: Path, results: dict) -> None:
    """결과를 JSON 파일로 저장한다 (케이스 하나 끝날 때마다 호출)."""
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


def print_summary(results: dict, cases: list[dict]) -> None:
    """케이스별 통과/실패 표와 카테고리별 통과 건수를 출력한다."""
    done = results["cases"]
    print(f"\n{'id':>3} {'분류':<9} {'결과':<4} {'점수':<3} 실패 이유")
    by_cat: dict[str, list[int]] = {}
    for case in cases:
        r = done.get(case["id"])
        cat = case["category"]
        by_cat.setdefault(cat, [0, 0])[1] += 1
        if r is None:
            print(f"{case['id']:>3} {cat:<9} {'-':<4} {'-':<3} (미실행)")
            continue
        by_cat[cat][0] += r["pass"]
        reasons = [f"{k}" for k, c in r["deterministic"]["checks"].items() if not c["pass"]]
        if r["judge"]["forbidden_violations"]:
            reasons.append("forbidden: " + "; ".join(r["judge"]["forbidden_violations"]))
        if r["judge"]["score"] < 4 or not r["judge"]["is_correct"]:
            reasons.append("judge: " + r["judge"]["reason"])
        print(f"{case['id']:>3} {cat:<9} {'통과' if r['pass'] else '실패':<4} {r['judge']['score']:<3} {' | '.join(reasons)[:150]}")
    total = sum(v[0] for v in by_cat.values())
    print("\n" + " / ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_cat.items()) + f" / 전체 {total}/{len(cases)}")
    scores = [r["judge"]["score"] for r in done.values()]
    if scores:
        print(f"Judge 평균 {sum(scores) / len(scores):.2f} ({len(scores)}건)")


def main() -> None:
    """명령줄 인자를 읽어 남은 케이스를 평가하고, 한 건씩 결과를 저장한다."""
    parser = argparse.ArgumentParser(description="웨더핏 인-아웃 세트 평가")
    parser.add_argument("--round", type=int, default=1, help="평가 차수 (결과 파일 이름에 쓰임)")
    parser.add_argument("--only", nargs="*", help="이 id들만 실행")
    parser.add_argument("--rerun", action="store_true", help="저장된 결과를 무시하고 다시 실행")
    parser.add_argument("--url", help="지정하면 이 /query 주소로 보내고, 없으면 같은 프로세스에서 실행")
    args = parser.parse_args()

    out_path = ROOT / "evaluation" / f"round{args.round}_results.json"
    results = load_results(out_path)
    results["meta"] = {
        "round": args.round,
        "agent_model": os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0 (기본값)"),
        "judge_model": os.getenv("JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL_ID),
        "mode": args.url or "in-process",
    }

    cases = load_cases()
    targets = [c for c in cases if not args.only or c["id"] in args.only]
    if not args.rerun:
        targets = [c for c in targets if c["id"] not in results["cases"]]
    print(f"평가 {len(targets)}건 실행 · agent={results['meta']['agent_model']} · judge={results['meta']['judge_model']}")

    call = call_agent_over_http(args.url) if args.url else (call_agent_in_process() if targets else None)
    for case in targets:
        try:
            result = evaluate_case(case, call)
        except Exception as e:  # 한도 초과 등: 지금까지 결과를 저장하고 멈춘다
            print(f"[{case['id']}] 중단: {type(e).__name__}: {str(e)[:150]}")
            print("남은 케이스는 같은 명령으로 다시 실행하면 이어서 돈다.")
            break
        results["cases"][case["id"]] = result
        save_results(out_path, results)
        print(f"[{case['id']}] {'통과' if result['pass'] else '실패'} (score {result['judge']['score']})")

    save_results(out_path, results)
    print_summary(results, cases)


if __name__ == "__main__":
    main()
