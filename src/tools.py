"""도메인 도구(날씨 조회, 체감온도 계산 등)를 정의하는 모듈."""

import json
from datetime import date, timedelta
from pathlib import Path

from langchain_core.tools import tool

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "weather_mock.json"

# 상대 날짜 표현 → 기준일로부터 며칠 뒤인지
RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2}

# 강수 유형별 필수 준비물 (진눈깨비는 비와 눈 규칙을 모두 적용)
REQUIRED_BY_TYPE = {
    "rain": ["우산"],
    "snow": ["장갑", "목도리"],
    "sleet": ["우산", "장갑", "목도리"],
}
RAIN_PROB_THRESHOLD = 50


def load_weather_data() -> dict:
    """mock 날씨 JSON 파일을 읽어 딕셔너리로 돌려준다."""
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_date(date_text: str, mock_today: str) -> str | None:
    """'오늘/내일/모레' 또는 'YYYY-MM-DD'를 기준일에 맞춰 ISO 날짜 문자열로 바꾼다. 해석할 수 없으면 None."""
    text = date_text.strip()
    if text in RELATIVE_DAYS:
        base = date.fromisoformat(mock_today)
        return (base + timedelta(days=RELATIVE_DAYS[text])).isoformat()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def required_items(precip_type: str, precip_prob: int) -> list[str]:
    """강수 유형과 강수확률로 그날의 필수 준비물 목록을 정한다.

    강수 유형이 비·눈·진눈깨비면 그 유형으로 판정하고,
    유형이 없을 때만 강수확률 50% 이상을 비로 판정한다.
    """
    if precip_type in REQUIRED_BY_TYPE:
        return list(REQUIRED_BY_TYPE[precip_type])
    if precip_prob >= RAIN_PROB_THRESHOLD:
        return list(REQUIRED_BY_TYPE["rain"])
    return []


@tool
def get_weather(city: str, date_text: str) -> dict:
    """도시와 날짜로 하루 기준 날씨(기온, 강수확률, 강수 유형, 풍속)를 조회한다.

    Args:
        city: 도시 이름 (예: 부산)
        date_text: '오늘', '내일', '모레' 또는 'YYYY-MM-DD'
    """
    data = load_weather_data()
    iso_date = resolve_date(date_text, data["mock_today"])
    city_data = data["cities"].get(city)
    if city_data is None:
        return {"error": "unsupported_city", "city": city}
    if iso_date is None or iso_date not in city_data:
        return {"error": "date_out_of_range", "date": date_text}
    return {"city": city, "date": iso_date, **city_data[iso_date]}
