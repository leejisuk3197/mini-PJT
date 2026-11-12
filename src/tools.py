"""도메인 도구(날씨 조회, 체감온도 계산 등)를 정의하는 모듈."""

import json
import re
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
    """날짜 표현을 기준일에 맞춰 ISO 날짜 문자열로 바꾼다. 해석할 수 없으면 None.

    지원 표현: '오늘/내일/모레', 'YYYY-MM-DD', '12월 16일', '12/16', '16일'
    """
    text = date_text.strip()
    base = date.fromisoformat(mock_today)
    if text in RELATIVE_DAYS:
        return (base + timedelta(days=RELATIVE_DAYS[text])).isoformat()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    month_day = re.fullmatch(r"(\d{1,2})\s*(?:월\s*|/)(\d{1,2})\s*일?", text)
    day_only = re.fullmatch(r"(\d{1,2})\s*일", text)
    try:
        if month_day:
            return date(base.year, int(month_day.group(1)), int(month_day.group(2))).isoformat()
        if day_only:
            return date(base.year, base.month, int(day_only.group(1))).isoformat()
    except ValueError:
        return None
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
    supported_cities = list(data["cities"])
    city_data = data["cities"].get(city.strip())
    if city_data is None:
        return {
            "error": "unsupported_city",
            "city": city,
            "supported_cities": supported_cities,
            "message": f"{city}은(는) 지원하지 않는 도시입니다. 날씨 수치를 제공하지 말고 지원 도시를 안내하세요.",
        }
    available_dates = list(city_data)
    iso_date = resolve_date(date_text, data["mock_today"])
    if iso_date is None or iso_date not in city_data:
        return {
            "error": "date_out_of_range",
            "date": date_text,
            "available_dates": available_dates,
            "message": "제공 범위 밖의 날짜입니다. 날씨 수치를 지어내지 말고 가능한 날짜를 안내하세요.",
        }
    return {"city": city.strip(), "date": iso_date, **city_data[iso_date]}


@tool
def calc_feels_like(temp: float, wind_speed: float) -> dict:
    """기온과 풍속으로 체감온도를 계산한다 (기상청 겨울철 풍냉지수 공식).

    공식은 기온 10℃ 이하, 풍속 4.8km/h 이상에서만 쓰며, 그 밖에서는 기온을 그대로 체감온도로 본다.

    Args:
        temp: 기온(℃). 출근 시간 기준이므로 보통 최저기온을 넣는다
        wind_speed: 풍속(m/s)
    """
    wind_kmh = wind_speed * 3.6
    if temp > 10 or wind_kmh < 4.8:
        return {"feels_like": float(temp), "temp": temp, "wind_speed": wind_speed,
                "formula": "적용 범위 밖(기온>10℃ 또는 풍속<4.8km/h)이라 기온을 그대로 사용"}
    v = wind_kmh ** 0.16
    feels = 13.12 + 0.6215 * temp - 11.37 * v + 0.3965 * temp * v
    return {"feels_like": round(feels, 1), "temp": temp, "wind_speed": wind_speed,
            "formula": "13.12 + 0.6215T - 11.37V^0.16 + 0.3965TV^0.16 (T: ℃, V: km/h)"}
