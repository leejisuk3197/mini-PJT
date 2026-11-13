"""도메인 도구(날씨 조회, 체감온도 계산 등)를 정의하는 모듈."""

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langchain_core.tools import tool

import regions
from retriever import search

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "weather_mock.json"

# 날씨 출처: .env 의 WEATHER_SOURCE=live(기본, Open-Meteo) / mock(데모·평가용)
KST = timezone(timedelta(hours=9))
FORECAST_DAYS = 3  # 오늘~모레
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_DAILY = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_probability_max",
                    "rain_sum", "showers_sum", "snowfall_sum", "wind_speed_10m_max"]
CACHE_SECONDS = 30 * 60
_weather_cache: dict[tuple[float, float], tuple[float, dict]] = {}

# Open-Meteo 날씨 코드(WMO) → 강수 유형 판정 (SERVICE.md §3)
FREEZING_CODES = {56, 57, 66, 67}
SNOW_CODES = set(range(71, 78)) | {85, 86}
RAIN_CODES = set(range(51, 66)) | {80, 81, 82} | set(range(95, 100))
CONDITION_BY_CODE = [
    ({0}, "맑음"), ({1, 2}, "구름조금"), ({3}, "흐림"), ({45, 48}, "안개"), ({51, 53, 55}, "이슬비"),
    (FREEZING_CODES, "진눈깨비"), ({61, 63, 65}, "비"), (SNOW_CODES, "눈"), ({80, 81, 82}, "소나기"),
    (set(range(95, 100)), "뇌우"),
]

# 상대 날짜 표현 → 기준일로부터 며칠 뒤인지
RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2}

# 강수 유형별 필수 준비물 (진눈깨비는 비와 눈 규칙을 모두 적용)
REQUIRED_BY_TYPE = {
    "rain": ["우산"],
    "snow": ["장갑", "목도리"],
    "sleet": ["우산", "장갑", "목도리"],
}
RAIN_PROB_THRESHOLD = 50
WEEKDAYS = "월화수목금토일"


def date_with_weekday(iso_date: str) -> str:
    """ISO 날짜에 요일을 붙인다 (예: 2026-12-16(수))."""
    return f"{iso_date}({WEEKDAYS[date.fromisoformat(iso_date).weekday()]})"


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


def weather_source() -> str:
    """지금 쓰는 날씨 출처. WEATHER_SOURCE=mock 이면 'mock', 그 밖에는 'live'."""
    return "mock" if os.getenv("WEATHER_SOURCE", "live").strip().lower() == "mock" else "live"


def today_kst() -> date:
    """한국 시간 기준 오늘 날짜."""
    return datetime.now(KST).date()


def current_today() -> date:
    """서버가 쓰는 '오늘'. 데모 모드는 mock 기준일, 실시간 모드는 한국 시간 오늘."""
    if weather_source() == "mock":
        return date.fromisoformat(load_weather_data()["mock_today"])
    return today_kst()


def classify_precip(weather_code: int, rain_mm: float, snow_cm: float) -> str:
    """Open-Meteo 날씨 코드와 하루 강수량·적설량으로 강수 유형(none/rain/snow/sleet)을 정한다.

    비와 눈이 둘 다 오거나 어는 비 코드면 진눈깨비, 눈 코드이거나 눈만 쌓이면 눈, 비 코드면 비다.
    """
    code = int(weather_code or 0)
    rain, snow = float(rain_mm or 0), float(snow_cm or 0)
    if code in FREEZING_CODES or (rain > 0 and snow > 0):
        return "sleet"
    if code in SNOW_CODES or snow > 0:
        return "snow"
    if code in RAIN_CODES:
        return "rain"
    return "none"


def condition_label(weather_code: int) -> str:
    """날씨 코드를 '맑음', '비' 같은 한국어 날씨 이름으로 바꾼다."""
    for codes, label in CONDITION_BY_CODE:
        if int(weather_code or 0) in codes:
            return label
    return "흐림"


def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Open-Meteo 에서 좌표의 3일치 하루 예보를 받는다 (한국 시간, 풍속 m/s). 일시적인 실패에 대비해 한 번 더 시도한다."""
    params = {"latitude": lat, "longitude": lon, "daily": ",".join(OPEN_METEO_DAILY), "timezone": "Asia/Seoul",
              "forecast_days": FORECAST_DAYS, "wind_speed_unit": "ms"}
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=8) as res:
                return json.load(res)
        except OSError:
            if attempt == 1:
                raise


def clear_weather_cache() -> None:
    """30분 예보 저장소를 비운다."""
    _weather_cache.clear()


def cached_forecast(lat: float, lon: float) -> dict:
    """같은 좌표의 예보는 30분 동안 저장해 두고 다시 쓴다."""
    key = (round(lat, 4), round(lon, 4))
    hit = _weather_cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    data = fetch_open_meteo(lat, lon)
    _weather_cache[key] = (time.time(), data)
    return data


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
def get_weather(region: str, date_text: str) -> dict:
    """지역과 날짜 하루의 날씨(기온, 강수확률, 강수 유형, 풍속)와 규칙으로 정한 필수 준비물을 조회한다.

    한 번에 하루만 조회한다. '오늘부터 모레까지'처럼 기간을 물으면 사이 날짜까지 하루씩 모두 호출한다.
    찾을 수 없는 지역이나 범위 밖 날짜도 이 도구로 확인하면 가능한 지역·날짜가 담긴 error를 돌려준다.

    Args:
        region: 지역 이름 (예: 서울특별시 강남구, 부산)
        date_text: '오늘', '내일', '모레' 또는 'YYYY-MM-DD'
    """
    if weather_source() == "mock":
        return mock_weather(region, date_text)
    return live_weather(region, date_text)


def live_weather(region: str, date_text: str) -> dict:
    """실시간 모드: 지역의 대표 좌표로 Open-Meteo 예보를 조회해 하루치 날씨를 돌려준다."""
    found = regions.resolve_region(region.strip())
    if found["status"] != "ok":
        result = {"error": "region_not_found", "region": region,
                  "message": "국내 시·군·구만 지원합니다. 화면의 지역 변경에서 골라 주세요. 날씨 수치를 지어내지 마세요."}
        if found["status"] == "ask":
            result.update(error="ambiguous_region", candidates=found["candidates"],
                          message="같은 이름의 지역이 여러 곳입니다. 어느 시/도인지 되물으세요. 날씨 수치를 지어내지 마세요.")
        return result
    today = today_kst()
    available = [today + timedelta(days=i) for i in range(FORECAST_DAYS)]
    available_labels = [date_with_weekday(d.isoformat()) for d in available]
    iso_date = resolve_date(date_text, today.isoformat())
    if iso_date is None or date.fromisoformat(iso_date) not in available:
        return {"error": "date_out_of_range", "date": date_text, "today": date_with_weekday(today.isoformat()),
                "available_dates": available_labels,
                "message": "오늘~모레만 안내할 수 있습니다. 날씨 수치와 요일을 지어내지 말고 가능한 날짜를 안내하세요."}
    try:
        daily = cached_forecast(found["lat"], found["lon"])["daily"]
        i = daily["time"].index(iso_date)
    except Exception:
        return {"error": "weather_unavailable", "region": found["name"],
                "message": "날씨 서비스에 연결하지 못했습니다. 날씨 수치를 지어내지 말고 잠시 뒤 다시 시도하라고 안내하세요."}
    code = daily["weather_code"][i]
    rain = (daily["rain_sum"][i] or 0) + (daily["showers_sum"][i] or 0)
    precip_type = classify_precip(code, rain, daily["snowfall_sum"][i])
    precip_prob = int(daily["precipitation_probability_max"][i] or 0)
    temp_min = round(float(daily["temperature_2m_min"][i]), 1)
    wind = round(float(daily["wind_speed_10m_max"][i] or 0), 1)
    return {
        "city": found["name"], "region": found["name"], "date": iso_date, "date_label": date_with_weekday(iso_date),
        "condition": condition_label(code), "temp_min": temp_min,
        "temp_max": round(float(daily["temperature_2m_max"][i]), 1), "precip_prob": precip_prob,
        "precip_type": precip_type, "wind_speed": wind,
        "feels_like": calc_feels_like.invoke({"temp": temp_min, "wind_speed": wind})["feels_like"],
        "required_items": required_items(precip_type, precip_prob),
        "required_items_note": "시스템 규칙으로 정한 그날의 필수 준비물. 본문에서 '필수'는 이 품목에만 쓴다. 빈 목록이면 필수 준비물이 없는 날이다.",
        "source": "Open-Meteo 예보 (하루 기준 값)",
    }


def mock_weather(region: str, date_text: str) -> dict:
    """데모 모드: data/weather_mock.json 의 5개 도시·고정 기준일로 조회한다 (평가용)."""
    data = load_weather_data()
    supported_cities = list(data["cities"])
    city = region.strip()
    if city not in data["cities"] and " " in city:  # '강원특별자치도 강릉시' 처럼 넘어오면 데모 도시 이름을 찾는다
        city = next((c for c in supported_cities if c in city), city)
    city_data = data["cities"].get(city)
    if city_data is None:
        return {
            "error": "unsupported_city",
            "city": city,
            "supported_cities": supported_cities,
            "message": f"{city}은(는) 지원하지 않는 도시입니다. 날씨 수치를 제공하지 말고 지원 도시를 안내하세요.",
        }
    available_dates = [date_with_weekday(d) for d in city_data]
    iso_date = resolve_date(date_text, data["mock_today"])
    if iso_date is None or iso_date not in city_data:
        return {
            "error": "date_out_of_range",
            "date": date_text,
            "today": date_with_weekday(data["mock_today"]),
            "available_dates": available_dates,
            "message": "제공 범위 밖의 날짜입니다. 날씨 수치와 요일을 지어내지 말고 가능한 날짜를 안내하세요.",
        }
    weather = city_data[iso_date]
    return {
        "city": city.strip(), "date": iso_date, "date_label": date_with_weekday(iso_date), **weather,
        "required_items": required_items(weather["precip_type"], weather["precip_prob"]),
        "required_items_note": "시스템 규칙으로 정한 그날의 필수 준비물. 본문에서 '필수'는 이 품목에만 쓴다. 빈 목록이면 필수 준비물이 없는 날이다.",
    }


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


@tool
def search_outfit_docs(query: str) -> dict:
    """옷차림 가이드 문서(기온 구간, 비, 눈·진눈깨비, 체감온도, 활동별, 아이템)를 검색해 근거 청크를 돌려준다.

    Args:
        query: 검색 질의 (예: '진눈깨비 출근 옷차림', '영하 5도 추위 타는 사람', '오후 외근')
    """
    return {
        "results": [
            {"doc_id": d.metadata["doc_id"], "section": d.metadata["section"], "text": d.page_content}
            for d in search(query)
        ]
    }
