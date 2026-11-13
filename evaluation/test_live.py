"""실시간 위치·날씨 기능(TASKS 묶음 8)의 자동 테스트.

- 인터넷과 AI 모델을 쓰지 않는다. Open-Meteo 응답은 가짜 데이터로 바꿔 넣는다.
- 기준 문서: SERVICE.md §3, §3-1, §4 / TASKS.md 8-1 ~ 8-8
- 실행: python -m pytest evaluation/test_live.py -v

구현 전에 먼저 만든 테스트다. 아래 이름의 함수·파일이 생기면 통과해야 한다.
    data/regions.json                     지역 목록과 대표 좌표 (8-1)
    src/regions.py   resolve_region()     질문 속 지역 해석 (8-5)
    src/tools.py     classify_precip()    날씨 코드 → 강수 유형 (8-3)
                     fetch_open_meteo()   Open-Meteo 호출 (테스트에서 가짜로 바꿈)
                     clear_weather_cache()  30분 예보 저장소 비우기
                     today_kst()          한국 시간 오늘 (8-4)
                     get_weather(region, date_text)  WEATHER_SOURCE 에 따라 실시간/데모 (8-2, 8-3)
    src/agent.py     plan_weather_calls(question, location, date)  조회할 지역·날짜를 코드로 정함 (8-5b, 8-6)
    src/server.py    GET /regions, GET /weather, POST /query 의 location·date (8-6, 8-7)
    web/regions.js   nearestRegion()      브라우저의 가장 가까운 시·군·구 계산 (8-8)
"""

import importlib
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

REGIONS_PATH = ROOT / "data" / "regions.json"
NEAREST_JS = ROOT / "web" / "regions.js"
KST = timezone(timedelta(hours=9))
WEATHER_NUMBER_KEYS = ("temp_min", "temp_max", "precip_prob", "wind_speed", "feels_like")


# ---------------------------------------------------------------- 도우미


def load(module_name: str):
    """src 아래 모듈을 불러온다. 아직 없으면 그 테스트만 실패한다."""
    return importlib.import_module(module_name)


def kst_today() -> date:
    """테스트 기준 한국 시간 오늘."""
    return datetime.now(KST).date()


def fake_forecast(days: list[dict]) -> dict:
    """Open-Meteo daily 응답 모양의 가짜 데이터를 만든다. days[0] 이 오늘이다."""
    start = kst_today()
    daily = {"time": [(start + timedelta(days=i)).isoformat() for i in range(len(days))]}
    keys = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_probability_max",
            "rain_sum", "showers_sum", "snowfall_sum", "wind_speed_10m_max"]
    for k in keys:
        daily[k] = [d[k] for d in days]
    return {"timezone": "Asia/Seoul", "daily": daily}


def day(code=3, tmax=14.0, tmin=4.0, prob=20, rain=0.0, showers=0.0, snow=0.0, wind=6.0) -> dict:
    """가짜 예보 하루치."""
    return {"weather_code": code, "temperature_2m_max": tmax, "temperature_2m_min": tmin,
            "precipitation_probability_max": prob, "rain_sum": rain, "showers_sum": showers,
            "snowfall_sum": snow, "wind_speed_10m_max": wind}


# 오늘: 구름 / 내일: 비 80% / 모레: 비와 눈이 함께 (진눈깨비)
THREE_DAYS = [day(), day(code=61, prob=80, rain=8.0, tmin=6.0), day(code=71, prob=70, rain=1.0, snow=2.0, tmin=-1.0)]


@pytest.fixture
def live(monkeypatch):
    """실시간 모드로 두고 Open-Meteo 호출을 가짜로 바꾼다. 호출 횟수를 calls 에 남긴다."""
    tools = load("tools")
    calls = []

    def fake_fetch(lat, lon):
        calls.append((lat, lon))
        return fake_forecast(THREE_DAYS)

    monkeypatch.setenv("WEATHER_SOURCE", "live")
    monkeypatch.setattr(tools, "fetch_open_meteo", fake_fetch)
    tools.clear_weather_cache()
    yield tools, calls
    tools.clear_weather_cache()


def weather(tools, region: str, date_text: str) -> dict:
    """get_weather 도구를 한 번 부른다."""
    return tools.get_weather.invoke({"region": region, "date_text": date_text})


def has_no_numbers(result: dict) -> bool:
    """오류 결과에 날씨 수치가 하나도 없는지 본다."""
    return all(k not in result for k in WEATHER_NUMBER_KEYS)


# ---------------------------------------------------------------- 8-1 지역 데이터


class TestRegionsData:
    """data/regions.json 이 SERVICE.md §3 데이터 항목대로 만들어졌는지 본다."""

    @pytest.fixture(scope="class")
    def regions(self):
        return json.loads(REGIONS_PATH.read_text(encoding="utf-8"))

    def test_시도_16곳(self, regions):
        # 2026-07-01 광주광역시·전라남도가 전남광주통합특별시로 합쳐져 16곳이다
        assert len(regions["sido"]) == 16

    def test_시군구_약_256곳(self, regions):
        count = sum(len(s["sigungu"]) for s in regions["sido"])
        assert 240 <= count <= 280, count

    def test_구가_있는_시는_구까지_나눈다(self, regions):
        gyeonggi = next(s for s in regions["sido"] if s["name"] == "경기도")
        names = [g["name"] for g in gyeonggi["sigungu"]]
        assert "수원시 영통구" in names
        assert "수원시" not in names

    def test_강남구_대표_좌표(self, regions):
        seoul = next(s for s in regions["sido"] if s["name"] == "서울특별시")
        gangnam = next(g for g in seoul["sigungu"] if g["name"] == "강남구")
        assert abs(gangnam["lat"] - 37.5) < 0.1 and abs(gangnam["lon"] - 127.0) < 0.1

    def test_모든_좌표가_국내_범위(self, regions):
        for s in regions["sido"]:
            for place in [s, *s["sigungu"]]:
                assert 33.0 <= place["lat"] <= 38.7, place
                assert 124.5 <= place["lon"] <= 131.9, place


# ---------------------------------------------------------------- 8-5 질문 속 지역 해석


RESOLVE_CASES = [
    # (질문, 화면에서 고른 시/도, 기대 상태, 기대 시/도, 기대 시·군·구)
    ("중구 출근 옷차림", "서울특별시", "ok", "서울특별시", "중구"),
    ("중구 출근 옷차림", "경기도", "ask", None, None),
    ("중구 출근 옷차림", None, "ask", None, None),
    ("해운대 출근하는데 뭐 입지?", None, "ok", "부산광역시", "해운대구"),
    ("광주 날씨 어때?", "경기도", "ok", "경기도", "광주시"),
    ("광주 날씨 어때?", "서울특별시", "ask", None, None),
    ("고성 내일 옷차림", "경상남도", "ok", "경상남도", "고성군"),
    ("부산 내일 출근", None, "ok", "부산광역시", None),
    ("강릉 모레 옷차림", None, "ok", "강원특별자치도", "강릉시"),
    ("영통구 오늘 뭐 입어?", None, "ok", "경기도", "수원시 영통구"),
    ("강서구 출근", "서울특별시", "ok", "서울특별시", "강서구"),    # '서구' 로 잘못 읽지 않는다
    ("천안 동남구 출근", None, "ok", "충청남도", "천안시 동남구"),  # '남구' 로 잘못 읽지 않는다
    ("도쿄 출근 옷차림", "서울특별시", "not_found", None, None),
    ("출근하는데 뭐 입을까?", "서울특별시", "none", None, None),   # 질문에 지역 없음 → 화면 선택을 쓴다
]


@pytest.mark.parametrize("question,selected,status,sido,sigungu", RESOLVE_CASES)
def test_질문_속_지역_해석(question, selected, status, sido, sigungu):
    regions = load("regions")
    result = regions.resolve_region(question, selected_sido=selected)
    assert result["status"] == status, result
    if status == "ok":
        assert result["sido"] == sido
        assert result["sigungu"] == sigungu
    if status == "ask":
        assert len(result["candidates"]) >= 1


# ---------------------------------------------------------------- 8-5b, 8-6 에이전트 연결 (조회 계획)


GANGNAM = {"sido": "서울특별시", "sigungu": "강남구"}

PLAN_CASES = [
    # (설명, 질문, 화면 위치, 화면 날짜, 기대 상태, 기대 조회 [(지역, 날짜)])
    ("질문에 지역 없음 → 화면 위치", "출근하는데 뭐 입을까?", GANGNAM, "오늘", "ok", [("서울특별시 강남구", "오늘")]),
    ("같은 이름의 구 → 화면 시/도", "중구 출근 옷차림", GANGNAM, "오늘", "ok", [("서울특별시 중구", "오늘")]),
    ("화면 날짜가 없으면 오늘", "출근하는데 뭐 입을까?", GANGNAM, None, "ok", [("서울특별시 강남구", "오늘")]),
    ("화면 날짜 사용", "출근하는데 뭐 입을까?", GANGNAM, "모레", "ok", [("서울특별시 강남구", "모레")]),
    ("질문 날짜가 화면 날짜보다 우선", "내일 출근하는데 뭐 입을까?", GANGNAM, "오늘", "ok", [("서울특별시 강남구", "내일")]),
    ("질문 지역이 화면 위치보다 우선", "해운대 출근 옷차림", GANGNAM, "오늘", "ok", [("부산광역시 해운대구", "오늘")]),
    ("화면 위치 없이 같은 이름의 구 → 되묻기", "중구 출근 옷차림", None, None, "ask", []),
    ("지역이 어디에도 없음 → 되묻기", "출근하는데 뭐 입을까?", None, None, "ask", []),
    ("국내가 아님 → 안내", "도쿄 출근 옷차림", GANGNAM, "오늘", "not_found", []),
]


@pytest.mark.parametrize("label,question,location,date_text,status,calls", PLAN_CASES)
def test_조회_계획(live, label, question, location, date_text, status, calls):
    agent = load("agent")
    plan = agent.plan_weather_calls(question, location=location, date=date_text)
    assert plan["status"] == status, f"{label}: {plan}"
    assert [(c["region"], c["date_text"]) for c in plan["calls"]] == calls, label
    if status in ("ask", "not_found"):
        assert plan["message"], label  # 모델이 아니라 코드가 정한 안내 문구가 있다


# ---------------------------------------------------------------- 8-3 강수 유형 판정


PRECIP_CASES = [
    # (날씨 코드, 비 mm, 눈 cm, 기대 강수 유형)
    (61, 5.0, 0.0, "rain"),
    (80, 2.0, 0.0, "rain"),
    (95, 3.0, 0.0, "rain"),
    (73, 0.0, 2.0, "snow"),
    (0, 0.0, 0.4, "snow"),     # 코드는 맑음이어도 적설량만 있으면 눈
    (71, 2.0, 1.0, "sleet"),   # 비와 눈이 둘 다 있으면 진눈깨비
    (66, 3.0, 0.0, "sleet"),   # 어는 비
    (56, 0.5, 0.0, "sleet"),   # 어는 이슬비
    (3, 0.0, 0.0, "none"),
]


@pytest.mark.parametrize("code,rain,snow,expected", PRECIP_CASES)
def test_강수_유형_판정(code, rain, snow, expected):
    tools = load("tools")
    assert tools.classify_precip(code, rain, snow) == expected


# ---------------------------------------------------------------- 8-3, 8-4 실시간 날씨 조회


class TestLiveWeather:
    """WEATHER_SOURCE=live 일 때 get_weather 가 SERVICE.md 규칙대로 답하는지 본다."""

    def test_오늘은_한국_시간_오늘(self, live):
        tools, _ = live
        assert tools.today_kst() == kst_today()
        assert weather(tools, "서울특별시 강남구", "오늘")["date"] == kst_today().isoformat()

    def test_구름_낀_날은_필수_준비물_없음(self, live):
        tools, _ = live
        result = weather(tools, "서울특별시 강남구", "오늘")
        assert result["precip_type"] == "none"
        assert result["required_items"] == []

    def test_비_오는_날은_우산(self, live):
        tools, _ = live
        result = weather(tools, "서울특별시 강남구", "내일")
        assert result["precip_type"] == "rain"
        assert result["required_items"] == ["우산"]

    def test_진눈깨비_날은_셋_다(self, live):
        tools, _ = live
        result = weather(tools, "서울특별시 강남구", "모레")
        assert result["precip_type"] == "sleet"
        assert sorted(result["required_items"]) == sorted(["우산", "장갑", "목도리"])

    def test_유형이_없어도_강수확률_50_이상이면_우산(self, live, monkeypatch):
        tools, _ = live
        monkeypatch.setattr(tools, "fetch_open_meteo", lambda lat, lon: fake_forecast([day(prob=60)] * 3))
        tools.clear_weather_cache()
        assert weather(tools, "서울특별시 강남구", "오늘")["required_items"] == ["우산"]

    def test_체감온도는_기상청_공식_최저기온_최대풍속(self, live):
        tools, _ = live
        result = weather(tools, "서울특별시 강남구", "오늘")
        expected = tools.calc_feels_like.invoke({"temp": 4.0, "wind_speed": 6.0})["feels_like"]
        assert result["feels_like"] == expected

    def test_모레_다음날은_범위_밖(self, live):
        tools, _ = live
        later = (kst_today() + timedelta(days=3)).isoformat()
        result = weather(tools, "서울특별시 강남구", later)
        assert result["error"] == "date_out_of_range"
        assert has_no_numbers(result)

    def test_국내가_아닌_지역은_수치_없이_안내(self, live):
        tools, _ = live
        result = weather(tools, "도쿄", "오늘")
        assert result["error"] == "region_not_found"
        assert has_no_numbers(result)

    def test_날씨_서비스_연결_실패는_수치_없이_안내(self, live, monkeypatch):
        tools, _ = live

        def broken(lat, lon):
            raise ConnectionError("network down")

        monkeypatch.setattr(tools, "fetch_open_meteo", broken)
        tools.clear_weather_cache()
        result = weather(tools, "서울특별시 강남구", "오늘")
        assert result["error"] == "weather_unavailable"
        assert has_no_numbers(result)

    def test_같은_지역은_30분_동안_한_번만_조회(self, live):
        tools, calls = live
        weather(tools, "서울특별시 강남구", "오늘")
        weather(tools, "서울특별시 강남구", "내일")
        assert len(calls) == 1
        weather(tools, "부산광역시 해운대구", "오늘")
        assert len(calls) == 2

    def test_모드를_정하지_않으면_실시간(self, live, monkeypatch):
        tools, calls = live
        monkeypatch.delenv("WEATHER_SOURCE", raising=False)
        weather(tools, "서울특별시 강남구", "오늘")
        assert len(calls) == 1


# ---------------------------------------------------------------- 8-2 데모 모드 회귀


class TestDemoMode:
    """WEATHER_SOURCE=mock 이면 지금까지와 같은 데이터·기준일로 답하는지 본다 (평가용)."""

    @pytest.fixture(autouse=True)
    def mock_mode(self, monkeypatch):
        tools = load("tools")

        def must_not_call(lat, lon):
            raise AssertionError("데모 모드에서 Open-Meteo 를 부르면 안 된다")

        monkeypatch.setenv("WEATHER_SOURCE", "mock")
        monkeypatch.setattr(tools, "fetch_open_meteo", must_not_call, raising=False)
        self.tools = tools

    def test_케이스1_부산_내일은_비_80(self):
        result = weather(self.tools, "부산", "내일")
        assert result["date"] == "2026-12-16"
        assert result["precip_type"] == "rain" and result["precip_prob"] == 80
        assert result["required_items"] == ["우산"]

    def test_5개_도시_밖은_지원_도시_안내(self):
        result = weather(self.tools, "인천", "내일")
        assert "error" in result and "temp_min" not in result
        assert "부산" in result["supported_cities"]


# ---------------------------------------------------------------- 8-6, 8-7 서버 경로


class TestServer:
    """화면용 경로와 /query 의 location·date 전달을 AI 모델 없이 확인한다."""

    @pytest.fixture
    def client(self, live, monkeypatch):
        from fastapi.testclient import TestClient

        server = load("server")
        seen = {}

        def fake_run_query(agent, question, location=None, date=None):
            seen.update(question=question, location=location, date=date)
            return {"answer": "ok", "contexts": [], "trace": []}

        monkeypatch.setattr(server, "run_query", fake_run_query)
        return TestClient(server.app), seen

    def test_regions_에_오늘_날짜와_지역_목록(self, client):
        c, _ = client
        body = c.get("/regions").json()
        assert body["today"] == kst_today().isoformat()
        assert len(body["sido"]) == 16
        assert all("lat" in s and "sigungu" in s for s in body["sido"])

    def test_weather_는_AI_없이_날씨와_필수_준비물(self, client):
        c, seen = client
        res = c.get("/weather", params={"sido": "서울특별시", "sigungu": "강남구", "date": "내일"})
        assert res.status_code == 200
        body = res.json()
        assert body["precip_type"] == "rain" and body["required_items"] == ["우산"]
        assert seen == {}  # /query 경로(AI 모델)를 거치지 않았다

    def test_weather_없는_지역은_오류(self, client):
        c, _ = client
        body = c.get("/weather", params={"sido": "서울특별시", "sigungu": "없는구", "date": "오늘"}).json()
        assert body["error"] == "region_not_found"
        assert has_no_numbers(body)

    def test_query_에_위치와_날짜를_넘긴다(self, client):
        c, seen = client
        res = c.post("/query", json={"question": "뭐 입을까?",
                                     "location": {"sido": "서울특별시", "sigungu": "강남구"}, "date": "내일"})
        assert res.status_code == 200
        assert set(res.json()) >= {"answer", "contexts", "trace"}
        assert seen["location"] == {"sido": "서울특별시", "sigungu": "강남구"}
        assert seen["date"] == "내일"

    def test_query_는_question_만_보내도_된다(self, client):
        c, seen = client
        res = c.post("/query", json={"question": "내일 부산 출근 옷차림"})
        assert res.status_code == 200
        assert seen["location"] is None and seen["date"] is None


# ---------------------------------------------------------------- 8-8 브라우저의 가장 가까운 시·군·구 계산


NEAREST_CASES = [
    # (설명, 위도, 경도, 기대 시/도, 기대 시·군·구. None 이면 '가까운 지역을 찾지 못했어요')
    ("강남구청", 37.5172, 127.0473, "서울특별시", "강남구"),
    ("해운대 해수욕장", 35.1587, 129.1604, "부산광역시", "해운대구"),
    ("도쿄", 35.6812, 139.7671, None, None),
    ("백령도 (군청이 먼 섬)", 37.9670, 124.6300, None, None),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node 가 없어 브라우저 계산을 검사할 수 없다")
@pytest.mark.parametrize("label,lat,lon,sido,sigungu", NEAREST_CASES)
def test_브라우저_가장_가까운_시군구(label, lat, lon, sido, sigungu):
    script = (
        "const fs = require('fs');"
        f"const {{ nearestRegion }} = require({json.dumps(str(NEAREST_JS))});"
        f"const regions = JSON.parse(fs.readFileSync({json.dumps(str(REGIONS_PATH))}, 'utf8'));"
        f"console.log(JSON.stringify(nearestRegion(regions, {lat}, {lon})));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    if sido is None:
        assert result is None, f"{label}: {result}"
    else:
        assert result["sido"] == sido and result["sigungu"] == sigungu, f"{label}: {result}"
