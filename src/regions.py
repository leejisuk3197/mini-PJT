"""지역 데이터(data/regions.json)를 읽고, 질문 속 지역 이름을 시/도·시·군·구로 해석하는 모듈.

해석 규칙은 SERVICE.md §3-1 의 4번을 따른다.
1. 시·군·구 이름에 맞으면 그 시·군·구로 본다 ("구·시·군" 글자는 빼고 써도 된다)
   - 여러 시/도에 같은 이름이 있으면 질문에 쓴 시/도 → 화면에서 고른 시/도 순서로 좁힌다
   - 그래도 하나로 정해지지 않으면 되묻는다 (status=ask)
2. 시·군·구에 맞지 않고 시/도 이름에만 맞으면 그 시/도의 대표 위치(시청·도청)로 본다
"""

import json
import re
from functools import lru_cache
from pathlib import Path

REGIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "regions.json"

# 시/도 줄임말과 옛 이름 (2026-07-01 광주광역시·전라남도 → 전남광주통합특별시)
SIDO_ALIASES = {
    "서울특별시": ["서울시", "서울"],
    "부산광역시": ["부산시", "부산"],
    "대구광역시": ["대구시", "대구"],
    "인천광역시": ["인천시", "인천"],
    "전남광주통합특별시": ["전남광주특별시", "광주광역시", "전라남도", "광주전남", "전남광주", "광주", "전남"],
    "대전광역시": ["대전시", "대전"],
    "울산광역시": ["울산시", "울산"],
    "세종특별자치시": ["세종시", "세종"],
    "경기도": ["경기"],
    "강원특별자치도": ["강원도", "강원"],
    "충청북도": ["충북"],
    "충청남도": ["충남"],
    "경상북도": ["경북"],
    "경상남도": ["경남"],
    "제주특별자치도": ["제주도", "제주"],
    "전북특별자치도": ["전라북도", "전북"],
}

# '구·시·군'을 뺀 줄임말이 흔한 낱말과 같아 잘못 읽힐 수 있는 것 (이것들은 '양산시'처럼 전체 이름으로만 찾는다)
AMBIGUOUS_SHORT = {
    "양산", "연수", "고양", "진주", "구리", "화성", "음성", "예산", "영광", "장수", "진도", "성주", "거창", "보은",
    "수성", "동작", "광산", "남동", "이천", "양주", "구미", "상주", "공주", "기장", "수영", "달성", "유성", "동해",
    "남해", "인제", "영양", "고령", "영동", "완주", "서해", "만세", "오정", "소사", "수정", "동안", "단원", "수지",
    "장안", "상당", "청원", "서북", "사상", "강화",
}

# 국내 시·군·구가 아닌 지명 (질문에 이것만 있으면 '국내 시·군·구만 지원' 안내)
FOREIGN_PLACES = [
    "도쿄", "오사카", "교토", "후쿠오카", "삿포로", "오키나와", "베이징", "상하이", "홍콩", "마카오", "타이베이",
    "방콕", "싱가포르", "하노이", "호치민", "다낭", "마닐라", "세부", "발리", "뉴욕", "워싱턴", "로스앤젤레스",
    "샌프란시스코", "시애틀", "시카고", "하와이", "괌", "사이판", "런던", "파리", "베를린", "로마", "마드리드",
    "시드니", "멜버른", "밴쿠버", "토론토", "두바이", "일본", "중국", "미국", "영국", "프랑스", "독일", "베트남",
    "태국", "대만", "필리핀", "호주", "캐나다", "해외",
]

HANGUL = re.compile(r"[가-힣]")


@lru_cache(maxsize=1)
def load_regions() -> dict:
    """data/regions.json 을 읽는다 (한 번만 읽고 재사용)."""
    return json.loads(REGIONS_PATH.read_text(encoding="utf-8"))


def regions_payload() -> dict:
    """화면에 내려줄 지역 목록(시/도 → 시·군·구, 대표 좌표)을 돌려준다."""
    return {"sido": load_regions()["sido"]}


def full_name(sido: str, sigungu: str | None) -> str:
    """'서울특별시 강남구'처럼 시/도와 시·군·구를 이어 붙인 이름을 만든다."""
    return f"{sido} {sigungu}" if sigungu else sido


def _place(sido: dict, sigungu: dict | None) -> dict:
    """지역 하나를 해석 결과에 쓰는 모양으로 만든다."""
    p = sigungu or sido
    return {"sido": sido["name"], "sigungu": sigungu["name"] if sigungu else None,
            "name": full_name(sido["name"], sigungu["name"] if sigungu else None), "lat": p["lat"], "lon": p["lon"]}


def lookup(sido: str, sigungu: str | None = None) -> dict | None:
    """시/도와 시·군·구 이름이 정확히 맞는 지역을 찾는다. 없으면 None."""
    for s in load_regions()["sido"]:
        if s["name"] != sido:
            continue
        if not sigungu:
            return _place(s, None)
        for g in s["sigungu"]:
            if g["name"] == sigungu:
                return _place(s, g)
    return None


def _short(name: str) -> str | None:
    """'해운대구' → '해운대'처럼 끝의 구·시·군을 뗀 줄임말. 두 글자 미만이거나 흔한 낱말이면 None."""
    short = name[:-1]
    if len(short) < 2 or short in AMBIGUOUS_SHORT:
        return None
    return short


@lru_cache(maxsize=1)
def _index() -> tuple[dict, dict]:
    """낱말 → 지역 이름 집합, 지역 이름 → 지역 정보 사전을 만든다."""
    tokens: dict[str, set] = {}
    places: dict[str, dict] = {}

    def add(token, key):
        if token:
            tokens.setdefault(token, set()).add(key)

    for s in load_regions()["sido"]:
        sido_place = _place(s, None)
        places[sido_place["name"]] = {**sido_place, "level": "sido"}
        for t in [s["name"], *SIDO_ALIASES.get(s["name"], [])]:
            add(t, sido_place["name"])
        for g in s["sigungu"]:
            p = _place(s, g)
            places[p["name"]] = {**p, "level": "sigungu"}
            name = g["name"]
            if " " in name:  # 구가 있는 시: '수원시 영통구'
                city, gu = name.split(" ", 1)
                for t in (name, name.replace(" ", ""), gu, _short(gu), city, _short(city)):
                    add(t, p["name"])
            else:
                add(name, p["name"])
                add(_short(name), p["name"])
    return tokens, places


def _find_spans(text: str) -> list[tuple[int, int, set]]:
    """질문에서 지역 낱말이 나온 위치와 후보 지역들을 찾는다.

    낱말 바로 앞이 한글이면 다른 낱말의 일부로 본다 ('강서구'의 '서구', '동남구'의 '남구').
    겹치는 위치에서는 긴 낱말을 쓰고, 같은 위치·같은 길이의 낱말은 후보를 합친다 ('광주').
    """
    tokens, _ = _index()
    found = []
    for token, keys in tokens.items():
        for m in re.finditer(re.escape(token), text):
            if m.start() > 0 and HANGUL.match(text[m.start() - 1]):
                continue
            found.append((m.start(), m.end(), keys))
    found.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    spans: list[list] = []
    for start, end, keys in found:
        same = next((s for s in spans if s[0] == start and s[1] == end), None)
        if same:
            same[2] |= keys
        elif all(end <= s[0] or start >= s[1] for s in spans):
            spans.append([start, end, set(keys)])
    return sorted((s[0], s[1], s[2]) for s in spans)


def find_regions(text: str, selected_sido: str | None = None) -> list[dict]:
    """질문에 나온 지역을 모두 해석해 순서대로 돌려준다.

    결과 하나는 status 가 ok(지역 확정), ask(후보가 여럿이라 되묻기), not_found(국내 지역이 아님) 중 하나다.
    질문에 지역이 없으면 빈 목록이다.
    """
    _, places = _index()
    spans = _find_spans(text)
    sido_only = [(a, b, keys) for a, b, keys in spans if all(places[k]["level"] == "sido" for k in keys)]
    named_sidos = {places[k]["sido"] for _, _, keys in sido_only for k in keys}

    results = []
    for start, end, keys in spans:
        if (start, end, keys) in sido_only:
            continue
        cands = set(keys)
        # '제주'처럼 시/도와 그 안의 시·군·구가 함께 후보면 시·군·구로 본다
        inner = {places[k]["sido"] for k in cands if places[k]["level"] == "sigungu"}
        cands = {k for k in cands if not (places[k]["level"] == "sido" and places[k]["sido"] in inner)}
        if len(cands) > 1 and named_sidos:  # 질문에 쓴 시/도로 좁힌다 ('서울 중구')
            cands = {k for k in cands if places[k]["sido"] in named_sidos} or cands
        if len(cands) > 1:  # 화면에서 고른 시/도로 좁힌다. 고른 시/도에 그 이름이 없거나 고른 곳이 없으면 되묻는다
            cands = {k for k in cands if places[k]["sido"] == selected_sido}
        word = text[start:end]
        if len(cands) == 1:
            results.append({"status": "ok", "text": word, **places[next(iter(cands))]})
        else:
            results.append({"status": "ask", "text": word, "candidates": sorted(keys)})

    ok_names = {r["name"] for r in results if r["status"] == "ok"}
    ok_sidos = {r["sido"] for r in results if r["status"] == "ok"}
    # '천안 동남구'처럼 넓은 이름의 후보 안에 이미 확정된 지역이 있으면 넓은 이름은 버린다
    results = [r for r in results if r["status"] == "ok" or not ok_names & set(r["candidates"])]
    # 시/도 이름만 나왔고 그 안의 시·군·구가 따로 확정되지 않았으면 시/도 대표 위치로 본다
    for start, end, keys in sido_only:
        for k in keys:
            if places[k]["sido"] not in ok_sidos:
                results.append({"status": "ok", "text": text[start:end], **places[k]})
                ok_sidos.add(places[k]["sido"])

    unique, seen = [], set()
    for r in results:
        key = (r["status"], r.get("name") or r["text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    if unique:
        return unique
    foreign = next((f for f in FOREIGN_PLACES if f in text), None)
    return [{"status": "not_found", "text": foreign}] if foreign else []


def resolve_region(text: str, selected_sido: str | None = None) -> dict:
    """질문 속 첫 번째 지역을 해석한다. 지역이 없으면 status=none."""
    results = find_regions(text, selected_sido)
    asks = [r for r in results if r["status"] == "ask"]
    return asks[0] if asks else (results[0] if results else {"status": "none"})
