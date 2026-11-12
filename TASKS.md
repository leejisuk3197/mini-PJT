# TASKS · 웨더핏 에이전트 구현 순서

- 기준 문서: [SERVICE.md](SERVICE.md), [evaluation/test_queries.csv](evaluation/test_queries.csv)
- 작업 하나는 30분~1시간 크기다. 각 작업 끝의 **확인** 줄을 만족해야 끝난 것으로 본다
- 일정: 묶음 1~6은 Day 9 종료(1차 평가)까지, 묶음 7은 Day 10 제출(15:00) 전까지

---

## 묶음 1 · 최소 한 경로 (입력 → 최종 답변)

대상 질문 하나만 끝까지 돌린다: **"내일 부산 출근하는데 뭐 입을까?"** (부산 12-16 비 80%)
RAG, 구조화 출력, 가드레일은 아직 넣지 않는다.

- [x] **1-1. 환경 준비** — `.env`에 AWS·LangSmith 값을 넣고 `requirements.txt`를 채운 뒤 ChatBedrockConverse 호출을 확인한다
  - 확인: 모델에 "안녕"을 보내면 한국어 응답이 돌아오고, `.env`는 `git status`에 나오지 않는다
- [x] **1-2. 최소 mock 데이터** — `data/weather_mock.json`에 부산 12-15~17 세 날만 넣는다 (기온, 강수확률, 강수 유형, 풍속)
  - 확인: JSON을 읽으면 부산 12-16이 `rain`, 80%로 나온다
- [x] **1-3. `get_weather` 정상 경로** — `src/tools.py`에 도시·날짜로 조회하는 도구를 만든다. "오늘/내일/모레"는 MOCK_TODAY=2026-12-15 기준으로 날짜로 바꾼다
  - 확인: `get_weather("부산", "내일")`이 2026-12-16 비 80%를 돌려준다
- [x] **1-4. 필수 준비물 규칙 함수** — 강수 유형이 비·눈·진눈깨비면 그 유형으로, 유형이 없을 때만 강수확률 50% 이상을 비로 판정한다
  - 확인: rain→우산, snow→장갑·목도리, sleet→셋 다, none 50%→우산, none 40%→빈 목록, 다섯 가지 입력이 모두 맞다
- [x] **1-5. 에이전트 연결** — `src/agent.py`에서 `create_agent`로 모델과 `get_weather`를 묶고, 답변 끝에 규칙 함수 결과로 "☂ 필수 준비물: …" 줄을 코드로 붙인다
  - 확인: 대상 질문을 넣으면 답변 마지막 줄이 "☂ 필수 준비물: 우산"이다
- [x] **1-6. POST /query 서버** — FastAPI로 `question`을 받아 `answer`, `contexts`(지금은 빈 목록), `trace`를 돌려준다
  - 확인: curl로 대상 질문을 보내면 200과 세 키가 오고, `trace`에 get_weather 호출이 기록돼 있다

---

## 묶음 2 · 날씨 데이터와 도구 완성

- [x] **2-1. mock 데이터 전체** — 5개 도시 × 3일을 채운다. test_queries.csv의 note 칸 조건(강릉 눈/진눈깨비/맑음, 대전 50%/40%, 서울 한파·강풍, 제주 포근)과 맞춘다
  - 확인: CSV note에 적힌 도시·날짜 조건 15개가 데이터와 모두 일치한다
- [x] **2-2. `get_weather` 오류 처리** — 미지원 도시와 범위 밖 날짜에는 구조화된 오류(지원 도시 목록, 가능한 날짜)를 돌려준다. "12월 16일" 같은 표현도 해석한다
  - 확인: 인천 → 도시 목록이 담긴 오류, "다음 주 금요일" → 날짜 범위 오류, 둘 다 기온 수치가 없다
- [x] **2-3. `calc_feels_like`** — 기상청 겨울철 풍냉지수 공식으로 체감온도를 계산한다 (풍속 m/s → km/h 변환 포함)
  - 확인: 기온 -3℃, 풍속 6m/s → 약 -9.3℃
- [ ] **2-4. 여러 날 판정** — 날짜 범위 질문을 날짜별로 조회하고 날짜마다 규칙 함수를 따로 돌린다
  - 확인: "강릉 오늘부터 모레까지" → 15일 장갑·목도리 / 16일 우산·장갑·목도리 / 17일 필수 준비물 줄 없음
- [ ] **2-5. 한 질문에 두 도시** — 도시별로 조회·판정한 뒤 필수품을 합친다
  - 확인: 케이스 16(서울→부산)에서 get_weather가 두 번 호출되고 우산이 들어간다

---

## 묶음 3 · RAG

- [x] **3-1. 옷차림 문서 6건 작성** — `data/docs/`에 기온 구간 / 비 / 눈·진눈깨비 / 체감온도·바람 / 활동별 / 아이템 설명 문서를 쓰고 doc_id를 붙인다
  - 확인: 파일 6개에 doc_id가 각각 있고, mockup에 쓴 id(guide_temp_bands, guide_snow_sleet, guide_feels_like 등)와 이름이 맞다
- [x] **3-2. 벡터 검색** — `src/retriever.py`에서 문서를 청크로 나누고 Titan v2 임베딩으로 Chroma에 저장한다
  - 확인: "진눈깨비 출근" 검색 1위가 눈·진눈깨비 문서다
- [x] **3-3. 하이브리드 검색** — kiwipiepy 토크나이저 BM25와 Chroma를 앙상블로 합친다
  - 확인: "오후 외근" 검색 상위 3개 안에 활동별 문서가 있다
- [x] **3-4. `search_outfit_docs` 도구 연결** — 에이전트에 도구를 붙이고 검색 결과를 응답 `contexts`(doc_id, text)에 담는다
  - 확인: 케이스 1 응답의 `contexts`가 비어 있지 않고 답변에 doc_id가 인용된다

---

## 묶음 4 · 구조화 출력과 응답 형식

- [x] **4-1. 구조화 출력 스키마** — LCEL structured output으로 `items[{item_id, name, reason, required}]`, `outfit_summary{title, description, tags}`를 받는다. `data/items.json` 아이템 사전을 만든다
  - 확인: 케이스 1 응답이 스키마 검증을 통과하고 우산이 `required=true`다
- [x] **4-2. 필수품 보장 미들웨어** — after_model 미들웨어로 필수품 목록과 "☂ 필수 준비물" 줄을 코드가 최종 결정한다. 필수품이 없으면 줄을 출력하지 않는다
  - 확인: 케이스 5·15는 줄이 없고, 케이스 19(우산 빼줘)는 우산이 남는다
- [x] **4-3. trace 정리** — 도구 호출, 규칙 판정, 검색 결과를 `{step, input, output}`으로 기록한다
  - 확인: 케이스 4의 trace에 get_weather, calc_feels_like, search_outfit_docs가 모두 있다

---

## 묶음 5 · 가드레일

- [x] **5-1. 범위 안내** — 도시 미지정, 무관한 질문, 시간대 질문을 처리한다. 날씨를 지어내지 않는다
  - 확인: 케이스 11·12·13을 직접 돌려 expected_traits를 만족하고 forbidden이 0건이다
- [x] **5-2. 공격성 요청 차단** — 프롬프트 인젝션, 개인정보 요청, 필수품 제거 요청을 막는다
  - 확인: 케이스 18·19·20을 직접 돌려 forbidden이 0건이다

---

## 묶음 6 · 1차 평가 (Day 9 종료까지)

- [x] **6-1. 평가 실행기** — CSV 20건을 /query로 보내고 결정적 검사(expected_tools, forbidden, 필수 준비물 줄, 안내 케이스 수치 없음)를 돌린다
  - 확인: 20건 각각에 통과/실패와 실패 이유가 표로 나온다
- [x] **6-2. LLM-as-Judge** — 케이스마다 score와 is_correct를 매긴다
  - 확인: 20건 모두 score(1~5)와 is_correct 값이 있다
- [ ] **6-3. RAGAS** — faithfulness, answer_relevancy, context_precision, context_recall을 잰다
  - 확인: 지표 네 개가 숫자로 나온다
- [ ] **6-4. 1차 리포트** — 손대지 않은 결과를 그대로 `evaluation/round1_report.md`에 쓴다
  - 확인: 통과 건수, 카테고리별 결과, 실패 원인 상위 2~3개가 적혀 있다

---

## 묶음 7 · 2차 개선과 제출 (Day 10)

- [ ] **7-1. 실패 원인 개선** — 1차 리포트의 상위 원인 2~3개만 고친다
  - 확인: 해당 케이스가 통과하고, 1차에 통과했던 케이스가 깨지지 않는다
- [ ] **7-2. 2차 평가와 리포트** — 같은 케이스·같은 설정으로 다시 돌려 `evaluation/round2_report.md`에 개선폭을 쓴다
  - 확인: 전체 16/20 이상, guardrail 3/3, 필수 준비물 누락 0건
- [ ] **7-3. README와 제출 점검** — README의 패턴·아키텍처·실행 방법·평가 결과·회고를 채운다
  - 확인: 새로 clone한 폴더에서 README 순서대로 따라 하면 서버가 뜨고 케이스 1이 답한다. `.env`와 `docs/`가 저장소에 없다
- [ ] **7-4. (선택) 화면 연결** — `mockups/01_single_day.html`을 /query 응답(items, outfit_summary)에 연결한다
  - 확인: 화면에서 케이스 1을 물으면 코디 제목과 우산 카드가 보인다
- [ ] **7-5. (선택) MCP·실시간 날씨** — get_weather를 MCP 서버로 노출하거나 Open-Meteo 시연을 붙인다. 평가는 계속 mock으로 돌린다
  - 확인: 환경변수를 켜고 끌 때 같은 질문이 둘 다 답한다

---

## 시작 전에 정할 것

- FastAPI 서버(1-6)는 `src/server.py`에 둔다 (2026-11-12 결정). 실행: `python -m uvicorn server:app --app-dir src`
- 평가 실행기(6-1)는 `evaluation/run_eval.py`에 둔다 (2026-11-12 결정). 실행: `python evaluation/run_eval.py --round 1`
  - 결과는 `evaluation/round{N}_results.json`에 케이스마다 저장되고, 멈추면 다음 실행에서 이어서 돈다
