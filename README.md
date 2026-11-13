# 미니 PJT: 웨더핏 에이전트

## 무엇을 푸나
현재 위치(전국 시·군·구)의 오늘 실시간 날씨를 바로 보여 주고, 지역·날짜를 말하지 않아도 날씨·옷차림·근거를 한 번에 알려 준다. 출근하는 직원이 비·눈 오는 날 우산이나 장갑·목도리를 빠뜨리지 않도록 해준다.

## 활용한 패턴 (Day 1~7)
12개 패턴 중 8개를 썼다. 필수 1·3·11·12번을 모두 포함한다.

- **Day 1 · #1 LCEL 구조화 출력**: `프롬프트 | 모델.with_structured_output(Pydantic)` 체인을 세 곳에 쓴다.
  - 화면용 코디 카드 `OutfitCard`
  - 주입 판별 `InjectionCheck`
  - 평가 Judge `JudgeResult`
  - 카드의 아이템은 사전(`data/items.json`)에 있는 id만 고르도록 `Literal`로 묶었다
- **Day 3 · #2 ReAct**: `create_agent`가 질문에 맞춰 도구를 스스로 고르고 여러 번 부른다. 여러 날, 여러 도시를 물으면 날짜·도시마다 따로 호출한다
- **Day 2 · #3 RAG (하이브리드 검색)**: 직접 쓴 옷차림 가이드 6건을 `##` 소제목 단위로 자른다.
  - kiwi 형태소 BM25와 Chroma(Titan v2 임베딩)를 0.3:0.7로 합친 하이브리드 검색을 쓴다
  - 답변에 근거 doc_id를 인용하고, `contexts`로 돌려준다
- **Day 4 · #4 도구 다중**: 날씨 조회(`get_weather`), 체감온도 계산(`calc_feels_like`, 기상청 풍냉지수), 문서 검색(`search_outfit_docs`)을 한 질의에서 결합한다
- **Day 5 · #6 가드레일**
  - 입력: 정규식 다음에 Haiku 구조화 판별로 프롬프트 인젝션을 막는다
  - 출력: 전화번호·이메일을 가린다
  - 필수 준비물 제거 요청("우산 빼줘")은 따르지 않고, 날짜·도시·강수확률로 이유를 코드가 붙인다
- **Day 5 · #8 미들웨어**: `AgentMiddleware` 다섯 개를 쓴다.
  - `InputGuardMiddleware`: `before_model`에서 주입이면 `jump_to: end`
  - `ToolEnforcementMiddleware`: `after_model`에서 빠진 도구가 있으면 `jump_to: model`로 한 번 되돌린다
  - `OutputPIIMiddleware`: 개인정보 패턴을 가린다
  - `DailyValueNoticeMiddleware`: 시간대 질문에 하루 기준 값 안내를 붙인다
  - `RequiredItemsMiddleware`: `after_model`에서 필수 준비물 줄을 규칙 결과로 강제한다
- **Day 7 · #11 Observability · Trace**
  - LangSmith 트레이스: `.env`의 `LANGSMITH_TRACING`
  - 응답 `trace`: 입력 가드 → 도구 호출 → 필수품 규칙 → 미들웨어 → 카드 생성 → 아이템 확정을 `{step, input, output}`으로 남긴다
- **Day 7 · #12 평가**: `evaluation/run_eval.py`가 채점한다.
  - 인-아웃 세트 20건: 결정적 검사 + LLM-as-Judge
  - RAGAS 4개 지표

## 아키텍처
```
POST /query {question}
  └─ src/server.py ── run_query()
       └─ src/agent.py  create_agent (Claude on Bedrock)
            ├─ InputGuardMiddleware     before_model: 정규식 → Haiku 판별, 주입이면 고정 안내로 종료
            ├─ 모델 ⇄ 도구 (ReAct)
            │    ├─ get_weather          data/weather_mock.json (5개 도시 × 12/15~17) + 필수품 규칙
            │    ├─ calc_feels_like      기상청 겨울철 풍냉지수
            │    └─ search_outfit_docs   src/retriever.py  BM25(kiwi) + Chroma(Titan v2) 하이브리드
            ├─ ToolEnforcementMiddleware  after_model: 빠진 도구가 있으면 모델로 한 번 되돌림
            ├─ OutputPIIMiddleware      after_model: 전화번호·이메일 가림
            ├─ DailyValueNoticeMiddleware after_model: 시간대 질문에 하루 기준 값 안내
            └─ RequiredItemsMiddleware  after_model: "☂ 필수 준비물" 줄을 규칙 결과로 다시 씀
       ├─ LCEL 구조화 출력 (Haiku)    답변 → 코디 카드(title, description, tags, items)
       └─ 응답 {answer, contexts, trace, items, outfit_summary}
```

**실시간 모드 (기본, `WEATHER_SOURCE=live`)** — 묶음 8에서 추가했다. 위 구조에서 날씨 조회 부분이 바뀐다.
```
화면 web/index.html
  ├─ 위치 권한 → web/regions.js nearestRegion()   좌표 → 가장 가까운 시·군·구 (30km 넘으면 직접 선택). 좌표는 서버로 보내지 않음
  ├─ GET /regions   data/regions.json (시/도 16 · 시·군·구 256, 청사 좌표) + 서버 기준 오늘(한국 시간)
  ├─ GET /weather   get_weather → Open-Meteo 3일 예보 (30분 저장) → 강수 유형 판정 → 필수품 규칙 (AI 없음)
  └─ POST /query {question, location?, date?}
       └─ run_query()
            ├─ 입력 가드
            ├─ plan_weather_calls()   src/regions.py 가 질문 속 지역을 해석 (질문 > 화면, 같은 이름은 고른 시/도, 아니면 되묻기)
            ├─ 코드가 get_weather 를 미리 호출해 대화에 넣음 (모델은 지역·날짜를 고르지 않음)
            └─ create_agent (실시간용 지시문) → 체감온도·문서 검색 → 답변 → 위와 같은 미들웨어·코디 카드
```
- 되묻기, 국내 지역 아님, 날씨 서비스 연결 실패는 AI 모델을 부르지 않고 코드가 정한 안내로 바로 답한다.
- 평가는 계속 데모 모드로 잰다. 실시간 기능은 `evaluation/test_live.py` 자동 테스트로 검사한다.
- **필수 준비물은 코드로 보장한다.**
  - 규칙: 강수 유형이 비·눈·진눈깨비면 그 유형으로 판정하고, 유형이 없을 때만 강수확률 50% 이상을 비로 본다. 비 → 우산, 눈 → 장갑·목도리, 진눈깨비 → 셋 다
  - 날짜별로 따로 판정한다. 같은 날짜 안에서만 여러 도시를 합친다
  - 모델이 쓴 필수 준비물 줄은 지우고 규칙 결과로 다시 붙인다
- 아이템의 이름과 필수 여부도 모델이 아니라 코드가 채운다.

## 실행 방법
```bash
# 1. 패키지 (Python 3.14 기준)
pip install -r requirements.txt
pip install --no-deps ragas==0.4.3      # 평가용. 이유는 requirements.txt 주석 참고

# 2. .env (저장소에 없음)
#    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=us-east-1
#    LANGSMITH_TRACING, LANGSMITH_API_KEY, LANGSMITH_PROJECT
#    BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0   # 없으면 Sonnet 4.5
#    WEATHER_SOURCE=live    # 기본값. Open-Meteo 실시간 날씨 (키 필요 없음). mock 이면 데모 데이터(5개 도시, 기준일 12/15)

# 3. 서버 (실시간 모드)
python -m uvicorn server:app --app-dir src --port 8000
#    Windows 에서는 run.bat 을 더블클릭하면 서버가 켜지고 화면(http://127.0.0.1:8000/)이 자동으로 열린다
#    화면을 열면 위치 권한을 묻고, 허용하면 가장 가까운 시·군·구의 오늘 날씨를 보여 준다 (localhost 에서만 동작)

# 4. 질의 (요청 본문은 UTF-8). location·date 는 선택 항목
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
     -d '{"question": "출근하는데 뭐 입을까?", "location": {"sido": "서울특별시", "sigungu": "강남구"}, "date": "내일"}'
curl "http://127.0.0.1:8000/weather?sido=서울특별시&sigungu=강남구&date=오늘"   # AI 없이 날씨만
curl "http://127.0.0.1:8000/regions"                                            # 지역 목록 + 서버 기준 오늘

# 5. 실시간 기능 자동 테스트 (60건, 인터넷·AI 모델 없이 몇 초)
python -m pytest evaluation/test_live.py -v

# 6. 평가 (항상 데모 모드로 돈다. run_eval.py 가 WEATHER_SOURCE=mock 으로 바꾼다)
python evaluation/run_eval.py --round 2            # 인-아웃 세트 20건 (멈추면 같은 명령으로 이어서)
python evaluation/run_eval.py --round 2 --ragas    # RAGAS 4개 지표
#    --url 로 서버에 보내려면 서버를 데모 모드로 띄운다: WEATHER_SOURCE=mock python -m uvicorn server:app --app-dir src
```
- 처음 검색할 때 `chroma_db/`에 임베딩을 만들고, 이후에는 재사용한다
- `DAILY_QUERY_LIMIT`(기본 100): 하루 `/query` 질문 수 한도. 넘으면 429 로 안내한다. 0 이면 제한 없음. `/weather` 는 제한하지 않는다
- `BM25_TOKENIZER`(기본 kiwi): `light` 면 형태소 분석기 없이 낱말·두 글자 조각으로 BM25 검색을 한다 (메모리 약 820MB → 200MB)

### 배포 (Render 무료 요금제)
- 설정 파일: `Dockerfile`(서버 실행 패키지만, `BM25_TOKENIZER=light`), `render.yaml`(Blueprint), `.dockerignore`
- Render 대시보드 > **New > Blueprint** > 이 GitHub 저장소 선택 > `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` 입력 > Apply
  - AWS 키는 Bedrock 호출 권한만 있는 별도 키를 권장한다. 키는 저장소에 넣지 않는다
- 주소는 `https://<서비스 이름>.onrender.com`. 15분 동안 접속이 없으면 잠들고, 다음 접속 때 1분쯤 걸려 깨어난다
- 하루 질문 수 카운터와 `chroma_db/`는 서버가 다시 켜질 때 초기화된다 (문서 임베딩은 첫 검색 때 다시 만든다)
- Windows Git Bash에서 curl에 한글을 직접 넣으면 400이 날 수 있다. 그때는 요청 본문을 UTF-8 파일로 만들어 `--data-binary @파일`로 보낸다

## RAGAS 평가 결과
2차 최종, 문서 검색을 한 13건 기준이다. 괄호는 목표.

- context_recall: 0.67 (0.70)
- context_precision: 0.66 (0.70)
- faithfulness: 0.78 (0.80)
- answer_relevancy: 0.46 (0.80)

측정 설정:

- 평가 LLM: Haiku 4.5 / 임베딩: Titan v2
- reference: `evaluation/ragas_references.json`의 모범 답변
- contexts: 검색 문서 + 날씨 도구 결과
- 프롬프트: `evaluation/ragas_prompts_ko/`에 고정한 한국어 프롬프트

같은 11건으로 1차와 비교하면 거의 같다 (0.638→0.644 / 0.609→0.609 / 0.775→0.789 / 0.539→0.475). 이번 개선이 검색 품질이 아니라 도구 호출과 필수품 규칙을 고친 것이기 때문이다. 4개 모두 목표에 못 미친다. 자세한 내용은 `evaluation/round2_report.md`에 있다.

## 인-아웃 세트 통과율 (자체 평가)
- 1차 (Day 9 종료): 13 / 20 통과
- 2차 (Day 10 개선 후): **17 / 20 통과** (목표 16/20 · guardrail 3/3 · 필수 준비물 누락 0건 달성)
- 개선폭: +4 건
  - 주요 개선 사항: 날짜 범위를 빠짐없이 조회, 도구 결과에 필수품 규칙 포함, 도구 호출을 미들웨어로 강제
- 에이전트·Judge 모두 Claude Haiku 4.5. 남은 실패 3건 중 2건은 Judge 오판으로 보인다 (`evaluation/round2_report.md`)

### 추가 개선: 한 번에 하나씩 고치는 반복 루프 (2차 이후)
2차 이후, 다음 순서를 3회 반복했다.

1. 20건 전체를 채점한다
2. 실패 문항 하나를 고른다
3. 원인이 모델 답변인지 판정 기준인지 진단한다
4. 한 가지만 고친다

결과는 `evaluation/round3_results.json`에 있다.

| 반복 | 채점 | 고른 문항 | 진단 | 고친 것 |
|---|---|---|---|---|
| 1 | 17/20 | 8 (강릉 3일) | 판정 기준: Judge가 체감온도 계산 결과를 받지 못해, 도구로 계산한 값을 "지어낸 수치"로 봄 | `run_eval.py`: Judge 참고 데이터에 `calc_feels_like` 결과 추가 |
| 2 | 17/20 | 8 (강릉 3일) | 판정 기준: 금지 문구 `17일 필수 준비물 줄 출력`이 "줄 출력 자체 금지"로 읽힘 | `test_queries.csv`: 문구를 `☂ 필수 준비물 줄에 12/17 품목 표기`로 바꿈 |
| 3 | 18/20 | 9 (인천) | 모델 답변: get_weather를 부르라는 안내를 세 번 모두 무시 | `ToolEnforcementMiddleware`: get_weather가 빠지면 코드가 직접 호출 |
| 확인 | **19/20** | | | |

- **최종 19/20**: positive 7/8, negative 4/4, edge 5/5, guardrail 3/3, Judge 평균 4.75
- **남은 실패**: 케이스 5. 맑은 날 본문에 "장갑과 목도리는 꼭 챙기세요"라고 써서 금지 항목을 어겼다. 모델 답변 문제이며, 3회 제한으로 고치지 않았다
- **주의**: 1·2회째는 채점기와 테스트 문구를 고친 것이다. 따라서 19/20은 공식 1차(13/20)·2차(17/20)와 채점 조건이 다르다. 에이전트 코드 변경은 3회째 하나뿐이다
- **RAGAS (13건)**: 0.754 / 0.463 / 0.656 / 0.688 (faithfulness / answer_relevancy / context_precision / context_recall). 2차와 거의 같다

## 트라이앤에러 회고
**시도했지만 실패한 접근**
- **프롬프트로 도구 호출 지시**
  - 2차 1회째에 "반드시 문서를 검색하라", "미지원 도시도 get_weather로 확인하라"를 넣었다
  - 도구 누락은 4건에서 5건으로 오히려 늘었고, Haiku는 매번 다른 도구를 빠뜨렸다
  - 지시를 늘려도 모델이 지킨다는 보장은 없었다
- **모델이 쓴 본문에 필수품 판단 맡기기**
  - 코드가 마지막 줄에 "☂ 필수 준비물: 우산"을 붙여도, 본문에는 "우산은 꼭 챙길 필요 없다"고 써서 서로 모순됐다(1차 케이스 14)
- **RAGAS 첫 측정 (모든 지표 0.18~0.38)**: 검색이 나빠서가 아니라 측정 방법 문제였다
  - reference에 형식 체크리스트(expected_traits)를 넣었다
  - contexts에서 날씨 도구 결과를 뺐다
  - 영어 프롬프트를 그대로 썼다
  - 프롬프트를 실행마다 새로 번역해서, 같은 답변인데 실행마다 점수가 달랐다

**최종 채택한 접근**
- **반드시 지켜야 하는 것은 코드로 보장한다**
  - 필수 준비물 판정, 필수 준비물 줄, 제거 요청 거절 문구, 도구 호출 누락 되돌림, 시간대 안내는 모두 미들웨어와 규칙 함수가 맡는다
  - 모델은 옷차림 설명만 쓴다
  - 그 결과 필수 준비물 누락 0건, 도구 누락 1건이 됐다
- **도구 결과에 규칙 결과를 같이 담는다**: get_weather가 `required_items`를 돌려주니 본문과 마지막 줄이 일치했다
- **평가는 되풀이할 수 있게 만든다**
  - 케이스마다 저장하고 이어서 돌린다 (한도 대응)
  - RAGAS 프롬프트를 고정한다
  - 1·2차를 같은 모델·같은 Judge로 돌린다
- **모델 선택**: Sonnet 4.5로 시작했지만 Bedrock 하루 토큰 한도에 계속 걸렸다. 그래서 1·2차 평가 모두 Haiku 4.5로 통일했다

**남은 한계 · 향후 개선 방향**
- RAGAS 4개 지표 모두 목표 미달이다
  - 답변이 길어 answer_relevancy가 낮다 → 질문의 핵심을 먼저 쓰는 구조로 바꾼다
  - context_precision은 여러 날·여러 도시 질문에서 0이다 → 원인 분석이 필요하다
  - BM25가 숫자에 끌려가는 문제 → 숫자 토큰 제외, 가중치 조정
- Judge 오판
  - Judge가 날씨 결과만 받아 체감온도를 "지어냄"으로 본다
  - 필수 준비물 줄을 잘못 읽는다
  - → 모든 도구 결과와 필수 준비물 줄을 따로 넘기고, 1·2차를 다시 채점한다
- 되돌림을 무시하는 경우(케이스 9) → 도시 판별을 코드로 먼저 하고, 미지원 도시면 도구 결과로만 답한다
- mock 날씨(5개 도시 × 3일)만 쓴다. Open-Meteo 실시간 연동과 MCP 노출은 하지 않았다

## 핵심 코드 위치
- `src/agent.py:142` `build_agent` — 메인 에이전트 그래프 (모델 + 도구 3개 + 미들웨어 5개, 모드별 지시문)
- `src/agent.py:672` `run_query` — 한 질의 처리: trace, 필수품 판정, 코디 카드, 응답 조립
- `src/agent.py:596` `plan_weather_calls` · `:640` `run_live_query` — 실시간 모드: 조회할 지역·날짜를 코드로 정하고 미리 조회
- `src/agent.py:522` `RequiredItemsMiddleware` — 필수 준비물 줄 강제
- `src/agent.py:279` `ToolEnforcementMiddleware` — 빠진 도구 호출 시 모델로 되돌림
- `src/agent.py:326` `DailyValueNoticeMiddleware` — 시간대 질문에 하루 기준 값 안내
- `src/agent.py:227` `InputGuardMiddleware` · `:356` `OutputPIIMiddleware` — 입력·출력 가드레일
- `src/tools.py:153` `required_items` — 비·눈·진눈깨비 필수품 규칙
- `src/tools.py:167` `get_weather` · `:182` `live_weather` · `:104` `classify_precip` — 실시간(Open-Meteo)/데모 날씨 조회, 강수 유형 판정
- `src/tools.py` — 도메인 도구 `get_weather` · `calc_feels_like` · `search_outfit_docs`
- `src/regions.py:154` `find_regions` — 질문 속 지역 해석 (같은 이름의 구, 줄임말, 국내 아님)
- `src/retriever.py:112` `build_hybrid_retriever` — RAG 파이프라인 (BM25 + Chroma)
- `src/server.py` — POST /query, GET /weather, GET /regions
- `web/index.html` · `web/regions.js` — 화면 (위치·날짜·날씨 상자, 질문, 코디 결과), 가장 가까운 시·군·구 계산
- `data/regions.json` — 시/도 16 · 시·군·구 256 대표 좌표 (출처는 파일 머리의 source)
- `evaluation/run_eval.py` — 인-아웃 세트 채점, LLM-as-Judge, RAGAS (데모 모드)
- `evaluation/test_live.py` — 실시간 기능 자동 테스트 59건
