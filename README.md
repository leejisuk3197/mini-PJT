# 미니 PJT: 웨더핏 에이전트

## 무엇을 푸나
출근하는 직원이 비·눈 오는 날 우산이나 장갑·목도리를 빠뜨리지 않도록, 도시와 날짜만 말하면 날씨·옷차림·근거를 한 번에 알려 준다.

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

# 3. 서버
python -m uvicorn server:app --app-dir src --port 8000

# 4. 질의 (요청 본문은 UTF-8)
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
     -d '{"question": "내일 부산 출근하는데 뭐 입을까?"}'

# 5. 평가
python evaluation/run_eval.py --round 2            # 인-아웃 세트 20건 (멈추면 같은 명령으로 이어서)
python evaluation/run_eval.py --round 2 --ragas    # RAGAS 4개 지표
```
- 처음 검색할 때 `chroma_db/`에 임베딩을 만들고, 이후에는 재사용한다
- Windows Git Bash에서 curl에 한글을 직접 넣으면 400이 날 수 있다. 그때는 요청 본문을 UTF-8 파일로 만들어 `--data-binary @파일`로 보낸다

## RAGAS 평가 결과
2차 최종, 문서 검색을 한 13건 기준이다. 괄호는 목표.

- context_recall: 0.67 (0.70)
- context_precision: 0.66 (0.70)
- faithfulness: 0.78 (0.80)
- answer_relevancy: 0.45 (0.80)

측정 설정:

- 평가 LLM: Haiku 4.5 / 임베딩: Titan v2
- reference: `evaluation/ragas_references.json`의 모범 답변
- contexts: 검색 문서 + 날씨 도구 결과
- 프롬프트: `evaluation/ragas_prompts_ko/`에 고정한 한국어 프롬프트

같은 11건으로 1차와 비교하면 거의 같다 (0.638→0.644 / 0.609→0.609 / 0.775→0.788 / 0.539→0.474). 이번 개선이 검색 품질이 아니라 도구 호출과 필수품 규칙을 고친 것이기 때문이다. 4개 모두 목표에 못 미친다. 자세한 내용은 `evaluation/round2_report.md`에 있다.

## 인-아웃 세트 통과율 (자체 평가)
- 1차 (Day 9 종료): 13 / 20 통과
- 2차 (Day 10 개선 후): **17 / 20 통과** (목표 16/20 · guardrail 3/3 · 필수 준비물 누락 0건 달성)
- 개선폭: +4 건
  - 주요 개선 사항: 날짜 범위를 빠짐없이 조회, 도구 결과에 필수품 규칙 포함, 도구 호출을 미들웨어로 강제
- 에이전트·Judge 모두 Claude Haiku 4.5. 남은 실패 3건 중 2건은 Judge 오판으로 보인다 (`evaluation/round2_report.md`)

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
- `src/agent.py:112` `build_agent` — 메인 에이전트 그래프 (모델 + 도구 3개 + 미들웨어 5개)
- `src/agent.py:476` `run_query` — 한 질의 처리: trace, 필수품 판정, 코디 카드, 응답 조립
- `src/agent.py:432` `RequiredItemsMiddleware` — 필수 준비물 줄 강제
- `src/agent.py:233` `ToolEnforcementMiddleware` — 빠진 도구 호출 시 모델로 되돌림
- `src/agent.py:267` `DailyValueNoticeMiddleware` — 시간대 질문에 하루 기준 값 안내
- `src/agent.py:193` `InputGuardMiddleware` · `:297` `OutputPIIMiddleware` — 입력·출력 가드레일
- `src/tools.py:63` `required_items` — 비·눈·진눈깨비 필수품 규칙
- `src/tools.py` — 도메인 도구 `get_weather` · `calc_feels_like` · `search_outfit_docs`
- `src/retriever.py:112` `build_hybrid_retriever` — RAG 파이프라인 (BM25 + Chroma)
- `src/server.py` — POST /query
- `evaluation/run_eval.py` — 인-아웃 세트 채점, LLM-as-Judge, RAGAS
