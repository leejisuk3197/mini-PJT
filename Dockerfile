# 웨더핏 에이전트 서버 이미지 (Render 무료 요금제 512MB 에 맞춘 배포용)
FROM python:3.14-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    WEATHER_SOURCE=live \
    BM25_TOKENIZER=light \
    DAILY_QUERY_LIMIT=100

# 서버 실행에 필요한 패키지만 설치한다.
# 형태소 분석기 kiwipiepy(메모리 약 500MB), 테스트·평가(RAGAS) 패키지는 빼고, BM25 는 light 토크나이저를 쓴다.
COPY requirements.txt .
RUN sed -n '1,/^# 실시간 기능 자동 테스트/p' requirements.txt | grep -v -e '^#' -e '^kiwipiepy' > requirements-server.txt \
 && pip install --no-cache-dir -r requirements-server.txt

COPY src ./src
COPY data ./data
COPY web ./web

# Render 가 PORT 환경변수로 포트를 알려 준다 (없으면 8000)
CMD ["sh", "-c", "python -m uvicorn server:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
