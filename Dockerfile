FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY common ./common
COPY reinforce ./reinforce
COPY dqn ./dqn
COPY ppo ./ppo
COPY scripts ./scripts
COPY tests ./tests
COPY train.py evaluate.py record_video.py ./

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest"]
