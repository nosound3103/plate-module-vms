# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes -r requirements.lock \
    && useradd --uid 10001 --create-home plate

COPY --chown=plate:plate plate_ai ./plate_ai
USER plate
CMD ["python", "-m", "plate_ai"]
