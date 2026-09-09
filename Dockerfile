# Один образ на два контейнера: плейграунд и штатный сервис MagicBrain API.
# Так они гарантированно работают на одной и той же ревизии библиотеки, и
# модель, обученная плейграундом, читается рантаймом без сюрпризов.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_STORAGE_PATH=/data/models \
    MBP_DATA_DIR=/data \
    MBP_CORPORA_DIR=/corpora

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Снимок MagicBrain фиксированной ревизии кладёт scripts/prepare_build.sh.
COPY build/magicbrain-src/ /src/magicbrain/
# balansis ставится релизом с PyPI, а не редактируемой копией из лаборатории:
# иначе «сравнение с ACT» зависело бы от чужого рабочего дерева.
RUN pip install --no-cache-dir "/src/magicbrain[service]" "balansis==1.1.0" \
    && cp -r /src/magicbrain/api /app/api \
    && rm -rf /src/magicbrain

COPY mbplay/ /app/mbplay/

RUN useradd --system --uid 10001 --create-home mbplay \
    && mkdir -p /data/models /data/runs \
    && chown -R mbplay:mbplay /app /data
USER mbplay

EXPOSE 8000
CMD ["uvicorn", "mbplay.app:app", "--host", "0.0.0.0", "--port", "8000"]
