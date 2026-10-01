FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
# Hugging Face runs the container as uid 1000; sessions write temp files here.
RUN chmod -R 777 /app

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    AGENT_HEADLESS=1 \
    AGENT_REQUIRE_AUTH=1

EXPOSE 7860
CMD ["python", "server.py"]
