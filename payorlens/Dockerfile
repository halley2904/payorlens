FROM python:3.11-slim


#RUN apt-get update && apt-get install -y --no-install-recommends \
#   libgomp1 \
#   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


RUN mkdir -p data/raw/cms data/processed reports

ENV PAYORLENS_PIPELINE_DIR=/app \
    PAYORLENS_RAW_DIR=/app/data/raw/cms \
    PAYORLENS_PROC_DIR=/app/data/processed \
    PAYORLENS_REPORTS_DIR=/app/reports \
    PAYORLENS_DB_PATH=/app/data/payorlens_runs.db \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "payorlens_api:app", "--host", "0.0.0.0", "--port", "8000"]
