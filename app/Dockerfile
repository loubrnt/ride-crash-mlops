# Web app image (build from the repository root):
#   docker build -f app/Dockerfile -t loubrunet/ride-crash-app .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/streamlit_app.py ./app/streamlit_app.py
COPY samples ./samples

RUN useradd --create-home appuser
USER appuser

ENV API_URL=http://api:8000 SAMPLES_DIR=/app/samples
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
