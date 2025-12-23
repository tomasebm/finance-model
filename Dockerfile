FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY daily_merval_signal.py .

CMD ["python", "daily_merval_signal.py"]