FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Override --paradigm at runtime (docker-compose / docker run).
CMD ["python", "main.py", "--paradigm", "hardcoded"]
