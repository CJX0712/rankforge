FROM python:3.13-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Smoke: run the bundled benchmark (writes benchmark.json)
RUN python examples/run_demo.py

CMD ["python", "examples/run_demo.py"]
