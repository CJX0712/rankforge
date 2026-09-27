FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest ruff

COPY . .

# 单元冒烟 + 端到端演示（落盘 benchmark.json）
RUN python -m pytest -q -W ignore::UserWarning \
 && python -m rankforge.examples.run_demo --quick --out benchmark.json

CMD ["python", "-m", "rankforge.cli", "--demo"]
