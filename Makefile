.PHONY: help install test lint demo ci lock

help:
	@echo "RankForge — make 目标"
	@echo "  install  安装依赖（建议先用 venv）"
	@echo "  test     pytest 单测"
	@echo "  lint     ruff 静态检查"
	@echo "  demo     端到端演示 -> benchmark.json"
	@echo "  ci       lint + test + demo"
	@echo "  lock     重新冻结 requirements.lock.txt"

install:
	pip install -r requirements.txt pytest ruff

test:
	python -m pytest -q -W ignore::UserWarning

lint:
	ruff check .

demo:
	python -m rankforge.examples.run_demo --out benchmark.json

ci: lint test demo

lock:
	pip freeze > requirements.lock.txt
