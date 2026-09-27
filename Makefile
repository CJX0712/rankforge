# RankForge Makefile
PY ?= python
VENV ?= .venv

.PHONY: venv install demo test ci clean

venv:
	$(PY) -m venv $(VENV)

install:
	$(VENV)/Scripts/python.exe -m pip install --upgrade pip
	$(VENV)/Scripts/python.exe -m pip install -r requirements.txt

demo:
	$(VENV)/Scripts/python.exe examples/run_demo.py

test:
	$(VENV)/Scripts/python.exe -m pytest -q -W ignore::UserWarning

ci:
	$(VENV)/Scripts/python.exe -m py_compile rankforge tests examples
	$(VENV)/Scripts/python.exe -m pytest -q -W ignore::UserWarning
	$(VENV)/Scripts/python.exe cli.py bench --queries 20 --seeds 2

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache benchmark.json
