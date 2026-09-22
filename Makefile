.PHONY: install test demo
install:
	python3 -m venv .venv && .venv/bin/pip install -e '.[api,dev]'
test:
	.venv/bin/python -m pytest
demo:
	.venv/bin/faultgate check examples/traces.json --judge laya
