.PHONY: install test demo demo-real

REAL_OUT ?= .demo/real_traces.json

install:
	python3 -m venv .venv && .venv/bin/pip install -e '.[api,dev,demo]'
test:
	.venv/bin/python -m pytest
demo:
	.venv/bin/faultgate check examples/traces.json --judge laya

# Downloads 9 public dataset shards (~220 MB) on first run; cached after
demo-real:
	@.venv/bin/python -c 'import faultgate, pyarrow, huggingface_hub' 2>/dev/null || $(MAKE) install
	@mkdir -p $$(dirname $(REAL_OUT))
	PATH="$(CURDIR)/.venv/bin:$$PATH" python scripts/fetch_real_traces.py --n 30 --out $(REAL_OUT)
	PATH="$(CURDIR)/.venv/bin:$$PATH" faultgate check $(REAL_OUT) || [ $$? -eq 1 ]
