.PHONY: test demo lint

test:
	python3 -m pytest -q

demo:
	python3 examples/run_demo.py

lint:
	python3 -m compileall -q src tests examples
