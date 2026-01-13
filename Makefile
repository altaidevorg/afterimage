.PHONY: docs clean

docs:
	uv run sphinx-build -b html docs docs/_build/html

clean:
	rm -rf docs/_build
