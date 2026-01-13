@ECHO OFF

IF "%1" == "docs" (
	uv run sphinx-build -b html docs docs/_build/html
	GOTO :EOF
)

IF "%1" == "clean" (
	rmdir /s /q docs\_build
	GOTO :EOF
)

ECHO Usage: make docs
