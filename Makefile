

install:
	uv sync
run: install
	uv run python -m src
debug:

clean:
	

lint:
	flake8 .
	mypy . --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
fclean: clean

.PHONY=install run debug clean fclean