# `uv run` syncs the environment from uv.lock before it runs anything, so
# these targets work from a fresh checkout with no install step in front of
# them. That is also why nothing here activates a venv: uv owns .venv.
lint:
	uv run ruff check --fix .
	uv run ruff format .
	uv run ty check .


# Same checks as `lint`, but reports instead of fixing (what CI runs)
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .


test:
	uv run pytest .


.PHONY: lint check test
