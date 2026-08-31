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


# Fit a model and write models/{league}.json. Reads the bucket by default,
# which needs AWS_PROFILE and ~/.aws-batch/config.json; pass `--root` for a
# local copy of the processed tree, e.g.
#   make train ARGS="--league nfl --seasons 2022-2025 --root ./plays"
ARGS ?=
train:
	uv run python train.py train $(ARGS)


# One game's curve and its game control, as JSON. The eyeball check on a
# fresh release: `make curve ARGS="401671789 --week 3"`.
curve:
	uv run python train.py curve $(ARGS)


# A local copy of the processed play-by-play, so a fit can be re-run without
# credentials. The prefix is endgame's; BUCKET comes from the same config the
# store reads.
BUCKET ?= $(shell jq -r .bucket.value $(HOME)/.aws-batch/config.json)
plays:
	aws s3 sync s3://${BUCKET}/processed/plays ./plays


.PHONY: lint check test train curve plays
