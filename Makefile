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


# What the wheel actually contains, which is the check that the fits in
# lucky_ones/releases are still in it. hatchling reads .gitignore when it
# decides, so a fit that stopped being tracked would also stop shipping.
wheel:
	uv build --wheel --out-dir dist
	@python -c "import zipfile, sys; \
	  names = zipfile.ZipFile(sorted(__import__('glob').glob('dist/*.whl'))[-1]).namelist(); \
	  print('\n'.join(sorted(n for n in names if n.endswith('.json'))) or 'NO FITS IN THE WHEEL'); \
	  sys.exit(0 if any(n.endswith('.json') for n in names) else 1)"


# Fit a model and rewrite lucky_ones/releases/{league}.json -- the fit that
# ships in the package, so commit what this changes. Reads the bucket by
# default, which needs AWS_PROFILE and ~/.aws-batch/config.json; pass `--root`
# for a local copy of the processed tree, e.g.
#   make train ARGS="--league nfl --seasons 2022-2025 --root ./plays"
ARGS ?=
train:
	uv run python train.py train $(ARGS)


# The other fit: expected points, written to
# lucky_ones/releases/{league}-ep.json. Half of what `epa` needs, and a
# separate file so a retrain of one model is a diff of one model.
train-ep:
	uv run python train.py train_ep $(ARGS)


# The rates behind DEFAULT_RETAINED, per season, as JSON -- what a `retained`
# is worth arguing about and what it isn't:
#   make rates ARGS="--league ncaafb --seasons 2006-2025 --root ./plays"
rates:
	uv run python train.py rates $(ARGS)


# One game's curve and its game control, as JSON, through the bundled fit --
# the eyeball check on a release: `make curve ARGS="401671789 --week 3"`.
# `--model PATH` reads one from a file instead.
curve:
	uv run python train.py curve $(ARGS)


# The same eyeball check for the expected points fit: one game's EPA per play
# and every snap behind it. `make epa ARGS="401671789 --week 3"`.
epa:
	uv run python train.py epa $(ARGS)


# The EPA distribution behind DEFAULT_CLIP and DEFAULT_WEIGHT_POWER, per
# season -- where the tail is, how much of a season the weighting keeps, and
# whether either changes a game's number:
#   make bounds ARGS="--league nfl --seasons 2009-2025 --root ./plays"
bounds:
	uv run python train.py bounds $(ARGS)


# Are the models any good? Refits both on 2014-2022 and reports calibration,
# skill and split-half stability on seasons neither fit has seen, writing the
# JSON and the charts that docs/epa-validation.md is a view of:
#   make validate ARGS="--league nfl --root ./plays"
# Rerun it for both leagues when a release or a feature list changes.
validate:
	uv run python validate.py validate $(ARGS)


# A local copy of the processed play-by-play, so a fit can be re-run without
# credentials. The prefix is endgame's; BUCKET comes from the same config the
# store reads.
BUCKET ?= $(shell jq -r .bucket.value $(HOME)/.aws-batch/config.json)
plays:
	aws s3 sync s3://${BUCKET}/processed/plays ./plays


.PHONY: lint check test wheel train train-ep curve epa rates bounds validate plays
