# Contributing

Use Python 3.11, install `.[dev]`, and run `ruff check .`, `pytest`, and the CPU smoke command before
a pull request. Scientific changes must include a test and explain whether they alter a frozen
estimand, split, endpoint, exclusion, or multiplicity family. Never change a locked confirmatory
configuration in place; create a new study version.
