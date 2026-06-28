.PHONY: install install-all test lint format smoke doctor aim-up aim-down package clean

install:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[all,dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts
	mypy src/behavior2weights

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

smoke:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 b2w smoke --output artifacts/smoke --overwrite

doctor:
	b2w doctor

aim-up:
	docker compose -f infra/aim/docker-compose.yml up -d --build

aim-down:
	docker compose -f infra/aim/docker-compose.yml down

package:
	bash scripts/package_repo.sh

clean:
	rm -rf artifacts/* dist .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
