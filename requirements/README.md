# Dependency snapshots

`verified-cpu-py313.lock` is the exact dependency closure used for the checked-in CPU verification on June 21, 2026. It covers core execution, tests, Ruff, and mypy. It does not claim cross-platform reproducibility for CUDA, Hugging Face, AIM, Slurm, or AWS images.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/verified-cpu-py313.lock
pip install -e . --no-deps
pytest -q
```

For each production study, create and archive a separate fully resolved lock for the selected Python/CUDA platform and extras, plus the immutable container-image digest. Do not reuse the CPU lock for GPU wheels.
