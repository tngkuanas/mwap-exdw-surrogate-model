# Reproducibility Guide: Production-Anchored Conditional Model Research

To reproduce all experiments, tables, figures, and verification audits:

```bash
# 1. Master experiment execution
PYTHONPATH=src ../.venv/bin/python src/conditional_challengers/run_conditional_experiments.py

# 2. Unit and integration tests
PYTHONPATH=src ../.venv/bin/pytest tests/test_conditional_challengers.py -v

# 3. Full codebase regression suite
PYTHONPATH=src ../.venv/bin/pytest tests/ -v
```
