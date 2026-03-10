# 2404.11373v3 Reproduction

Reproduction workspace for:

- Pacilio, Bhagwat, Cotesta (2024-12-03)
- "Simulation-based inference of black hole ringdowns in the time domain"
- arXiv:2404.11373v3

## Scope

- Injection studies: `Kerr220`, `Kerr221`, `Kerr330`
- Real data analysis: `GW150914`
- Baseline comparison against `pyRing + cpnest`
- Coverage diagnostics and paper-style figures

## Project Layout

```text
configs/           Runtime configuration files
data/              Raw, processed and cached data
docs/              Engineering and reproducibility policies
reports/           Generated figures, tables and logs
scripts/           Task entry-point scripts
src/rd_sbi/        Core package
tests/             Unit tests
```

## Quick Start

1. Create environment:

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

2. Run bootstrap checks:

```bash
python scripts/check_m0.py
```

3. Run tests:

```bash
pytest -q
```

## Current Milestone

- `M0`: completed project bootstrap (scaffold, environment pinning, config + seed, artifact policy)
