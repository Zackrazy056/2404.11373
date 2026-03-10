# Artifact And Logging Policy

This policy keeps generated outputs reproducible and auditable.

## 1. Output locations

- Figures: `reports/figures/`
- Tables: `reports/tables/`
- Logs: `reports/logs/`
- Large intermediate cache: `data/cache/`

## 2. Naming format

Use:

`<task_id>__<run_id>__<timestamp_utc>.<ext>`

Example:

`T035__kerr220_round3__20260304T121500Z.json`

## 3. Required metadata sidecar

Every generated artifact must have a sidecar metadata file:

`<artifact_name>.meta.json`

Minimum fields:

- `task_id`
- `script`
- `git_commit` (or `"unknown"`)
- `seed`
- `config_path`
- `created_utc`

## 4. Logging levels

- Default: `INFO`
- Debug runs: `DEBUG`
- Warnings/errors must always be written to stderr and log file.

## 5. Reproducibility checklist

Before marking a task as done, keep:

1. Input config snapshot.
2. Seed value.
3. Key command line invocation.
4. Output paths.
