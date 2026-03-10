"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PAPER_CASES = ("kerr220", "kerr221", "kerr330")
PAPER_MODE_SETS = {
    "kerr220": {(2, 2, 0)},
    "kerr221": {(2, 2, 0), (2, 2, 1)},
    "kerr330": {(2, 2, 0), (3, 3, 0)},
}
PAPER_MODE_PARAMS = {
    "kerr220": {(2, 2, 0): {"amplitude": 5.0e-21, "phase": 1.047}},
    "kerr221": {(2, 2, 0): {"amplitude": 8.92e-21, "phase": 1.047}, (2, 2, 1): {"amplitude": 9.81e-21, "phase": 4.19}},
    "kerr330": {(2, 2, 0): {"amplitude": 30.0e-21, "phase": 1.047}, (3, 3, 0): {"amplitude": 3.0e-21, "phase": 5.014}},
}
PAPER_SOURCE = {
    "ra_rad": 1.95,
    "dec_rad": -1.27,
    "psi_rad": 0.82,
    "gps_h1": 1126259462.42323,
}
PAPER_INCLINATION = {
    "kerr220": 3.141592653589793,
    "kerr221": 3.141592653589793,
    "kerr330": 0.7853981633974483,
}
PAPER_TARGET_SNR = {
    "kerr220": 14.0,
    "kerr221": 14.0,
    "kerr330": 53.0,
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config into a dictionary.

    Parameters
    ----------
    path:
        File path to a YAML file.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at root of {config_path}, got {type(data)!r}")
    return data


def project_root_from_file(file_path: str | Path) -> Path:
    """Infer project root as two levels above a file under src/rd_sbi."""
    return Path(file_path).resolve().parents[2]


def validate_paper_case_config(cfg: dict[str, Any], case: str | None = None) -> list[str]:
    """Return strict paper-case mismatches for Table I style injection configs.

    This validates the fixed injection and run-contract fields that should not
    drift when reproducing Fig.1 cases.
    """

    inferred_case = str(case or cfg.get("name", "")).strip().lower()
    if inferred_case not in PAPER_CASES:
        return [f"unknown_paper_case:{inferred_case or '<missing>'}"]

    issues: list[str] = []
    remnant = cfg.get("remnant", {})
    data = cfg.get("data", {})
    source = cfg.get("source", {})
    qnm = cfg.get("qnm", {})

    if abs(float(remnant.get("mass_msun", float("nan"))) - 67.0) >= 1e-9:
        issues.append(f"mass_msun={remnant.get('mass_msun')}")
    if abs(float(remnant.get("chi_f", float("nan"))) - 0.67) >= 1e-9:
        issues.append(f"chi_f={remnant.get('chi_f')}")
    if abs(float(data.get("sample_rate_hz", float("nan"))) - 2048.0) >= 1e-9:
        issues.append(f"sample_rate_hz={data.get('sample_rate_hz')}")
    if abs(float(data.get("duration_s", float("nan"))) - 0.1) >= 1e-9:
        issues.append(f"duration_s={data.get('duration_s')}")
    if abs(float(data.get("t_start_s", float("nan"))) - 0.0) >= 1e-9:
        issues.append(f"t_start_s={data.get('t_start_s')}")

    for key, expected in PAPER_SOURCE.items():
        value = float(source.get(key, float("nan")))
        if abs(value - expected) >= 1e-9:
            issues.append(f"{key}={source.get(key)}")
    if abs(float(source.get("inclination_rad", float("nan"))) - PAPER_INCLINATION[inferred_case]) >= 1e-9:
        issues.append(f"inclination_rad={source.get('inclination_rad')}")

    modes = cfg.get("modes", [])
    mode_keys = {(int(m["l"]), int(m["m"]), int(m["n"])) for m in modes}
    if mode_keys != PAPER_MODE_SETS[inferred_case]:
        issues.append(f"modes={sorted(mode_keys)}")
    mode_lookup = {(int(m["l"]), int(m["m"]), int(m["n"])): m for m in modes}
    for mode_id, expected in PAPER_MODE_PARAMS[inferred_case].items():
        mode = mode_lookup.get(mode_id, {})
        if abs(float(mode.get("amplitude", float("nan"))) - expected["amplitude"]) >= 1e-24:
            issues.append(f"{mode_id}.amplitude={mode.get('amplitude')}")
        if abs(float(mode.get("phase", float("nan"))) - expected["phase"]) >= 1e-9:
            issues.append(f"{mode_id}.phase={mode.get('phase')}")

    if tuple(cfg.get("detectors", [])) != ("H1", "L1"):
        issues.append(f"detectors={cfg.get('detectors')}")
    if bool(cfg.get("use_detector_time_delay", False)) is not True:
        issues.append(f"use_detector_time_delay={cfg.get('use_detector_time_delay')}")
    if str(cfg.get("reference_detector", "")) != "H1":
        issues.append(f"reference_detector={cfg.get('reference_detector')}")
    if abs(float(cfg.get("noise_std", float("nan"))) - 0.0) >= 1e-12:
        issues.append(f"noise_std={cfg.get('noise_std')}")
    if abs(float(cfg.get("target_snr", float("nan"))) - PAPER_TARGET_SNR[inferred_case]) >= 1e-9:
        issues.append(f"target_snr={cfg.get('target_snr')}")

    method = str(qnm.get("method", "")).strip().lower()
    if method != "fit":
        issues.append(f"qnm.method={qnm.get('method')}")
    if abs(float(qnm.get("alpha_r", float("nan"))) - 0.0) >= 1e-12:
        issues.append(f"qnm.alpha_r={qnm.get('alpha_r')}")
    if abs(float(qnm.get("alpha_i", float("nan"))) - 0.0) >= 1e-12:
        issues.append(f"qnm.alpha_i={qnm.get('alpha_i')}")
    if int(qnm.get("spin_weight", 0)) != -2:
        issues.append(f"qnm.spin_weight={qnm.get('spin_weight')}")

    return issues
