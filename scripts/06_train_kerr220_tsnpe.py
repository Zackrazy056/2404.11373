"""Train Kerr220 posterior model from cached index and export posterior samples.

Usage:
    python scripts/06_train_kerr220_tsnpe.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sbi.utils import BoxUniform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config  # noqa: E402
from rd_sbi.inference.embedding_net import EmbeddingConfig, build_nsf_density_estimator  # noqa: E402
from rd_sbi.inference.sbi_loss_patch import NoiseResamplingConfig, NoiseResamplingSNPE  # noqa: E402
from rd_sbi.utils.seed import set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kerr220 SNPE model from cached index")
    parser.add_argument("--index-csv", type=Path, default=Path("data/cache/injections/index.csv"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/tsnpe_nsf.yaml"))
    parser.add_argument("--inj-config", type=Path, default=Path("configs/injections/kerr220.yaml"))
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all train samples")
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/models/kerr220_tsnpe"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=240411373)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--stop-after-epochs", type=int, default=0, help="Override training stop_after_epochs if > 0")
    parser.add_argument("--max-num-epochs", type=int, default=0, help="Override training max_num_epochs if > 0")
    parser.add_argument("--show-progress-bars", action="store_true", help="Enable sbi progress bars")
    parser.add_argument(
        "--sample-source",
        type=str,
        choices=["estimator", "posterior"],
        default="estimator",
        help="Sample from raw density estimator (fast) or built posterior (can be slow).",
    )
    return parser.parse_args()


def _read_index_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _select_kerr220_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    selected = [r for r in rows if r.get("config_name") == "Kerr220" and r.get("split") == split]
    return selected


def _extract_mode_220_from_meta(meta: dict[str, Any], fallback_amp: float, fallback_phase: float) -> tuple[float, float]:
    for mode in meta.get("modes", []):
        if int(mode.get("l", -1)) == 2 and int(mode.get("m", -1)) == 2 and int(mode.get("n", -1)) == 0:
            amp = float(mode.get("amplitude", fallback_amp))
            phase = float(mode.get("phase", fallback_phase))
            return amp, phase
    return fallback_amp, fallback_phase


def _load_xy_theta(
    rows: list[dict[str, str]],
    inj_cfg: dict[str, Any],
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fallback_mode = inj_cfg["modes"][0]
    fallback_amp = float(fallback_mode["amplitude"])
    fallback_phase = float(fallback_mode["phase"])

    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        raise RuntimeError("No rows selected for loading x/theta")

    xs: list[np.ndarray] = []
    thetas: list[np.ndarray] = []

    for row in rows:
        npz_path = Path(row["npz_path"])
        meta_path = Path(row["meta_path"])
        data = np.load(npz_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        x = np.concatenate([data["strain_H1"], data["strain_L1"]], axis=0).astype(np.float32)
        amp220, phase220 = _extract_mode_220_from_meta(meta, fallback_amp=fallback_amp, fallback_phase=fallback_phase)
        theta = np.array(
            [
                float(meta["mass_msun"]),
                float(meta["chi_f"]),
                amp220,
                phase220,
            ],
            dtype=np.float32,
        )
        xs.append(x)
        thetas.append(theta)

    x_tensor = torch.from_numpy(np.stack(xs, axis=0))
    theta_tensor = torch.from_numpy(np.stack(thetas, axis=0))
    return theta_tensor, x_tensor


def _build_prior() -> BoxUniform:
    low = torch.tensor([20.0, 0.0, 0.1e-21, 0.0], dtype=torch.float32)
    high = torch.tensor([300.0, 0.99, 50.0e-21, 2.0 * np.pi], dtype=torch.float32)
    return BoxUniform(low=low, high=high)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    index_path = ROOT / args.index_csv
    model_cfg = load_yaml_config(ROOT / args.model_config)
    inj_cfg = load_yaml_config(ROOT / args.inj_config)
    rows = _read_index_rows(index_path)

    train_rows = _select_kerr220_rows(rows, split="train")
    val_rows = _select_kerr220_rows(rows, split="val")
    if not train_rows:
        raise RuntimeError("No Kerr220 train rows found in cache index")

    theta_train, x_train = _load_xy_theta(train_rows, inj_cfg=inj_cfg, max_samples=args.max_train_samples)
    theta_val, x_val = _load_xy_theta(val_rows if val_rows else train_rows[:1], inj_cfg=inj_cfg, max_samples=1)
    x_observed = x_val[0]

    emb_cfg_dict = model_cfg.get("embedding", {})
    density_cfg = model_cfg.get("density_estimator", {})
    training_cfg = model_cfg.get("training", {})

    embedding_cfg = EmbeddingConfig(
        input_dim=int(emb_cfg_dict.get("input_dim", 408)),
        num_hidden_layers=int(emb_cfg_dict.get("num_hidden_layers", 2)),
        hidden_dim=int(emb_cfg_dict.get("hidden_dim", 150)),
        output_dim=int(emb_cfg_dict.get("output_dim", 128)),
    )
    z_score_theta = density_cfg.get("z_score_theta", "independent")
    theta_std = torch.std(theta_train, dim=0)
    if z_score_theta == "independent" and torch.any(theta_std < 1e-14):
        z_score_theta = None
        print("Detected near-constant theta dimensions; override z_score_theta to None for stability.")

    density_builder = build_nsf_density_estimator(
        embedding_config=embedding_cfg,
        hidden_features=int(density_cfg.get("hidden_features", 150)),
        num_transforms=int(density_cfg.get("num_transforms", 5)),
        num_bins=int(density_cfg.get("num_bins", 10)),
        num_blocks=int(density_cfg.get("num_blocks", 2)),
        batch_norm=bool(density_cfg.get("batch_norm", True)),
        z_score_theta=z_score_theta,
        z_score_x=density_cfg.get("z_score_x", "independent"),
    )

    inference = NoiseResamplingSNPE(
        prior=_build_prior(),
        density_estimator=density_builder,
        device=args.device,
        show_progress_bars=args.show_progress_bars,
        noise_config=NoiseResamplingConfig(enabled=True, noise_std=args.noise_std, apply_in_validation=False),
    )
    inference.append_simulations(theta_train, x_train)
    stop_after = int(training_cfg.get("stop_after_epochs", 20))
    max_epochs = int(training_cfg.get("max_num_epochs", 1000))
    if args.stop_after_epochs > 0:
        stop_after = args.stop_after_epochs
    if args.max_num_epochs > 0:
        max_epochs = args.max_num_epochs

    density_estimator = inference.train(
        training_batch_size=int(training_cfg.get("training_batch_size", 512)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        validation_fraction=float(training_cfg.get("validation_fraction", 0.1)),
        stop_after_epochs=stop_after,
        max_num_epochs=max_epochs,
        show_train_summary=True,
    )
    if args.sample_source == "estimator":
        cond = x_observed.reshape(1, -1)
        samples_tensor = density_estimator.sample((args.posterior_samples,), condition=cond)
        samples = samples_tensor.squeeze(1).detach().cpu().numpy()
    else:
        posterior = inference.build_posterior(sample_with="direct")
        posterior.set_default_x(x_observed)
        samples = posterior.sample((args.posterior_samples,), x=x_observed).detach().cpu().numpy()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "posterior_samples_kerr220.npz"
    np.savez_compressed(
        samples_path,
        samples=samples.astype(np.float32),
        x_observed=x_observed.detach().cpu().numpy().astype(np.float32),
        theta_observed=theta_val[0].detach().cpu().numpy().astype(np.float32),
        theta_train=theta_train.detach().cpu().numpy().astype(np.float32),
    )

    summary = {
        "index_csv": str(index_path),
        "n_train": int(theta_train.shape[0]),
        "n_observed_candidates": int(len(val_rows)),
        "theta_dim": int(theta_train.shape[1]),
        "x_dim": int(x_train.shape[1]),
        "posterior_samples": int(args.posterior_samples),
        "sample_source": args.sample_source,
        "noise_std_train": float(args.noise_std),
        "samples_path": str(samples_path),
    }
    summary_path = out_dir / "train_summary_kerr220.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved posterior samples: {samples_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Train size: {theta_train.shape[0]}; x_dim={x_train.shape[1]}; theta_dim={theta_train.shape[1]}")


if __name__ == "__main__":
    main()
