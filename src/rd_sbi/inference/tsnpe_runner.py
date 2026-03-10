"""Truncated Sequential Neural Posterior Estimation (TSNPE) runner."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable
import time
import threading

import torch
from sbi.inference import SNPE
from torch import Tensor
from torch.distributions import Distribution

from .sbi_loss_patch import NoiseResamplingConfig, NoiseResamplingSNPE


@dataclass(frozen=True)
class TSNPEConfig:
    """TSNPE loop configuration."""

    num_simulations_first_round: int = 50_000
    num_simulations_per_round: int = 100_000
    max_rounds: int = 8

    trunc_quantile: float = 1e-4
    stopping_ratio: float = 0.8

    posterior_samples_for_hpd: int = 8_192
    prior_volume_mc_samples: int = 16_384
    rejection_candidate_batch: int = 16_384
    rejection_max_batches: int = 1_024

    training_batch_size: int = 512
    learning_rate: float = 1e-3
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    max_num_epochs: int = 1_000
    show_train_summary: bool = False

    discard_round1_after_first_update: bool = True
    device: str = "cpu"
    show_progress_bars: bool = True
    log_round_timing: bool = True
    truncation_device: str = "cpu"
    truncation_probe_samples: int = 16_384
    min_rounds_before_stopping: int = 3
    max_volume_for_stopping: float = 0.95
    require_volume_shrink_for_stopping: bool = True
    fail_on_no_truncation_for_next_round: bool = False
    no_truncation_volume_threshold: float = 0.999
    varying_noise_enabled: bool = False
    varying_noise_std: float = 1.0
    varying_noise_apply_in_validation: bool = False
    resume_training_state_across_rounds: bool = False
    auto_reduce_round_simulations_on_low_acceptance: bool = True
    rejection_acceptance_safety_factor: float = 0.8
    min_simulations_per_round_after_reduction: int = 4_096
    adaptive_truncation_relaxation_enabled: bool = True
    min_probe_acceptance_rate_for_next_round: float = 0.002


@dataclass(frozen=True)
class RoundSimulationBudget:
    """Planned vs feasible simulation budget for a round."""

    requested_num_simulations: int
    effective_num_simulations: int
    estimated_rejection_acceptance_rate: float | None
    rejection_budget_max_draws: int | None
    estimated_max_accepted_under_budget: int | None
    simulation_budget_adjusted: bool
    simulation_budget_adjustment_reason: str | None


@dataclass(frozen=True)
class TruncationPlan:
    """Chosen truncation threshold and probe diagnostics for a round."""

    threshold: float
    probe_acceptance_rate: float
    probe_logp_q01: float
    probe_logp_q50: float
    probe_logp_q99: float
    truncation_relaxed: bool
    truncation_relaxation_reason: str | None
    target_min_probe_acceptance_rate: float | None


@dataclass(frozen=True)
class RoundDiagnostics:
    """Per-round diagnostics for truncation and stopping."""

    round_index: int
    num_simulations: int
    requested_num_simulations: int
    hpd_log_prob_threshold: float
    truncated_prior_volume: float
    acceptance_rate_in_round: float
    stop_by_ratio: bool
    simulation_seconds: float
    training_seconds: float
    truncation_seconds: float
    round_total_seconds: float
    volume_ratio_to_previous: float | None
    stop_eligible: bool
    stop_reason: str
    probe_samples: int
    probe_acceptance_rate: float
    probe_logp_q01: float
    probe_logp_q50: float
    probe_logp_q99: float
    appended_simulations: int
    inferred_num_epochs: int | None
    epochs_trained_this_round: int | None
    estimated_rejection_acceptance_rate: float | None
    rejection_budget_max_draws: int | None
    estimated_max_accepted_under_budget: int | None
    simulation_budget_adjusted: bool
    simulation_budget_adjustment_reason: str | None
    truncation_relaxed: bool
    truncation_relaxation_reason: str | None
    target_min_probe_acceptance_rate: float | None


def should_stop_by_volume_ratio(previous_volume: float, current_volume: float, stopping_ratio: float) -> bool:
    """Implements stopping criterion from the paper.

    Stop when current truncated volume is larger than `stopping_ratio` times
    the previous truncated volume, i.e. little additional truncation gain.
    """
    if previous_volume <= 0.0:
        return False
    return bool((current_volume / previous_volume) > stopping_ratio)


class TSNPERunner:
    """Round-based TSNPE trainer with explicit truncation and stopping."""

    def __init__(
        self,
        *,
        prior: Distribution,
        simulator: Callable[[Tensor], Tensor],
        x_observed: Tensor,
        density_estimator_builder,
        config: TSNPEConfig | None = None,
        heartbeat_callback: Callable[[dict[str, Any]], None] | None = None,
        heartbeat_interval_seconds: float = 0.0,
    ) -> None:
        self.prior = prior
        self.simulator = simulator
        self.x_observed = x_observed.detach().clone()
        self.density_estimator_builder = density_estimator_builder
        self.config = config or TSNPEConfig()
        self.heartbeat_callback = heartbeat_callback
        self.heartbeat_interval_seconds = max(float(heartbeat_interval_seconds), 0.0)

        if self.config.num_simulations_first_round <= 0 or self.config.num_simulations_per_round <= 0:
            raise ValueError("number of simulations per round must be positive")
        if not (0.0 < self.config.trunc_quantile < 1.0):
            raise ValueError("trunc_quantile must satisfy 0 < trunc_quantile < 1")
        if self.config.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.config.truncation_probe_samples <= 0:
            raise ValueError("truncation_probe_samples must be positive")
        if self.config.min_rounds_before_stopping < 1:
            raise ValueError("min_rounds_before_stopping must be >= 1")
        if not (0.0 < self.config.max_volume_for_stopping <= 1.0):
            raise ValueError("max_volume_for_stopping must satisfy 0 < value <= 1")
        if not (0.0 < self.config.no_truncation_volume_threshold <= 1.0):
            raise ValueError("no_truncation_volume_threshold must satisfy 0 < value <= 1")
        if not (0.0 < self.config.rejection_acceptance_safety_factor <= 1.0):
            raise ValueError("rejection_acceptance_safety_factor must satisfy 0 < value <= 1")
        if self.config.min_simulations_per_round_after_reduction <= 0:
            raise ValueError("min_simulations_per_round_after_reduction must be positive")
        if not (0.0 < self.config.min_probe_acceptance_rate_for_next_round < 1.0):
            raise ValueError("min_probe_acceptance_rate_for_next_round must satisfy 0 < value < 1")
        self.last_density_estimator = None
        self.last_diagnostics: list[RoundDiagnostics] = []
        self._status_lock = threading.Lock()
        self.current_status: dict[str, Any] = {
            "state": "initialized",
            "device": self.config.device,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
        }

    def _snapshot_status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self.current_status)

    def _emit_status(self, **kwargs: Any) -> None:
        with self._status_lock:
            self.current_status.update(kwargs)
            self.current_status["timestamp"] = time.time()
            payload = dict(self.current_status)
        if self.heartbeat_callback is not None:
            self.heartbeat_callback(payload)

    @contextmanager
    def _phase_heartbeat(self, *, round_index: int, phase: str, num_simulations: int) -> Any:
        phase_start = time.time()
        stop_event = threading.Event()

        def _beat() -> None:
            while not stop_event.wait(self.heartbeat_interval_seconds):
                self._emit_status(
                    state="running",
                    event="heartbeat",
                    round_index=round_index,
                    phase=phase,
                    num_simulations=num_simulations,
                    phase_elapsed_seconds=time.time() - phase_start,
                )

        self._emit_status(
            state="running",
            event="phase_start",
            round_index=round_index,
            phase=phase,
            num_simulations=num_simulations,
            phase_elapsed_seconds=0.0,
        )
        worker: threading.Thread | None = None
        if self.heartbeat_callback is not None and self.heartbeat_interval_seconds > 0.0:
            worker = threading.Thread(target=_beat, daemon=True)
            worker.start()
        try:
            yield
        finally:
            stop_event.set()
            if worker is not None:
                worker.join(timeout=0.2)
            self._emit_status(
                state="running",
                event="phase_end",
                round_index=round_index,
                phase=phase,
                num_simulations=num_simulations,
                phase_elapsed_seconds=time.time() - phase_start,
            )

    def _sample_prior(self, n: int) -> Tensor:
        theta = self.prior.sample((n,))
        return theta.detach()

    def _proposal_marker_for_round(self, round_idx: int):
        if round_idx <= 1:
            return None
        # Mark later TSNPE rounds as proposal-driven for `sbi` bookkeeping
        # (dataset round indices / discard_prior_samples) while still training
        # with explicit truncated-prior samples and first-round MLE loss.
        return copy.deepcopy(self.prior)

    def _simulate(self, theta: Tensor) -> Tensor:
        x = self.simulator(theta)
        if x.shape[0] != theta.shape[0]:
            raise ValueError(f"simulator output batch {x.shape[0]} does not match theta batch {theta.shape[0]}")
        return x.detach()

    def _maybe_sync_cuda(self) -> None:
        if str(self.config.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _condition_for_estimator(self, density_estimator) -> Tensor:
        try:
            device = next(density_estimator.parameters()).device
        except StopIteration:
            device = self.x_observed.device
        return self.x_observed.reshape(1, -1).to(device)

    def _estimator_for_truncation(self, density_estimator):
        mode = str(self.config.truncation_device).lower()
        if mode == "cpu":
            # Keep training on original device, but evaluate HPD threshold/volume on CPU
            # to avoid occasional CUDA-side spline sampling instability.
            est = copy.deepcopy(density_estimator).to("cpu")
            est.eval()
            return est
        return density_estimator

    def _cpu_safe_estimator_copy(self):
        if self.last_density_estimator is None:
            raise RuntimeError("TSNPE produced no density estimator")
        est = copy.deepcopy(self.last_density_estimator).to("cpu")
        est.eval()
        return est

    def _estimator_log_prob(self, density_estimator, theta: Tensor) -> Tensor:
        cond = self._condition_for_estimator(density_estimator)
        theta = theta.to(cond.device)
        theta_in = theta.unsqueeze(1)
        logp = density_estimator.log_prob(theta_in, condition=cond)
        if logp.ndim == 2:
            return logp[:, 0]
        return logp

    def _truncation_probe(self, density_estimator, threshold: float, sample_count: int) -> tuple[float, float, float, float]:
        theta_probe = self._sample_prior(sample_count)
        log_prob_probe = self._estimator_log_prob(density_estimator, theta_probe)
        return self._truncation_probe_from_log_prob(log_prob_probe, threshold)

    def _truncation_probe_from_log_prob(self, log_prob_probe: Tensor, threshold: float) -> tuple[float, float, float, float]:
        keep = log_prob_probe >= threshold
        acceptance = float(keep.float().mean().item())
        q = torch.tensor([0.01, 0.50, 0.99], device=log_prob_probe.device, dtype=log_prob_probe.dtype)
        q01, q50, q99 = torch.quantile(log_prob_probe, q).detach().cpu().tolist()
        return acceptance, float(q01), float(q50), float(q99)

    def _posterior_threshold_for_hpd(self, density_estimator) -> float:
        cond = self._condition_for_estimator(density_estimator)
        theta_post = density_estimator.sample((self.config.posterior_samples_for_hpd,), condition=cond).squeeze(1)
        log_prob_post = self._estimator_log_prob(density_estimator, theta_post)
        threshold = torch.quantile(log_prob_post, self.config.trunc_quantile)
        return float(threshold.item())

    def _estimate_truncated_prior_volume(self, density_estimator, threshold: float) -> float:
        volume, _, _, _ = self._truncation_probe(
            density_estimator,
            threshold=threshold,
            sample_count=self.config.prior_volume_mc_samples,
        )
        return volume

    def _build_truncation_plan_from_log_prob(
        self,
        *,
        threshold: float,
        log_prob_probe: Tensor,
        round_idx: int,
    ) -> TruncationPlan:
        probe_acceptance, q01, q50, q99 = self._truncation_probe_from_log_prob(log_prob_probe, threshold)
        truncation_relaxed = False
        relaxation_reason: str | None = None
        target_min_probe_acceptance_rate: float | None = None

        if (
            round_idx > 1
            and self.config.adaptive_truncation_relaxation_enabled
            and probe_acceptance < self.config.min_probe_acceptance_rate_for_next_round
        ):
            target_acceptance = float(self.config.min_probe_acceptance_rate_for_next_round)
            relaxed_threshold = torch.quantile(
                log_prob_probe,
                max(0.0, min(1.0, 1.0 - target_acceptance)),
            )
            threshold = min(float(threshold), float(relaxed_threshold.item()))
            probe_acceptance, q01, q50, q99 = self._truncation_probe_from_log_prob(log_prob_probe, threshold)
            truncation_relaxed = True
            target_min_probe_acceptance_rate = target_acceptance
            relaxation_reason = (
                "relaxed_to_probe_acceptance_floor:"
                f"target_min_accept={target_acceptance:.6f},"
                f"observed_accept={probe_acceptance:.6f}"
            )

        return TruncationPlan(
            threshold=float(threshold),
            probe_acceptance_rate=probe_acceptance,
            probe_logp_q01=q01,
            probe_logp_q50=q50,
            probe_logp_q99=q99,
            truncation_relaxed=truncation_relaxed,
            truncation_relaxation_reason=relaxation_reason,
            target_min_probe_acceptance_rate=target_min_probe_acceptance_rate,
        )

    def _build_truncation_plan(self, density_estimator, *, round_idx: int) -> TruncationPlan:
        threshold = self._posterior_threshold_for_hpd(density_estimator)
        theta_probe = self._sample_prior(self.config.truncation_probe_samples)
        log_prob_probe = self._estimator_log_prob(density_estimator, theta_probe)
        return self._build_truncation_plan_from_log_prob(
            threshold=threshold,
            log_prob_probe=log_prob_probe,
            round_idx=round_idx,
        )

    def _rejection_sample_truncated_prior(self, density_estimator, threshold: float, n_target: int) -> tuple[Tensor, float]:
        accepted: list[Tensor] = []
        accepted_count = 0
        total_drawn = 0

        for _ in range(self.config.rejection_max_batches):
            theta_cand = self._sample_prior(self.config.rejection_candidate_batch)
            logp = self._estimator_log_prob(density_estimator, theta_cand)
            keep = logp >= threshold
            kept = theta_cand[keep]

            total_drawn += theta_cand.shape[0]
            if kept.shape[0] > 0:
                accepted.append(kept)
                accepted_count += kept.shape[0]
            if accepted_count >= n_target:
                break

        if accepted_count < n_target:
            raise RuntimeError(
                f"TSNPE rejection sampler could not collect {n_target} points "
                f"(got {accepted_count}, total_drawn={total_drawn})."
            )

        theta_out = torch.cat(accepted, dim=0)[:n_target]
        acceptance_rate = accepted_count / max(total_drawn, 1)
        return theta_out, float(acceptance_rate)

    def _plan_round_simulation_budget(
        self,
        *,
        requested_num_simulations: int,
        estimated_acceptance_rate: float | None,
    ) -> RoundSimulationBudget:
        if requested_num_simulations <= 0:
            raise ValueError("requested_num_simulations must be positive")
        if estimated_acceptance_rate is None:
            return RoundSimulationBudget(
                requested_num_simulations=requested_num_simulations,
                effective_num_simulations=requested_num_simulations,
                estimated_rejection_acceptance_rate=None,
                rejection_budget_max_draws=None,
                estimated_max_accepted_under_budget=None,
                simulation_budget_adjusted=False,
                simulation_budget_adjustment_reason=None,
            )

        acceptance = max(float(estimated_acceptance_rate), 0.0)
        max_draws = int(self.config.rejection_candidate_batch * self.config.rejection_max_batches)
        estimated_max_accepted = int(max_draws * acceptance * self.config.rejection_acceptance_safety_factor)
        effective_num_simulations = requested_num_simulations
        adjusted = False
        reason: str | None = None

        if (
            self.config.auto_reduce_round_simulations_on_low_acceptance
            and estimated_max_accepted < requested_num_simulations
        ):
            if estimated_max_accepted < self.config.min_simulations_per_round_after_reduction:
                raise RuntimeError(
                    "TSNPE rejection budget infeasible for next round: "
                    f"requested={requested_num_simulations}, "
                    f"estimated_max_accepted={estimated_max_accepted}, "
                    f"acceptance_estimate={acceptance:.6f}, "
                    f"max_draws={max_draws}. "
                    "Increase rejection budget or relax truncation."
                )
            effective_num_simulations = estimated_max_accepted
            adjusted = True
            reason = (
                "reduced_to_fit_rejection_budget:"
                f"requested={requested_num_simulations},"
                f"effective={effective_num_simulations},"
                f"acceptance_estimate={acceptance:.6f},"
                f"max_draws={max_draws},"
                f"safety_factor={self.config.rejection_acceptance_safety_factor:.3f}"
            )

        return RoundSimulationBudget(
            requested_num_simulations=requested_num_simulations,
            effective_num_simulations=effective_num_simulations,
            estimated_rejection_acceptance_rate=acceptance,
            rejection_budget_max_draws=max_draws,
            estimated_max_accepted_under_budget=estimated_max_accepted,
            simulation_budget_adjusted=adjusted,
            simulation_budget_adjustment_reason=reason,
        )

    def sample_posterior(self, num_samples: int, *, cpu_safe: bool = False) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.last_density_estimator is None:
            raise RuntimeError("TSNPE produced no density estimator")

        density_estimator = self._cpu_safe_estimator_copy() if cpu_safe else self.last_density_estimator
        cond = self._condition_for_estimator(density_estimator)
        with torch.no_grad():
            samples = density_estimator.sample((num_samples,), condition=cond).squeeze(1)
        return samples.detach().cpu()

    def run(self) -> tuple[object, list[RoundDiagnostics]]:
        """Run TSNPE training rounds and return final posterior + diagnostics."""
        self._emit_status(state="starting", event="run_start", round_index=0, phase="setup", num_simulations=0)
        if self.config.varying_noise_enabled:
            inference = NoiseResamplingSNPE(
                prior=self.prior,
                density_estimator=self.density_estimator_builder,
                device=self.config.device,
                show_progress_bars=self.config.show_progress_bars,
                noise_config=NoiseResamplingConfig(
                    enabled=True,
                    noise_std=self.config.varying_noise_std,
                    apply_in_validation=self.config.varying_noise_apply_in_validation,
                ),
            )
        else:
            inference = SNPE(
                prior=self.prior,
                density_estimator=self.density_estimator_builder,
                device=self.config.device,
                show_progress_bars=self.config.show_progress_bars,
            )

        diagnostics: list[RoundDiagnostics] = []
        previous_volume: float | None = None
        previous_probe_acceptance: float | None = None
        posterior = None
        threshold_for_next_round: float | None = None

        for round_idx in range(1, self.config.max_rounds + 1):
            round_start = time.time()
            budget_plan = RoundSimulationBudget(
                requested_num_simulations=self.config.num_simulations_first_round,
                effective_num_simulations=self.config.num_simulations_first_round,
                estimated_rejection_acceptance_rate=None,
                rejection_budget_max_draws=None,
                estimated_max_accepted_under_budget=None,
                simulation_budget_adjusted=False,
                simulation_budget_adjustment_reason=None,
            )
            if round_idx == 1:
                n_sim = self.config.num_simulations_first_round
                theta = self._sample_prior(n_sim)
                acceptance_rate = 1.0
            else:
                if self.last_density_estimator is None or threshold_for_next_round is None:
                    raise RuntimeError("Missing estimator/threshold for round > 1")
                if (
                    self.config.fail_on_no_truncation_for_next_round
                    and previous_volume is not None
                    and previous_volume >= self.config.no_truncation_volume_threshold
                ):
                    raise RuntimeError(
                        "Aborting next round: previous round produced no effective truncation "
                        f"(volume={previous_volume:.6f} >= {self.config.no_truncation_volume_threshold:.6f})."
                    )
                budget_plan = self._plan_round_simulation_budget(
                    requested_num_simulations=self.config.num_simulations_per_round,
                    estimated_acceptance_rate=previous_probe_acceptance,
                )
                n_sim = budget_plan.effective_num_simulations
                theta, acceptance_rate = self._rejection_sample_truncated_prior(
                    density_estimator=self.last_density_estimator,
                    threshold=threshold_for_next_round,
                    n_target=n_sim,
                )
            if self.config.log_round_timing:
                n_sim_text = f"{n_sim}/{budget_plan.requested_num_simulations}"
                print(
                    f"[TSNPE] round={round_idx}/{self.config.max_rounds} "
                    f"n_sim={n_sim_text} accept={acceptance_rate:.4f}",
                    flush=True,
                )
                if budget_plan.simulation_budget_adjusted:
                    print(
                        f"[TSNPE] round={round_idx} budget_adjustment "
                        f"{budget_plan.simulation_budget_adjustment_reason}",
                        flush=True,
                    )
            self._emit_status(
                state="running",
                event="round_start",
                round_index=round_idx,
                phase="prepare",
                num_simulations=n_sim,
                requested_num_simulations=budget_plan.requested_num_simulations,
                acceptance_rate_in_round=acceptance_rate,
                estimated_rejection_acceptance_rate=budget_plan.estimated_rejection_acceptance_rate,
                rejection_budget_max_draws=budget_plan.rejection_budget_max_draws,
                estimated_max_accepted_under_budget=budget_plan.estimated_max_accepted_under_budget,
                simulation_budget_adjusted=budget_plan.simulation_budget_adjusted,
                simulation_budget_adjustment_reason=budget_plan.simulation_budget_adjustment_reason,
            )

            sim_start = time.time()
            with self._phase_heartbeat(round_index=round_idx, phase="simulation", num_simulations=n_sim):
                self._maybe_sync_cuda()
                x = self._simulate(theta)
                self._maybe_sync_cuda()
            sim_seconds = time.time() - sim_start
            inference.append_simulations(theta, x, proposal=self._proposal_marker_for_round(round_idx))
            appended_simulations = int(theta.shape[0])

            train_start = time.time()
            with self._phase_heartbeat(round_index=round_idx, phase="training", num_simulations=n_sim):
                self._maybe_sync_cuda()
                resume_training = bool(
                    round_idx > 1 and self.config.resume_training_state_across_rounds
                )
                epoch_before = int(getattr(inference, "epoch", 0))
                density_estimator = inference.train(
                    training_batch_size=self.config.training_batch_size,
                    learning_rate=self.config.learning_rate,
                    validation_fraction=self.config.validation_fraction,
                    stop_after_epochs=self.config.stop_after_epochs,
                    max_num_epochs=self.config.max_num_epochs,
                    show_train_summary=self.config.show_train_summary,
                    resume_training=resume_training,
                    force_first_round_loss=True,
                    discard_prior_samples=(round_idx > 1 and self.config.discard_round1_after_first_update),
                )
                epoch_after = int(getattr(inference, "epoch", 0))
                self._maybe_sync_cuda()
            train_seconds = time.time() - train_start
            inferred_num_epochs: int | None = None
            epochs_trained_this_round: int | None = None
            summary_obj = getattr(inference, "summary", None)
            if isinstance(summary_obj, dict):
                epochs_candidate = summary_obj.get("epochs_trained")
                if isinstance(epochs_candidate, list) and epochs_candidate:
                    try:
                        inferred_num_epochs = int(epochs_candidate[-1])
                    except Exception:  # noqa: BLE001
                        inferred_num_epochs = None
            if resume_training:
                epochs_trained_this_round = max(epoch_after - epoch_before, 0)
            else:
                epochs_trained_this_round = max(epoch_after, 0)
            self.last_density_estimator = density_estimator
            trunc_start = time.time()
            with self._phase_heartbeat(round_index=round_idx, phase="truncation", num_simulations=n_sim):
                self._maybe_sync_cuda()
                trunc_estimator = self._estimator_for_truncation(self.last_density_estimator)
                truncation_plan = self._build_truncation_plan(trunc_estimator, round_idx=round_idx)
                threshold = truncation_plan.threshold
                volume = self._estimate_truncated_prior_volume(trunc_estimator, threshold=threshold)
                probe_acceptance = truncation_plan.probe_acceptance_rate
                q01 = truncation_plan.probe_logp_q01
                q50 = truncation_plan.probe_logp_q50
                q99 = truncation_plan.probe_logp_q99
                self._maybe_sync_cuda()
            trunc_seconds = time.time() - trunc_start

            stop_flag = False
            volume_ratio: float | None = None
            stop_eligible = False
            stop_reason = "first_round"
            if previous_volume is not None:
                if previous_volume > 0.0:
                    volume_ratio = float(volume / previous_volume)
                enough_rounds = round_idx >= self.config.min_rounds_before_stopping
                shrink_ok = (not self.config.require_volume_shrink_for_stopping) or (volume < previous_volume)
                volume_ok = volume < self.config.max_volume_for_stopping
                stop_eligible = bool(enough_rounds and shrink_ok and volume_ok)
                if stop_eligible:
                    stop_flag = should_stop_by_volume_ratio(
                        previous_volume=previous_volume,
                        current_volume=volume,
                        stopping_ratio=self.config.stopping_ratio,
                    )
                    stop_reason = "eligible_ratio_check"
                else:
                    reasons: list[str] = []
                    if not enough_rounds:
                        reasons.append("insufficient_rounds")
                    if not shrink_ok:
                        reasons.append("no_volume_shrink")
                    if not volume_ok:
                        reasons.append("volume_too_large")
                    stop_reason = ",".join(reasons) if reasons else "not_eligible"

            diagnostics.append(
                RoundDiagnostics(
                    round_index=round_idx,
                    num_simulations=n_sim,
                    requested_num_simulations=budget_plan.requested_num_simulations,
                    hpd_log_prob_threshold=threshold,
                    truncated_prior_volume=volume,
                    acceptance_rate_in_round=acceptance_rate,
                    stop_by_ratio=stop_flag,
                    simulation_seconds=sim_seconds,
                    training_seconds=train_seconds,
                    truncation_seconds=trunc_seconds,
                    round_total_seconds=time.time() - round_start,
                    volume_ratio_to_previous=volume_ratio,
                    stop_eligible=stop_eligible,
                    stop_reason=stop_reason,
                    probe_samples=self.config.truncation_probe_samples,
                    probe_acceptance_rate=probe_acceptance,
                    probe_logp_q01=q01,
                    probe_logp_q50=q50,
                    probe_logp_q99=q99,
                    appended_simulations=appended_simulations,
                    inferred_num_epochs=inferred_num_epochs,
                    epochs_trained_this_round=epochs_trained_this_round,
                    estimated_rejection_acceptance_rate=budget_plan.estimated_rejection_acceptance_rate,
                    rejection_budget_max_draws=budget_plan.rejection_budget_max_draws,
                    estimated_max_accepted_under_budget=budget_plan.estimated_max_accepted_under_budget,
                    simulation_budget_adjusted=budget_plan.simulation_budget_adjusted,
                    simulation_budget_adjustment_reason=budget_plan.simulation_budget_adjustment_reason,
                    truncation_relaxed=truncation_plan.truncation_relaxed,
                    truncation_relaxation_reason=truncation_plan.truncation_relaxation_reason,
                    target_min_probe_acceptance_rate=truncation_plan.target_min_probe_acceptance_rate,
                )
            )
            self.last_diagnostics = list(diagnostics)
            self._emit_status(
                state="running",
                event="round_complete",
                round_index=round_idx,
                phase="complete",
                num_simulations=n_sim,
                requested_num_simulations=budget_plan.requested_num_simulations,
                truncated_prior_volume=volume,
                probe_acceptance_rate=probe_acceptance,
                stop_by_ratio=stop_flag,
                stop_eligible=stop_eligible,
                stop_reason=stop_reason,
                volume_ratio_to_previous=volume_ratio,
                inferred_num_epochs=inferred_num_epochs,
                epochs_trained_this_round=epochs_trained_this_round,
                estimated_rejection_acceptance_rate=budget_plan.estimated_rejection_acceptance_rate,
                rejection_budget_max_draws=budget_plan.rejection_budget_max_draws,
                estimated_max_accepted_under_budget=budget_plan.estimated_max_accepted_under_budget,
                simulation_budget_adjusted=budget_plan.simulation_budget_adjusted,
                simulation_budget_adjustment_reason=budget_plan.simulation_budget_adjustment_reason,
                truncation_relaxed=truncation_plan.truncation_relaxed,
                truncation_relaxation_reason=truncation_plan.truncation_relaxation_reason,
                target_min_probe_acceptance_rate=truncation_plan.target_min_probe_acceptance_rate,
            )
            if self.config.log_round_timing:
                print(
                    f"[TSNPE] round={round_idx} done "
                    f"sim={sim_seconds:.1f}s train={train_seconds:.1f}s "
                    f"trunc={trunc_seconds:.1f}s total={diagnostics[-1].round_total_seconds:.1f}s "
                    f"volume={volume:.6f} probe_accept={probe_acceptance:.6f} "
                    f"epochs_this_round={epochs_trained_this_round} "
                    f"stop={stop_flag} stop_eligible={stop_eligible} reason={stop_reason}",
                    flush=True,
                )
                if truncation_plan.truncation_relaxed:
                    print(
                        f"[TSNPE] round={round_idx} truncation_relaxed "
                        f"{truncation_plan.truncation_relaxation_reason}",
                        flush=True,
                    )

            previous_volume = volume
            previous_probe_acceptance = probe_acceptance
            threshold_for_next_round = threshold
            if stop_flag and round_idx > 1:
                break

        if self.last_density_estimator is None:
            raise RuntimeError("TSNPE produced no density estimator")
        posterior = inference.build_posterior(sample_with="direct")
        posterior.set_default_x(self.x_observed)
        self._emit_status(
            state="completed",
            event="run_complete",
            round_index=len(diagnostics),
            phase="complete",
            num_simulations=0,
        )
        return posterior, diagnostics
