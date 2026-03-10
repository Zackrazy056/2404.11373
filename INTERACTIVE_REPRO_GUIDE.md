# 2404.11373v3 Interactive Reproduction Guide
下面这版是针对 **arXiv:2404.11373v3（2024-12-03）** 的“全量复现方案 + 工程架构”。

**复现目标（必须交付）**
1. 复现注入实验 `Kerr220/Kerr221/Kerr330`，并与 `pyRing v2.3.0 + cpnest` 对比。  
2. 复现真实事件 `GW150914`（220 单模）后验结果。  
3. 复现覆盖率检验（coverage cdf + KS p-value统计）。  
4. 复现论文主图：Fig.1/2/3/4 + Appendix Fig.5/6/7。  
5. 形成可重复运行的工程（固定配置、固定随机种子、可追踪产物）。

**推荐工程架构**
```text
2404.11373-repro/
  README.md
  pyproject.toml
  Makefile
  configs/
    model/tsnpe_nsf.yaml
    injections/kerr220.yaml
    injections/kerr221.yaml
    injections/kerr330.yaml
    real/gw150914_220.yaml
    baseline/pyring.yaml
  data/
    raw/gwosc/
    raw/psd_segments/
    processed/
    cache/simulations/
  src/rd_sbi/
    waveforms/ringdown.py
    qnm/kerr_fits.py
    detector/patterns.py
    noise/psd.py
    noise/whitening.py
    simulator/injection.py
    inference/embedding_net.py
    inference/tsnpe_runner.py
    inference/sbi_loss_patch.py
    baseline/pyring_runner.py
    eval/posterior_compare.py
    eval/coverage_test.py
    plots/reproduce_figs.py
    io/artifacts.py
    utils/seed.py
  scripts/
    01_prepare_psd.py
    02_make_injections.py
    03_train_tsnpe.py
    04_run_pyring_baseline.py
    05_coverage_eval.py
    06_make_figures.py
  tests/
    test_waveform.py
    test_whitening.py
    test_tspe_rounding.py
    test_coverage.py
  reports/
    figures/
    tables/
    logs/
```

**关键参数（按论文锁死）**
1. 采样与片段：`2048 Hz`，`0.1 s`，每台探测器 `204 bins`。  
2. 真实数据：GWOSC `4096 Hz` 下采样到 `2048 Hz`。  
3. 白化：时域 Cholesky（ACF -> Toeplitz covariance -> `L^-1 h`）。  
4. TSNPE：`epsilon=1e-4`，`stopping_ratio=0.8`，round1 `50k`，后续每轮 `100k`。  
5. NSF：`num_transforms=5`，`hidden_features=150`，`num_blocks=2`，`num_bins=10`，`batch_norm=True`。  
6. 嵌入网络：`408 -> 150 -> 150 -> 128`。  
7. 训练：`batch_size=512`，`lr=1e-3`，`validation_fraction=0.1`，每个 epoch 重采样噪声（需 patch `sbi` loss）。  
8. 注入固定参数：  
   - `Kerr220`: `Mf=67, χf=0.67, A220=5e-21, φ220=1.047, ι=π, SNR≈14`  
   - `Kerr221`: `A220=8.92e-21, A221=9.81e-21, φ220=1.047, φ221=4.19, ι=π, SNR≈14`  
   - `Kerr330`: `A220=30e-21, A330=3e-21, φ220=1.047, φ330=5.014, ι=π/4, SNR≈53`  
9. 固定天区/极化：`α=1.95, δ=-1.27, ψ(文中记ϕ)=0.82`。  
10. 基线采样：`cpnest live points=4096`，`max mcmc steps=4094`，约 `20k` posterior samples。

**执行流水线**
1. 数据准备：下载 GWOSC 数据、选取 PSD 估计段、产出 whiten 工具输入。  
2. 前向仿真：构造 ringdown waveform + 噪声注入 + 参数标准化。  
3. TSNPE 训练：逐轮截断训练，记录每轮有效先验体积与停止条件。  
4. 基线运行：同一配置跑 pyRing，导出对齐后的 posterior。  
5. 评估：参数恢复、二维可信域、振幅小提琴图、coverage cdf。  
6. 绘图与报告：自动生成与论文对应图版。

**里程碑（单人）**
1. 环境、数据、白化、波形模块。  
2. TSNPE + `sbi` loss patch + Kerr220 跑通。  
3. ：Kerr221/Kerr330 + pyRing 对比。  
4.GW150914 实数据信号 + coverage 全量。  
5. 图表复现、回归测试、文档与一键脚本。

**验收标准**
1. 注入参数真值落在论文同等级可信区间内（重点 `Mf, χf, A_lmn`）。  
2. SBI 与 pyRing 后验形状一致（主观图检 + 数值距离指标）。  
3. coverage cdf 落在理论带内，KS p-value 分布与论文量级一致。  
4. 全流程从空目录可一键复跑并复现主要图。

如果你要，我下一步可以直接在你当前目录把这套骨架和首批可运行脚本（`01~03`）搭起来。


This guide defines a round-based collaboration workflow:
- One round = one task.
- You assign the task each round.
- I execute, verify, and report evidence.
- We iterate until full reproduction is complete.

## 1. Objective

Reproduce the full workflow and results of arXiv:2404.11373v3, including:
- Injection experiments: `Kerr220`, `Kerr221`, `Kerr330`.
- Real-data analysis: `GW150914`.
- Baseline comparison: `pyRing + cpnest`.
- Coverage diagnostics and figure reproduction.
- Reproducible engineering pipeline (code, config, logs, artifacts).

## 2. Round Protocol (Core Rule)

Each round must include only one deliverable task.

### 2.1 Your Task Assignment Template

Copy and fill this each round:

```md
[Round N]
Task ID: Txxx
Goal: <one clear outcome>
Inputs: <files/data/refs>
Constraints: <time, tools, style, limits>
Definition of Done:
1) <checkable condition>
2) <checkable condition>
Output Required: <what you want back>
```

Minimal fast form:

```md
R<N> -> Txxx
DoD: <one sentence>
```

### 2.2 My Round Response Contract

For every assigned task, I will return:
1. Task understanding.
2. Actions executed (files, commands, changes).
3. Verification evidence (tests/metrics/plots/logs).
4. Completion status: `Done` or `Blocked`.
5. If blocked: exact blocker + next best fallback.
6. Suggested next 1-3 task IDs.

## 3. Global Milestones

- `M0` Project bootstrap and reproducible environment.
- `M1` Data pipeline (GWOSC, PSD, whitening, preprocessing).
- `M2` Ringdown model + simulator.
- `M3` TSNPE/SBI training pipeline.
- `M4` Baseline inference (`pyRing`) and posterior alignment.
- `M5` Evaluation (coverage, comparisons, plots).
- `M6` Real-data run (`GW150914`) and final report.

## 4. Task Backlog (Atomic, One-Round Friendly)

### M0: Bootstrap

- `T001` Create project scaffold and module layout.
- `T002` Create pinned environment (`pyproject` or `requirements`).
- `T003` Add config system and seed control.
- `T004` Add artifact/log directory policy.

### M1: Data + Noise

- `T010` Add GWOSC data fetch script for GW150914.
- `T011` Add PSD estimation script.
- `T012` Implement ACF -> Toeplitz covariance.
- `T013` Implement Cholesky whitening in time domain.
- `T014` Add data normalization utilities.
- `T015` Add unit tests for whitening and covariance consistency.

### M2: Waveform + Injection

- `T020` Implement ringdown waveform model (time domain).
- `T021` Implement detector response projection (`H1`, `L1`).
- `T022` Implement Kerr QNM mapping from `(Mf, chi_f)` fits.
- `T023` Add injection configs: `Kerr220`, `Kerr221`, `Kerr330`.
- `T024` Add SNR computation and amplitude sanity checks.
- `T025` Add injection generation script and cached datasets.

### M3: SBI / TSNPE

- `T030` Build embedding network (`408 -> 128`).
- `T031` Build NSF posterior model with paper hyperparameters.
- `T032` Implement TSNPE truncation loop (`epsilon=1e-4`).
- `T033` Implement stopping rule (`stopping_ratio=0.8`).
- `T034` Patch training loss to vary noise each epoch.
- `T035` Add train runner for `Kerr220`.
- `T036` Add train runner for `Kerr221`.
- `T037` Add train runner for `Kerr330`.

### M4: Baseline

- `T040` Integrate `pyRing` run wrapper.
- `T041` Configure `cpnest` baseline settings.
- `T042` Standardize posterior export format.
- `T043` Add 20k sample alignment utilities for fair comparison.

### M5: Evaluation + Figures

- `T050` Add posterior comparison metrics and corner generation.
- `T051` Reproduce paper-like Fig.1 (mass-spin posteriors).
- `T052` Reproduce paper-like Fig.2 (amplitude violin plots).
- `T053` Implement coverage statistic `gamma(theta*, x*)`.
- `T054` Run coverage catalogs and cdf/KS summary.
- `T055` Reproduce coverage plots (main + appendix style).

### M6: Real Data + Finalization

- `T060` Build `GW150914` real-data preprocessing pipeline.
- `T061` Run `GW150914` SBI inference (220-only model).
- `T062` Run baseline comparison on `GW150914`.
- `T063` Generate final result pack (plots, tables, config snapshot).
- `T064` Write reproducibility report and runbook.

## 5. Definition of Done (Strict)

A task is `Done` only if all are satisfied:
1. Code/config/docs changed and saved in repository.
2. Verification executed (test/script/check).
3. Evidence included in round report (key output summary).
4. No unresolved ambiguity for this task scope.

If any condition fails, status must be `Blocked` (not `Done`).

## 6. Round Output Format (Standard)

I will use this structure:

```md
Round N - Txxx
Status: Done | Blocked

What I changed:
- ...

Verification:
- Command: ...
- Result: ...

Artifacts:
- /abs/path/to/file1
- /abs/path/to/file2

Next suggested tasks:
1) T...
2) T...
```

## 7. Blocker Handling Rules

If blocked, I will provide:
1. Exact blocker.
2. Why it blocks completion.
3. Fast fallback option.
4. What you should assign next to unblock.

## 8. Progress Tracking Board (You Can Paste and Update)

```md
| Task ID | Milestone | Status | Evidence | Notes |
|---|---|---|---|---|
| T001 | M0 | Todo | - | - |
```

Status values:
- `Todo`
- `In Progress`
- `Done`
- `Blocked`

## 9. Suggested Start Sequence

Use this order unless you want to reprioritize:
1. `T001`
2. `T002`
3. `T003`
4. `T010`
5. `T011`
6. `T012`
7. `T013`
8. `T020`

## 10. First Round Example

```md
[Round 1]
Task ID: T001
Goal: Create reproducible project scaffold and initial package layout.
Inputs: Current folder with paper PDFs.
Constraints: Keep structure minimal and clean.
Definition of Done:
1) Base folders and package skeleton created.
2) README includes project objective and run entry points.
Output Required: Tree summary + created file list.
```

When ready, send `Round 1` with your chosen `Task ID`.
