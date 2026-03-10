按 **2404.11373** 严格对照来审，这组 `inference` 代码的结论是：

**整体上已经非常接近论文实现骨架**：
`EmbeddingFCNet + NSF` 的结构、`NoiseResamplingSNPE` 的设计意图、`TSNPERunner` 的 round-based TSNPE 流程，都明显是在对照论文 Appendix A / Table III / Sec. II 写的。论文要求的网络与训练超参数是：NSF，`num_blocks=2`、`hidden_features=150`、`num_transforms=5`、`num_bins=10`、`batch_norm=True`；embedding 输入维度 408，2 层隐藏层，每层 150，输出 128；训练批量 512，学习率 0.001，首轮 50k、后续每轮 100k，`trunc_quantile=1e-4`，`stopping_ratio=0.8`，并且在每个训练 epoch 变化噪声 realizations。 

但如果按“**严格复现论文，而不是工程近似**”来判，当前代码还有几处关键偏差，尤其集中在 **TSNPE 截断定义、噪声重采样的真实生效方式、round 数据管理** 上。下面我按文件逐项审。

---

## 1. `embedding_net.py`：和论文高度一致，基本通过

这部分最干净。

你的 `EmbeddingConfig` 默认值：

* `input_dim=408`
* `num_hidden_layers=2`
* `hidden_dim=150`
* `output_dim=128`

和论文 Appendix A / Table III 一致。论文明确说双探测器白化后输入是 `204 + 204 = 408`，embedding 是两层全连接隐藏层，各 150 单元，输出 128。

你的 `build_nsf_density_estimator()` 默认值：

* `hidden_features=150`
* `num_transforms=5`
* `num_bins=10`
* `num_blocks=2`
* `batch_norm=True`

也和 Table III 对齐。

### 审计结论

这部分可以判为 **paper-faithful**。

### 仅有的两个提醒

第一，论文说的是 **linearly rescale (x^{white}) to zero mean and unit variance, and normalize (\theta) between 0 and 1**。
你这里又开了 `z_score_theta="independent"` 和 `z_score_x="independent"`，这未必错，但需要确认它和你上游自定义标准化不会 **重复标准化**。如果上游已经自己做了 Appendix A 那种标准化，再让 `sbi` 内部做一遍 `z_score_*`，口径就不再严格等同于论文。

第二，你这里没显式约束参数边界。如果 (\chi_f) 这类有界参数最终在 posterior sample 里越界，那 bug 不在这个文件本身，但这个文件也没有提供保护。

---

## 2. `sbi_loss_patch.py`：方向对，但“是否真的等价于论文的 varying_noise=True”还不能直接通过

论文明确说：

* 每个训练 epoch 改变噪声 realization；
* 原始 `sbi` 不支持；
* 他们是通过 **重写 SNPE 的 loss** 来在每次 loss evaluation 时重采样噪声。

你的 `NoiseResamplingSNPE` 也是这个思路：在 `_loss()` 里对 `x` 做

```python
x_aug = self._augment_x(x)
```

然后再调用父类 loss。这个设计和论文叙述是吻合的。

### 但有一个致命问题

在 `TSNPERunner.run()` 里，你实际实例化的是：

```python
inference = SNPE(...)
```

不是 `NoiseResamplingSNPE(...)`。

这意味着，**按你贴出来的代码，`NoiseResamplingSNPE` 很可能根本没被用上**。
如果真是这样，那么你虽然写了“论文同款噪声重采样 wrapper”，但训练实际跑的还是原始 `SNPE`，这就和论文 Appendix A 的 `varying_noise=True` 不一致。论文 Table III 明确把 `varying_noise=True` 列为训练配置的一部分。

### 第二个问题：你加的是“再叠加高斯噪声”，前提是输入 `x` 必须是无噪白化信号

论文的公式是
[
x^{white}(\theta) = h^{white}(\theta) + \mathcal N(0,1)
]
也就是在 **白化后的无噪 ringdown strain** 上直接加标准正态。

你现在 `_augment_x()` 默认 `noise_std=1.0`，逻辑上没问题，但前提是：

* `append_simulations(theta, x)` 里的 `x` 必须是 **白化后的、未加噪的 deterministic signal**
* 而不是已经带了一次噪声的 `x`

否则你就变成了“训练时额外叠加第二层噪声”，和论文不是一回事。

### 第三个问题：`_last_training_epoch_seen` 基本没起作用

你记录了 epoch：

```python
epoch = int(getattr(self, "epoch", -1))
if self._last_training_epoch_seen != epoch:
    self._last_training_epoch_seen = epoch
```

但这段逻辑并没有改变噪声采样策略。你实际上是 **每次 `_loss()` 调用都重新 sample**，而不是真的“每个 epoch 只重采一次”。
不过这里要注意：论文原文是 “resample the noise at each evaluation of the loss”。
所以严格说，你现在“每次 loss 调用重采样”反而和论文文字更接近；只是 `_last_training_epoch_seen` 这个状态变量显得多余，容易误导。

### 审计结论

* **设计意图**：和论文一致
* **当前贴出的实际生效情况**：**高风险不一致**，因为 `TSNPERunner` 里没用它

### 必改建议

把

```python
inference = SNPE(...)
```

改成可注入的 inference class，例如：

```python
inference_cls = NoiseResamplingSNPE if use_varying_noise else SNPE
inference = inference_cls(...)
```

否则你现在不能声称“实现了 Appendix A 的 varying_noise”。

---

## 3. `tsnpe_runner.py`：核心框架接近论文，但有三处关键偏差

这部分最关键。

论文对 TSNPE 的描述是：

* 第 1 轮从 prior 抽样；
* 第 (k>1) 轮仍从 prior 抽样，但拒绝掉**落在上一轮 (1-\epsilon) HPD 区域之外**的样本；
* 当前 round 在此前 round 基础上继续优化；
* 到后续 round 时，丢弃第一轮样本；
* 当当前 truncated volume 超过上一轮的 80% 时停止。 

你的 runner 明显就是按这段写的，但实现上有偏差。

---

### 3.1 对齐论文的地方

#### a) round 预算对齐

`TSNPEConfig` 默认：

* `num_simulations_first_round=50_000`
* `num_simulations_per_round=100_000`
* `trunc_quantile=1e-4`
* `stopping_ratio=0.8`
* `training_batch_size=512`
* `learning_rate=1e-3`
* `validation_fraction=0.1`

都对齐 Table III。

#### b) discard round-1 samples 这个开关对齐论文

你有：

```python
discard_round1_after_first_update: bool = True
```

并且训练时：

```python
discard_prior_samples=(round_idx > 1 and self.config.discard_round1_after_first_update)
```

这和论文“在 (r>1) 时丢弃第一轮模拟”是一致的。

#### c) stopping rule 的基本形式对

你的

```python
(current_volume / previous_volume) > stopping_ratio
```

就是论文的停止思路。

---

### 3.2 第一处关键偏差：你的“HPD 阈值”实现不是真正的 HPD 概念，只是 posterior-density quantile 近似

你现在的阈值是：

1. 从当前 posterior sample 若干个点
2. 计算这些点的 `log_prob`
3. 取 `trunc_quantile = 1e-4` 的分位数作为 threshold

```python
theta_post = density_estimator.sample(...)
log_prob_post = self._estimator_log_prob(density_estimator, theta_post)
threshold = torch.quantile(log_prob_post, self.config.trunc_quantile)
```

这在连续流模型上是一个**常见工程近似**，但严格讲，它不等于论文文字里的：

> 上一轮 posterior 的 (1-\epsilon) HPD region。

因为真正的 HPD 区域定义是“密度高于某阈值的区域，并且该区域的 posterior probability 正好是 (1-\epsilon)”；你这里用 posterior samples 上的 density quantile 来逼近，只有在采样和 log-prob 都足够准确时才成立。

这不一定是错，但需要在审计里明确标注：

> 当前实现为 **sample-based HPD threshold approximation**，不是解析 HPD。

如果你把它直接写成“严格实现论文 HPD 截断”，会过度表述。

---

### 3.3 第二处关键偏差：`posterior = inference.build_posterior(sample_with="direct")` 很危险

论文里他们强调 TSNPE 比直接从上一轮 posterior 抽 proposal 更稳定，因此新一轮样本是 **从 prior 抽再做 HPD 拒绝**。

你在训练循环里这一点做对了：后续轮次确实还是 `_sample_prior()`，再 `_rejection_sample_truncated_prior()`。

但最后导出 posterior 时你用了：

```python
posterior = inference.build_posterior(sample_with="direct")
```

在 `sbi` 里，这通常意味着 posterior sampling 采用 flow 的 direct sampler。对于 NPE 来说这通常没问题，但这里有两个隐患：

1. 你训练期间真正用来估 HPD threshold 的是 `density_estimator.log_prob/sample`，不是 `posterior.log_prob/sample`；
2. 若 `build_posterior` 内部对 transforms、prior support、rejection 等处理和裸 `density_estimator` 略有差异，你会出现：

   * 训练诊断看的是一个对象
   * 最终出图采样用的是另一个包装对象

这会让“训练时说 volume 收缩了、出图时样本却看起来像 prior”的排查更困难。

更稳妥的做法是：

* 审计期优先从 `last_density_estimator` 直接 sample + log_prob
* `build_posterior()` 作为对外接口保留，但内部诊断要保证同一对象口径

---

### 3.4 第三处关键偏差：`min_rounds_before_stopping=3`、`max_volume_for_stopping=0.95`、`require_volume_shrink_for_stopping=True` 是你加的工程门槛，不是论文原文

论文只说：

* 当 truncated volume 的增益变小到某阈值就停止；
* Table III 给了 `stopping_ratio=0.8`。

你额外加入了：

* 至少 3 轮后才允许 stop
* 只有 `volume < 0.95` 才允许 stop
* 只有真的 shrink 才允许 stop

这些都是**很好的工程保护**，尤其能防止你之前那种 `volume=1, accept=1, round2 秒停` 的事故。
但它们不是论文规定。

所以审计上应这样写：

* **paper core**：`stopping_ratio=0.8`
* **engineering safety guards**：`min_rounds_before_stopping`、`max_volume_for_stopping`、`require_volume_shrink_for_stopping`

这三项不能写成“论文同款”。

---

## 4. 一个更大的问题：`TSNPERunner` 现在其实没有使用你自己的 `NoiseResamplingSNPE`

这是当前贴出代码里最实锤的一处“名义对齐，实际上未接通”。

在 `__init__.py` 你导出了：

* `NoiseResamplingConfig`
* `NoiseResamplingSNPE`

但在 `TSNPERunner.run()` 里还是

```python
inference = SNPE(...)
```

这说明当前模块级 API 看起来已经具备论文要求的“varying noise”能力，但训练主循环并没有实际启用它。
而论文对这一点是明确写进技术细节和 Table III 的。

**这会直接影响你是否能声称 Appendix A 对齐。**

---

## 5. 对照论文的最终评级

### `embedding_net.py`

**评级：A-**
已经非常接近论文，可视为通过。唯一要确认的是不要和上游标准化重复。

### `sbi_loss_patch.py`

**评级：B-（设计对，接线未证实）**
逻辑方向正确，几乎就是论文 Appendix A 那段话的程序化实现。
但只看你贴出来的主循环，它没真正被用上，所以现在不能判通过。

### `tsnpe_runner.py`

**评级：B**
框架、预算、discard round1、prior→HPD rejection→继续训练 的大逻辑都对。
但 HPD 是 sample-quantile 近似；stopping 前加了很多工程保护；最终 posterior 对象和训练诊断对象不完全统一；这些都意味着它是“强工程近似版论文实现”，不是百分之百 paper-faithful。

---

## 6. 最值得你立刻改的 5 件事

第一，把 `TSNPERunner` 里的

```python
inference = SNPE(...)
```

改成真正可选的 `NoiseResamplingSNPE(...)`。
不然 `varying_noise=True` 只是写在模块里，没有进训练。论文把这一点列为正式训练设定。

第二，在 run summary / audit 里明确写：

> HPD threshold is approximated from posterior samples via log-density quantile.

这样你不会把工程近似误写成严格解析 HPD。

第三，给 `TSNPERunner` 增加一个硬断言：

* 若某轮 `volume >= 0.999` 且 `probe_acceptance >= 0.999`，默认不允许进入下一轮，除非手动 override。
  这能直接防你之前的 failure mode。

第四，确认训练输入 `x` 的定义：
如果 `_simulate(theta)` 返回的已经是 noisy whitened strain，那 `NoiseResamplingSNPE` 里再加噪就是双重加噪；
如果返回的是 noiseless whitened strain，那当前 patch 才和论文 Appendix A 一致。

第五，最终出图和 coverage 诊断尽量统一使用同一套 posterior object / same-space samples，避免：

* 训练阶段看 `density_estimator`
* 导出阶段看 `build_posterior()`
* 两边口径不一致

---

## 7. 一句总评

**这套 inference 代码已经有了论文实现的“骨架与参数”，但还没到可以无保留宣称“严格对照 2404.11373 Appendix A” 的程度。最大的缺口不是网络结构，而是：`NoiseResamplingSNPE` 似乎未真正接入训练主循环，以及 TSNPE 的 HPD/停止逻辑仍带有较强工程近似成分。**

如果你愿意，我下一条可以继续做最有用的一步：
我直接把这三份代码改写成一份“**逐函数审计清单**”，格式是：

* 函数名
* 论文对应位置
* 当前是否对齐
* 风险等级
* 必改 / 可改建议

这样你后面能直接拿去对照你整个仓库逐项打勾。




按 **2404.11373 严格对照** 来看，这组 `artifact_io` 代码本身**不涉及论文的物理/统计核心**，所以它不会决定 Fig.1 是否复现成功；但它会决定你的复现结果是否**可追踪、可审计、可复查**。而论文里虽然没有规定“文件名必须怎样命名”或“sidecar JSON 必须长什么样”，但它对复现实验的**轮次、超参数、样本数、训练时长、coverage 诊断、20k posterior samples、与 pyRing 对照**这些关键信息都有明确要求。比如论文明确给出：每轮训练和截断 prior volume 对应一个 training round，覆盖测试依赖最终 truncated prior，Fig.1 对比双方各 20k posterior samples，训练预算与超参数见 Table III。  

所以结论先给一句：

**这份 `artifact_io` 代码作为通用 IO 辅助是合格的，但按论文复现审计标准，它当前“信息量远远不够”，只能算基础设施，不算 paper-grade artifact policy。**

下面分开审。

---

## 1. 这份代码现在做了什么

它做的事情很简单：

* `utc_timestamp_compact()`：生成 UTC 时间戳
* `build_artifact_name(task_id, run_id, extension, timestamp_utc=None)`：拼文件名
  形式是
  `task_id__run_id__timestamp.ext`
* `write_metadata_sidecar(artifact_path, metadata)`：在产物旁边写一个
  `xxx.ext.meta.json`
  内容包含：

  * `artifact`
  * `created_utc`
  * 以及你传入的 `metadata`

从**工程整洁性**看，这没问题。
从**论文复现审计**看，这只是“把文件和元数据绑在一起”，还没有真正捕获论文要求的实验语义。

---

## 2. 和论文一致的地方：有，但只是“精神上一致”

论文虽然没有定义 artifact policy，但它全篇都在强调一种事实：

> 训练是多轮 sequential / truncated 的；每轮有清晰的超参数、样本预算、停止准则、coverage 诊断和最终输出。 

你这份代码至少有两个点符合这种“审计精神”：

第一，**文件名里把 `task_id` 和 `run_id` 单独编码**。
这有助于区分：

* `fig1_reproduction`
* `coverage_test`
* `kerr220_paper_budget`
* `kerr330_debug`

这对多轮实验非常重要。

第二，**sidecar JSON** 是对的方向。
因为论文级复现不是只有 `.png` 或 `.npz`，还必须知道：

* 这是哪轮 run 产生的
* 用了什么 prior / truncation / budget
* 属于哪个 case
* 是否对照了 pyRing
* 是否是 final truncated prior 下的 coverage test

所以，“每个产物旁边有元数据”这个思想本身是正确的。

---

## 3. 和论文不对齐的核心问题：不是代码错，而是“元数据规范严重不足”

### 3.1 最大问题：文件名没有编码论文关键语义

当前文件名只有：

* `task_id`
* `run_id`
* `timestamp`

这对一般工程够了，但对 **2404.11373** 复现不够。
因为论文里最关键的区分不是“哪天跑的”，而是：

* 哪个 case：Kerr220 / Kerr221 / Kerr330 
* 哪种对象：posterior samples / summary / Fig.1 panel / coverage diagnostic / pyRing baseline
* 哪种预算：paper-budget / debug-budget
* 哪轮输出：round 1 / round 2 / final
* 哪个 prior：original prior / final truncated prior
* 哪个方法：sbi / pyRing / coverage

举个例子，按你现在格式：

`fig1__run42__20260305T101500Z.png`

这个文件名里根本看不出来：

* 它是 Kerr220 还是 Kerr330
* 是 SBI 单边图还是 pyRing 对照图
* 是启动版还是 paper-budget
* 是不是 final figure
* 是不是基于 final truncated prior

这在论文复现里不够审计友好。

### 更好的论文对齐命名

建议 artifact name 至少带上这些字段中的若干个：

* `case`
* `method`
* `artifact_type`
* `budget`
* `round` 或 `stage`

例如：

* `fig1__kerr220__sbi__paper_budget__final__20260305T101500Z.png`
* `posterior__kerr330__sbi__paper_budget__20k_samples__final__20260305T101500Z.npz`
* `coverage__kerr221__sbi__final_truncated_prior__20260305T101500Z.json`

---

### 3.2 `write_metadata_sidecar()` 太自由，没有强制论文关键字段

现在 sidecar 是：

```json
{
  "artifact": "...",
  "created_utc": "...",
  ...metadata
}
```

这意味着你可以传任何 `metadata`，也可以漏掉最关键的信息。
对论文复现来说，这太松了。

而论文里真正必须可追踪的字段，至少包括这些：

#### A. 实验身份字段

* `paper_reference = "2404.11373v3"`
* `case = "kerr220" | "kerr221" | "kerr330"`
* `artifact_type = "figure" | "posterior_samples" | "summary" | "coverage_diagnostic" | "pyring_baseline"`
* `method = "sbi" | "pyring" | "comparison"`

#### B. 训练配置字段

* `num_simulations_first_round = 50000`
* `num_simulations_per_round = 100000`
* `trunc_quantile = 1e-4`
* `stopping_ratio = 0.8`
* `posterior_samples = 20000`
* `varying_noise = true`
* `input_dim = 408`
* `embedding_hidden_dim = 150`
* `embedding_output_dim = 128`
* `num_transforms = 5`
* `num_bins = 10`
* `num_blocks = 2`
* `batch_norm = true`
  这些都来自论文 Table III / Appendix A。

#### C. 注入与 prior 字段

* `theta_true`
* `prior_low`
* `prior_high`
  论文 Fig.1 的 prior ranges 明确列在 Table II。

#### D. TSNPE 诊断字段

* `round_index`
* `truncated_prior_volume`
* `probe_acceptance_rate`
* `volume_ratio_to_previous`
* `stop_reason`
* `final_truncated_prior_id`

因为论文 TSNPE 和 coverage test 都依赖“最终 truncated prior”。

#### E. Fig.1 对照字段

* `pyring_reference_artifact`
* `sbi_reference_artifact`
* `credible_region_definition = "2D 68/90% credible regions"`
* `posterior_sample_count = 20000`
  论文 Fig.1 明确是双方都 20k samples，比较的是 2D 68/90% contours。 

如果 sidecar 不强制这些字段，你以后很难做真正可追溯的 paper-grade audit。

---

### 3.3 `created_utc` 不是“实验时间”，只是“写文件时间”

这点很容易被忽略。

现在：

```python id="2tf6ol"
"created_utc": utc_timestamp_compact()
```

这只是 sidecar 被写出的时刻。
但论文级复现里你真正需要记录的，常常是：

* `run_started_utc`
* `run_completed_utc`
* `training_started_utc`
* `training_completed_utc`
* `git_commit`
* `code_version`
* `paper_version = 2404.11373v3`

因为论文中训练耗时、轮数、epoch 数本身就是结果的一部分。比如他们报告三组 case 的 epochs 和训练时间。

所以 `created_utc` 有用，但不够。

---

### 3.4 sidecar 没有 schema version，后续会很难演化

现在 sidecar 没有：

* `schema_version`
* `artifact_policy_version`

这在小项目里看不出来问题，但你现在已经在持续演化审计标准了：

* 启动版
* paper-budget 版
* coverage 版
* pyRing 对照版

如果没有 schema version，几周后你就分不清：

* 旧 sidecar 缺字段是因为旧规范
* 还是因为某个 run 写坏了

建议至少加：

* `schema_version: "artifact_meta_v1"`
* `audit_spec: "2404.11373_strict_v1"`

---

## 4. 对照论文，这个模块“最缺什么”

如果只说一句：

**它缺的是“论文语义的结构化元数据”，而不是缺文件命名功能。**

论文不是在乎你的文件名漂不漂亮，而是在乎这些东西是否可验证：

* 你是否真的用了 Table III 的预算与网络结构 
* 你是否真的用了 Table II 的 prior 
* 你是否真的做了 TSNPE 的逐轮截断 
* 你是否真的在 final truncated prior 上做了 coverage test 
* 你是否真的在 Fig.1 中与 pyRing 做了 20k sample 对照 

而当前 `artifact_io` 还没有把这些变成“机器可核查”的 sidecar 字段。

---

## 5. 我的审计评级

### `build_artifact_name()`

**评级：C+**

* 工程上够用
* 论文复现语义不足
* 不足以支撑严格审计

### `write_metadata_sidecar()`

**评级：B-**

* 方向对
* 但缺强制 schema，缺论文关键字段
* 现在更像“任意 metadata dump”，还不是“paper-grade artifact manifest”

### 整个 `artifact_io`

**总评级：B-**

* 适合作为基础设施
* 不足以单独支撑 “strictly aligned with 2404.11373”

---

## 6. 最值得你立刻改的版本

我建议你把这个模块从“自由写 metadata”升级成“论文审计专用 manifest”。

### 最小升级方案

把 sidecar 统一成下面这种结构：

```json
{
  "schema_version": "artifact_meta_v1",
  "audit_spec": "2404.11373_strict_v1",
  "paper_reference": "2404.11373v3",
  "artifact": "fig1__kerr220__sbi__paper_budget__final__20260305T101500Z.png",
  "artifact_type": "figure",
  "created_utc": "20260305T101500Z",
  "task_id": "fig1",
  "run_id": "kerr220_paper_budget",
  "case": "kerr220",
  "method": "sbi",
  "stage": "final",
  "paper_budget": true,
  "params": {
    "num_simulations_first_round": 50000,
    "num_simulations_per_round": 100000,
    "trunc_quantile": 1e-4,
    "stopping_ratio": 0.8,
    "posterior_samples": 20000
  },
  "prior": {
    "low": [20.0, 0.0],
    "high": [300.0, 0.99]
  },
  "theta_true": [67.0, 0.67],
  "diagnostics_ref": "kerr220_summary.json",
  "comparison_ref": "kerr220_pyring_posterior.npz"
}
```

这就已经非常接近你需要的 paper-grade 审计形态了。

---

## 7. 一句落地建议

这份 `artifact_io` 现在不用大改逻辑，**重点是把“任意 metadata”升级成“强约束的论文复现 manifest schema”**。
一旦你做了这个，你后面的 `quality_gates`、`Fig.1`、`coverage test`、`pyRing 对照` 就都能通过 sidecar 自动串起来，而不是靠人脑记忆。

如果你愿意，我下一条可以直接给你写出一版 **`ArtifactMetadata` dataclass + `write_paper_sidecar()` 规范草案**，字段专门对齐 2404.11373。


按 **2404.11373 Appendix A** 严格对照来审，这组 `noise` 模块里，**`whitening.py` 的总体路线是对的，而且是目前你仓库里少数真正“贴着论文原文写”的部分**；但 `psd.py` 和 `whitening.py` 之间还存在几处会影响“是否能宣称严格复现”的细节点，尤其是 **PSD 估计口径、ACF 数值归一化、时域白化是否与论文/参考文献完全同义、以及训练时噪声注入所需的数据约定**。

先给一句总评：

**总评**

* `whitening.py`：**方向正确，接近论文 Appendix A。**
* `psd.py`：**工程上可用，但离“严格对照论文数据准备”还差几条关键信息。**
* 整个 `noise` 模块：**可以作为你当前复现管线的白化核心，但还不能无保留地写成“严格复现 2404.11373 的 whitening implementation”。**

论文在 Appendix A 里对这部分说得很清楚：模拟注入时数据以 **2048 Hz** 采样并截断到 **0.1 s**，每台探测器 **204 bins**；白化是**直接在时域**做的，方法是 **PSD → ACF → Toeplitz covariance → Cholesky**，然后
[
h^{\rm white}=L^{-1}h,
]
噪声再在白化空间里以
[
x^{\rm white}(\theta)=h^{\rm white}(\theta)+\mathcal N(0,1)
]
加入。 

下面逐项审。

---

## 1. `__init__.py`：接口层没有问题

它只是把：

* `load_gwosc_strain_hdf5`
* `estimate_psd_welch`
* `acf_from_one_sided_psd`
* `covariance_from_acf`
* `covariance_from_one_sided_psd`
* `whiten_strain_from_covariance`
* `whiten_strain_from_psd`

统一导出。

从论文对齐角度，它本身不构成风险。

---

## 2. `psd.py`：能用，但“论文口径”还有几处不够硬

### 2.1 `load_gwosc_strain_hdf5()`：工程上合理，但要确认 GWOSC 文件结构假设

你假设：

* strain 数据在 `strain/Strain`
* 采样间隔在 `Xspacing`
* `meta/GPSstart` 和 `meta/Detector` 可选

这对常见 GWOSC HDF5 是合理的工程实现。论文确实说真实数据来自 GWOSC，并把 4096 Hz 下采样到 2048 Hz。

**风险点**
这不是论文的核心数学问题，但如果你后面用的 GWOSC 文件版本字段略有差异，这个 loader 会静默失败或读出不一致的元数据。
建议加：

* dataset/key existence checks
* 如果 `Detector` 缺失，报更具体的提示
* 记录 `sample_rate_hz` 与论文目标 2048 Hz 是否一致

---

### 2.2 `estimate_psd_welch()`：论文允许用 PSD 估计，但这里还不是“严格对照论文设置”

论文最后的软件说明里写，他们用 `scipy` 和 `pycbc` 从 noisy time series 估计 PSD 与 correlation matrix。
同时正文说，注入用到的 SNR 是由 **GW150914 附近数据段估计的 PSD** 得到的。

你这里采用 Welch 是合理的，但问题在于：**论文没有在你贴出的这些段落里明确给出 Welch 的具体窗、分段长度、overlap、detrend 参数**。所以：

* 你现在默认 `window="hann"`，`detrend="constant"`，`nperseg≈4s`
* 这些都是合理工程选择
* 但它们是**你自己的实现口径**，不能直接说“就是论文设置”

### 审计结论

`estimate_psd_welch()` 可以标成：

> “consistent with a standard PSD-estimation approach compatible with the paper, but exact Welch settings are not explicitly fixed by the quoted paper text.”

### 建议

你最好把 PSD 估计参数显式写进 run metadata，否则以后出现以下问题很难审：

* 同样的波形/白化代码，SNR 却和论文对不上
* 仅仅因为 PSD 参数不同，后验宽度就发生变化

至少要写：

* `window`
* `nperseg`
* `noverlap`
* `detrend`
* PSD 使用的数据时间段

---

### 2.3 缺一个对“采样率和目标输入长度”的硬检查

论文输入是：

* 2048 Hz
* 0.1 s
* 204 bins / detector。

你 `psd.py` 并没有检查：

* 读取的原始真实数据是不是 4096 Hz
* 是否已正确下采样到 2048 Hz
* 用于白化/推断的 ringdown 段长度是不是 0.1 s

虽然这不该全部写在 `psd.py` 里，但从审计角度，当前模块没有提供“论文输入形状一致性”的任何强约束。

---

## 3. `whitening.py`：这是对论文最对齐的部分

### 3.1 主路线和论文完全一致

你实现的是：

1. 从 one-sided PSD 得到 ACF
2. 用 ACF 生成 Toeplitz covariance
3. Cholesky 分解 covariance
4. 做
   [
   h^{\rm white}=L^{-1}h
   ]

这和论文 Appendix A 的文字是直接对齐的：先从 PSD 的逆傅里叶得到 ACF，再由 ACF 构造 Toeplitz covariance，再以 covariance 的 Cholesky 因子做时域白化。 

这部分可以明确说：
**你的白化数学路线与论文 Appendix A 一致。**

---

### 3.2 `_one_sided_to_two_sided_density()`：思路正确

你把 Welch 的 one-sided PSD 内部频点除以 2，再做 `irfft`。
这是因为 one-sided PSD 的中间频率已经把正负频功率折叠在一起了；如果直接 inverse transform，不还原成 two-sided 正频支，就会把 ACF 归一化搞错。

这一步在数学上是对的，而且正是从 `scipy.signal.welch(..., return_onesided=True)` 过渡到时域协方差所需要的。

### 审计建议

这里最好补一句更强的注释，说明：

* DC 和 Nyquist 不除 2
* 中间频点除 2
* 对应的是 one-sided → two-sided spectral density conversion

因为这正是最容易被以后改坏的地方。

---

### 3.3 `acf_from_one_sided_psd()`：整体正确，但这里是“最值得你再做一轮数值验证”的点

你现在用：

```python
acf_full = np.fft.irfft(two_sided_pos, n=n_fft) * sample_rate_hz
```

这在离散谱密度到自协方差的近似关系里是合理的：`irfft` 给出的是带 (1/N) 因子的离散和，而连续版本里需要乘 (\Delta f\sim f_s/N)，所以会出现一个 `* sample_rate_hz` 的因子。

**但这里非常值得做单元测试。**
因为即使数学上方向对，数值上仍可能差一个：

* (N)
* 2
* (\Delta t)
* (\Delta f)

你以后如果出现“白化后噪声方差不是 1”“训练输入缩放不对”“NoiseResamplingSNPE 里再加 N(0,1) 后整体噪声尺度不一致”，这里就是第一嫌疑点。

### 必做的两个 sanity tests

第一，对一个已知 PSD（例如常数 PSD）构造 covariance，检查得到的 ACF 是否近似 delta-like。
第二，用该 covariance 生成高斯噪声，再白化，看 `whitened` 是否近似 iid (N(0,1))。

如果这两条不过，你的时域白化虽然“形式像论文”，但不是真正等价。

---

### 3.4 `covariance_from_acf()`：与论文完全同路

Toeplitz covariance 正是论文说的构造方式。
这部分没问题。

---

### 3.5 `cholesky_lower_with_jitter()`：工程上很值钱，但不是论文原文内容

论文只说用 covariance 的 Cholesky factor，没有展开数值稳定性细节。
你这里加了 jitter escalation，这在工程上非常合理，尤其 Toeplitz covariance 可能因数值误差接近半正定边界。

### 审计表述

这属于：

* **与论文不冲突**
* **但属于工程稳定性增强，而非论文原生细节**

这点要分清，别在文档里误写成“论文就是这么做的”。

---

### 3.6 `whiten_strain_from_covariance()`：核心公式正确

你做的是：

```python
whitened = solve_triangular(l_factor, x, lower=True)
```

即 (L y = x)，所以 (y=L^{-1}x)。
这和论文的 Eq. (A1) 一致。

### 一个小问题

论文文字里说 `L` 是 **upper-triangular** Cholesky factor。
而你这里是 lower-triangular。
这不构成数学问题，因为：

* 若 (C=LL^\top) 用 lower
* 或 (C=U^\top U) 用 upper
  本质是同一个分解约定，只要你 consistently solve 就行。

但文档里最好别写“Eq. (A1): (h_{\rm white}=L^{-1}h)”却同时把 `L` 存成 lower 而不说明。
建议注明：

> Here we use the lower-triangular Cholesky factor returned by NumPy, which is equivalent up to convention.

---

### 3.7 `whiten_strain_from_psd()`：符合论文管线，但缺“输出检验”

这就是论文 Appendix A 那条链的便捷封装。
问题不在逻辑，而在**缺验证**：

论文后面立刻用白化后的表示定义
[
x^{white}=h^{white}+\mathcal N(0,1),
]
这意味着你的 `whitened` 输出应当已经满足一个很强的约定：
**在纯噪声情况下，白化后的噪声协方差应近似单位阵。** 

当前模块没有提供任何 helper 去检查这一点。

### 建议你必须加的函数

例如：

* `validate_whitening_identity_covariance(...)`
* `estimate_whitened_noise_stats(...)`

输出至少包括：

* mean
* variance
* lag-1/lag-k autocorrelation
* covariance condition number before/after whitening

这会直接帮你判断“当前白化是否足以支撑 Appendix A 的 (+\mathcal N(0,1)) 噪声注入”。

---

## 4. 对照论文，当前 `noise` 模块最大的两个风险

### 风险 1：你实现了“论文形式”，但还没证明“白化后噪声真是 (N(0,1))”

论文 Appendix A 的训练数据构造真正依赖的是：
[
x^{white}=h^{white}+\mathcal N(0,1).
]


这不是一句装饰性公式。
它要求你的 whitening 结果已经把 colored detector noise 映射成近似白噪声、单位方差尺度。否则：

* NoiseResamplingSNPE 里再加 `torch.randn_like(x)` 就不再等价于论文
* 训练/验证分布都会偏

而当前代码还没有把这一点验证闭环做出来。

### 风险 2：PSD 估计和白化是“兼容论文”，但未锁定到论文的具体估计口径

论文只明确说：

* PSD 来自 GW150914 附近数据段
* 用它来估计 correlation matrix 和时域白化。 

你现在的 Welch 设置是合理的，但如果没有固定：

* 取哪段数据
* Welch 参数
* 下采样顺序
* ringdown 截断顺序
  那么同一个项目里的不同 run 就可能得到不同白化矩阵和不同 SNR 标定。

---

## 5. 我的评级

### `psd.py`

**评级：B**

* 工程上合理
* 与论文不冲突
* 但不是“严格锁定论文口径”的 PSD 估计规范

### `whitening.py`

**评级：A-**

* 白化数学路线与 Appendix A 高度一致 
* 主要差距不在公式，而在缺少“白化后噪声确实 ~ (N(0,1))”的数值验证

### 整个 `noise` 模块

**总评级：A-/B+ 之间**

* 已经足够作为你当前论文复现主线的白化基础
* 但若要写“严格对照 2404.11373 Appendix A”，还需要补验证与元数据固定

---

## 6. 你下一步最值得立刻加的 6 个审计项

第一，固定并记录 PSD 估计配置。
至少把 `window`、`nperseg`、`noverlap`、`detrend`、所用 GW150914 邻域时间段写进 summary。论文只说 PSD 来自 GW150914 邻域，但没有替你固定这些实现细节。

第二，给 `whiten_strain_from_psd()` 加一个“纯噪声白化测试”。
目标是检验白化后噪声的均值约 0、方差约 1、自相关接近 0。只有这样，你后面在训练时加 (\mathcal N(0,1)) 才和论文一致。

第三，明确 lower/upper Cholesky 约定。
避免未来读代码的人误以为你和论文公式不一致。

第四，检查 white-noise injection 的时机。
论文是在 **白化后** 加噪声。
所以你要确保 simulator / training pipeline 里不是“先加 colored noise 再白化”和“白化后再额外加一层标准正态”同时发生。

第五，增加对输入长度的硬校验。
论文要求每台探测器 204 bins，对应 2048 Hz × 0.1 s。
这应当在 whitening 入口就能验证。

第六，给 whitening sidecar / run summary 写入：

* PSD source segment
* covariance size
* Cholesky jitter used
* whitened noise validation stats
  这样以后一旦 Fig.1 变形，你能快速判断问题在不在 whitening。

---

## 7. 一句最终结论

**这套 `noise` 代码已经抓住了 2404.11373 Appendix A 最关键的白化思想：时域白化、PSD→ACF→Toeplitz→Cholesky、再在白化空间加入标准正态噪声。真正还差的不是公式，而是“把这种实现证明为数值上等价于论文设定”的验证层。**

如果你愿意，我下一条可以继续最有价值的一步：
我直接把这组 `noise` 代码整理成一份“逐函数审计清单 + 必做单元测试列表”，专门服务你快速形成这部分的专业审计能力。






按 **2404.11373** 的复现目标来审，这份 `kerr` 模块的定位是：

**它作为“GR Kerr 基线映射器”是合理的，但它还不是一份“严格贴合论文具体实验对象”的 QNM 模块。**
原因不在基础公式，而在于：论文这篇工作的核心推断对象是 ((M_f,\chi_f)) 加各模的 **振幅 (A_{lmn})** 和 **相位 (\phi_{lmn})**，三组注入只涉及 ((2,2,0))、((2,2,1))、((3,3,0))，而 Fig.1/2 的对照也是围绕这些模式做的。论文 Table I 明确列出 Kerr220、Kerr221、Kerr330 三组注入及其模式内容，且固定 (M_f=67M_\odot,\chi_f=0.67)。 论文 Table II 只对 (M_f,\chi_f,A_{220},A_{221},A_{330},\phi_{lmn}) 设先验范围，并没有在这篇文章里直接引入你代码中的 (\alpha_r,\alpha_i) 偏差参数。

所以一句话先定性：

**如果目标是复现 2404.11373 本文结果，这个模块“基础上够用，但偏研究扩展版，不是论文最小同款版”。**

下面分开审。

---

## 1. 这份代码做的物理事是什么

这段代码的核心任务是：

给定

* 黑洞末态质量 (M_f)
* 末态自旋 (\chi_f)
* 模式标签 ((l,m,n))

返回该 Kerr QNM 的

* 复无量纲频率 (\omega M)
* 物理频率 (f_{lmn})
* 阻尼时间 (\tau_{lmn})

这在 ringdown 里是非常基础的一层。因为真正的时间域 ringdown 模型，形式上就是“每个模式一个阻尼正弦”，而这个阻尼正弦的频率和阻尼时间由 Kerr QNM 谱给定；再乘上模式振幅和相位，就得到论文里推断的时间域信号对象。论文的三组注入正是围绕 220、221、330 三个模式构建的。

所以这个模块在你的整条管线里，属于：

> 从 ((M_f,\chi_f)) 到 QNM 频率/阻尼时间的“谱映射层”。

这层必须对，否则后面的波形、SNR、后验都会全偏。

---

## 2. 和论文一致的地方

### 2.1 `mass_seconds_from_msun()` 的思路是对的

你用

```python
MTSUN_SI = 4.925490947e-6
```

把太阳质量换成几何单位秒，这正是从无量纲 Kerr 频率 (\omega M) 转成物理频率/时间尺度所需的标准常数。

对初学者来说，这一步的物理意义是：

* 在 GR 的 Kerr QNM 文献里，频率通常先算成“乘了黑洞质量后的无量纲量”
* 真正变成 Hz 时，要再除以黑洞质量对应的时间尺度 (M_f G/c^3)

你的实现：

* `frequency_hz = omega_r / (2π m_sec)`
* `damping_time_s = m_sec / |omega_i|`

这两条都是标准关系。

### 2.2 用 `qnm` 包作为主求值器，方向是对的

论文引用的 QNM 理论背景包含 Berti 等综述和 Kerr/QNM 相关文献。
你直接调用 `qnm` 包拿 Kerr 模式频率，是很合理的工程做法。对于 2404.11373 这种以时间域 ringdown 推断为主的论文，代码层并不要求你必须自己解 Teukolsky 方程；只要模式频率是对的即可。

### 2.3 sign convention 处理是合理的

你强制

```python
if np.imag(omega) >= 0.0:
    omega = complex(np.real(omega), -abs(np.imag(omega)))
```

这保证采用
[
e^{-|\Im\omega|, t/M}
]
的阻尼号约定。
这对 ringdown 实现非常重要，因为很多库/文献的复频率号约定不完全一样。你这里统一成负虚部，工程上是正确的。

---

## 3. 和论文不一致、或不够“严格对照”的地方

### 3.1 最大偏差：论文这篇文章本身并没有用 (\alpha_r,\alpha_i) 作为推断参数

你在

```python
kerr_qnm_physical(..., alpha_r=0.0, alpha_i=0.0)
```

里引入了对实部和虚部的分数偏差：

* `omega_r -> omega_r (1 + alpha_r)`
* `|omega_i| -> |omega_i| (1 + alpha_i)`

这很适合你自己的课题方向，尤其你要做“可解释的 QNM 偏差参数 (\alpha)”的黑洞光谱学推断。
但**它不是 2404.11373 本文的实验设定**。这篇文章的主要结果里，Fig.1/2/coverage test 都是在 **GR Kerr 基线** 下做的，即从 (M_f,\chi_f) 唯一确定模式频率和阻尼时间，再推断振幅与相位。论文 Table I / Table II 里没有 (\alpha_r,\alpha_i) 这一类偏差参数。 

所以审计结论是：

* 对**你的课题扩展**：这是好设计
* 对**“严格复现 2404.11373”**：这是超出论文设定的扩展接口，不应默认暴露在论文复现主路径里

更稳妥的做法是把它标成：

* `GR baseline mode`: `alpha_r=alpha_i=0`
* `beyond-GR experimental mode`: 非零 alpha

---

### 3.2 第二个偏差：fallback 的 Berti fit 只覆盖 `(2,2,0)` 和 `(3,3,0)`，但论文 Fig.1 还需要 `(2,2,1)`

这是非常关键的一条。

论文三组 case 分别用到：

* Kerr220: 220
* Kerr221: 220 + 221
* Kerr330: 220 + 330 

而你当前 `_BERTI_FIT_COEFFS` 只给了：

* `(2,2,0)`
* `(3,3,0)`

没有 `(2,2,1)`。

这意味着只要发生以下任一情况：

* `qnm` 包不可用
* `method="fit"`
* `method="auto"` 但 `qnm` 调用异常

那么 **Kerr221 case 会直接失败**。

这和论文复现目标直接冲突，因为 Kerr221 是三大主 case 之一。

这是我认为这段代码里**最值得立刻修的硬问题**。

---

### 3.3 第三个偏差：`_validate_spin()` 只允许 `0 <= chi < 1`，不支持反向自旋

对 2404.11373 本文本身，这不构成问题，因为它固定 (\chi_f=0.67)。
但从一般 Kerr QNM 工具角度看，这个约束太窄。物理上 Kerr 自旋通常允许 (-1<\chi<1)，只是模式频率会随 (m) 与自旋号一起变化。

如果你未来要做更广泛的推断、模拟或 coverage tests，这个限制会妨碍你。
所以：

* **对论文复现**：可以接受
* **对通用 QNM 库**：不够完整

---

### 3.4 第四个偏差：`method="auto"` 的回退策略对论文复现不够透明

你现在逻辑是：

* 先试 `qnm`
* 出错就 silent fallback 到 Berti fit

这在工程上很方便，但对论文级审计不够好。
因为一旦 fallback 发生，结果质量就变成：

* 220 / 330 还能算
* 221 直接报错
* 或不同模式精度不一致

而你 summary/日志 里如果没记录 fallback 是否发生，后续就很难解释：

* 为什么某轮 Kerr221 崩了
* 为什么某个 case 与 pyRing 有系统偏差
* 到底是推断器问题还是 QNM 频率映射器问题

### 更适合论文审计的策略

对于 2404.11373 复现主线，我建议：

* 默认 `method="qnm"`
* 若 `qnm` 不可用，直接 fail fast
* 只有在“debug / no-external-deps 模式”下，才允许 fallback，而且要明确写进 metadata

因为这篇论文的三大 case 里包含 overtone 221，fallback 不完整时静默回退不是好主意。

---

### 3.5 第五个偏差：没有显式审计“模式集合是否与论文 case 匹配”

你有 `map_modes_to_qnms()`，但它只是把传进来的 modes 全部映射。
从工程复现角度，更好的做法是提供论文级显式 helper，例如：

* `kerr220_modes() -> [(2,2,0)]`
* `kerr221_modes() -> [(2,2,0),(2,2,1)]`
* `kerr330_modes() -> [(2,2,0),(3,3,0)]`

因为论文 Table I 就是按这三类 case 定义的。
这样可以减少上游误传 mode list 的风险。

---

## 4. 从“完全不懂引力波分析”的角度，怎样理解这段代码是否靠谱

你可以用三个非常实用的问题来审这类 QNM 模块。

### 第一问：这个模块是不是把黑洞“振铃音高和衰减速度”算出来了？

是的。
`frequency_hz` 对应“音高”，`damping_time_s` 对应“多久衰减掉”。

### 第二问：它是不是对论文关心的三个模式都支持？

当前答案是：**主路径支持，fallback 不完整。**

* 220：支持
* 330：支持
* 221：依赖 `qnm` 包；fallback 不支持

这就是最关键的审计点。

### 第三问：它是不是和论文实际推断参数一一对应？

部分对应，部分超出。

* 对应：(M_f,\chi_f) → (f_{lmn},\tau_{lmn})
* 超出：(\alpha_r,\alpha_i) 不是这篇论文 Fig.1/2 的原生参数

---

## 5. 对你当前课题方向，这段代码反而有一个很好的“研究接口”

虽然它不完全等于 2404.11373 的最小同款实现，但对你的课题——“端到端黑洞光谱学推断并估计可解释的 QNM 偏差参数 (\alpha)”——它有一个明显优点：

你已经把 beyond-GR 偏差写成了

* 实部偏差 `alpha_r`
* 虚部偏差 `alpha_i`

这等价于对

* 模式频率偏差
* 阻尼时间偏差

做最直接的参数化。
这和你后面想做“QNM 偏差参数可解释推断”非常契合。只是你需要在工程上明确分层：

* **论文复现层**：固定 `alpha_r = alpha_i = 0`
* **研究扩展层**：开放 `alpha_r, alpha_i` 或更细的 mode-dependent (\alpha_{lmn})

否则复现与研究扩展会混在一起，审计会变得很乱。

---

## 6. 我给这份代码的评级

### 如果标准是“能否作为 2404.11373 复现中的 Kerr 谱映射器”

**评级：B+**

优点：

* 基本公式对
* `qnm` 主路径合理
* sign convention 合理
* mass-to-time conversion 合理

缺点：

* fallback 不支持 221，这对论文主 case 是硬伤
* 默认暴露 alpha 偏差，超出论文最小设定
* silent fallback 不利于论文级审计

### 如果标准是“是否适合作为你未来黑洞光谱学/偏差参数课题的基础库”

**评级：A-**

因为它已经把 GR Kerr 基线和 beyond-GR 偏差接口连起来了，只是还需要更严格的模式覆盖和审计元数据。

---

## 7. 最值得你立刻改的 5 件事

第一，**补 `(2,2,1)` 的 fallback 支持**。
否则 Kerr221 无法在无 `qnm` 包情形下稳定跑完，而 Kerr221 是论文三大核心注入之一。

第二，把 `method="auto"` 改成对论文复现更严格的策略：
复现主线默认 `method="qnm"`，失败就报错；只有 debug 模式才允许 fallback。

第三，把 `alpha_r, alpha_i` 从“论文复现主接口”里剥离出来，至少在命名或文档上标明：

* `kerr_qnm_physical_gr(...)`
* `kerr_qnm_physical_biased(...)`
  这样更不容易混淆“复现”和“扩展研究”。

第四，增加 `case -> modes` 的显式 helper。
这会直接减少你上游脚本把 Kerr221 / Kerr330 模式列表传错的概率。

第五，给这个模块加最关键的数值单测：

* 对 (M_f=67,\chi_f=0.67) 下的 220 / 221 / 330，检查频率和阻尼时间是否单调合理
* 220 与 330 的频率大小规律是否合理
* `alpha_r>0` 是否只提高频率，`alpha_i>0` 是否只缩短阻尼时间

---

## 8. 一句结论

**这份 `kerr` 代码已经是一个合格的 Kerr QNM 映射层，但按 2404.11373 严格复现标准，它还差最后一步“把论文三大 case 的模式支持、复现主线的 GR-only 设定、以及 fallback 审计透明化”做扎实。最大硬问题是 `(2,2,1)` fallback 缺失。**

如果你愿意，我下一条可以直接把这份 `kerr.py` 变成“逐函数审计清单”，格式和前面一样：
函数名、论文对应关系、当前对齐度、风险等级、必改建议。



这批新文件里，**最关键的是 `injection.py` 和 `ringdown.py`**。它们基本把 2404.11373 的“注入生成 + 时间域 ringdown 波形 + 双探测器投影”主链路补齐了。总体结论是：

**这组代码已经很接近论文主实验对象，但仍然是“带你自己课题扩展接口的工程实现”，不是完全剥离扩展项后的论文最小同款。**
最强的部分是：它确实对准了论文的三大核心设定——2048 Hz、0.1 s、204 bins、H1/L1 双探测器、时间域 ringdown 模式叠加。论文 Appendix A 明确写了输入是 2048 Hz 采样、0.1 s 截断、每台探测器 204 bins、双探测器拼接后 408 维；正文的 Table I 也明确了三组注入 Kerr220 / 221 / 330 的模式内容与固定的 (M_f=67M_\odot,\chi_f=0.67)。 

下面我按模块给你一版“对照论文的审计结论”。

---

## 1. `runtime_env.py` 与 `seed.py`：不属于论文核心，但对可复现性是加分项

### `runtime_env.py`

`ensure_local_runtime_home()` 的作用是把 `HOME/USERPROFILE/ARVIZ_DATA` 指到项目内可写目录，避免第三方库在系统环境下写缓存失败。这个完全属于工程稳健性措施，不是论文内容，但对长跑训练和本地复现很有价值。

**审计判断**：

* 和论文不冲突
* 不增加物理风险
* 属于“工程增强，不是论文要求”

### `seed.py`

`set_global_seed()` 同时设置 Python / NumPy / Torch 的随机种子，并可选启用 deterministic cuDNN。对 SBI 训练的复现实验很重要。

**审计判断**：

* 论文没有专门要求这部分
* 但对你的“审计能力”和可复跑性很关键
* 建议每个 run summary 都记下归一化后的 seed

---

## 2. `config.py`：基础合格，但还不够“论文复现配置规范”

`load_yaml_config()` 和 `project_root_from_file()` 都很简单直接，逻辑没问题。

但从论文复现角度，这个模块现在只是“能读 config”，还没有把论文关键字段做成强约束。
按 2404.11373 严格对照，config 至少应该强制包含：

* case 名称：`kerr220 | kerr221 | kerr330`
* 注入真值：`mass_msun=67`, `chi_f=0.67`
* 固定天空位置与极化
* `sample_rate_hz=2048`
* `duration_s=0.1`
* modes 列表与 Table I 对齐
* TSNPE / Table III 预算
* prior / whitening / PSD 来源

你现在这些约束主要散落在 `injection.py` 里，而不是在 config 层统一验收。

**建议**：后面可以加一个 `validate_paper_case_config()`，专门检查 Table I / II / III 对齐。

---

## 3. `ringdown.py`：这是目前最值得肯定的一份，但也藏着一个关键风险

### 3.1 对齐论文的地方很多

这份文件的 docstring 已经明确写了自己在实现论文对象：

* Eq. (1): 探测器投影
* Eq. (2a, 2b): 时间域阻尼正弦模态叠加
* Eq. (3): 相位模型
* Eq. (5): (Y_+,Y_\times) from spin-weighted spherical harmonics 

#### `build_time_array()`

默认 2048 Hz、0.1 s，产生 204 bins，这与论文 Appendix A 完全对齐。

#### `qnm_complex_frequency()`

把物理频率和阻尼时间组合成
[
\tilde\omega = 2\pi f + i/\tau
]
再交给后面的时间域阻尼项使用。这个结构是合理的，而且和 ringdown 的标准写法一致。

#### `spin_weighted_spherical_harmonic()` 与 `y_plus_y_cross()`

你显式实现了 Wigner small-d，再构造 ({}*{-2}Y*{lm})，最后生成 (Y_+) 和 (Y_\times)。对论文这种只涉及少量低阶模式、固定倾角的时间域注入来说，这是完全正当的实现路线。

#### `generate_ringdown_polarizations()`

这里是核心：

* 先根据 `t_start_s` 做 Heaviside gate
* 每个 mode 构造指数衰减包络
* 再按 cos/sin 与 (Y_+,Y_\times) 叠加成 (h_+, h_\times)

这正是论文三组注入所需的时间域波形层。

### 3.2 最大风险：`h_cross` 的符号/相位约定必须与参考实现一致

你这里写的是：

* `h_plus += A * envelope * cos(phase_t) * y_plus`
* `h_cross += A * envelope * sin(phase_t) * y_cross` 

这在某种约定下是完全合理的，但 `h_cross` 的正负号、`sin` 前是否有额外负号、`y_cross` 是否需要乘 (i) 或取虚部，往往和具体文献/代码库约定有关。
这不会让代码“跑不起来”，但会影响：

* 极化与探测器响应的一致性
* 和 pyRing / LAL 对照时的相位约定
* 某些模式组合下的干涉结构

**所以这份波形代码最该做的单测不是数值稳定性，而是“和参考实现对照”**。
至少要对 220 / 221 / 330 三个 case，在固定 (M_f=67,\chi_f=0.67)、固定倾角下，与 pyRing 或你参考实现的 (h_+, h_\times) 做时域对比。否则它只能算“形式正确”。

### 3.3 第二个风险：你保留了 `QNMBias`

`RingdownMode` 里还带了 `bias: QNMBias | None`，而 `qnm_complex_frequency()` 也支持 `alpha_r, alpha_i` 偏差。
这和你自己的课题方向高度匹配，但不是 2404.11373 本文的最小对象。论文正文和 Table II 的主设定没有这两个偏差参数。
因此这和前面 `kerr.py` 一样：

* 对你的研究扩展是优点
* 对“严格复现论文”则应默认关闭

---

## 4. `injection.py`：这是整批文件里最关键、也最接近论文实验主线的一份

这份代码几乎把论文 Table I 的注入管线串起来了。

### 4.1 明显对齐论文的地方

#### a) 2048 Hz × 0.1 s × 204 bins 被硬检查

```python
if time_s.shape[0] != 204 and np.isclose(sample_rate_hz, 2048.0) and np.isclose(duration_s, 0.1):
    raise RuntimeError(...)
```

这正对齐 Appendix A 的输入尺寸。

#### b) 双探测器 H1 / L1 投影

它调用了：

* `antenna_pattern`
* `detector_strain`
* `gmst_from_gps`
* `time_delay_from_geocenter_s`
* `h1_geometry`, `l1_geometry` 

这和论文 Fig.1/Appendix A 的双探测器输入口径一致。

#### c) 注入参数结构与论文对象一致

读取：

* `mass_msun`
* `chi_f`
* sky position / polarization / inclination
* H1 GPS time
* `modes` 中各个 mode 的 `l,m,n, amplitude, phase`

这正是论文 Table I + Table II 的实验对象：固定 (M_f,\chi_f) 与 sky / (\psi) / (\iota)，不同 case 开启不同模式，并推断 (A_{lmn},\phi_{lmn})。

#### d) 显式保存模式级 QNM 记录

`qnm_records` 会把每个 mode 的

* `frequency_hz`
* `damping_time_s`
* `omega_dimensionless_real/imag`

都写进 metadata。
这对审计非常有价值，因为以后你可以直接从 sidecar 判断“某次注入到底用了哪一组 QNM 频率”。

---

### 4.2 最大偏差：当前 `noise_std` 加噪方式不是论文 Appendix A 的白化空间噪声模型

这里要非常明确：

你在 `injection.py` 里做的是：

```python
if noise_std > 0.0:
    strain = strain + rng.normal(...)
```

也就是**直接在探测器时域 strain 上加独立同分布高斯噪声**。

而论文 Appendix A 的训练数据构造是：

1. 从真实 PSD 得到 covariance
2. 时域白化
3. 在白化空间里加 (\mathcal N(0,1))
   即
   [
   x^{white}=h^{white}+\mathcal N(0,1).
   ]


这两者不是一回事。

所以审计结论必须分清：

* **如果 `noise_std=0`**：这份 `generate_injection()` 很适合生成论文那种“零噪声注入”的 deterministic 信号模板
* **如果 `noise_std>0`**：它生成的是“简单白高斯噪声上的注入”，不是论文 Appendix A 的真实训练/推断输入模型

这条非常关键。你后面如果还要做 Fig.1 论文级复现，就不应该把 `injection.py` 的这个 `noise_std` 路径当成最终训练数据来源。

### 4.3 第二个偏差：默认 `method="auto"` 和 `alpha_*` 继承了前面 `kerr.py` 的审计问题

`generate_injection()` 会把 `qnm.method`、`alpha_r`、`alpha_i` 传给 `kerr_qnm_physical()`。
这意味着：

* 复现主线下你必须确保 `alpha_r=alpha_i=0`
* 最好让论文复现 case 强制 `method="qnm"` 而不是 `auto`

否则你有可能在自己都没注意时走到 fallback / beyond-GR 分支。

### 4.4 第三个偏差：`reference_detector="H1"`、`gps_h1` 是合理的，但必须和论文时间约定保持一致

你这里通过 `gps_h1` 算 GMST，再以 H1 为 reference detector 计算其他台站时间延迟。
论文 Table I 确实固定了 H1 ringdown 起始时间，并从天空位置推导到 L1 的时差。
所以这条总体是对的。
但它的正确性强依赖你前面 detector response/time-delay 代码的约定。如果那边有号约定问题，这里会把错传播到注入层。

---

## 5. 新增 `__init__.py` 文件们：问题不大

这些 `__init__.py` 做的事情都比较简单：

* 暴露 `generate_injection` 
* 暴露 ringdown waveform 接口 
* core package 暴露 config 
* utils 暴露 `set_global_seed` 

没什么论文层面的风险。

---

## 6. 对照论文的综合评级

### `ringdown.py`

**评级：A-**

* 已经非常接近论文的时间域 ringdown 波形对象
* 最大风险是极化/相位约定要和参考实现对照确认 

### `injection.py`

**评级：B+**

* 零噪声注入主链路非常接近论文 Table I / Appendix A
* 但 `noise_std` 路径不是论文白化空间噪声模型，不能直接当论文训练输入 

### `runtime_env.py`, `seed.py`, `config.py`

**评级：B 到 A-**

* 工程基础设施合格
* 对论文复现本身不构成核心差异

---

## 7. 我最建议你立刻改的 6 件事

第一，把 `injection.py` 里的 `noise_std` 明确改名或注释成：

> debug/simple additive white noise only; not paper Appendix-A whitened noise model
> 否则未来很容易误用。

第二，为论文复现主线加一个强制模式：

* `alpha_r = alpha_i = 0`
* `qnm.method = "qnm"`
* `noise_std = 0`
  这样能把“论文复现”和“课题扩展”彻底分开。

第三，给 `generate_ringdown_polarizations()` 做一个与 pyRing / 基准实现的时域对照测试，重点检查 `h_cross` 号约定和 221/330 的模式叠加相位。

第四，把论文 Table I 的三种 case 做成显式配置模板，而不是完全靠自由 YAML 组合。
这样你可以减少“模式表写错、倾角写错、GPS 写错”的低级错误。

第五，让 injection metadata 额外记录：

* `case`
* `paper_reference`
* `paper_faithful=true/false`
* `uses_whitened_noise_model=false`
  这会极大提升审计透明度。

第六，把 `config.py` 升级成“论文 case 验证器”，而不只是 YAML loader。

---

## 8. 一句总的判断

**这批新增文件把你的工程从“有 QNM/白化/推断组件”推进到了“已经能生成论文对象级别的注入与时间域波形”。真正离 2404.11373 严格复现还差的，不是主链路，而是把“零噪声注入”“白化空间噪声”“论文 GR 基线”和“你自己的 QNM 偏差扩展”这四条路径彻底拆清楚。**

如果你愿意，下一步最有价值的是：我可以把你到目前为止贴过的所有模块，整理成一份**总表式审计矩阵**，按“模块—论文对应段落—当前评级—最大风险—优先修复项”给你一页式汇总。
