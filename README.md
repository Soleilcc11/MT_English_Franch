# MT_English_Franch

英语 → 法语 Transformer：双框架实现、注意力消融与推理优化。

从随机初始化实现 Encoder–Decoder Transformer，以 PyTorch 为主实现、TensorFlow 为独立复现。覆盖数据审计、共享 BPE、30 epoch 训练、注意力消融、跨框架输出和梯度对齐、KV Cache、SDPA 对照、冻结后的测试评分、bootstrap 和错误分析。

## A0 解码扩展（2026-11-04）

新增 PyTorch beam search、长度归一化与 `compare-decode` 一键对照。直接使用已完成的 A0 seed 42 最佳 checkpoint，在验证集选择 beam 参数后冻结，并完成测试、bootstrap 和自然生成测速；无需重新训练。旧 A0–A3 结果只读，新结果保存至 `output/host4090-torch27-decode/A0-seed42/`。

完整服务器更新、验证、后台启动、恢复和翻译命令见 [Greedy / Beam Search 操作指南](scripts/BEAM_SEARCH.md)。GPU 上的 BLEU、延迟和显存由服务器实际运行后生成。

本次开发机 CPU 验证通过 45 项单元/流程测试（含 19 项新增解码与评估流程测试），并通过 399 项 PyTorch 2.7.0 / TensorFlow 2.15.0 前向、梯度和 SGD 更新对齐。服务器 CUDA/BF16 的质量与速度以实际评估为准。

## 当前状态

代码和真实数据准备已在开发机完成。GitHub 仓库包含代码、配置、安装脚本和验证摘要；原始语料、预处理数据、虚拟环境及 checkpoint 不纳入 Git，需要按下文生成。本次开发机器为 Apple Silicon Mac，**没有 CUDA GPU**。开发机 `output/smoke/` 中的训练、翻译和性能结果均属于合成数据上的 CPU 冒烟测试，不能写成 IWSLT 正式成绩或 RTX 4090 性能。

已提交的验证摘要见 [output/verification/summary.json](output/verification/summary.json)。执行检查后，详细日志和报告写入本地 `output/`；开发机的完整记录包括 `output/implementation_status.json`、`output/tests.log`、`output/parity/report.json` 和 `output/overfit/report.json`。10 次正式训练及 GPU 性能实验必须在 4090 主机执行；正式结果将由实际训练后生成，项目不会预填 BLEU 或加速倍数。

## 目录

```text
scripts/
  run.py                    # 全部命令入口
  configs/                  # 正式配置、CPU 冒烟配置
  enfr/                     # 数据、模型、训练、评估、诊断和报告
  tests/                    # 数值正确性、数据与断点恢复测试
cache/
  raw/                      # 固定 revision 的原始压缩包
  prepared/                 # 审计、8k BPE、完整三划分、batch 清单
  smoke/                    # 显式标记的合成小数据
output/
  runs/                     # A0/A1/A2 × 3 seeds + A3 × 1 seed
  parity/                   # 正式模型配置的 CPU FP32 数值对齐
  overfit/                  # 32 个真实句对的小模型记忆测试
  scaling/                  # 缩放因子诊断
  smoke/                    # 不作为正式结果
  report/                   # 汇总、bootstrap、人工复核表
```

项目目录名为 `MT_English_Franch`，下面的命令均从项目根目录执行。默认路径相对于项目根目录定位，不依赖开发机绝对路径。若同步已有项目副本，可保留 `cache/prepared/`，排除 Mac 的全部 `.venv*/` 目录。

## 从 GitHub 获取项目

在训练服务器上先通过平台说明与 `df -h` 确认数据盘挂载目录，进入该目录后执行：

```bash
git clone https://github.com/Soleilcc11/MT_English_Franch.git
cd MT_English_Franch
```

随后按下面的主机配置安装环境、准备数据并验证。GitHub 中的 `cache/` 只有目录占位文件，首次克隆后必须执行数据准备命令。

## 当前选定主机：RTX 4090 / PyTorch 2.7.0 / Python 3.11

当前部署目标使用镜像 `NVIDIA/cuda-samples/CUDA13.0-Torch2.7.0-Python3.11:v2`，硬件为 RTX 4090 24GB × 1、16 vCPU、120GB 内存，系统盘 30GB、数据盘 50GB。目标服务器上的实际驱动、框架构建及 GPU 可用性仍需检查。

本部署配置使用 **PyTorch 2.7.0 + 官方 CUDA 12.8 构建**，并在另一环境中使用 **TensorFlow 2.15.0.post1**。PyTorch 2.7.0 的官方 CUDA 构建提供 11.8、12.6、12.8；镜像名称中的 CUDA 13.0 不能直接视作 `torch.version.cuda`。参见 [PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/#v270)。安装脚本固定使用 cu128，并保存实际环境信息。

2026-11-02 已在本地 Mac CPU 上用 PyTorch 2.7.0 通过全部 26 项测试，并与 TensorFlow 2.15.0 通过 399 项 FP32 数值对齐。记录位于 `output/host4090-torch27/local-cpu-tests.log`、`local-cpu-parity/report.json` 和 `local-cpu-environment.json`。这验证了代码在新 PyTorch 版本下的 CPU 行为；Linux CUDA/BF16 行为和性能仍待服务器实测。

在服务器先用 `df -h` 确认数据盘挂载位置，将项目副本放在数据盘上，在项目根目录运行：

```bash
bash scripts/setup_4090_torch27.sh --dry-run
bash scripts/setup_4090_torch27.sh
```

默认用 `python3` 创建 Python 3.11 环境；若其不是 3.11，可指定 `ENFR_PYTHON=python3.11 bash scripts/setup_4090_torch27.sh`。脚本创建 `.venv-torch27/` 和 `.venv-tf215/` 两个独立环境，不使用镜像内预装的第三方包；关闭 pip 下载缓存，降低磁盘占用。它只安装依赖和检查 GPU，不自动启动正式训练。

首次克隆后，使用固定的数据集 revision 下载和预处理；若已同步完整的 `cache/prepared/`，则无需重新生成：

```bash
.venv-torch27/bin/python scripts/run.py prepare \
  --revision c18a4f81a47ae6fa079fe9d32db288ddde38451d
```

该命令下载英法压缩包，完成清洗、分词器训练、编码及数据审计。两框架与全部消融实验必须共用这一次生成的 prepared 数据，不要分别训练分词器。

随后运行跨框架对齐、两环境测试及 GPU 训练吞吐检查：

```bash
if [ ! -f cache/smoke/audit.json ]; then
  .venv-torch27/bin/python scripts/run.py smoke-data
fi
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=scripts \
  .venv-torch27/bin/python -m unittest discover -s scripts/tests -v
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=scripts:scripts/tests \
  .venv-tf215/bin/python -m unittest \
  test_resume.ResumeTests.test_tensorflow_optimizer_and_dropout_rng_resume \
  test_pipeline.PipelineTests.test_tensorflow_full_pipeline_resume -v
.venv-torch27/bin/python scripts/run.py check-parity \
  --torch-python .venv-torch27/bin/python --tf-python .venv-tf215/bin/python \
  --output output/host4090-torch27/parity
.venv-torch27/bin/python scripts/run.py overfit --device cuda \
  --output output/host4090-torch27/overfit
.venv-torch27/bin/python scripts/run.py profile --framework torch \
  --output output/host4090-torch27/profile-torch.json
.venv-tf215/bin/python scripts/run.py profile --framework tensorflow \
  --output output/host4090-torch27/profile-tensorflow.json
```

这些检查全部通过后，运行以下队列。模型与训练配置仍为 base.json：4+4 层、d_model=256、4 heads、micro-batch=32、有效 batch=128、BF16、30 epoch。显存调整必须在全部正式实验前完成。

```bash
.venv-torch27/bin/python scripts/run.py run-suite --finalize \
  --runs output/host4090-torch27/runs \
  --torch-python .venv-torch27/bin/python --tf-python .venv-tf215/bin/python
```

该队列将训练、冻结清单、测试结果和报告保存到 `output/host4090-torch27/` 下，与旧环境的产物区分。全套 PyTorch 实验须统一使用这一版本；升级框架后需重新验证数值对齐和速度，不能沿用 PyTorch 2.1.0 的性能结论。下面通用步骤中的 `.venv-torch` / `.venv-tf` 在本部署中分别替换为 `.venv-torch27` / `.venv-tf215`；默认输出路径也需相应替换。

## 原始 PyTorch 2.1.0 复现环境（保留）

需要 Python 3.11 和可用的 NVIDIA 驱动。两个框架分别建环境，避免其 CUDA 依赖混装。框架 API 以本地验证的 PyTorch 2.1.0 / TensorFlow 2.15.0 为基础；Linux TensorFlow 锁定为 2.15.0.post1，采用官方发布的 [and-cuda 安装修复](https://blog.tensorflow.org/2023/12/tensorflow-215-update-hot-fix-linux-installation-issue.html)，避免 2.15.0 的 TensorRT 依赖问题。GPU 运行需要执行 doctor、profile 和 GPU 测试，不能由 CPU 测试替代。

在训练机的项目根目录执行：

```bash
python3.11 -m venv .venv-torch
python3.11 -m venv .venv-tf
.venv-torch/bin/python -m pip install --upgrade pip
.venv-tf/bin/python -m pip install --upgrade pip
.venv-torch/bin/python -m pip install -r scripts/requirements-torch.txt
.venv-tf/bin/python -m pip install -r scripts/requirements-tensorflow.txt
.venv-torch/bin/python scripts/run.py doctor --framework torch --require-gpu --output output/doctor-torch.json
.venv-tf/bin/python scripts/run.py doctor --framework tensorflow --require-gpu --output output/doctor-tf.json
.venv-torch/bin/python -m pip freeze > output/torch-environment.lock.txt
.venv-tf/bin/python -m pip freeze > output/tensorflow-environment.lock.txt
```

本地验证使用 `.venv/bin/python`，该环境仅供 Mac CPU 验证。正式 `base.json` 的 BF16 训练会拒绝在没有 GPU 的环境运行，防止误启动大量 CPU 训练。

## 数据来源和审计

数据为 [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017) 的英法双语压缩包，固定仓库 revision，并记录 SHA-256。验证集为 dev2010，测试集为 tst2010–2015 合集；不是单独的 tst2017。

本次实际审计：

| 项目 | 数量 |
|---|---:|
| 原始训练句对 | 232,825 |
| 删除重复句对 | 2,062 |
| 删除与验证/测试英文源句重合的训练样本 | 168 |
| 删除任一端含 BOS/EOS 后超过 128 token 的训练句对 | 172 |
| 最终训练句对 | 230,423 |
| 验证句对 | 890 |
| 测试句对 | 8,597 |

保留参考法语原文用于评分。仅模型输入执行 NFC 和空白标准化，保留大小写、重音和撇号。分词器仅用清理后的训练文本训练，之后才过滤长度。共享 8,000 词表；PAD/UNK/BOS/EOS 分别为 0/1/2/3。

原始法语中存在句内 Unicode 行分隔符；解析按物理换行读取，不能用 `str.splitlines()` 拆开后重新配对。测试已覆盖此问题。

从头准备数据：

```bash
.venv-torch/bin/python scripts/run.py prepare \
  --revision c18a4f81a47ae6fa079fe9d32db288ddde38451d
```

已有 `cache/prepared/audit.json` 时不会覆盖。可使用 `--output` 生成另一份数据，或通过 `--archive` 和 `--revision` 使用已有压缩包。两框架共享同一份 prepared 数据与 tokenizer；训练前重新校验文件哈希。

数据保持上游署名和用途约束；仓库分发代码和复现脚本，原始语料留在缓存中。数据许可信息以来源数据卡为准。

## 正确性验证

```bash
PYTHONPATH=scripts OMP_NUM_THREADS=1 .venv-torch/bin/python -m unittest discover -s scripts/tests -v
.venv-torch/bin/python scripts/run.py check-parity \
  --torch-python .venv-torch/bin/python --tf-python .venv-tf/bin/python
PYTHONPATH=scripts:scripts/tests .venv-tf/bin/python -m unittest \
  test_resume.ResumeTests.test_tensorflow_optimizer_and_dropout_rng_resume -v
.venv-torch/bin/python scripts/run.py overfit --device cuda
```

`check-parity` 分别启动两种 Python 环境，在 CPU FP32 下导出共享初始化的前向中间量、loss、参数梯度和一次 SGD 更新，再执行对齐。正式 4+4 层、256 维、8k 词表配置有 399 项检查；前向容差 `atol=1e-5, rtol=1e-4`，梯度容差 `atol=1e-5, rtol=1e-3`。

对齐测试保留了一次定位记录：原始测试输入使一个 FFN 神经元在两框架中分别为 `1.7e-7` 和 `0.0`，跨过 ReLU 不可导点，引起梯度差异。最终测试使用避开该边界的公共输入，模型权重和误差阈值均未放宽；初次失败与诊断保存在 `output/parity/`。

`overfit` 使用 32 个真实训练句对、1+1 层/64 维小模型，关闭 dropout 和 label smoothing，要求 token accuracy 超过 99%。该结果是实现诊断，不是完整翻译模型成绩。

## 正式训练协议

正式配置在 `scripts/configs/base.json`：4 层 Encoder + 4 层 Decoder，d_model=256，4 heads，FFN=1024，Pre-LN、正弦位置编码、dropout=0.1，源/目标 embedding 与输出矩阵共享。参数约 943 万，以运行日志统计为准。

AdamW：峰值学习率 3e-4，warmup 1000 update，cosine 到 3e-5；betas=(0.9,0.98)，eps=1e-9；weight decay=0.01，bias/LayerNorm 不衰减。BF16 训练，softmax/loss/归一化统计为 FP32；全局梯度裁剪 1.0；label smoothing=0.1，仅向非 PAD 词表分配平滑质量。

有效 batch=128，默认 micro-batch=32。按整个有效 batch 的非 PAD token 总数归一化。最后不满 batch 的样本仍被处理。按长度分桶并保存每个 epoch 的样本 ID 清单，同 seed 在所有变体和两框架间共用。PyTorch 使用 DataLoader，TensorFlow 使用 tf.data 预取及 tf.function 自定义梯度循环。

先测吞吐与峰值显存：

```bash
.venv-torch/bin/python scripts/run.py profile --framework torch --output output/profile-torch.json
.venv-tf/bin/python scripts/run.py profile --framework tensorflow --output output/profile-tensorflow.json
```

profile 预热 20 update、测量 200 update；估算只包含训练，不包含验证解码和磁盘写入。若 micro-batch=32 显存不足，将 base.json 的 micro_batch 统一改成 16；有效 batch 仍为 128。在全部正式实验启动前冻结此修改。

| 实验 | 变动 | seeds |
|---|---|---|
| A0 | PyTorch 基线 | 42/43/44 |
| A1 | 所有注意力改成 1 head，总隐藏维度不变 | 42/43/44 |
| A2 | 仅移除 Encoder 位置编码 | 42/43/44 |
| A3 | TensorFlow 基线 | 42 |

各组完成相同 30 epoch。每个 epoch 完整验证并保存预测。以验证 BLEU 最高选择 checkpoint，同分选择 NLL 更低者；正式测试不参与选择。持续记录未平滑 NLL、学习率、梯度范数、吞吐、显存和验证结果。

启动单个实验：

```bash
.venv-torch/bin/python scripts/run.py train --framework torch --variant A0 --seed 42
.venv-tf/bin/python scripts/run.py train --framework tensorflow --variant A3 --seed 42
```

每 250 update 及每个 epoch 原子发布 checkpoint 指针，仅保留最佳和最近快照。包括模型、优化器、数据进度、随机状态及调度所需配置/step。TensorFlow dropout 使用可保存状态的 Generator。遇中断，从最近已发布快照继续，不假装恢复了未落盘的 update：

```bash
.venv-torch/bin/python scripts/run.py train --framework torch --variant A0 --seed 42 \
  --resume output/runs/A0-torch-seed42/latest.json
```

队列预览及完整执行：

```bash
.venv-torch/bin/python scripts/run.py run-suite --dry-run --finalize \
  --torch-python .venv-torch/bin/python --tf-python .venv-tf/bin/python
.venv-torch/bin/python scripts/run.py run-suite --finalize \
  --torch-python .venv-torch/bin/python --tf-python .venv-tf/bin/python
```

run-suite 串行使用一张 GPU，先做两框架 GPU 检查与数值对齐，再运行 10 次训练。重启时跳过已完成实验、恢复中断实验。`--finalize` 在全部训练完成后冻结配置和 checkpoint、完成测试、bootstrap、性能对照、注意力诊断及报告。人工复核字段保持“未复核”，需要逐条填写，工具不会生成虚假的人工结论。

## 评估和推理

贪心解码以 BOS 开始，禁止生成 PAD/BOS，保留 UNK，EOS 停止；最多 256 个新 token。验证/测试源句不截断，超过训练长度的样本用单句 batch。输出上限标记、空输出率与长度比一并报告。

```bash
.venv-torch/bin/python scripts/run.py freeze
.venv-torch/bin/python scripts/run.py evaluate \
  --checkpoint output/runs/A0-torch-seed42 --split test --device cuda --precision bf16 \
  --freeze output/freeze.json --output output/runs/A0-torch-seed42/test
.venv-torch/bin/python scripts/run.py translate \
  --checkpoint output/runs/A0-torch-seed42 --text "The red car is expensive." --device cuda
```

测试评估要求 checkpoint 和数据包含在冻结清单中。BLEU 使用 SacreBLEU、13a、大小写敏感；chrF2 使用 char_order=6、beta=2。保存反分词预测、原始参考、指标签名，以及按年份和英文子词长度分桶的结果。PPL 由未平滑 NLL 计算。

bootstrap 固定 seed 42，对同序同 ID 的预测进行 1000 次句级配对重采样，报告右侧系统减左侧系统的 BLEU 差与区间。它不替代训练种子方差，也不是文档聚类 bootstrap。

真实测试集的长度桶数量为 3363 / 3274 / 1746 / 213 / 1。错误复核先每桶至多选 20 条，再从剩余桶循环补足 100 条，不复制唯一长句；保存实际分配和不足数量。复核表包含漏译、重复、数字专名、否定角色、法语一致性及不确定标记。

## 推理和原理实验

```bash
.venv-torch/bin/python scripts/run.py benchmark \
  --checkpoint output/runs/A0-torch-seed42 --output output/benchmark.json
.venv-torch/bin/python scripts/run.py diagnose --kind scaling --output output/scaling
.venv-torch/bin/python scripts/run.py diagnose --kind attention \
  --checkpoint output/runs/A0-torch-seed42 --output output/attention
```

benchmark 固定同一 checkpoint，在手写/SDPA × 无缓存/有缓存的四个组合上测量。B∈{1,8}，源/生成长度∈{32,128}，统一 BF16，20 预热+50 次测量。固定目标轨迹，Encoder 均只运行一次；记录编码、首 token、剩余解码、吞吐、P95、显存和实际 SDPA 算子。另记录自然 EOS 生成的文本一致率和延迟，清楚区分是否包含分词。

缓存按层保存 Decoder 历史 K/V 及 Cross-Attention 的 Encoder K/V。增量步使用正确的显式有效 key mask，不依赖非方形 is_causal 默认对齐。缓存和 SDPA 的 FP32 正确性由回归测试验证；BF16 另报告数值/输出差异。

缩放实验使用独立正态 Q/K，观察 logits 标准差、熵和梯度 RMS。位置置换实验检查无 Encoder PE 的置换等变/翻译 logits 不变性质。注意力图固定验证样本，保存所有层/头的原始矩阵和热力图，作为诊断而非因果解释。

## 开发用端到端冒烟

```bash
.venv/bin/python scripts/run.py smoke-data
OMP_NUM_THREADS=1 .venv/bin/python scripts/run.py train \
  --config scripts/configs/smoke.json --prepared cache/smoke --device cpu --run output/my-smoke
.venv/bin/python scripts/run.py evaluate \
  --checkpoint output/my-smoke --prepared cache/smoke --split test --smoke --output output/my-smoke/test
.venv/bin/python scripts/run.py benchmark \
  --checkpoint output/my-smoke --prepared cache/smoke --device cpu --smoke --output output/my-smoke/benchmark.json
```

合成数据评估的 `--smoke` 只对明确标记的 fixture 解除正式测试冻结要求，不能用于绕过真实测试集协议。

## 简历填写

只在正式训练后填入真实数值：独立实现约 943 万参数的英法 Transformer；通过跨框架前向/梯度对齐；完成三种子的头数与位置编码消融；在明确 batch、长度和精度条件下测量 KV Cache/SDPA 的收益。保留失败结果和无显著收益的实验，不把冒烟成绩、过拟合准确率或接口调用写成正式翻译性能。
