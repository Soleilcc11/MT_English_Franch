# English → French Transformer

从随机初始化训练英语到法语的 Encoder–Decoder Transformer，使用 **PyTorch / TensorFlow 独立实现**，研究多头注意力、位置编码、R-Drop 与解码策略。项目包括数据审计、跨框架数值对齐、可恢复训练、KV Cache / SDPA 性能测试，以及冻结协议后的翻译质量评估。

**已完成 18 次正式训练**：A0–A3 共 10 次，A4/A5 扩展共 8 次；另完成 A0 的 greedy / beam search 对照，无新增训练。结果来自单卡 RTX 4090 实验，公开记录见 [2026-11-06 结果快照](output/published-results/20261106-155127/)。仓库名称沿用 `MT_English_Franch`。

## 导航

- [已完成的实验与结果](#已完成的实验与结果)
- [模型与实验协议](#模型与实验协议)
- [从零复现](#从零复现)：环境 → 数据 → 冒烟与正确性检查 → 正式队列
- [评估与翻译](#评估与翻译)
- [进度检查与断点恢复](#进度检查与断点恢复)
- [产物与验收标准](#产物与验收标准)
- [代码导航与参考](#代码导航与参考)

## 已完成的实验与结果

测试集为 **IWSLT2017 英法数据包的 tst2010–2015 合集，共 8,597 句**，不是单独的 tst2017。下表均使用 greedy；除 A3 仅 seed 42 外，报告 seed 42/43/44 的均值 ± **样本标准差**。

| 实验 | 改动 | 参数量 | BLEU ↑ | chrF2 ↑ |
|---|---|---:|---:|---:|
| A0 | PyTorch 基线，4 个注意力头 | 9,429,824 | 35.7995 ± 0.3115 | 60.2361 ± 0.3023 |
| A1 | 所有注意力改为单头，总隐藏维度不变 | 9,429,824 | 35.0446 ± 0.0912 | 59.5868 ± 0.0984 |
| A2 | 移除 Encoder 绝对位置编码，保留 Decoder 位置编码 | 9,429,824 | 14.9020 ± 0.1733 | 43.7496 ± 0.3312 |
| A3 | TensorFlow 独立基线，仅一个种子 | 9,429,824 | 35.7804 | 60.2209 |
| A4 | Encoder 使用相对位置 K/V，替代绝对位置编码 | 9,446,720 | 36.1810 ± 0.1809 | 60.4529 ± 0.1666 |
| A5 | R-Drop，验证集选择 α=1 | 9,429,824 | 36.1392 ± 0.1072 | 60.3750 ± 0.0492 |

逐种子原始数值：[A0–A3 CSV](output/published-results/20261106-155127/baseline/report/experiments.csv)、[A0/A4/A5 CSV](output/published-results/20261106-155127/extensions/report/experiments.csv)、[A5 验证集选参](output/published-results/20261106-155127/extensions/report/a5-tuning.csv)。

这些结果支持以下范围内的结论：

- 本配置下，多头相对单头的平均收益为 **+0.7549 BLEU**；移除源位置编码使平均 BLEU 降低 **20.8975**。A2 的置换性质另由计算测试验证，热力图只用于观察。
- A4 相对 A0 的平均收益为 **+0.3816 BLEU**。预先指定的 seed 42 配对差值为 +0.3019，句子 bootstrap 95% 区间为 **[0.0824, 0.5236]**；A4 多出 16,896 个参数。
- A5 相对 A0 的平均收益为 **+0.3397 BLEU**，但 seed 42 的差值 +0.1044 对应区间 **[-0.0922, 0.3099]**，跨过 0，不能据此宣称该种子下显著改善。α=1 优于本次搜索的 α=3、5，不能推广为所有任务的最优值。
- A3 只验证独立实现和完整训练的可行性，不用一次运行判断框架质量高低。已发布的 [跨框架报告](output/published-results/20261106-155127/baseline/parity/report.json) 通过全部 **399 项**前向、loss、梯度及 SGD 更新检查。

### 同一 A0 checkpoint 的解码对照

使用 A0 seed 42 的最佳 checkpoint，在完整验证集搜索 `beam_size ∈ {4,8}`、`length_penalty ∈ {0,0.6,1.0}`，选择 **beam=8、α=0**。两种系统均为 BF16、手写 attention、KV Cache、最多生成 256 token。

| 解码方式 | BLEU ↑ | chrF2 ↑ | PPL ↓ | 生成/参考词数比 |
|---|---:|---:|---:|---:|
| Greedy，重新计算的配对基线 | 35.9150 | 60.3755 | 3.9644 | 0.9885 |
| Beam=8，α=0 | 36.5723 | 60.8261 | 3.9644 | 0.9652 |

Beam 提升 **+0.6573 BLEU**，95% 配对 bootstrap 区间为 **[0.4912, 0.8253]**。PPL 来自同一模型的 teacher forcing，因此不随搜索策略改善。固定前 20 条验证样本、batch=8 的整组生成延迟中位数由 **412.50 ms 增至 564.69 ms**，约增加 36.9%；该数据包含编码和生成，不是单句延迟。

完整记录：[解码报告](output/published-results/20261106-155127/beam/report/report.md)、[对照 CSV](output/published-results/20261106-155127/beam/report/comparison.csv)、[测速 JSON](output/published-results/20261106-155127/beam/benchmark.json)。开启缓存的 BF16 greedy 与历史无缓存路径有 217/8,597 条文本不同，因此 beam 收益相对本次配对基线计算。

### 结论边界与复现范围

- A4/A5 和 beam 是查看原测试成绩之后设计的扩展实验；选参仅用验证集，但不能称为全新盲测。bootstrap 为 1,000 次句子配对抽样，不替代训练种子不确定性或 TED 文档聚类抽样。
- A5 共训练 5 个模型：seed 42 的三个 α 候选，加选中 α 的 seed 43/44。五次 `training_seconds` 合计 **16.06 小时**，只计训练更新，不含验证、测试和 I/O；不能把它当作整个项目用时，也不能声称与 A0 等计算预算。不同阶段的历史训练耗时不能直接证明 A4 加速。
- 按长度分桶中 `>128` 桶仅有 1 条，不能据此推断长句总体能力。A4 seed 43 有 2 条达到生成上限，仍计入评分；语言学错误分类需要人工复核。
- **GitHub 提供代码、配置、训练日志、指标和冻结清单，不提供原始语料、分词器二进制、checkpoint 或虚拟环境。**新克隆需按下文重新准备数据、训练；不能仅用公开报告目录进行翻译。
- 下面复现的是完整方法与评估流程。不同驱动、设备、BF16 计算路径及分词器文件元数据可能影响哈希或数值，不承诺逐位复现历史权重与分数。使用旧 checkpoint 时必须保留其配套的完整 `cache/prepared/`，不能重新训练分词器后混用。

## 模型与实验协议

基线配置见 [base.json](scripts/configs/base.json)。架构采用 Transformer Encoder–Decoder，并使用 Pre-LN，不是原论文的逐项复现。

| 项目 | 配置 |
|---|---|
| Encoder / Decoder | 4 / 4 层；`d_model=256`；4 heads；FFN `256 → 1024 → 256`，ReLU |
| 位置与归一化 | 正弦/余弦绝对位置；Pre-LN；Encoder/Decoder 各有末尾 LayerNorm，ε=`1e-5` |
| 参数共享 | 源 embedding、目标 embedding、输出投影共享矩阵，输出另有词表 bias |
| Dropout | 0.1，覆盖 embedding、attention probability、FFN 中间层及子层输出 |
| 优化器 | AdamW；β=(0.9,0.98)，ε=`1e-9`；weight decay=0.01，bias/LayerNorm 不衰减 |
| 学习率 | 前 1,000 次 optimizer update warmup 至 `3e-4`，cosine 降至 `3e-5` |
| Batch / 精度 | 有效 batch=128；micro-batch=32，累积 4 次；BF16，softmax/loss/归一化统计为 FP32 |
| 损失 | Label smoothing=0.1，平滑质量分配到非 PAD 词表；全局梯度裁剪 1.0 |
| 预算与选模 | 每次 30 epoch；每轮验证；最高验证 greedy BLEU，平局时选较低 NLL；不早停 |

三类注意力均手写 Q/K/V 投影、分头、缩放、mask、softmax 和合头。内部 `True=允许关注`：Encoder 屏蔽源 PAD，Decoder self-attention 屏蔽目标 PAD 和未来位置，cross-attention 屏蔽源 PAD。PAD 查询通过损失屏蔽，避免产生全负无穷的注意力行。

训练使用 `tgt[:-1]` 作为 Decoder 输入，`tgt[1:]` 作为标签；EOS 参与监督，PAD 不参与。梯度累积按整个有效 batch 的非 PAD 目标 token 总数归一化。两个框架共用数据、分词器、初始化权重交换格式与按 seed 保存的 epoch batch 清单。

A4 仅改变 Encoder：使用 Shaw 风格相对 K/V，距离截断为 16，各层相对位置表在头间共享。A5 保持 A0 架构，两次独立 dropout 前向，优化平均平滑交叉熵加 `α × 0.5 × (KL(p‖q)+KL(q‖p))`，KL 按非 PAD 目标位置归约。A4/A5 仅在 PyTorch 实现；A4 相对注意力不支持 SDPA 路径。

## 从零复现

以下命令面向 **Linux x86_64 / WSL2、单张 RTX 4090 24GB、Python 3.11**。PyTorch 和 TensorFlow 分环境安装。完整顺序为 **A0–A3 → beam → A4/A5**；每一阶段完成后再启动下一阶段，避免同卡并发影响训练和测速。

### 1. 获取代码并固定路径

在数据盘目录执行；AutoDL 通常是 `/root/autodl-tmp`，先用 `df -h` 核对实际挂载点。

```bash
cd /root/autodl-tmp
git clone https://github.com/Soleilcc11/MT_English_Franch.git
cd MT_English_Franch

export PROJECT="$PWD"
export PY="$PROJECT/.venv-torch27/bin/python"
export TFPY="$PROJECT/.venv-tf215/bin/python"
export PREPARED="$PROJECT/cache/prepared"
export CFG="$PROJECT/scripts/configs/base.json"
export REPRO="$PROJECT/output/reproduction"
export BASE="$REPRO/baseline"
export DECODE="$REPRO/decode/A0-seed42"
export EXT="$REPRO/extensions"
mkdir -p "$REPRO" "$BASE" "$DECODE" "$EXT"
git rev-parse HEAD > "$REPRO/checkout-at-start.txt"
```

新开终端需要先回到项目目录，再重新执行以上 `export` 行。后文所有命令依赖这些变量；`PY` 是 Python 可执行文件，不是虚拟环境目录。新实验统一写入 `output/reproduction/`，不覆盖已发布快照或旧的 `output/host4090-torch27*` 结果。

已有仓库可在所有队列停止后执行 `git status --short`、`git pull --ff-only`；先保留本地修改，不使用强制重置覆盖文件。准备恢复旧训练时，先看[断点恢复约束](#进度检查与断点恢复)，不要先更新训练源码。

### 2. 安装并检查环境

安装脚本要求 Python **3.11**，不能直接使用 3.12。若系统已有 `python3.11`：

```bash
python3.11 --version
ENFR_PYTHON=python3.11 bash scripts/setup_4090_torch27.sh --dry-run
ENFR_PYTHON=python3.11 bash scripts/setup_4090_torch27.sh
```

若镜像只有 Python 3.12，可先用 Conda 创建解释器，然后执行安装脚本：

```bash
conda create -n enfr-py311 python=3.11 -y
conda activate enfr-py311
ENFR_PYTHON="$(command -v python)" bash scripts/setup_4090_torch27.sh
```

以上两条安装路径选择其一。`--dry-run` 只打印命令，**不会创建环境**。安装脚本固定 PyTorch `2.7.0+cu128` 和 TensorFlow `2.15.0.post1`；后者的 `tf.__version__` 可能显示 `2.15.0`。不要把系统 CUDA 或 `nvidia-smi` 的 CUDA Version 当作框架的实际运行库版本。

```bash
"$PY" scripts/run.py doctor --framework torch --require-gpu \
  --output "$REPRO/doctor-torch.json"
"$TFPY" scripts/run.py doctor --framework tensorflow --require-gpu \
  --output "$REPRO/doctor-tensorflow.json"
"$PY" -m pip freeze > "$REPRO/environment-torch.txt"
"$TFPY" -m pip freeze > "$REPRO/environment-tensorflow.txt"
nvidia-smi > "$REPRO/nvidia-smi.txt"
```

`doctor` 必须成功识别 GPU。脚本自身的安装记录另保存在 `output/host4090-torch27/`，正式实验使用上面指定的新目录。依赖见 [PyTorch 2.7](scripts/requirements-torch27.txt)、[TensorFlow](scripts/requirements-tensorflow.txt)、[公共依赖](scripts/requirements-common.txt)；历史真实环境见 [A0 environment.json](output/published-results/20261106-155127/baseline/runs/A0-torch-seed42/environment.json)。

### 3. 小数据检查流程

先用合成数据验证训练、保存、加载、评分和翻译是否可运行；此步骤使用 CPU FP32，**不产生正式翻译成绩**。

```bash
if [ ! -f "$PROJECT/cache/smoke/audit.json" ]; then
  "$PY" scripts/run.py smoke-data
fi

OMP_NUM_THREADS=1 "$PY" scripts/run.py train \
  --config scripts/configs/smoke.json --prepared "$PROJECT/cache/smoke" \
  --framework torch --variant A0 --seed 42 --device cpu \
  --run "$REPRO/smoke/A0"

OMP_NUM_THREADS=1 "$PY" scripts/run.py evaluate \
  --checkpoint "$REPRO/smoke/A0" --prepared "$PROJECT/cache/smoke" \
  --split test --smoke --device cpu --precision fp32 \
  --output "$REPRO/smoke/test"

"$PY" scripts/run.py translate \
  --checkpoint "$REPRO/smoke/A0" --prepared "$PROJECT/cache/smoke" \
  --device cpu --text "The red bird sings number 7."
```

冒烟配置只训练 2 轮小模型。若已执行过，不重复启动同一训练目录；检查现有产物，或选一个新的 `--run` 路径。

### 4. 准备真实数据

```bash
if [ ! -f "$PREPARED/audit.json" ]; then
  "$PY" scripts/run.py prepare --config "$CFG" \
    --revision c18a4f81a47ae6fa079fe9d32db288ddde38451d \
    --output "$PREPARED"
fi
cat "$PREPARED/audit.json"
```

数据来自 [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017) 的 `iwslt2017-en-fr`。下载固定 revision 的原始压缩包，记录 SHA-256、句对 ID、文档 ID 和测试年份；已知原始包 SHA-256 为 `aa5b3b622e9389dfebc9b99ad22d7b21cdccb49be4c67e3e0a9f22573a1b500d`。

| 划分 | 来源 | 原始句对 | 最终句对 |
|---|---|---:|---:|
| Train | 官方训练集 | 232,825 | 230,423 |
| Validation | dev2010 | 890 | 890 |
| Test | tst2010–2015 合集 | 8,597 | 8,597 |

模型输入执行 Unicode NFC 和空白标准化，保留大小写、重音、撇号及标点；评分保留原始法语参考。训练集删除空句对（本次 0 条）、重复句对（2,062 条）、与验证/测试英文源句重合的样本（168 条）。仅用清理后的训练文本训练共享 **SentencePiece BPE 8,000 词表**，使用 identity normalization、确定性分词，特殊 ID 为 PAD=0、UNK=1、BOS=2、EOS=3。

分词器冻结后，去掉源或目标含 BOS/EOS 后超过 128 token 的训练句对（172 条，约占长度过滤前数据的 0.0746%）。验证和测试不截断源句，长句使用较小 batch。两个框架和所有变体必须使用同一份 prepared 数据，不要分别运行分词器训练。

已有 `audit.json` 时准备程序拒绝覆盖；需要重做时用新的 `--output`。旧 checkpoint 的 `data_id` 和分词器哈希必须精确匹配，不能复制一份旧 `freeze.json` 给新数据使用。

### 5. 正确性与 GPU 预算检查

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH="$PROJECT/scripts" \
  "$PY" -m unittest discover -s "$PROJECT/scripts/tests" -v

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT/scripts:$PROJECT/scripts/tests" \
  "$TFPY" -m unittest \
  test_resume.ResumeTests.test_tensorflow_optimizer_and_dropout_rng_resume \
  test_pipeline.PipelineTests.test_tensorflow_full_pipeline_resume -v

"$PY" scripts/run.py check-parity --config "$CFG" \
  --torch-python "$PY" --tf-python "$TFPY" --output "$REPRO/parity"

"$PY" scripts/run.py overfit --prepared "$PREPARED" \
  --device cuda --output "$REPRO/overfit"

"$PY" scripts/run.py profile --config "$CFG" --prepared "$PREPARED" \
  --framework torch --device cuda --output "$REPRO/profile-torch.json"
"$TFPY" scripts/run.py profile --config "$CFG" --prepared "$PREPARED" \
  --framework tensorflow --device cuda --output "$REPRO/profile-tensorflow.json"
"$PY" scripts/run.py profile --config "$CFG" --prepared "$PREPARED" \
  --framework torch --variant A4 --device cuda --output "$REPRO/profile-A4.json"
"$PY" scripts/run.py profile --config "$CFG" --prepared "$PREPARED" \
  --framework torch --variant A5 --rdrop-alpha 1 --device cuda \
  --output "$REPRO/profile-A5.json"
```

检查因果性、PAD mask、标签右移、梯度累积、KV Cache/beam、权重共享、断点恢复和扩展流程。PyTorch 环境中 TensorFlow 专属测试可能显示 `skipped`，所以另用 TensorFlow 环境执行恢复测试和跨框架对齐。

`check-parity` 使用 CPU FP32、关闭 dropout、显式映射同一权重；前向 `atol=1e-5, rtol=1e-4`，梯度 `atol=1e-5, rtol=1e-3`。`overfit` 使用 32 个真实句对、1+1 层的小模型，关闭 dropout / label smoothing，要求非 PAD token accuracy >99%。

`profile` 预热 20 次更新、计时 200 次；估算只包含训练更新，不是完成所有验证与测试的总用时。如需把 micro-batch 改为 16，必须在本轮**所有正式训练前**统一修改配置，保持有效 batch=128，并记录差异；不要在队列中途修改。

### 6. A0–A3：10 次正式训练与评估

先预览；确认检查通过后启动后台队列：

```bash
"$PY" scripts/run.py run-suite --config "$CFG" --prepared "$PREPARED" \
  --runs "$BASE/runs" --torch-python "$PY" --tf-python "$TFPY" \
  --device cuda --finalize --dry-run

nohup "$PY" -u scripts/run.py run-suite \
  --config "$CFG" --prepared "$PREPARED" --runs "$BASE/runs" \
  --torch-python "$PY" --tf-python "$TFPY" --device cuda --finalize \
  >> "$BASE/suite.log" 2>&1 < /dev/null &
echo $! > "$BASE/suite.pid"
```

队列串行训练 A0/A1/A2 各 3 个种子、A3 一个种子，每次 30 epoch。`--finalize` 随后生成冻结清单、10 组测试成绩、bootstrap、KV Cache × SDPA 测速、注意力诊断和汇总报告。**训练完成不等于这些后处理完成**；等待队列退出且报告生成，再进行下一阶段。

### 7. A0：Greedy / Beam Search，无需训练

此阶段需要上一阶段的 checkpoint 和 `freeze.json`，因此不能在新克隆后直接运行。

```bash
"$PY" scripts/run.py compare-decode \
  --checkpoint "$BASE/runs/A0-torch-seed42" --prepared "$PREPARED" \
  --freeze "$BASE/freeze.json" --device cuda --precision bf16 \
  --output "$DECODE" --dry-run

nohup "$PY" -u scripts/run.py compare-decode \
  --checkpoint "$BASE/runs/A0-torch-seed42" --prepared "$PREPARED" \
  --freeze "$BASE/freeze.json" --device cuda --precision bf16 \
  --output "$DECODE" >> "$DECODE/compare.log" 2>&1 < /dev/null &
echo $! > "$DECODE/compare.pid"
```

顺序执行验证集 greedy 和 6 个 beam 组合、按验证 BLEU/chrF2/较小 beam/较小 α 选参、独立冻结、测试评分、bootstrap、固定前 20 条验证样本的真实生成测速。新复现实验必须采用其自身选参结果，不能直接假定仍为 beam=8、α=0。该目录内保存 `decode-freeze.json`，不修改基线冻结清单。

### 8. A4/A5：8 次扩展训练与评估

```bash
"$PY" scripts/run.py run-extensions --config "$CFG" --prepared "$PREPARED" \
  --baseline-runs "$BASE/runs" --baseline-freeze "$BASE/freeze.json" \
  --torch-python "$PY" --device cuda --output "$EXT" --finalize --dry-run

nohup "$PY" -u scripts/run.py run-extensions \
  --config "$CFG" --prepared "$PREPARED" \
  --baseline-runs "$BASE/runs" --baseline-freeze "$BASE/freeze.json" \
  --torch-python "$PY" --device cuda --output "$EXT" --finalize \
  >> "$EXT/queue.log" 2>&1 < /dev/null &
echo $! > "$EXT/queue.pid"
```

A4 训练 seed 42/43/44；A5 在 seed 42 搜索 α=1/3/5，按验证 BLEU、较低 NLL、较小 α 依次选优，再补跑该 α 的 seed 43/44。`--finalize` 只读加载旧 A0，统一重算 A0/A4/A5 的 greedy，生成 9 组测试指标、配对 bootstrap、100 条错误复核样本及报告。搜索未选中的 A5 候选也保留训练成本与验证记录。

## 评估与翻译

正式质量口径：SacreBLEU 2.6.0、大小写敏感、`tokenize=13a`；chrF2 使用 `char_order=6, beta=2`；对反分词法语评分并保存指标签名。PPL=`exp(NLL)`，NLL 来自不含平滑/正则项的 teacher forcing，排除 PAD、包含 EOS。

Greedy / beam 禁止生成 PAD/BOS，允许 UNK，遇 EOS 停止，最多生成 256 个 token。beam 使用 `logP / ((5+L)/6)^α`，L 包括 EOS、不含 BOS/PAD；不使用 coverage penalty；beam=1 直接复用 greedy。KV Cache 缓存每层历史 self-attention K/V 和固定的 cross-attention K/V，候选重排时同步重排缓存。

三条队列已自动执行正式测试。需要单独检查验证集时，使用独立输出目录：

```bash
"$PY" scripts/run.py evaluate \
  --checkpoint "$BASE/runs/A0-torch-seed42" --prepared "$PREPARED" \
  --split validation --device cuda --precision bf16 --decode greedy --no-cache \
  --output "$REPRO/manual-validation/A0-seed42"

"$PY" scripts/run.py translate \
  --checkpoint "$BASE/runs/A0-torch-seed42" --prepared "$PREPARED" \
  --device cuda --precision bf16 --decode greedy --cache \
  --text "The red car is expensive."
```

若当前 `decode-freeze.json` 也选中 beam=8、α=0，可使用下例；否则将两个参数改为本轮选中值：

```bash
"$PY" scripts/run.py translate \
  --checkpoint "$BASE/runs/A0-torch-seed42" --prepared "$PREPARED" \
  --device cuda --precision bf16 --decode beam --beam-size 8 \
  --length-penalty 0 --cache --text "The red car is expensive."
```

TensorFlow 翻译使用 `"$TFPY"` 和 A3 checkpoint，只支持 greedy。手动执行 `evaluate --split test` 必须提供匹配协议的 `--freeze`：基线用 `freeze.json`，解码对照用 `decode-freeze.json`，A4/A5 用 `extension-freeze.json`。不能通过修改冻结文件绕过数据、权重或解码参数校验。

## 进度检查与断点恢复

以下是只读检查，按当前阶段选择对应日志；尚未启动的阶段不会有状态文件。

```bash
pgrep -af 'scripts/run.py'
nvidia-smi
tail -n 30 "$BASE/suite.log"
tail -n 30 "$DECODE/compare.log"
tail -n 30 "$EXT/queue.log"

"$PY" - <<'PY'
import json, os
from pathlib import Path
for name in ("BASE", "EXT"):
    root = Path(os.environ[name])
    for path in sorted((root / "runs").glob("*/status.json")):
        s = json.loads(path.read_text())
        print(path.parent.name, s.get("status"),
              "updates=", s.get("completed_updates"), "/", s.get("total_updates"))
for name in ("DECODE", "EXT"):
    path = Path(os.environ[name]) / "status.json"
    if path.exists():
        print(path, path.read_text())
PY
```

基线每次正式训练预期为 30 epoch、54,030 次更新（以上数据与 batch 配置下）。A0–A3 全部训练完成后，队列还可能在跑测试或 benchmark。最终应看到[产物表](#产物与验收标准)中对应报告、日志无异常且父/子进程均已退出；`compare-decode` 和 `run-extensions` 的根 `status.json` 应为 `complete`。

恢复时遵循以下规则：

1. 先确认旧父进程和训练子进程都已退出，再重新执行**同一条**后台队列命令。不要只依据 PID 文件判断，也不要启动两个队列操作同一目录。
2. 保持源码、配置、数据与路径一致。`run-suite` 跳过已完成训练，并从 `latest.json` 恢复未完成训练；其 `--finalize` 可能重新执行已完成的评分/测速。`compare-decode` / `run-extensions` 对已完成阶段做哈希核验后跳过，未完成阶段重新执行，并使用目录锁防止并发。
3. checkpoint 每 250 次 update 及 epoch 结束发布，恢复模型、优化器、随机状态和数据进度；尚未落盘的更新会重做。只复制 `best` 权重不足以恢复训练，迁移时保留完整 run 目录和 prepared 数据；虚拟环境在新主机重新安装。
4. 已有训练的源码哈希不匹配时，不修改元数据强行续跑。恢复原训练版本，或使用当前版本新开实验目录。README 修改本身不参与训练源码哈希。

`nohup ... >> log 2>&1 < /dev/null &` 表示后台运行、追加 stdout/stderr 到日志并断开终端输入；`echo $!` 保存最近后台进程的 PID。出现 `Exit 127` 时先检查 `echo "$PY"`、`test -x "$PY"` 和日志，通常是解释器路径不存在或变量未设置，并不代表训练已开始。

## 产物与验收标准

| 阶段 | 本轮复现产物（相对项目根目录） | 内容 |
|---|---|---|
| 数据 | `cache/prepared/audit.json`、`tokenizer.model`、三划分 JSONL、batch 清单 | 来源、校验和、过滤计数及样本 ID |
| 基线 | `output/reproduction/baseline/runs/<run>/` | config、environment、source、status、log、best/latest 指针及 checkpoint |
| 基线评分 | `output/reproduction/baseline/report/experiments.csv`、`summary.json` | 10 次训练、逐年/长度桶指标、bootstrap |
| 推理优化 | `output/reproduction/baseline/benchmark.json` | 手写/SDPA × 无/有缓存，BF16，20 次预热 + 50 次计时 |
| 解码 | `output/reproduction/decode/A0-seed42/report/comparison.csv`、`report.md` | 验证选参、greedy/beam 测试与性能差异 |
| 扩展 | `output/reproduction/extensions/report/experiments.csv`、`a5-tuning.csv`、`summary.json` | A0/A4/A5 三种子及 A5 搜索记录 |
| 错误复核 | 各训练队列的 `report/error_review.jsonl` | 100 条固定抽样，需人工填写错误类型 |

```bash
cat "$BASE/report/experiments.csv"
cat "$DECODE/decode-freeze.json"
cat "$DECODE/report/comparison.csv"
cat "$DECODE/report/bootstrap-greedy-beam.json"
cat "$EXT/report/experiments.csv"
cat "$EXT/report/a5-tuning.csv"
cat "$EXT/report/bootstrap-A0-A4.json"
cat "$EXT/report/bootstrap-A0-A5.json"
cat "$EXT/report/report.md"
```

验收以实现正确、协议一致、记录完整为准，不要求复现时每个分数等于历史值或每个改进都提升 BLEU：

- 正确性测试、跨框架对齐、32 句过拟合通过；所有正式训练达到 30 epoch。
- 参数只在验证集选择；冻结后再评估测试；报告整体、各年份、长度桶及指标签名。五个源 BPE 长度桶为 `≤16 / 17–32 / 33–64 / 65–128 / >128`，本数据样本数为 `3363 / 3274 / 1746 / 213 / 1`。
- 同时报告均值、样本标准差、配对差值与区间，保留负结果；空输出、长度比、截断案例计入诊断，人工标注前不声称完成语言学错误分析。
- 性能测量固定设备、精度、batch、长度和计时范围，记录 GPU 同步、实际 SDPA 后端及生成长度；质量收益与延迟/显存代价一起报告。

公开结果快照保留原始报告，不用新的复现产物覆盖。可以在无需 GPU 的情况下验证上传文件的完整性：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("output/published-results/20261106-155127")
manifest = json.loads((root / "export-manifest.json").read_text())
files = manifest["files_sha256"]
for name, expected in files.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch: {name}")
print(f"Verified {len(files)} published artifacts")
PY
```

导出时 checkout 为 `0651535a50c74e4ed0227700736630d88db14d88`；这不代表所有历史实验在同一源码版本训练，各自实际代码哈希以 `source.json` 和冻结清单为准。已发布快照不含完整逐句预测和 checkpoint，进一步复算分数/案例需服务器原始产物或从头复现。

## 代码导航与参考

| 文件 | 职责 |
|---|---|
| [scripts/run.py](scripts/run.py) | 统一命令入口、基线实验队列 |
| [data.py](scripts/enfr/data.py) | 原始语料解析、清洗、共享 BPE、数据审计和 batch 清单 |
| [torch_model.py](scripts/enfr/torch_model.py) / [tf_model.py](scripts/enfr/tf_model.py) | 两套独立模型；PyTorch 的 KV Cache、beam 和相对注意力 |
| [training.py](scripts/enfr/training.py) | 两框架训练循环、token 归约、R-Drop、保存与恢复 |
| [parity.py](scripts/enfr/parity.py) | 权重交换、逐层前向/梯度/更新对齐 |
| [evaluation.py](scripts/enfr/evaluation.py) / [reporting.py](scripts/enfr/reporting.py) | 生成、指标、分组评分、冻结校验、bootstrap 与汇总 |
| [decode_comparison.py](scripts/enfr/decode_comparison.py) / [extensions.py](scripts/enfr/extensions.py) | beam 与 A4/A5 的独立实验流程和阶段恢复 |
| [benchmark.py](scripts/enfr/benchmark.py) / [diagnostics.py](scripts/enfr/diagnostics.py) | 推理性能、缩放因子、位置置换及注意力诊断 |
| [scripts/tests/](scripts/tests/) | 数值正确性、beam、恢复与端到端流程测试 |

补充的历史服务器操作指南：[Greedy / Beam Search](scripts/BEAM_SEARCH.md)、[A4 / A5](scripts/EXTENSIONS_A4_A5.md)。其中 `output/host4090-torch27*` 是原实验路径；从零复现统一使用本 README 的 `output/reproduction/` 路径。

方法与工具参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[Relative Position Representations](https://aclanthology.org/N18-2074/)、[R-Drop](https://arxiv.org/abs/2106.14448)、[GNMT 长度归一化](https://arxiv.org/abs/1609.08144)、[SentencePiece](https://github.com/google/sentencepiece)、[SacreBLEU](https://github.com/mjpost/sacrebleu)。

原始数据的许可、署名和使用条件以 [IWSLT 数据卡](https://huggingface.co/datasets/IWSLT/iwslt2017) 为准。本仓库不重新分发原始训练语料。
