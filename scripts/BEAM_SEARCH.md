# A0：Greedy / Beam Search 服务器操作指南

直接复用 **A0 seed 42 的最佳验证 checkpoint**，不重新训练。新增功能仅支持 PyTorch；A0–A3 原结果保持原样，A4/A5 不包含在本轮中。

## 实验口径

- 完整验证集 dev2010（890 句）：greedy，以及 beam size 4/8 × length penalty 0/0.6/1.0，共 7 组。
- 最佳 beam 按验证 BLEU 选择；并列依次比较 chrF2、更小 beam、更小 length penalty。即使 beam 不及 greedy，也保留结果。
- 写入独立解码冻结清单后，在 tst2010–2015 合集（8,597 句）测试 greedy 与选定 beam。
- 两种策略均使用同一 checkpoint、手写 attention、BF16、KV Cache、相同 batch 和 256 token 生成上限。
- 保存 BLEU/chrF2 签名、年份/长度分桶、长度比、空输出和截断比例，并做 1,000 次 paired bootstrap。
- 旧 greedy 没有使用 cache；本轮重新计算缓存 greedy，记录与历史预测的文本、截断标记及指标差异。
- NLL/PPL 使用相同 teacher forcing，必须一致；它们不是 beam search 的收益。
- 自然生成测速：固定前 20 条验证样本，batch 1/8，预热 3 次、计时 10 次，CUDA 同步。耗时包含编码及生成，排除分词、NLL、加载和磁盘读写。延迟是**完整 20 句集合**的耗时，不是单句延迟。输出 token/s 包含 EOS，排除 BOS/PAD，不统计内部候选扩展。

Beam 使用原词表的 FP32 log-softmax，然后屏蔽 PAD/BOS，不重新归一化；允许 UNK。每步按 `log_probability / ((5 + L) / 6)^alpha` 排序，L 包含 EOS、排除 BOS/PAD。完成候选仅通过零分数增量的 PAD 延续，长度和分数冻结。分数相同优先较早父候选，再优先较小 token ID。beam size 1 直接调用原 greedy。

长度归一化参考 [GNMT](https://arxiv.org/html/1609.08144v2)，不加入 coverage penalty。模型参数、前向和原 greedy 保持兼容。

## 1. 保存旧源码并更新

先确认原训练队列已结束，再在服务器执行：

```bash
cd /root/autodl-tmp/MT_English_Franch

PROJECT=/root/autodl-tmp/MT_English_Franch
OLD="$PROJECT/output/host4090-torch27"
OUT="$PROJECT/output/host4090-torch27-decode/A0-seed42"
PY="$PROJECT/.venv-torch27/bin/python"

test -x "$PY"
test -f "$OLD/freeze.json"
test -f "$OLD/runs/A0-torch-seed42/best.json"
mkdir -p "$OUT/archive"
if [ ! -f "$OUT/archive/source-before-beam.tar.gz" ]; then
  tar -czf "$OUT/archive/source-before-beam.tar.gz" scripts README.md .gitignore
fi
cp -n "$OLD/freeze.json" "$OUT/archive/baseline-freeze.json"
cp -n "$OLD/report/experiments.csv" "$OUT/archive/baseline-experiments.csv"

git status --short
git pull --ff-only
```

`git status` 若显示代码修改，先保存并解决冲突，再更新，不使用强制重置。既有虚拟环境可继续使用，没有新增第三方依赖。

旧结果只读，新输出目录位于其旁边。输出路径与旧目录、checkpoint 或 prepared 数据存在包含关系时，程序拒绝执行。

## 2. 环境和正确性检查

```bash
"$PY" "$PROJECT/scripts/run.py" doctor --framework torch --require-gpu

if [ ! -f "$PROJECT/cache/smoke/audit.json" ]; then
  "$PY" "$PROJECT/scripts/run.py" smoke-data
fi
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH="$PROJECT/scripts" \
  "$PY" -m unittest discover -s "$PROJECT/scripts/tests" -v

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH="$PROJECT/scripts:$PROJECT/scripts/tests" \
  "$PROJECT/.venv-tf215/bin/python" -m unittest \
  test_resume.ResumeTests.test_tensorflow_optimizer_and_dropout_rng_resume \
  test_pipeline.PipelineTests.test_tensorflow_full_pipeline_resume -v

"$PY" "$PROJECT/scripts/run.py" check-parity \
  --torch-python "$PY" --tf-python "$PROJECT/.venv-tf215/bin/python" \
  --output "$OUT/parity"
```

PyTorch 环境未安装 TensorFlow 时，两项 TF 测试会跳过；后续专门的 TF 测试与 parity 验证独立框架。

## 3. 预览完整流程

```bash
"$PY" "$PROJECT/scripts/run.py" compare-decode \
  --checkpoint "$OLD/runs/A0-torch-seed42" \
  --prepared "$PROJECT/cache/prepared" --freeze "$OLD/freeze.json" \
  --device cuda --precision bf16 --output "$OUT" --dry-run
```

预览返回 14 个阶段、7 组验证参数，以及 `training_runs: 0`；不创建或修改输出文件。输入必须匹配原数据、分词器、权重及冻结清单。

## 4. 后台启动评估

```bash
nohup "$PY" -u "$PROJECT/scripts/run.py" compare-decode \
  --checkpoint "$OLD/runs/A0-torch-seed42" \
  --prepared "$PROJECT/cache/prepared" --freeze "$OLD/freeze.json" \
  --device cuda --precision bf16 --output "$OUT" \
  >> "$OUT/compare.log" 2>&1 < /dev/null &

echo $! > "$OUT/compare.pid"
```

本任务无需运行 `train`、原 `run-suite` 或重新执行原 `freeze`。原训练恢复的源码哈希检查仍严格保留；新评估记录独立的源码版本和解码协议。

## 5. 查看进度与中断恢复

```bash
tail -n 50 "$OUT/compare.log"
cat "$OUT/status.json"
ps -p "$(cat "$OUT/compare.pid")" -o pid,etime,args
nvidia-smi
```

日志中 `stage_start` / `stage_complete` / `skip_complete` 对应阶段开始、完成、跳过。状态文件记录当前阶段和已处理句数。阶段顺序：7 组验证 → 冻结 → 2 组测试 → bootstrap → 测速 → 历史对比 → 报告。

中断后，确认旧进程退出，重新执行第 4 步同一命令。已完整落盘且校验通过的阶段不重算；未完成阶段从头重跑，这是阶段恢复，不是 token 级恢复。

目录锁随进程退出自动释放，不要删除 `.compare.lock` 绕过正在运行的进程。源码、输入、配置或已完成产物发生变化时，程序拒绝复用目录；保留旧目录并使用新的 `--output`。

## 6. 查看结果

```bash
cat "$OUT/decode-freeze.json"
cat "$OUT/report/comparison.csv"
cat "$OUT/report/bootstrap-greedy-beam.json"
cat "$OUT/report/report.md"
```

```text
experiment.json                 # 输入哈希、候选参数、源码版本、旧文件快照
status.json                     # 阶段进度和产物校验和
decode-freeze.json              # 验证选择与冻结协议
validation/*/                   # 全部验证候选的指标与预测
test/greedy/                    # 本次缓存 greedy 的测试结果
test/beam/                      # 选定 beam 的测试结果
historical-greedy.json           # 与旧 greedy 结果的差异
benchmark.json                  # 测速原始记录
report/comparison.csv
report/comparison.json
report/bootstrap-greedy-beam.json
report/report.md
```

报告保留正向或负向结果。测试集之前已经查看过，本轮描述为扩展实验，参数选择只使用验证集。

## 7. 使用选定 beam 翻译

```bash
BEAM_SIZE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["systems"]["beam"]["beam_size"])' "$OUT/decode-freeze.json")
BEAM_ALPHA=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["systems"]["beam"]["length_penalty"])' "$OUT/decode-freeze.json")

"$PY" "$PROJECT/scripts/run.py" translate \
  --checkpoint "$OLD/runs/A0-torch-seed42" --prepared "$PROJECT/cache/prepared" \
  --device cuda --precision bf16 \
  --decode beam --beam-size "$BEAM_SIZE" --length-penalty "$BEAM_ALPHA" --cache \
  --text "The red car is expensive."
```

`evaluate` 默认仍为无缓存 greedy，`translate` 默认仍为缓存 greedy。`--no-cache` 用于完整前缀参考解码。需要单独复核测试时，可以运行：

```bash
"$PY" "$PROJECT/scripts/run.py" evaluate \
  --checkpoint "$OLD/runs/A0-torch-seed42" --prepared "$PROJECT/cache/prepared" \
  --split test --device cuda --precision bf16 \
  --decode beam --beam-size "$BEAM_SIZE" --length-penalty "$BEAM_ALPHA" --cache \
  --freeze "$OUT/decode-freeze.json" --output "$OUT/manual-test-beam"
```

该手工复核会重新评估测试集；正常完成一键流程后无需重复执行。输出目录不能已有完成指标，解码参数、精度和 batch 必须匹配冻结协议。

## CPU 冒烟模式

`compare-decode --smoke --device cpu --precision fp32` 只接受审计中明确标记的合成数据，结果标为非正式。测试覆盖穷举参考、缓存重排、EOS、截断、长度惩罚、旧结果保护、冻结拒绝、目录锁和阶段恢复；冒烟指标不能用于声称正式 BLEU 或 GPU 性能。
