# A4 / A5：服务器训练、恢复、评估

本轮仅扩展 PyTorch。旧 A0–A3 训练与成绩、A0 beam search 结果保持原样。默认新增目录：

```text
output/host4090-torch27-extensions/
  experiment.json                  # 输入、源码、原始结果哈希、固定搜索计划
  status.json                      # 队列当前阶段及完成阶段校验和
  runs/A4-torch-seed42/             # A4: 42、43、44
  runs/A5-alpha1-torch-seed42/      # A5: alpha 1、3、5，先各跑 seed 42
  runs/A5-alpha<选中值>-torch-seed43/
  runs/A5-alpha<选中值>-torch-seed44/
  receipts/                        # 完整训练目录校验和
  a5-selection.json                # 验证集选参；seed 42 直接复用选中的候选
  extension-freeze.json            # 与旧 freeze.json 分开
  test/                            # 当前代码下 A0/A4/A5 各三种子 greedy 预测
  historical-a0.json               # 重算 A0 与原始 A0 的文本差异
  report/experiments.csv
  report/a5-tuning.csv
  report/summary.json              # 均值/样本标准差、整体/年份/长度桶及指标签名
  report/bootstrap-A0-A4.json
  report/bootstrap-A0-A5.json
  report/error_review.jsonl        # 100 条分析集；需人工标注
  report/report.md
```

## 方法与对照边界

**A4：Encoder-only Shaw 相对位置 K/V 注意力。**

Encoder 不再加正弦绝对位置编码。对每层每个头：

```text
r(i,j) = clip(j - i, -16, 16) + 16
logits(i,j) = [q_i · k_j + q_i · relative_k[r(i,j)]] / sqrt(d_k)
p(i,:) = softmax(logits(i,:) + mask)
out_i = Σ_j dropout(p(i,j)) [v_j + relative_v[r(i,j)]]
```

每层各有两个 [33, 64] 参数表，同层跨头共享、不同层不共享。两项使用同一份 dropout 后注意力概率。Decoder 的绝对位置、因果注意力和 Cross-Attention 不变，KV Cache 与 beam search 可直接使用。相对距离以 key 位置减 query 位置定义。所有原有参数使用原初始化次序，相同 seed 的公共初始参数与 A0 完全一致；新增表在最后初始化为标准差 1/sqrt(d_k) 的正态分布。

默认参数量：A0/A5 = 9,429,824；A4 = 9,446,720（增加 16,896）。A4 使用手写 attention，调用 SDPA 对照会明确报错，避免漏掉相对 V 项。本实验只替换 Encoder 位置机制，不宣称逐项复现原论文；A4 的机制收益与少量新增参数影响不能完全分离。来源：[Shaw et al., 2018](https://aclanthology.org/N18-2074/)。

**A5：保持 A0 网络结构，加入 R-Drop 训练目标。**

```text
CE = [LSCE(logits_1, y) + LSCE(logits_2, y)] / 2
KL_sym = [KL(p_1 || p_2) + KL(p_2 || p_1)] / 2
objective = CE + alpha * KL_sym
```

同一 micro-batch 做两次独立 dropout 前向，不重置 RNG，不 detach 任一路径。KL 在 FP32 上覆盖完整词表，只屏蔽标签为 PAD 的时间位置；label smoothing 仍沿用非 PAD 词表公式。CE 与 KL 都先求非 PAD token 总和，再除以整个有效 batch 的 token 数。日志的 loss 是总目标；ce、symmetric_kl、rdrop_alpha 单独记录；nll 是两次无平滑 NLL 的平均。验证/测试关闭 dropout，只做一次 teacher forcing，PPL=exp(NLL)，不包含 KL 或 label smoothing。alpha=0 的底层训练兼容路径仍仅进行一次前向；正式 A5 要求 alpha>0。

系数定义与 reduction 会影响 alpha 的含义，不能直接把其他代码中的 alpha 数值照搬为同等强度。来源：[R-Drop 原论文](https://arxiv.org/abs/2106.14448)。

**固定实验计划：8 次新增训练。**

1. A4：seed 42、43、44，各 30 epoch。
2. A5：alpha=1、3、5，各用 seed 42 训练 30 epoch。
3. 按最佳验证 BLEU 降序选择 alpha；并列时按验证 NLL 升序，再按 alpha 升序。选参不读测试成绩。
4. 选定 alpha 补跑 seed 43、44；seed 42 复用对应候选，不重复训练。
5. 全部完成后冻结，评估 A0/A4/A5 各三种子。A0 只加载旧最佳 checkpoint；六个新增入选模型与 A0 都使用 BF16、手写注意力、无缓存 greedy、上限 256，保持原质量评估口径。未选中的两个 A5 候选不跑最终测试。
6. seed 42 的 A4−A0、A5−A0 分别做 1,000 次 paired bootstrap；报告三种子均值与样本标准差、年份/长度桶、长度比/截断率，以及 A5 搜索成本。

旧测试成绩已被查看，本轮应标为扩展实验。A5 多出双前向和候选搜索，不能表述为等 GPU 时间对照。提升、退化、显存及耗时均需实测。TensorFlow 基线不新增 A4/A5 训练。

## 1. 更新服务器代码

进入项目并设置变量。新开终端也要重新执行本段：

```bash
cd /root/autodl-tmp/MT_English_Franch
PROJECT=/root/autodl-tmp/MT_English_Franch
OLD="$PROJECT/output/host4090-torch27"
OUT="$PROJECT/output/host4090-torch27-extensions"
PY="$PROJECT/.venv-torch27/bin/python"

pgrep -af 'scripts/run.py'
git status --short
```

如果旧 run-suite 或 compare-decode 还在运行，先让它完成；不要在运行过程中更换源码。训练恢复和阶段恢复都会校验源码哈希。旧 beam 实验冻结的是旧评估源码，更新后不能用新代码接着旧目录跑；保留旧 checkout/源码归档可重现旧协议。

备份源码与旧汇总，再普通更新：

```bash
mkdir -p "$OUT/archive"
if [ ! -f "$OUT/archive/source-before-a45.tar.gz" ]; then
  tar -czf "$OUT/archive/source-before-a45.tar.gz" scripts README.md .gitignore
fi
cp -n "$OLD/freeze.json" "$OUT/archive/baseline-freeze.json"
cp -n "$OLD/report/experiments.csv" "$OUT/archive/baseline-experiments.csv"
if [ ! -f "$OUT/archive/source-before-a45-commit.txt" ]; then
  git rev-parse HEAD > "$OUT/archive/source-before-a45-commit.txt"
fi
git pull --ff-only
```

Git 提示本地修改冲突时先处理冲突，不要强制 reset 或覆盖旧目录。需要保留多个备份时使用新的 archive 目录。

## 2. 环境检查与回归测试

沿用现有 Python 3.11 / PyTorch 2.7.0+cu128 环境，无新增依赖。

```bash
"$PY" scripts/run.py doctor --framework torch --require-gpu \
  --output "$OUT/doctor-torch.json"

if [ ! -f "$PROJECT/cache/smoke/audit.json" ]; then
  "$PY" scripts/run.py smoke-data
fi

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH="$PROJECT/scripts" \
  "$PY" -m unittest discover -s "$PROJECT/scripts/tests" -v

"$PY" scripts/run.py check-parity \
  --torch-python "$PY" \
  --tf-python "$PROJECT/.venv-tf215/bin/python" \
  --output "$OUT/parity"
```

PyTorch 环境未安装 TensorFlow 时，相关测试会标记 skipped；随后跨环境 check-parity 必须通过。新增模型/训练测试在 PyTorch 环境内完成。

## 3. 测量 A4/A5 的训练开销

不能直接沿用 A0 的秒/更新：A4 多了相对 K/V 运算，A5 多了一次前向与对称 KL。

```bash
"$PY" scripts/run.py profile --framework torch --variant A4 \
  --prepared "$PROJECT/cache/prepared" --device cuda \
  --output "$OUT/profile-A4.json"

"$PY" scripts/run.py profile --framework torch --variant A5 --rdrop-alpha 3 \
  --prepared "$PROJECT/cache/prepared" --device cuda \
  --output "$OUT/profile-A5-alpha3.json"

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("A4", "A5-alpha3"):
    p = json.loads((root / f"profile-{name}.json").read_text())
    print(name, "秒/更新 =", round(p["mean_update_seconds"], 3),
          "峰值 GiB =", round(p["peak_memory_bytes"] / 2**30, 2),
          "单次纯训练小时 =", round(p["estimated_training_only_hours_per_run"], 2))
a = json.loads((root / "profile-A4.json").read_text())
b = json.loads((root / "profile-A5-alpha3.json").read_text())
print("8 次纯训练粗估小时 =", round(
    3*a["estimated_training_only_hours_per_run"] +
    5*b["estimated_training_only_hours_per_run"], 2))
print("不含逐 epoch 验证、checkpoint IO、最终测试；A5 alpha=3 仅代表性估计。")
PY
```

默认 micro-batch 与原 A0 一致。若旧实验实际使用了改过的 base 配置，以下所有 profile/run-extensions 命令应加相同的 `--config /实际路径/base.json`。队列会拒绝与旧 A0 不可比的配置，包括 micro-batch 差异；不要在中途为某个变体单独改训练预算。OOM 时先检查并重新设计统一对照，不覆盖旧实验。

## 4. 预览并启动正式队列

```bash
"$PY" scripts/run.py run-extensions \
  --baseline-runs "$OLD/runs" --baseline-freeze "$OLD/freeze.json" \
  --prepared "$PROJECT/cache/prepared" --output "$OUT" \
  --torch-python "$PY" --device cuda --finalize --dry-run
```

应显示 `new_training_runs: 8`，最后两个 A5 的 alpha 显示 `validation-selected`。dry-run 核验数据、旧冻结及配置，不创建训练目录，不占用 GPU 训练。

正式后台启动：

```bash
nohup "$PY" -u "$PROJECT/scripts/run.py" run-extensions \
  --baseline-runs "$OLD/runs" --baseline-freeze "$OLD/freeze.json" \
  --prepared "$PROJECT/cache/prepared" --output "$OUT" \
  --torch-python "$PY" --device cuda --finalize \
  >> "$OUT/queue.log" 2>&1 < /dev/null &

echo $! > "$OUT/queue.pid"
```

`--finalize` 自动完成冻结、测试、bootstrap 和报告。省略它只完成 8 次训练及选参；之后用同一命令加 `--finalize` 即可继续。

## 5. 进度与恢复

```bash
tail -n 50 "$OUT/queue.log"
cat "$OUT/status.json"
pgrep -af 'scripts/run.py'
nvidia-smi
```

stage 为 `train/A4-torch-seed42` 等时，在训练对应实验；`select-alpha` 是选参；`test/...`、`bootstrap/...` 是最终评估。查看单次训练：

```bash
tail -n 5 "$OUT/runs/A4-torch-seed42/log.jsonl"
cat "$OUT/runs/A4-torch-seed42/status.json"
cat "$OUT/a5-selection.json"
```

选参文件只有三个 alpha 候选全部完成后才出现。完成训练数可查看：

```bash
"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
done = 0
for p in sorted((root / "runs").glob("*/status.json")):
    s = json.loads(p.read_text())
    done += s.get("status") == "complete"
    print(p.parent.name, s.get("status"), "updates", s.get("completed_updates"))
print("完成新增训练：", done, "/ 8")
PY
```

中断后先确认旧队列和训练子进程均已退出，再重新执行同一条后台启动命令。已完成训练和评估阶段校验哈希后跳过；未完成训练从 latest.json 恢复模型、优化器、RNG、epoch/batch 进度，最近未保存更新会重算；未完成评估阶段重新执行。目录锁阻止两个队列同时写同一输出目录。不要手工另起 train 指向队列正在使用的 run。

若在首个 checkpoint 写出前异常退出，run 可能已有 config.json 而没有 latest.json，程序会拒绝覆盖。保留日志，将**这个新实验的未完成 run 子目录**移到队列输出目录之外，再用原命令启动。不要移动或删除原 A0–A3 目录。源码、已完成模型、输入数据或配置改变时恢复会拒绝；使用原版本恢复或另开扩展实验目录。

训练 subprocess 日志沿用父进程的 queue.log。当前任务/报错写入 status.json；崩溃时队列状态可能暂留 running，检查进程而不是只看状态文本。

## 6. 完成后查看与演示

```bash
cat "$OUT/a5-selection.json"
cat "$OUT/report/a5-tuning.csv"
cat "$OUT/report/experiments.csv"
cat "$OUT/report/bootstrap-A0-A4.json"
cat "$OUT/report/bootstrap-A0-A5.json"
cat "$OUT/report/report.md"
```

A4 翻译（不训练、不改模型）：

```bash
"$PY" scripts/run.py translate \
  --checkpoint "$OUT/runs/A4-torch-seed42" \
  --prepared "$PROJECT/cache/prepared" --device cuda --precision bf16 \
  --decode greedy --cache \
  --text "The scientist did not publish the results in 2024."
```

选中的 A5 seed 42 路径记录在 `a5-selection.json` 的 `seed42_run`：

```bash
A5_RUN=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed42_run"])' "$OUT/a5-selection.json")
"$PY" scripts/run.py translate \
  --checkpoint "$OUT/runs/$A5_RUN" \
  --prepared "$PROJECT/cache/prepared" --device cuda --precision bf16 \
  --decode beam --beam-size 4 --length-penalty 0.6 --cache \
  --text "The scientist did not publish the results in 2024."
```

这条 beam 命令只是演示，不能作为 A4/A5 正式测试协议。extension-freeze.json 仅允许无缓存 greedy；A0 的 compare-decode 仍只接受 A0 seed 42，不能拿 A0 验证集选出的 beam 参数宣称完成 A4/A5 的独立解码选参。

所有统计使用实测成绩，不能提前填入“相对位置提升”“R-Drop 提升”。
