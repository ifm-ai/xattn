# Correctness and performance

Run commands from the checkout root, after building xattn for the target GPU.
The attention-family correctness modules cover causal attention, FlashSCA, FlashSWA,
and the three SoftDelta variants. SoftDelta exposes scalar gates;
`tests/softdelta/test_value_groups.py` also covers its internal value-group layout
(18 representative cases by default; the complete matrix with `--full-matrix`).

```bash
python -m pytest tests -q
python -m pytest tests -q --full-matrix --long
```

Run the consolidated precision validation separately with:

```bash
python -m pytest tests/test_high_precision.py -q
```

This module covers `high_precision_level` 1 against the level 0 baseline. It
checks API validation for every attention family, the SBS forward, and, for
causal, sliding-window and sliding-chunk attention, FP64 output/gradient
parity, chunk boundaries, padded GQA, live previous KV and precision-repair
regressions. SoftDelta numerics at both levels live in `tests/softdelta`.

The normal suite combines output and gradient checks, FP16/BF16, MHA/GQA/MQA,
short queries, previous KV, segment/BOS inputs, noncontiguous inputs, chunk
position resets, and the high-precision levels. Standalone forward/backward
and autograd use the same quantized-input FP64 reference. Forward and backward
have separate tolerances.

SoftDelta default tests use representative D/V shapes; `--full-matrix` also
includes its complete declared shape and internal value-group matrices. Large-batch
checks for all families, including explicit `k_read` gradients, live in
`tests/test_large_batch.py`.

`--full-matrix` adds every D/V bucket (32, 64, 96, 128, 160, 192, 256), both
precision settings and both backward determinism settings. The full D/V matrix
uses lengths 7, 193, 513, 4096, 16384, 32768, and 65536 for causal,
sliding-window and sliding-chunk attention.
Lengths above 513 additionally require `--long`: these use B=1, sampled-row
FP64 outputs against all keys, and full-sized gradients from sparse dO. Both
backward determinism settings are exercised, with exact repeatability checked
when enabled. Short lengths use a full FP64 oracle. Long cases do not silently
skip OOM or reduce sequence length; their batch is already at the minimum.
FlashSCA generic CUDA cases require an SM80/SM86/SM89 GPU and the matching build;
they are skipped on an SM90-only machine.

For multiple GPUs, run one process per physical GPU with different shard indices:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests --full-matrix --long \
  --shard-count=2 --shard-index=0 --inventory-output=shard0.json --junitxml=shard0.xml
CUDA_VISIBLE_DEVICES=1 python -m pytest tests --full-matrix --long \
  --shard-count=2 --shard-index=1 --inventory-output=shard1.json --junitxml=shard1.xml
```

Each inventory retains the complete declared node list and its coverage hash.
For result validation, augment each inventory with `exit_code` and the absolute
`junit` path, then run the validator below. Its checks reject incomplete or
duplicated shard partitions and incomplete pytest results.

```bash
python -m tests.validate_shards --kind correctness \
  --output validated.json shard0.json shard1.json
```

The three benchmark entrypoints measure xattn itself. Each timed case first checks
sampled outputs and sparse-dO gradients against FP64 using the actual timed
inputs. Reports include forward/backward latency, throughput, separate backward
determinism groups, physical GPU identity, binary hash and the case manifest.
Do not overlap benchmarks on the same physical GPU.

```bash
python -m tests.causal_flash_attn.benchmark --dtype fp16 bf16
python -m tests.sca.benchmark --cases dense segment previous short
python -m tests.swa.benchmark --length 4096 --dims 96 64
```

Use `--output` for an explicit report path; otherwise reports go to
`logs/benchmarks/` under the checkout root. `--warmup`, `--iterations`, `--repeats`,
`--kv-heads`, `--span` and `--dims D V` select the measurement setup.
Validate completed benchmark manifests with the same validator and `--kind speed`.
