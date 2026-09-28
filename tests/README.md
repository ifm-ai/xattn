# Correctness and performance

Run commands from the checkout root, after building xattn for the target GPU.
The four correctness modules cover causal attention, FlashSCA, FlashSWA,
and the three SoftDelta SM90 variants.

```bash
python -m pytest tests -q
python -m pytest tests -q --full-matrix --long
```

The normal suite combines output and gradient checks, FP16/BF16, MHA/GQA/MQA,
short queries, previous KV, segment/BOS inputs, noncontiguous inputs, chunk
position resets, and high-precision backward state. Standalone forward/backward
and autograd use the same quantized-input FP64 reference. Forward and backward
have separate tolerances.

`--full-matrix` adds every D/V bucket (32, 64, 96, 128, 160, 192, 256), both
precision settings and both backward determinism settings. SoftDelta also checks
the paired-reader backward kernel on the short lengths. The full D/V matrix
uses lengths 7, 193, 513, 4096, 16384, 32768, and 65536 for all six operations.
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
python -m tests.softdelta.validate_flash_softdelta_shards --kind correctness \
  --output validated.json shard0.json shard1.json
```

The four benchmark entrypoints measure xattn itself. Each timed case first checks
sampled outputs and sparse-dO gradients against FP64 using the actual timed
inputs. Reports include forward/backward latency, throughput, separate backward
determinism groups, physical GPU identity, binary hash and the case manifest.
Do not overlap benchmarks on the same physical GPU.

```bash
python -m tests.causal_flash_attn.benchmark --dtype fp16 bf16
python -m tests.sca.benchmark --cases dense segment previous short
python -m tests.swa.benchmark --length 4096 --dims 96 64
python -m tests.softdelta.benchmark --full-matrix --high-precision
```

Use `--output` for an explicit report path; otherwise reports go to
`~/codex_logs/xattn/benchmarks/`. `--warmup`, `--iterations`, `--repeats`,
`--kv-heads`, `--span` and `--dims D V` select the measurement setup.
Validate completed benchmark manifests with the same validator and `--kind speed`.
