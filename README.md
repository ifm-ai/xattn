<div align="center">

<h1>xattn</h1>

<p><strong>Efficient attention modules for xLLM.</strong></p>

<p>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.10+"></a>
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-007EC6" alt="Apache 2.0 License"></a>
</p>

<p>
  <a href="#highlights">Highlights</a> &nbsp;|&nbsp;
  <a href="#supported-attention">Supported Attention</a> &nbsp;|&nbsp;
  <a href="#installation">Installation</a> &nbsp;|&nbsp;
  <a href="https://github.com/ifm-ai/xattn/issues">Issues</a>
</p>

</div>

xattn provides efficient attention modules for the
[xLLM framework](https://github.com/ifm-ai/xllm), with Python APIs for use in
other PyTorch projects.

## Highlights

| Feature | Capabilities |
| --- | --- |
| **High-precision Flash Attention output state** | Retains FP32 attention output state for backward, reducing gradient error from BF16 output. Enable with `high_precision_output=True`. |
| **Flexible segment-aware attention masking** | Isolates packed sequences using segment IDs or BOS masks.|

## Supported Attention

Select an attention type for API examples, tensor layouts, and standalone
forward/backward usage.

| Attention |
| --- |
| [Causal Flash Attention](docs/causal_flash_attention.md) |
| [Sliding Window Attention](docs/sliding_window_attention.md) |
| [Sliding Chunk Attention](docs/sliding_chunk_attention.md) |
| [SoftDelta Attention](docs/softdelta_attention.md) |
| [Sliding Window SoftDelta Attention](docs/sliding_window_softdelta_attention.md) |
| [Sliding Chunk SoftDelta Attention](docs/sliding_chunk_softdelta_attention.md) |

All native attention paths accept FP16/BF16 inputs and support MHA, GQA, and MQA.

## Installation

Source builds require Linux, Python 3.10+, PyTorch 2.10+ with CUDA support, CUDA Toolkit 12.8+
(including `nvcc`), and a compatible host compiler. Set `CUDA_HOME` to the
selected toolkit. Its CUDA major version must match `torch.version.cuda`;
using the same major/minor version is recommended. PyTorch wheels do not
include the complete build toolkit. Install a PyTorch CUDA wheel matching your
toolkit before the commands below. For CUDA 12.8, for example:

```bash
python -m pip install "torch>=2.10" --index-url https://download.pytorch.org/whl/cu128
```

### Install from Source

```bash
git clone --recurse-submodules https://github.com/ifm-ai/xattn.git
cd xattn
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation .
```

### Install from the Git SSH URL

```bash
python -m pip install "setuptools>=64" wheel ninja "torch>=2.10"
python -m pip install --no-build-isolation "xattn @ git+ssh://git@github.com/ifm-ai/xattn.git"
```

<details>
<summary><strong>Build configuration</strong></summary>

For an existing checkout, run `git submodule update --init --recursive` before
building.

By default, the build targets visible GPUs; with no visible GPU it targets SM80.
Set `XATTN_TARGET_SM=90a` for a headless SM90 build, or
`XATTN_TARGET_SM=80,86,89,90a` for all supported targets. SM90 builds on an SM90
system include the SM90 kernels by default. SM100 and newer architectures
are currently unsupported.

Set `MAX_JOBS` to limit concurrent compilation jobs and `XATTN_NVCC_THREADS`
to set nvcc threads per job (default: 4).

Some Conda toolkits place CUDA headers under
`$CUDA_HOME/targets/x86_64-linux/include`. If `$CUDA_HOME/include` does not
contain `cuda_runtime.h`, set `CUDA_INC_PATH` to that target include directory
when building.

The build selects the default C++ standard from the installed PyTorch version:

| PyTorch version | Default C++ standard |
| --- | --- |
| 2.10 / 2.11 | C++17 |
| 2.12+ | C++20 |

Set `XATTN_CXX_STANDARD=17` or `20` to override the default, for example:

```bash
XATTN_CXX_STANDARD=20 python -m pip install --no-build-isolation .
```

The build rejects an explicit C++17 selection with PyTorch 2.14+.
Both host and CUDA compilation use the selected standard.

Rebuild xattn after changing PyTorch, the toolkit, or the C++ standard;
installed binaries are not promised to work across PyTorch versions.

</details>

## License

xattn is licensed under the [Apache License 2.0](LICENSE). Third-party components
retain their licenses; see [third-party notices](THIRD_PARTY_NOTICES).
