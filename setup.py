import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"


def _load_sm90_manifest_module():
    module_path = CSRC / "flash_sca" / "hopper" / "manifest" / "sm90.py"
    spec = importlib.util.spec_from_file_location(
        "xattn_sm90_build_manifest", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SM90 build manifest: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sm90_manifest = _load_sm90_manifest_module()
sm90_head_dims = sm90_manifest.sm90_head_dims
sm90_instantiation_sources = sm90_manifest.sm90_instantiation_sources

TORCH_LIB_DIR = Path(torch.__file__).resolve().parent / "lib"
TORCH_RPATHS = ["$ORIGIN/torch/lib"]
SOURCE_PREFIX_MAP_FLAG = f"-ffile-prefix-map={ROOT}=."
DEFAULT_NVCC_THREADS = 4
MAX_JOBS_CPU_DIVISOR = 2
MAX_JOBS_MEMORY_GB_PER_JOB = 8
SUPPORTED_TARGET_SMS = ("80", "86", "89", "90a")
SUPPORTED_GENERIC_SMS = ("80", "86", "89")
SUPPORTED_SM90_SMS = ("90a",)
ALL_TARGET_SMS = SUPPORTED_TARGET_SMS
DEFAULT_TARGET_SM_WITHOUT_VISIBLE_DEVICE = "80"


def _positive_env_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1
    except OSError:
        return os.cpu_count() or 1


def _resolve_host_compiler() -> str:
    requested = os.getenv("CUDAHOSTCXX") or os.getenv("CXX") or "c++"
    compiler = shutil.which(requested)
    if compiler is None:
        raise RuntimeError(f"Cannot resolve CUDA host compiler: {requested}")
    return compiler


def _cxx_standard() -> str:
    torch_version = _version_pair(torch.__version__)
    default = "20" if torch_version >= (2, 12) else "17"
    standard = os.getenv("XATTN_CXX_STANDARD", default)
    if standard not in {"17", "20"}:
        raise RuntimeError("XATTN_CXX_STANDARD must be 17 or 20")
    if standard == "17" and torch_version >= (2, 14):
        raise RuntimeError(
            "PyTorch 2.14+ headers require C++20; "
            "unset XATTN_CXX_STANDARD or set it to 20"
        )
    return f"c++{standard}"


def _version_pair(version: str):
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Cannot parse version: {version!r}")
    return tuple(map(int, match.groups()))


def _require_supported_toolchain() -> None:
    if _version_pair(torch.__version__) < (2, 10):
        raise RuntimeError("xattn requires PyTorch 2.10 or newer")
    if not torch.version.cuda or _version_pair(torch.version.cuda) < (12, 8):
        raise RuntimeError("xattn requires a CUDA 12.8+ PyTorch build")
    if CUDA_HOME is None:
        raise RuntimeError("xattn requires CUDA Toolkit 12.8+; set CUDA_HOME")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", output)
    if match is None or _version_pair(match.group(1)) < (12, 8):
        raise RuntimeError("xattn requires CUDA Toolkit 12.8 or newer")
    if _version_pair(match.group(1))[0] != _version_pair(torch.version.cuda)[0]:
        raise RuntimeError(
            "CUDA Toolkit and PyTorch CUDA major versions must match: "
            f"nvcc={match.group(1)}, torch.version.cuda={torch.version.cuda}"
        )


def _torch_cuda_major() -> Optional[int]:
    cuda_version = torch.version.cuda
    if not cuda_version:
        return None
    try:
        return int(cuda_version.split(".", 1)[0])
    except ValueError:
        return None


def _require_sm90_build_supported() -> None:
    cuda_major = _torch_cuda_major()
    if cuda_major is None or cuda_major < 12:
        raise RuntimeError(
            "xattn SM90 builds require a CUDA 12+ PyTorch build. "
            f"Detected torch.version.cuda={torch.version.cuda!r}."
        )


def _normalize_target_sm(raw_sm: str) -> str:
    sm = raw_sm.strip().lower()
    for prefix in ("sm_", "sm", "compute_", "compute"):
        if sm.startswith(prefix):
            sm = sm[len(prefix):]
            break
    sm = sm.replace(".", "")
    if sm in ("9", "90"):
        return "90a"
    if sm == "90a":
        return "90a"
    if sm in SUPPORTED_GENERIC_SMS:
        return sm
    raise RuntimeError(
        "Unsupported XATTN_TARGET_SM entry "
        f"{raw_sm!r}. Supported values are auto, all, "
        "or one or more of 80, 86, 89, and 90/90a."
    )


def _target_sm_to_torch_arch(sm: str) -> str:
    if sm == "90a":
        return "9.0a"
    return f"{sm[0]}.{sm[1:]}"


def _visible_target_sms():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return [DEFAULT_TARGET_SM_WITHOUT_VISIBLE_DEVICE]

    target_sms = []
    for device_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device_idx)
        if major == 9 and minor == 0:
            target_sm = "90a"
        else:
            target_sm = _normalize_target_sm(f"{major}{minor}")
        if target_sm not in target_sms:
            target_sms.append(target_sm)
    return target_sms


def _parse_target_sms():
    raw_target = os.getenv("XATTN_TARGET_SM")
    if raw_target is None:
        target_sms = _visible_target_sms()
    else:
        target = raw_target.strip().lower()
        if target in ("", "auto"):
            target_sms = _visible_target_sms()
        elif target == "all":
            target_sms = list(ALL_TARGET_SMS)
        else:
            target_sms = []
            for raw_sm in target.replace(";", ",").split(","):
                if raw_sm.strip():
                    target_sm = _normalize_target_sm(raw_sm)
                    if target_sm not in target_sms:
                        target_sms.append(target_sm)
            if not target_sms:
                raise RuntimeError("XATTN_TARGET_SM did not contain any targets")

    if any(sm in SUPPORTED_SM90_SMS for sm in target_sms):
        _require_sm90_build_supported()
    return target_sms


def _cgroup_memory_headroom_gb() -> Optional[float]:
    """Return the tightest remaining memory budget in the cgroup hierarchy."""
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None

    budgets = []
    for membership in memberships:
        _, controllers, group = membership.split(":", 2)
        v2 = not controllers
        if not v2 and "memory" not in controllers.split(","):
            continue
        for mount in mounts:
            before, after = mount.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            if filesystem[0] != ("cgroup2" if v2 else "cgroup"):
                continue
            if not v2 and "memory" not in filesystem[2].split(","):
                continue
            # mountinfo escapes whitespace and backslashes using octal codes.
            def unescape(value):
                for escaped, char in ((r"\040", " "), (r"\011", "\t"),
                                      (r"\012", "\n"), (r"\134", "\\")):
                    value = value.replace(escaped, char)
                return value

            mount_root = Path(unescape(fields[3]))
            mountpoint = Path(unescape(fields[4]))
            try:
                relative = Path(os.path.normpath(group)).relative_to(mount_root)
            except ValueError:
                # A cgroup namespace can report paths relative to its own root.
                relative = Path(os.path.normpath(group).lstrip("/"))
            current = mountpoint / relative
            limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
            usage_name = "memory.current" if v2 else "memory.usage_in_bytes"
            while True:
                try:
                    limit = int((current / limit_name).read_text().strip())
                    usage = int((current / usage_name).read_text().strip())
                    budgets.append(max(0, limit - usage) / (1024 ** 3))
                except (OSError, ValueError):
                    pass  # Includes cgroup v2's unlimited value, "max".
                if current == mountpoint:
                    break
                current = current.parent
    return min(budgets) if budgets else None


def _available_memory_gb() -> Optional[float]:
    budgets = []
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    budgets.append(float(line.split()[1]) / (1024 * 1024))
                    break
    except OSError:
        pass
    cgroup_budget = _cgroup_memory_headroom_gb()
    if cgroup_budget is not None:
        budgets.append(cgroup_budget)
    return min(budgets) if budgets else None


def _default_max_jobs(
    cpu_count: Optional[int] = None,
    available_memory_gb: Optional[float] = None,
) -> int:
    if cpu_count is None:
        cpu_count = _available_cpu_count()
    max_jobs_by_cpu = max(1, cpu_count // MAX_JOBS_CPU_DIVISOR)

    if available_memory_gb is None:
        available_memory_gb = _available_memory_gb()
        if available_memory_gb is None:
            return max_jobs_by_cpu

    max_jobs_by_memory = max(
        1, int(available_memory_gb // MAX_JOBS_MEMORY_GB_PER_JOB)
    )
    return max(1, min(max_jobs_by_cpu, max_jobs_by_memory))


NVCC_THREADS_REQUESTED = _positive_env_integer(
    "XATTN_NVCC_THREADS", DEFAULT_NVCC_THREADS
)
CPU_COUNT = _available_cpu_count()
if os.getenv("MAX_JOBS"):
    BUILD_MAX_JOBS = _positive_env_integer("MAX_JOBS", 1)
else:
    BUILD_MAX_JOBS = _default_max_jobs(
        cpu_count=CPU_COUNT,
    )
NVCC_THREADS = NVCC_THREADS_REQUESTED
CUDA_HOST_COMPILER = _resolve_host_compiler()


class XattnBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.getenv("MAX_JOBS"):
            os.environ["MAX_JOBS"] = str(BUILD_MAX_JOBS)
            print(
                f"Auto set MAX_JOBS={BUILD_MAX_JOBS} for xattn extension build. "
                f"NVCC threads per job: {NVCC_THREADS}. "
                "Override with MAX_JOBS=N if needed."
            )
        super().__init__(*args, **kwargs)


TARGET_SMS = _parse_target_sms()
TARGET_INCLUDES_SM90 = any(
    sm in SUPPORTED_SM90_SMS for sm in TARGET_SMS
)
TARGET_INCLUDES_GENERIC_CUDA = any(
    sm in SUPPORTED_GENERIC_SMS for sm in TARGET_SMS
)
BUILD_HAS_SM90 = TARGET_INCLUDES_SM90
BUILD_ONLY_SM90 = (
    BUILD_HAS_SM90 and not TARGET_INCLUDES_GENERIC_CUDA
)

target_cuda_arch_list = ";".join(
    _target_sm_to_torch_arch(target_sm) for target_sm in TARGET_SMS
)
os.environ["TORCH_CUDA_ARCH_LIST"] = target_cuda_arch_list

print(f"xattn target SMs: {', '.join(TARGET_SMS)}")

FLASH_SCA_HEAD_DIMS = [32, 64, 128, 256]


def _flash_sca_instantiation_sources(kind: str):
    assert kind in {"fwd", "bwd"}
    sources = []
    for dtype in ("fp16", "bf16"):
        for head_dim in FLASH_SCA_HEAD_DIMS:
            for head_dim_v in FLASH_SCA_HEAD_DIMS:
                name = (
                    f"sliding_chunk_attention_{kind}_hdim{head_dim}_"
                    f"vdim{head_dim_v}_{dtype}.cu"
                )
                sources.append(
                    CSRC
                    / "flash_sca"
                    / "cuda"
                    / "instantiations"
                    / name
                )
    return sources


def _flash_sca_sm90_instantiation_sources(kind: str):
    return sm90_instantiation_sources(kind, repo_root=ROOT)


def _flash_swa_sm90_instantiation_sources(kind: str):
    assert kind in {"fwd", "bwd"}
    sources = []
    for dtype in ("fp16", "bf16"):
        for head_dim in sm90_head_dims():
            for head_dim_v in sm90_head_dims():
                name = (
                    f"flash_swa_{kind}_hdim{head_dim}_"
                    f"vdim{head_dim_v}_{dtype}_sm90.cu"
                )
                sources.append(
                    CSRC / "flash_swa" / "hopper" / "instantiations" / name
                )
    return sources


FLASH_SCA_CSRCS = [
    CSRC / "xattn_extension.cc",
    CSRC / "flash_sca" / "sliding_chunk_attention.cc",
    CSRC / "flash_sca" / "sliding_chunk_attention_cuda.cu",
    CSRC / "flash_sca" / "cuda" / "dispatch.cc",
    *_flash_sca_instantiation_sources("fwd"),
    *_flash_sca_instantiation_sources("bwd"),
]

FLASH_SCA_SM90_CSRCS = [
    CSRC / "flash_sca" / "hopper" / "sliding_chunk_attention.cu",
    *_flash_sca_sm90_instantiation_sources("fwd"),
    *_flash_sca_sm90_instantiation_sources("bwd"),
]

CAUSAL_ATTENTION_CSRCS = [
    CSRC / "attention" / "causal_attention.cc",
    CSRC / "causal_flash_attn" / "causal_flash_attn.cc",
    CSRC / "flash_swa" / "flash_swa.cc",
]

FLASH_SWA_SM90_CSRCS = [
    CSRC / "causal_attention" / "launch.cu",
    *_flash_swa_sm90_instantiation_sources("fwd"),
    *_flash_swa_sm90_instantiation_sources("bwd"),
]

FLASH_SOFTDELTA_CSRCS = [
    CSRC / "softdelta" / "composition.cc",
    CSRC / "softdelta" / "gate.cu",
    CSRC / "softdelta" / "gate_bindings.cc",
    CSRC / "softdelta" / "gradient_merge.cu",
]


def _flash_softdelta_sm90_instantiation_sources():
    sources = []
    for dtype in ("fp16", "bf16"):
        for head_dim in sm90_head_dims():
            for head_dim_v in sm90_head_dims():
                name = (
                    f"softdelta_fwd_hdim{head_dim}_"
                    f"vdim{head_dim_v}_{dtype}_sm90.cu"
                )
                sources.append(
                    CSRC / "softdelta" / "instantiations" / name
                )
    return sources

FLASH_SOFTDELTA_SM90_CSRCS = [
    CSRC / "softdelta" / "fwd.cu",
    CSRC / "softdelta" / "fwd_bindings.cc",
    *_flash_softdelta_sm90_instantiation_sources(),
]

if BUILD_ONLY_SM90:
    CSRCS = [
        CSRC / "xattn_extension.cc",
        CSRC / "flash_sca" / "sliding_chunk_attention.cc",
        *CAUSAL_ATTENTION_CSRCS,
        *FLASH_SCA_SM90_CSRCS,
        *FLASH_SWA_SM90_CSRCS,
        *FLASH_SOFTDELTA_CSRCS,
        *FLASH_SOFTDELTA_SM90_CSRCS,
    ]
elif BUILD_HAS_SM90:
    CSRCS = [
        *FLASH_SCA_CSRCS,
        *CAUSAL_ATTENTION_CSRCS,
        *FLASH_SCA_SM90_CSRCS,
        *FLASH_SWA_SM90_CSRCS,
        *FLASH_SOFTDELTA_CSRCS,
        *FLASH_SOFTDELTA_SM90_CSRCS,
    ]
else:
    CSRCS = [
        *FLASH_SCA_CSRCS,
        *CAUSAL_ATTENTION_CSRCS,
        *FLASH_SOFTDELTA_CSRCS,
    ]

CUTLASS_INCLUDE_DIR = Path(
    os.getenv(
        "CUTLASS_INCLUDE_DIR",
        str(CSRC / "cutlass" / "include"),
    )
)
if not CUTLASS_INCLUDE_DIR.is_dir():
    raise RuntimeError(
        "Missing CUTLASS headers at "
        f"{CUTLASS_INCLUDE_DIR}. Set CUTLASS_INCLUDE_DIR to an external "
        "CUTLASS include directory or initialize csrc/cutlass/include "
        "in this checkout."
    )

INCLUDE_DIRS = [
    str(CSRC),
    str(CUTLASS_INCLUDE_DIR),
]

CXX_FLAGS = [
    "-O3",
    f"-std={_cxx_standard()}",
]

if BUILD_HAS_SM90:
    CXX_FLAGS.append("-DXATTN_HAS_SM90")
if BUILD_ONLY_SM90:
    CXX_FLAGS.append("-DXATTN_SM90_ONLY")

NVCC_FLAGS = [
    "-DNDEBUG",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--threads",
    str(NVCC_THREADS),
    "-ccbin",
    CUDA_HOST_COMPILER,
]

if BUILD_HAS_SM90:
    NVCC_FLAGS.extend(
        [
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-gencode",
            "arch=compute_90a,code=sm_90a",
        ]
    )


def main():
    _require_supported_toolchain()
    setup(
        name="xattn",
        version="1.0.0",
        description="Efficient attention modules for xLLM.",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://github.com/ifm-ai/xattn",
        license="Apache-2.0",
        license_files=["LICENSE", "THIRD_PARTY_NOTICES"],
        packages=find_packages(exclude=("tests", "tests.*")),
        ext_modules=[
            CUDAExtension(
                "xattn_cuda",
                [str(path.relative_to(ROOT)) for path in CSRCS],
                include_dirs=INCLUDE_DIRS,
                library_dirs=[str(TORCH_LIB_DIR)],
                extra_compile_args={
                    "cxx": CXX_FLAGS + [SOURCE_PREFIX_MAP_FLAG],
                    "nvcc": CXX_FLAGS
                    + [f"-Xcompiler={SOURCE_PREFIX_MAP_FLAG}"]
                    + NVCC_FLAGS,
                },
                extra_link_args=[
                    f"-Wl,-rpath,{rpath}"
                    for rpath in TORCH_RPATHS
                ],
            )
        ],
        cmdclass={"build_ext": XattnBuildExtension},
        python_requires=">=3.10",
        install_requires=["torch>=2.10"],
        extras_require={"test": ["pytest"]},
    )


if __name__ == "__main__":
    main()
