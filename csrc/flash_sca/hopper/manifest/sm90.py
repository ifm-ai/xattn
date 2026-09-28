from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from functools import lru_cache
from typing import Any


MANIFEST_PATH = pathlib.Path(__file__).with_name("sm90_instances.json")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
INSTANTIATION_MACRO_RE = re.compile(
    r"^\s*(XATTN_FLASH_SCA_(?:FWD|BWD)_SM90_INSTANTIATE"
    r"(?:_[A-Z_]+)?)\s*\(([^()]*)\)",
    re.MULTILINE,
)
INSTANTIATION_MACRO_START_RE = re.compile(
    r"^\s*XATTN_FLASH_SCA_", re.MULTILINE
)
COMPAT_REGISTRY_ENTRY_RE = re.compile(
    r"struct\s+ScaCompatRegistryEntrySm90<\s*"
    r"semantics::DirectionKind::(?P<direction>kForward|kBackward),\s*"
    r"semantics::RouteKind::(?P<route>"
    r"kDenseOrSegmentBundle|kVarlen)>\s*\{"
    r"(?P<body>.*?)\n\};",
    re.DOTALL,
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_raw_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _compat_registry_records(
    manifest: dict[str, Any], repo_root: pathlib.Path
) -> list[dict[str, Any]]:
    registry_spec = manifest["compat_registry"]
    source = repo_root / registry_spec["source"]
    content = source.read_text(encoding="utf-8")
    records = []
    for match in COMPAT_REGISTRY_ENTRY_RE.finditer(content):
        body = match.group("body")

        def required(pattern: str, field: str) -> str:
            value = re.search(pattern, body)
            if value is None:
                raise RuntimeError(
                    f"compat registry entry is missing {field}: "
                    f"{match.group(0)}"
                )
            return value.group(1)

        records.append(
            {
                "direction_kind": match.group("direction"),
                "route_kind": match.group("route"),
                "semantic_type": required(
                    r"using\s+Semantic\s*=\s*semantics::(\w+)\s*;",
                    "Semantic",
                ),
                "plan_type": required(
                    r"using\s+Plan\s*=\s*(\w+)\s*;", "Plan"
                ),
                "stable_plan_id": required(
                    r'kStablePlanId\s*=\s*"([^"]+)"\s*;',
                    "kStablePlanId",
                ),
                "stable_instance_route_bundle_id": required(
                    r'kStableInstanceRouteBundleId\s*=\s*"([^"]+)"\s*;',
                    "kStableInstanceRouteBundleId",
                ),
            }
        )
    return records


def _normalized_instantiation_macros(
    source: pathlib.Path, direction: str
) -> list[str]:
    content = source.read_text(encoding="utf-8")
    matches = list(INSTANTIATION_MACRO_RE.finditer(content))
    starts = list(INSTANTIATION_MACRO_START_RE.finditer(content))
    if len(matches) != len(starts):
        raise RuntimeError(
            f"unrecognized SM90 instantiation macro in {source}"
        )
    macros = []
    for match in matches:
        macro_direction = "fwd" if "_FWD_" in match.group(1) else "bwd"
        if macro_direction != direction:
            raise RuntimeError(
                f"{source} contains a {macro_direction} macro in a "
                f"{direction} instance"
            )
        arguments = re.sub(r"\s+", " ", match.group(2).strip())
        macros.append(f"{match.group(1)}({arguments})")
    if not macros:
        raise RuntimeError(f"{source} has no SM90 instantiation macro")
    return macros


def expand_sm90_instances(
    manifest: dict[str, Any], repo_root: pathlib.Path = REPO_ROOT
) -> list[dict[str, Any]]:
    axes = manifest["axes"]
    source_spec = manifest["source"]
    records = []
    for direction in axes["directions"]:
        bundle = manifest["route_bundles"][direction]
        determinism = (
            "not_applicable"
            if direction == "fwd"
            else "runtime_deterministic_and_nondeterministic"
        )
        for kernel_dtype in axes["kernel_dtypes"]:
            for head_dim in axes["head_dims"]:
                for value_head_dim in axes["value_head_dims"]:
                    filename = source_spec["filename"].format(
                        direction=direction,
                        head_dim=head_dim,
                        value_head_dim=value_head_dim,
                        kernel_dtype=kernel_dtype,
                    )
                    relative_source = (
                        pathlib.PurePosixPath(source_spec["directory"])
                        / filename
                    )
                    source = repo_root / relative_source
                    if not source.is_file():
                        raise RuntimeError(
                            f"missing SM90 instance source: {source}"
                        )
                    stable_id = (
                        f"sm90/{direction}/sca/{kernel_dtype}/"
                        f"d{head_dim}/v{value_head_dim}/route_bundle"
                    )
                    records.append(
                        {
                            "architecture": manifest["architecture"],
                            "d": head_dim,
                            "determinism_specialization": determinism,
                            "direction": direction,
                            "formula_resource_class": manifest[
                                "formula_resource_class"
                            ],
                            "instantiation_macros":
                                _normalized_instantiation_macros(
                                    source, direction
                                ),
                            "kernel_dtype": kernel_dtype,
                            "route_bundle": bundle["stable_id"],
                            "semantic_identity": manifest[
                                "semantic_identity"
                            ],
                            "source": relative_source.as_posix(),
                            "source_sha256": _sha256_file(source),
                            "stable_instance_id": stable_id,
                            "v": value_head_dim,
                            "variant": manifest["variant"],
                        }
                    )
    return records


def derive_sm90_manifest_metadata(
    manifest: dict[str, Any], repo_root: pathlib.Path = REPO_ROOT
) -> dict[str, Any]:
    records = expand_sm90_instances(manifest, repo_root)
    compat_registry_records = _compat_registry_records(
        manifest, repo_root
    )
    counts = {
        "total": len(records),
        "fwd": sum(record["direction"] == "fwd" for record in records),
        "bwd": sum(record["direction"] == "bwd" for record in records),
    }
    generator = manifest["generator"]
    return {
        "coverage_sha256": _canonical_sha256(records),
        "compat_registry_coverage_sha256": _canonical_sha256(
            compat_registry_records
        ),
        "compat_registry_source_sha256": _sha256_file(
            repo_root / manifest["compat_registry"]["source"]
        ),
        "group_counts": counts,
        "loader_source_sha256": _sha256_file(
            repo_root / generator["loader_source"]
        ),
        "validator_source_sha256": _sha256_file(
            repo_root / generator["validator_source"]
        ),
    }


def validate_sm90_manifest(
    manifest: dict[str, Any], repo_root: pathlib.Path = REPO_ROOT
) -> list[dict[str, Any]]:
    if manifest["schema_version"] != 1:
        raise RuntimeError(
            "unsupported FlashSCA SM90 manifest schema version: "
            f"{manifest['schema_version']}"
        )
    axes = manifest["axes"]
    for name in (
        "directions",
        "kernel_dtypes",
        "head_dims",
        "value_head_dims",
    ):
        values = axes[name]
        if not values or len(values) != len(set(values)):
            raise RuntimeError(
                f"SM90 manifest axis {name!r} is empty or has duplicates"
            )
    if axes["directions"] != ["fwd", "bwd"]:
        raise RuntimeError(
            "SM90 manifest must preserve production direction order "
            "['fwd', 'bwd']"
        )
    if axes["kernel_dtypes"] != ["fp16", "bf16"]:
        raise RuntimeError(
            "SM90 manifest must preserve production dtype order "
            "['fp16', 'bf16']"
        )

    registry_spec = manifest["compat_registry"]
    registry_records = _compat_registry_records(manifest, repo_root)
    if registry_records != registry_spec["entries"]:
        raise RuntimeError(
            "SM90 compatibility registry/manifest mismatch: "
            f"expected={registry_spec['entries']}, "
            f"derived={registry_records}"
        )
    stable_plan_ids = [
        record["stable_plan_id"] for record in registry_records
    ]
    if len(stable_plan_ids) != len(set(stable_plan_ids)):
        raise RuntimeError(
            "SM90 compatibility registry has duplicate stable plan IDs"
        )
    expected_registry_keys = {
        ("kForward", "kDenseOrSegmentBundle"),
        ("kForward", "kVarlen"),
        ("kBackward", "kDenseOrSegmentBundle"),
        ("kBackward", "kVarlen"),
    }
    actual_registry_keys = {
        (record["direction_kind"], record["route_kind"])
        for record in registry_records
    }
    if actual_registry_keys != expected_registry_keys:
        raise RuntimeError(
            "SM90 compatibility registry support matrix mismatch: "
            f"{sorted(actual_registry_keys)}"
        )
    route_bundle_ids = {
        bundle["stable_id"]
        for bundle in manifest["route_bundles"].values()
    }
    unknown_route_bundle_ids = {
        record["stable_instance_route_bundle_id"]
        for record in registry_records
    } - route_bundle_ids
    if unknown_route_bundle_ids:
        raise RuntimeError(
            "SM90 compatibility registry references unknown instance "
            f"route bundles: {sorted(unknown_route_bundle_ids)}"
        )

    records = expand_sm90_instances(manifest, repo_root)
    stable_ids = [record["stable_instance_id"] for record in records]
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("SM90 manifest contains duplicate stable IDs")

    expected_sources = {record["source"] for record in records}
    source_directory = repo_root / manifest["source"]["directory"]
    actual_sources = {
        path.relative_to(repo_root).as_posix()
        for path in source_directory.glob(
            "sliding_chunk_attention_*_sm90.cu"
        )
    }
    if expected_sources != actual_sources:
        missing = sorted(expected_sources - actual_sources)
        extra = sorted(actual_sources - expected_sources)
        raise RuntimeError(
            "SM90 manifest/source mismatch: "
            f"missing={missing}, extra={extra}"
        )

    derived = derive_sm90_manifest_metadata(manifest, repo_root)
    if derived["group_counts"] != manifest["group_counts"]:
        raise RuntimeError(
            "SM90 manifest group counts are stale: "
            f"expected={manifest['group_counts']}, "
            f"derived={derived['group_counts']}"
        )
    if derived["coverage_sha256"] != manifest["coverage_sha256"]:
        raise RuntimeError(
            "SM90 manifest coverage hash is stale: "
            f"expected={manifest['coverage_sha256']}, "
            f"derived={derived['coverage_sha256']}"
        )
    for key in (
        "compat_registry_coverage_sha256",
        "compat_registry_source_sha256",
    ):
        if derived[key] != registry_spec[key]:
            raise RuntimeError(
                f"SM90 compatibility registry field {key} is stale: "
                f"expected={registry_spec[key]}, derived={derived[key]}"
            )
    generator = manifest["generator"]
    for key in ("loader_source_sha256", "validator_source_sha256"):
        if derived[key] != generator[key]:
            raise RuntimeError(
                f"SM90 manifest generator field {key} is stale: "
                f"expected={generator[key]}, derived={derived[key]}"
            )
    return records


@lru_cache(maxsize=None)
def _validated_records(repo_root: str) -> tuple[dict[str, Any], ...]:
    manifest = _load_raw_manifest()
    return tuple(validate_sm90_manifest(manifest, pathlib.Path(repo_root)))


def sm90_head_dims() -> tuple[int, ...]:
    manifest = _load_raw_manifest()
    head_dims = tuple(manifest["axes"]["head_dims"])
    if head_dims != tuple(manifest["axes"]["value_head_dims"]):
        raise RuntimeError(
            "FlashSCA SM90 currently requires matching D/V bucket axes"
        )
    return head_dims


def sm90_instantiation_sources(
    kind: str, repo_root: pathlib.Path = REPO_ROOT
) -> list[pathlib.Path]:
    if kind not in {"fwd", "bwd"}:
        raise ValueError(f"unsupported SM90 instance direction: {kind!r}")
    records = _validated_records(str(repo_root.resolve()))
    return [
        repo_root / record["source"]
        for record in records
        if record["direction"] == kind
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--derive",
        action="store_true",
        help="print hashes/counts derived from the source tree",
    )
    args = parser.parse_args()
    manifest = _load_raw_manifest()
    if args.derive:
        result = derive_sm90_manifest_metadata(manifest)
    else:
        records = validate_sm90_manifest(manifest)
        result = {
            "coverage_sha256": manifest["coverage_sha256"],
            "instance_count": len(records),
            "status": "ok",
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
