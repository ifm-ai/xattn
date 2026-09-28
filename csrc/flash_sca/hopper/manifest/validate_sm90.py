from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from typing import Any


MACRO_RE = re.compile(
    r"^\s*(XATTN_FLASH_SCA_(?:FWD|BWD)_SM90_INSTANTIATE"
    r"(?:_[A-Z_]+)?)\s*\(([^()]*)\)",
    re.MULTILINE,
)
MACRO_START_RE = re.compile(r"^\s*XATTN_FLASH_SCA_", re.MULTILINE)
COMPAT_REGISTRY_ENTRY_RE = re.compile(
    r"struct\s+ScaCompatRegistryEntrySm90<\s*"
    r"semantics::DirectionKind::(?P<direction>kForward|kBackward),\s*"
    r"semantics::RouteKind::(?P<route>"
    r"kDenseOrSegmentBundle|kVarlen)>\s*\{"
    r"(?P<body>.*?)\n\};",
    re.DOTALL,
)


def _file_hash(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def independently_validate(
    manifest_path: pathlib.Path, repo_root: pathlib.Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    axes = manifest["axes"]

    registry_spec = manifest["compat_registry"]
    registry_source = repo_root / registry_spec["source"]
    registry_content = registry_source.read_text(encoding="utf-8")
    registry_records = []
    for match in COMPAT_REGISTRY_ENTRY_RE.finditer(registry_content):
        body = match.group("body")

        def required(pattern: str, field: str) -> str:
            value = re.search(pattern, body)
            if value is None:
                raise AssertionError(
                    f"registry entry missing {field}: {match.group(0)}"
                )
            return value.group(1)

        registry_records.append(
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
    if registry_records != registry_spec["entries"]:
        raise AssertionError(
            "compat registry records differ from manifest: "
            f"{registry_records} != {registry_spec['entries']}"
        )
    registry_keys = {
        (record["direction_kind"], record["route_kind"])
        for record in registry_records
    }
    expected_registry_keys = {
        ("kForward", "kDenseOrSegmentBundle"),
        ("kForward", "kVarlen"),
        ("kBackward", "kDenseOrSegmentBundle"),
        ("kBackward", "kVarlen"),
    }
    if registry_keys != expected_registry_keys:
        raise AssertionError(
            f"compat registry support mismatch: {registry_keys}"
        )
    stable_plan_ids = [
        record["stable_plan_id"] for record in registry_records
    ]
    if len(stable_plan_ids) != len(set(stable_plan_ids)):
        raise AssertionError("duplicate compatibility stable plan IDs")
    route_bundle_ids = {
        bundle["stable_id"]
        for bundle in manifest["route_bundles"].values()
    }
    if any(
        record["stable_instance_route_bundle_id"]
        not in route_bundle_ids
        for record in registry_records
    ):
        raise AssertionError(
            "compat registry references an unknown instance route bundle"
        )
    registry_coverage_hash = _canonical_hash(registry_records)
    if (
        registry_coverage_hash
        != registry_spec["compat_registry_coverage_sha256"]
    ):
        raise AssertionError(
            "compat registry coverage hash mismatch: "
            f"{registry_coverage_hash} != "
            f"{registry_spec['compat_registry_coverage_sha256']}"
        )
    registry_source_hash = _file_hash(registry_source)
    if (
        registry_source_hash
        != registry_spec["compat_registry_source_sha256"]
    ):
        raise AssertionError(
            "compat registry source hash mismatch: "
            f"{registry_source_hash} != "
            f"{registry_spec['compat_registry_source_sha256']}"
        )

    records = []
    source_names = set()
    stable_ids = set()

    for direction in axes["directions"]:
        for kernel_dtype in axes["kernel_dtypes"]:
            for d in axes["head_dims"]:
                for v in axes["value_head_dims"]:
                    relative = pathlib.PurePosixPath(
                        manifest["source"]["directory"]
                    ) / manifest["source"]["filename"].format(
                        direction=direction,
                        head_dim=d,
                        value_head_dim=v,
                        kernel_dtype=kernel_dtype,
                    )
                    source = repo_root / relative
                    if not source.is_file():
                        raise AssertionError(f"missing source: {relative}")
                    content = source.read_text(encoding="utf-8")
                    matches = list(MACRO_RE.finditer(content))
                    starts = list(MACRO_START_RE.finditer(content))
                    if len(matches) != len(starts):
                        raise AssertionError(
                            f"unknown macro in {relative}"
                        )
                    macro_calls = []
                    for match in matches:
                        observed_direction = (
                            "fwd" if "_FWD_" in match.group(1) else "bwd"
                        )
                        if observed_direction != direction:
                            raise AssertionError(
                                f"direction mismatch in {relative}"
                            )
                        arguments = re.sub(
                            r"\s+", " ", match.group(2).strip()
                        )
                        macro_calls.append(
                            f"{match.group(1)}({arguments})"
                        )
                    if not macro_calls:
                        raise AssertionError(f"no macro in {relative}")

                    stable_id = (
                        f"sm90/{direction}/sca/{kernel_dtype}/"
                        f"d{d}/v{v}/route_bundle"
                    )
                    if stable_id in stable_ids:
                        raise AssertionError(
                            f"duplicate stable ID: {stable_id}"
                        )
                    stable_ids.add(stable_id)
                    source_names.add(relative.as_posix())
                    records.append(
                        {
                            "architecture": manifest["architecture"],
                            "d": d,
                            "determinism_specialization": (
                                "not_applicable"
                                if direction == "fwd"
                                else
                                "runtime_deterministic_and_nondeterministic"
                            ),
                            "direction": direction,
                            "formula_resource_class": manifest[
                                "formula_resource_class"
                            ],
                            "instantiation_macros": macro_calls,
                            "kernel_dtype": kernel_dtype,
                            "route_bundle": manifest["route_bundles"][
                                direction
                            ]["stable_id"],
                            "semantic_identity": manifest[
                                "semantic_identity"
                            ],
                            "source": relative.as_posix(),
                            "source_sha256": _file_hash(source),
                            "stable_instance_id": stable_id,
                            "v": v,
                            "variant": manifest["variant"],
                        }
                    )

    actual_sources = {
        path.relative_to(repo_root).as_posix()
        for path in (
            repo_root / manifest["source"]["directory"]
        ).glob("sliding_chunk_attention_*_sm90.cu")
    }
    if source_names != actual_sources:
        raise AssertionError(
            "source partition mismatch: "
            f"missing={sorted(source_names - actual_sources)}, "
            f"extra={sorted(actual_sources - source_names)}"
        )

    counts = {
        "total": len(records),
        "fwd": sum(record["direction"] == "fwd" for record in records),
        "bwd": sum(record["direction"] == "bwd" for record in records),
    }
    if counts != manifest["group_counts"]:
        raise AssertionError(
            f"group count mismatch: {counts} != {manifest['group_counts']}"
        )
    coverage_hash = _canonical_hash(records)
    if coverage_hash != manifest["coverage_sha256"]:
        raise AssertionError(
            "coverage hash mismatch: "
            f"{coverage_hash} != {manifest['coverage_sha256']}"
        )

    generator = manifest["generator"]
    for source_key, hash_key in (
        ("loader_source", "loader_source_sha256"),
        ("validator_source", "validator_source_sha256"),
    ):
        observed = _file_hash(repo_root / generator[source_key])
        if observed != generator[hash_key]:
            raise AssertionError(
                f"{hash_key} mismatch: {observed} != "
                f"{generator[hash_key]}"
            )
    return {
        "compat_registry_coverage_sha256": registry_coverage_hash,
        "compat_registry_entry_count": len(registry_records),
        "coverage_sha256": coverage_hash,
        "group_counts": counts,
        "stable_id_count": len(stable_ids),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = pathlib.Path(__file__).resolve().parents[4]
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("sm90_instances.json"),
    )
    parser.add_argument(
        "--repo-root", type=pathlib.Path, default=default_root
    )
    args = parser.parse_args()
    print(
        json.dumps(
            independently_validate(args.manifest, args.repo_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
