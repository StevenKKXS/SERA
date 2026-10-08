#!/usr/bin/env python3
"""Gate and materialize a complete train+val evolution dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


NON_BEHAVIOR_EPISODE_CONFIG_KEYS = frozenset({
    "authentication",
    "codex_version",
    "created_at",
    "detailed_task_description",
    "harness_version",
    "planner_guidance",
    "prompt_profile",
    "provider_version",
    "rollout_behavior_id",
    "task",
})
EMPTY_INTERVENTION_FIELDS = (
    "runtime_experience_memory",
    "runtime_lesson_memory",
    "runtime_intervention_bundle",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_profile_contract(profile_name: str) -> dict[str, Any]:
    """Return the effective profile contract, ignoring its bookkeeping label."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from text_progress_planner import PROMPT_PROFILES  # pylint: disable=import-outside-toplevel

    if profile_name not in PROMPT_PROFILES:
        raise ValueError(f"Unknown planner profile in cache policy: {profile_name}")

    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return {"sha256": sha256_file(value.resolve())}
        if isinstance(value, dict):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items())
                if key != "harness_version"
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    contract = normalize(PROMPT_PROFILES[profile_name])
    return {
        "profile": profile_name,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
    }


def audit_episode_behavior(
    *, summary_path: Path, actual_profile: str, required_profile: str
) -> dict[str, Any]:
    """Audit a stored legacy rollout from its materialized runtime artifacts."""
    planner_dir = summary_path.parent / "codex_planner"
    config_path = planner_dir / "episode_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing planner episode config for cache audit: {config_path}")
    config = load_json(config_path)
    if config.get("prompt_profile") != actual_profile:
        raise ValueError(f"Planner episode profile mismatch: {config_path}")
    for field in EMPTY_INTERVENTION_FIELDS:
        if config.get(field) is not None:
            raise ValueError(
                f"Baseline cache has non-empty {field}: {config_path}"
            )
    if config.get("cross_episode_lesson_loading") not in (None, False):
        raise ValueError(f"Baseline cache loaded cross-episode lessons: {config_path}")
    if config.get("experience_transition_gate") not in (None, {}):
        raise ValueError(f"Baseline cache has an experience gate: {config_path}")
    if config.get("counting_event_ledger_guard_enabled") not in (None, False):
        raise ValueError(f"Baseline cache has a counting ledger: {config_path}")

    required_contract = prompt_profile_contract(required_profile)
    actual_contract = prompt_profile_contract(actual_profile)
    if actual_contract["contract_sha256"] != required_contract["contract_sha256"]:
        raise ValueError(
            "Legacy baseline profile is not behavior-equivalent to the required "
            f"profile: {actual_profile} != {required_profile}"
        )

    artifact_paths = {
        "system_prompt": planner_dir / "planner_system_prompt.txt",
        "user_prompt": planner_dir / "planner_user_prompt_template.txt",
        "output_schema": planner_dir / "planner_output_schema.json",
        "refine_check_criteria": planner_dir / "refine_check_criteria.json",
    }
    artifact_hashes: dict[str, str] = {}
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise ValueError(f"Missing planner artifact for cache audit: {path}")
        artifact_hashes[name] = sha256_file(path)

    required_profile_value = required_contract["contract"]
    for name in ("system_prompt", "user_prompt", "output_schema"):
        expected = required_profile_value[name]["sha256"]
        if artifact_hashes[name] != expected:
            raise ValueError(
                f"Stored {name} does not match required baseline behavior: {summary_path}"
            )
    expected_criteria = required_profile_value.get("refine_check_criteria_path", {}).get(
        "sha256"
    )
    if expected_criteria and artifact_hashes["refine_check_criteria"] != expected_criteria:
        raise ValueError(
            "Stored refine criteria do not match required baseline behavior: "
            f"{summary_path}"
        )

    normalized_config = {
        key: value
        for key, value in config.items()
        if key not in NON_BEHAVIOR_EPISODE_CONFIG_KEYS
    }
    fingerprint_payload = {
        "episode_runtime_config": normalized_config,
        "planner_artifacts": artifact_hashes,
    }
    return {
        "episode_config_path": str(config_path.resolve()),
        "episode_config_sha256": sha256_file(config_path),
        "profile_contract_sha256": required_contract["contract_sha256"],
        "planner_artifact_sha256": artifact_hashes,
        "runtime_behavior_fingerprint": canonical_sha256(fingerprint_payload),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def pool_ids(path: Path, expected_task: str, expected_split: str) -> list[int]:
    value = load_json(path)
    if value.get("task") != expected_task or value.get("dataset_split") != expected_split:
        raise ValueError(f"Pool identity mismatch: {path}")
    ids = [int(row["episode"]) for row in value.get("candidates") or []]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Pool contains duplicate episode ids: {path}")
    return ids


def normalized_run_roots(value: Path | list[Path] | tuple[Path, ...]) -> list[Path]:
    roots = [value] if isinstance(value, Path) else list(value)
    if not roots:
        raise ValueError("At least one rollout source root is required")
    resolved = [path.resolve() for path in roots]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate rollout source roots are not allowed")
    return resolved


def collect_split(
    *, run_root: Path | list[Path] | tuple[Path, ...], task: str, split: str,
    expected_ids: list[int], profile: str,
    required_behavior_id: str | None = None,
    compatible_profiles: frozenset[str] = frozenset(),
    require_episode_config_audit: bool = False,
) -> list[dict[str, Any]]:
    summaries: dict[int, tuple[Path, dict[str, Any]]] = {}
    for root in normalized_run_roots(run_root):
        for path in root.rglob("summary.json"):
            value = load_json(path)
            if value.get("task") != task:
                continue
            episode = int(value["episode"])
            if episode in summaries:
                raise ValueError(
                    f"Duplicate completed summary for {split} episode {episode}"
                )
            summaries[episode] = (path, value)
    missing = sorted(set(expected_ids) - set(summaries))
    extra = sorted(set(summaries) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"Incomplete {split} baseline: missing={missing}, extra={extra}")

    records = []
    for episode in expected_ids:
        path, value = summaries[episode]
        if value.get("complete") is not True:
            raise ValueError(f"Incomplete summary: {path}")
        if value.get("dataset_split") != split:
            raise ValueError(f"Dataset split mismatch: {path}")
        actual_profile = value.get("planner_prompt_profile")
        if actual_profile != profile and actual_profile not in compatible_profiles:
            raise ValueError(f"Planner profile mismatch: {path}")
        if value.get("experience_ids_used") not in (None, []):
            raise ValueError(f"Baseline unexpectedly loaded experience: {path}")
        recorded_behavior_id = value.get("planner_behavior_id")
        if (
            required_behavior_id is not None
            and recorded_behavior_id is not None
            and recorded_behavior_id != required_behavior_id
        ):
            raise ValueError(f"Planner behavior id mismatch: {path}")
        behavior_audit = None
        if require_episode_config_audit:
            behavior_audit = audit_episode_behavior(
                summary_path=path,
                actual_profile=str(actual_profile),
                required_profile=profile,
            )
        compatibility_mode = (
            "native_profile"
            if actual_profile == profile
            else "behavior_equivalent_legacy_profile"
        )
        records.append(
            {
                "record_id": f"{split}:{task}:{episode}",
                "task": task,
                "dataset_split": split,
                "episode": episode,
                "success": bool(value.get("success")),
                "result": value.get("result"),
                "steps_executed": value.get("steps_executed"),
                "rounds": value.get("rounds"),
                "summary_path": str(path.resolve()),
                "summary_sha256": sha256_file(path),
                "source_baseline_profile": actual_profile,
                "baseline_behavior_id": required_behavior_id or recorded_behavior_id,
                "cache_compatibility_mode": compatibility_mode,
                "behavior_audit": behavior_audit,
                "codex_usage_total": value.get("codex_usage_total") or {},
            }
        )
    return records


def prepare(
    *, config_path: Path, task: str, train_pool: Path, val_pool: Path,
    train_run_root: Path | list[Path] | tuple[Path, ...],
    val_run_root: Path | list[Path] | tuple[Path, ...], output_dir: Path
) -> dict[str, Any]:
    config = load_json(config_path)
    dataset_id = str(config["episode_sampling"]["dataset_id"])
    evolution_dataset_name = str(
        config["evolution_dataset_gate"]["dataset_name"]
    )
    expected_name = f"{dataset_id}_evolution50_dataset"
    if evolution_dataset_name != expected_name:
        raise ValueError(
            f"Evolution dataset name must be {expected_name!r}, got "
            f"{evolution_dataset_name!r}"
        )
    gate = config["evolution_dataset_gate"]
    train_ids = pool_ids(train_pool, task, "train")
    val_ids = pool_ids(val_pool, task, "val")
    if len(train_ids) != int(gate["required_train_episodes"]):
        raise ValueError("Train pool size does not match evolution dataset gate")
    if len(val_ids) != int(gate["required_val_episodes"]):
        raise ValueError("Val pool size does not match evolution dataset gate")
    profile = str(gate["required_baseline_profile"])
    required_behavior_id = gate.get("required_baseline_behavior_id")
    compatible_profiles = frozenset(
        str(value) for value in gate.get("compatible_baseline_profiles", [])
    )
    require_episode_config_audit = bool(
        gate.get("require_episode_config_behavior_audit", False)
    )
    records = collect_split(
        run_root=train_run_root, task=task, split="train",
        expected_ids=train_ids, profile=profile,
        required_behavior_id=required_behavior_id,
        compatible_profiles=compatible_profiles,
        require_episode_config_audit=require_episode_config_audit,
    ) + collect_split(
        run_root=val_run_root, task=task, split="val",
        expected_ids=val_ids, profile=profile,
        required_behavior_id=required_behavior_id,
        compatible_profiles=compatible_profiles,
        require_episode_config_audit=require_episode_config_audit,
    )
    expected_total = int(gate["required_train_episodes"]) + int(gate["required_val_episodes"])
    if len(records) != expected_total:
        raise ValueError("Evolution dataset is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_dir / f"{evolution_dataset_name}_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
    )
    by_split = Counter((row["dataset_split"], row["success"]) for row in records)
    source_profiles = sorted({str(row["source_baseline_profile"]) for row in records})
    compatibility_modes = sorted({str(row["cache_compatibility_mode"]) for row in records})
    behavior_fingerprints = sorted({
        str(row["behavior_audit"]["runtime_behavior_fingerprint"])
        for row in records
        if row.get("behavior_audit") is not None
    })
    if require_episode_config_audit and len(behavior_fingerprints) != 1:
        raise ValueError(
            "Evolution baseline episodes do not share one runtime behavior fingerprint"
        )
    aggregate = {
        "schema_version": "self-evolve-evolution-dataset-gate-v2",
        "dataset_id": dataset_id,
        "evolution_dataset_name": evolution_dataset_name,
        "task": task,
        "complete": True,
        "evolution_episode_count": len(records),
        "train_episode_count": len(train_ids),
        "val_episode_count": len(val_ids),
        "successes": sum(row["success"] for row in records),
        "failures": sum(not row["success"] for row in records),
        "split_summary": {
            split: {
                "successes": by_split[(split, True)],
                "failures": by_split[(split, False)],
            }
            for split in ("train", "val")
        },
        "baseline_profile": profile,
        "baseline_behavior_id": required_behavior_id,
        "source_baseline_profiles": source_profiles,
        "cache_compatibility_modes": compatibility_modes,
        "cross_campaign_cache_reused": (
            "behavior_equivalent_legacy_profile" in compatibility_modes
        ),
        "runtime_behavior_fingerprint": (
            behavior_fingerprints[0] if behavior_fingerprints else None
        ),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "train_pool_sha256": sha256_file(train_pool),
        "val_pool_sha256": sha256_file(val_pool),
        "analysis_unlocked": True,
    }
    cache_import = {
        "schema_version": "self-evolve-baseline-cache-import-v1",
        "task": task,
        "read_only": True,
        "required_baseline_profile": profile,
        "required_baseline_behavior_id": required_behavior_id,
        "source_baseline_profiles": source_profiles,
        "compatibility_modes": compatibility_modes,
        "runtime_behavior_fingerprint": aggregate["runtime_behavior_fingerprint"],
        "train_run_roots": [
            str(path) for path in normalized_run_roots(train_run_root)
        ],
        "val_run_roots": [
            str(path) for path in normalized_run_roots(val_run_root)
        ],
        "records": [
            {
                "record_id": row["record_id"],
                "summary_path": row["summary_path"],
                "summary_sha256": row["summary_sha256"],
                "episode_config_path": (
                    row["behavior_audit"]["episode_config_path"]
                    if row.get("behavior_audit") is not None else None
                ),
                "episode_config_sha256": (
                    row["behavior_audit"]["episode_config_sha256"]
                    if row.get("behavior_audit") is not None else None
                ),
            }
            for row in records
        ],
    }
    atomic_write(
        output_dir / "baseline_cache_import.json",
        json.dumps(cache_import, indent=2) + "\n",
    )
    atomic_write(
        output_dir / f"{evolution_dataset_name}_gate_report.json",
        json.dumps(aggregate, indent=2) + "\n",
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--val-pool", type=Path, required=True)
    parser.add_argument("--train-run-root", type=Path, action="append", required=True)
    parser.add_argument("--val-run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        config_path=args.config.resolve(), task=args.task,
        train_pool=args.train_pool.resolve(), val_pool=args.val_pool.resolve(),
        train_run_root=[path.resolve() for path in args.train_run_root],
        val_run_root=[path.resolve() for path in args.val_run_root],
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
