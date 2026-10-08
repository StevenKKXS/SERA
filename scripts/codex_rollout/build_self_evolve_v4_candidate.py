#!/usr/bin/env python3
"""Materialize and compose typed self-evolve v4 intervention candidates."""

from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from interventions import compose_candidate_manifests, validate_candidate_manifest
    from interventions.bundle import sha256_file
    from planner import write_json
except ModuleNotFoundError:  # Repository-root unittest imports.
    from scripts.codex_rollout.interventions import (
        compose_candidate_manifests,
        validate_candidate_manifest,
    )
    from scripts.codex_rollout.interventions.bundle import sha256_file
    from scripts.codex_rollout.planner import write_json


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Candidate artifact path must be source-relative: {path}")
    return path


def materialize_candidate(source_path: Path, output_dir: Path) -> Path:
    """Copy candidate artifacts into a content-addressed, self-contained directory."""
    source_path = source_path.resolve()
    source_dir = source_path.parent
    output_dir = output_dir.resolve()
    candidate = load_json(source_path)
    components = candidate.get("components")
    if not isinstance(components, list):
        raise ValueError("Candidate components must be a list")
    materialized_components = []
    for raw_component in components:
        if not isinstance(raw_component, dict):
            raise ValueError("Candidate component must be an object")
        component = json.loads(json.dumps(raw_component))
        component_id = str(component.get("component_id", ""))
        artifacts = component.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"{component_id}.artifacts must be an object")
        materialized_artifacts = {}
        for name, descriptor in artifacts.items():
            if not isinstance(descriptor, dict) or "path" not in descriptor:
                raise ValueError(f"Invalid source artifact {component_id}.{name}")
            source_artifact = (
                source_dir / _safe_relative(str(descriptor["path"]))
            ).resolve()
            try:
                source_artifact.relative_to(source_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    f"Candidate artifact escapes source directory: {source_artifact}"
                ) from error
            if not source_artifact.is_file():
                raise ValueError(f"Missing candidate artifact: {source_artifact}")
            suffix = "".join(source_artifact.suffixes)
            relative = Path("artifacts") / component_id / f"{name}{suffix}"
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_artifact, destination)
            materialized_artifacts[str(name)] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(destination),
            }
        component["artifacts"] = materialized_artifacts
        materialized_components.append(component)
    candidate["components"] = materialized_components
    validate_candidate_manifest(candidate, bundle_dir=output_dir)
    manifest_path = output_dir / "candidate_manifest.json"
    write_json(manifest_path, candidate)
    return manifest_path


def run_static_checks(candidate_path: Path, repo_root: Path | None = None) -> None:
    candidate_path = candidate_path.resolve()
    candidate = validate_candidate_manifest(
        load_json(candidate_path), bundle_dir=candidate_path.parent
    )
    for component in candidate["components"]:
        component_type = component["type"]
        if component_type == "python_plugin":
            module = candidate_path.parent / component["artifacts"]["module"]["path"]
            py_compile.compile(str(module), doraise=True)
        elif component_type == "core_harness_patch":
            patch = candidate_path.parent / component["artifacts"]["patch"]["path"]
            if repo_root is None:
                raise ValueError("core_harness_patch static checks require --repo-root")
            repo_root = repo_root.resolve()
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
            patch_config = component["config"]
            if patch_config["materialized_commit"] != head:
                raise ValueError(
                    "core_harness_patch materialized_commit must equal repository HEAD"
                )
            base_check = subprocess.run(
                ["git", "cat-file", "-e", f"{patch_config['base_commit']}^{{commit}}"],
                cwd=repo_root,
                text=True,
                capture_output=True,
            )
            if base_check.returncode != 0:
                raise ValueError("core_harness_patch base_commit is not available")
            completed = subprocess.run(
                ["git", "apply", "--numstat", str(patch)],
                cwd=repo_root,
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise ValueError(f"Invalid harness patch: {completed.stderr.strip()}")


def compose_candidates(
    candidate_paths: list[Path], output_dir: Path, *, bundle_id: str, revision: int
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for candidate_path in candidate_paths:
        candidate_path = candidate_path.resolve()
        candidate = validate_candidate_manifest(
            load_json(candidate_path), bundle_dir=candidate_path.parent
        )
        copied = json.loads(json.dumps(candidate))
        for component in copied["components"]:
            for name, descriptor in component["artifacts"].items():
                source = candidate_path.parent / descriptor["path"]
                relative = (
                    Path("artifacts")
                    / candidate["candidate_id"]
                    / component["component_id"]
                    / f"{name}{''.join(Path(descriptor['path']).suffixes)}"
                )
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                descriptor["path"] = relative.as_posix()
                descriptor["sha256"] = sha256_file(destination)
        candidates.append(copied)
    runtime = compose_candidate_manifests(
        candidates, bundle_id=bundle_id, revision=revision
    )
    manifest_path = output_dir / "runtime_intervention_bundle.json"
    write_json(manifest_path, runtime)
    # Reloading performs artifact, identity-isolation, and conflict checks on the
    # exact directory the planner will receive.
    try:
        from interventions import load_runtime_intervention_bundle
    except ModuleNotFoundError:
        from scripts.codex_rollout.interventions import load_runtime_intervention_bundle

    load_runtime_intervention_bundle(manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--candidate", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--repo-root", type=Path)

    compose = subparsers.add_parser("compose")
    compose.add_argument("--candidate", type=Path, action="append", required=True)
    compose.add_argument("--output-dir", type=Path, required=True)
    compose.add_argument("--bundle-id", required=True)
    compose.add_argument("--revision", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        path = materialize_candidate(args.candidate, args.output_dir)
        run_static_checks(path, args.repo_root)
    else:
        path = compose_candidates(
            args.candidate,
            args.output_dir,
            bundle_id=args.bundle_id,
            revision=args.revision,
        )
    print(path)


if __name__ == "__main__":
    main()
