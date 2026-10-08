#!/usr/bin/env python3
"""Build auditable failure attributions and candidate experiences from rollouts.

The default path is deterministic and never calls a model.  ``--run-codex`` adds a
second-pass semantic review using the local ChatGPT-authenticated Codex CLI.  Neither
path writes to a deployed experience memory: every generated lesson remains a
candidate until a separate held-out validation promotes it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from planner import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    require_chatgpt_login,
    run_codex_inference,
    sanitized_codex_environment,
    write_json,
    write_text,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "failure_experience_analyzer_system_v1.txt"
)
DEFAULT_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "failure_experience_analysis_v1.schema.json"
)
ANALYZER_VERSION = "failure-experience-analyzer-v1.1"

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
ORDINAL_NUMBERS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

FAILURE_TYPES = {
    "task_understanding_error",
    "planning_error",
    "grounding_error",
    "completion_error",
    "executor_lapse",
    "recovery_failure",
    "observation_insufficient",
    "capability_limit",
    "unknown",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected episode directory: {path}")
    return int(match.group(1))


def discover_episode_dirs(task_dir: Path, include_success: bool) -> list[Path]:
    episodes = []
    for path in task_dir.glob("episode_*"):
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            continue
        summary = load_json(summary_path)
        if include_success or not bool(summary.get("success")):
            episodes.append(path)
    return sorted(episodes, key=episode_number)


def assert_analysis_split(summary: dict[str, Any], expected_split: str) -> None:
    """Hard-stop semantic failure analysis on the final test split."""
    if expected_split not in {"train", "val"}:
        raise ValueError("Failure analysis is permitted only on train or val")
    actual = summary.get("dataset_split")
    if actual != expected_split:
        raise ValueError(
            "Rollout split audit failed: "
            f"expected {expected_split!r}, summary records {actual!r}"
        )


def normalize_env_stage(stage: str | None) -> str:
    value = str(stage or "").lower()
    if "pick up the peg" in value:
        return "peg_pickup"
    if "hook the cube" in value:
        return "peg_push"
    if "pick up the cube" in value:
        return "cube_pickup"
    if "place the cube" in value:
        return "cube_place"
    if "close the gripper" in value and "push" in value:
        return "direct_push"
    return value.replace(" ", "_") or "unknown"


def method_from_env_stage(stage: str) -> str:
    normalized = normalize_env_stage(stage)
    if normalized.startswith("peg_"):
        return "peg_push"
    if normalized.startswith("cube_"):
        return "pick_place"
    if normalized == "direct_push":
        return "direct_push"
    return "unknown"


def method_from_subtask(subtask: str | None) -> str:
    value = str(subtask or "").lower()
    if "peg" in value and ("pick up" in value or "hook" in value or "push" in value):
        return "peg_push"
    if "pick up the cube" in value or "place the cube" in value:
        return "pick_place"
    if "close the gripper" in value and "push the cube" in value:
        return "direct_push"
    return "unknown"


def method_from_guidance(summary: dict[str, Any]) -> str:
    guidance = summary.get("planner_guidance_for_planner") or {}
    facts = " ".join(str(value) for value in guidance.get("fixed_facts", [])).lower()
    description = str(summary.get("detailed_task_description_for_planner", "")).lower()
    joined = facts + " " + description
    if "critically ambiguous" in joined or "ambiguous between" in joined:
        return "ambiguous"
    if "required method: peg" in joined or "use the peg-pushing method" in joined:
        return "peg_push"
    if "use direct gripper pushing" in joined or "required method: direct" in joined:
        return "direct_push"
    if "use the pick-and-place method" in joined or "required method: pick-and-place" in joined:
        return "pick_place"
    if "grasp the peg before moving" in joined and "held peg" in joined:
        return "peg_push"
    if "grasp and transport the cube directly" in joined or "visibly lifted clear" in joined:
        return "pick_place"
    return "unknown"


def env_stage_by_round(diagnostics: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in diagnostics:
        round_index = int(row.get("round_index", 0))
        after = row.get("after") or {}
        result[round_index] = normalize_env_stage(after.get("current_task_name"))
    return result


def first_stagnant_streak(history: list[dict[str, Any]], length: int = 3) -> int | None:
    """Return the start of the longest meaningful no-progress streak.

    A rollout can temporarily stagnate and later make progress.  Treating that first
    transient as the causal divergence produces a misleading recovery lesson, so we
    select the longest streak instead (the earliest one wins a tie).
    """
    best_start = None
    best_length = 0
    current_start = None
    current_length = 0
    for row in history:
        if row.get("recent_status") in {"stagnant", "uncertain"}:
            if current_length == 0:
                current_start = int(row["round_index"])
            current_length += 1
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
        else:
            current_start = None
            current_length = 0
    return best_start if best_length >= length else None


def first_premature_peg_transition(
    history: list[dict[str, Any]], stages: dict[int, str]
) -> int | None:
    candidates = []
    for row in history:
        round_index = int(row["round_index"])
        subtask = str(row.get("next_grounded_subtask", "")).lower()
        if "hook the cube" in subtask and stages.get(round_index) == "peg_pickup":
            candidates.append(round_index)
    if not candidates:
        return None

    # A premature proposal is not the terminal cause if the rollout subsequently
    # repaired the prerequisite and the evaluator advanced to peg pushing.
    first = candidates[0]
    if any(
        round_index > first and stage == "peg_push"
        for round_index, stage in stages.items()
    ):
        return None
    if peg_method_became_operational(history, after_round=first):
        return None
    return first


def peg_method_became_operational(
    history: list[dict[str, Any]], *, after_round: int
) -> bool:
    """Use deployment-visible trace evidence to reject a stale pickup diagnosis."""
    cube_motion = re.compile(
        r"(?:moved|moving|pushed|displaced).*cube.*toward|"
        r"cube.*(?:moved|moving|pushed|displaced).*toward"
    )
    for row in history:
        if int(row["round_index"]) <= after_round:
            continue
        state = row.get("execution_state") or {}
        outcome = str(row.get("recent_outcome", "")).lower()
        if (
            state.get("held_object") == "peg"
            and cube_motion.search(outcome)
        ):
            return True
    return False


def compact_trace(
    history: list[dict[str, Any]], stages: dict[int, str], selected_rounds: Iterable[int]
) -> list[dict[str, Any]]:
    by_round = {int(row["round_index"]): row for row in history}
    compact = []
    for round_index in sorted(set(selected_rounds)):
        row = by_round.get(round_index)
        if row is None:
            continue
        compact.append(
            {
                "round": round_index,
                "recent_frame_ids": row.get("recent_rollout_frame_ids", []),
                "recent_status": row.get("recent_status"),
                "recent_outcome": row.get("recent_outcome"),
                "next_grounded_subtask": row.get("next_grounded_subtask"),
                "execution_state": row.get("execution_state"),
                "env_stage_evaluator_only": stages.get(round_index),
            }
        )
    return compact


def evidence_rounds(
    history: list[dict[str, Any]], divergence: int | None, stagnation: int | None
) -> list[int]:
    if not history:
        return []
    last = int(history[-1]["round_index"])
    selected = {0, last}
    for anchor in (divergence, stagnation):
        if anchor is None:
            continue
        selected.update({max(0, anchor - 1), anchor, min(last, anchor + 1)})
    for row in history:
        if row.get("recent_status") in {"progressing", "regressed"}:
            selected.add(int(row["round_index"]))
            if len(selected) >= 8:
                break
    return sorted(selected)[:10]


def empty_candidate() -> dict[str, Any]:
    return {
        "condition": "",
        "lesson": "",
        "recommended_skill_or_policy": "",
        "success_evidence": [],
        "forbidden_behavior": [],
        "scope": "none",
    }


def requested_swing_cycles(summary: dict[str, Any]) -> int | None:
    description = str(
        summary.get("detailed_task_description_for_planner", "")
    ).lower()
    values = []
    cycle_pattern = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)"
        r"(?:\s+[a-z-]+){0,3}\s+cycles?\b"
    )
    for match in cycle_pattern.finditer(description):
        value = match.group(1)
        parsed = int(value) if value.isdigit() else NUMBER_WORDS.get(value)
        if parsed is not None:
            values.append(parsed)
    if "single cycle" in description:
        values.append(1)
    if "exactly twice" in description:
        values.append(2)
    return max(values) if values else None


def swing_phase_index(value: str | None, *, required_cycles: int) -> int | None:
    normalized = " ".join(str(value or "").lower().replace("_", " ").split())
    if "pick up the" in normalized and "cube" in normalized:
        return 0
    side = None
    if "right-side target" in normalized:
        side = "right"
    elif "left-side target" in normalized:
        side = "left"
    if side is not None:
        ordinal = next(
            (number for word, number in ORDINAL_NUMBERS.items() if word in normalized),
            None,
        )
        if ordinal is None:
            return None
        return 2 * ordinal - (1 if side == "right" else 0)
    if "put" in normalized and "cube" in normalized and (
        "table" in normalized or "down" in normalized
    ):
        return 2 * required_cycles + 1
    if "press the button" in normalized:
        return 2 * required_cycles + 2
    return None


def first_unresolved_swing_advance(
    history: list[dict[str, Any]],
    stages: dict[int, str],
    *,
    required_cycles: int,
) -> dict[str, Any] | None:
    stage_indices = {
        round_index: swing_phase_index(stage, required_cycles=required_cycles)
        for round_index, stage in stages.items()
    }
    for row in history:
        round_index = int(row["round_index"])
        proposed = swing_phase_index(
            row.get("next_grounded_subtask"), required_cycles=required_cycles
        )
        evaluator = stage_indices.get(round_index)
        if proposed is None or evaluator is None or proposed <= evaluator:
            continue
        later_indices = [
            value
            for later_round, value in stage_indices.items()
            if later_round >= round_index and value is not None
        ]
        if later_indices and max(later_indices) >= proposed:
            continue
        return {
            "round": round_index,
            "proposed_phase_index": proposed,
            "evaluator_phase_index": evaluator,
            "proposed_subtask": row.get("next_grounded_subtask"),
            "evaluator_stage": stages.get(round_index),
        }
    return None


def swing_heuristic_analysis(
    summary: dict[str, Any],
    history: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    stages = env_stage_by_round(diagnostics)
    required_cycles = requested_swing_cycles(summary)
    stagnant_round = first_stagnant_streak(history)
    terminal_round = int(history[-1]["round_index"]) if history else None
    unresolved = (
        first_unresolved_swing_advance(
            history, stages, required_cycles=required_cycles
        )
        if required_cycles is not None
        else None
    )
    terminal_subtask = (
        str(history[-1].get("next_grounded_subtask", "")) if history else ""
    )
    terminal_evaluator_stage = stages.get(terminal_round or 0, "unknown")

    if unresolved is not None:
        failure_type = "completion_error"
        responsible = "completion_judge"
        divergence = int(unresolved["round"])
        eligibility = "eligible_candidate"
        analysis_summary = (
            "The planner irreversibly advanced beyond the evaluator's ordered "
            "SwingXtimes stage. Later execution never established the skipped "
            "right/left visit prerequisite."
        )
        expected = (
            "Preserve an evidence-backed right-then-left visit ledger and advance "
            "only after the expected target visit is confirmed."
        )
        observed = (
            f"Round {divergence} proposed {unresolved['proposed_subtask']!r} while "
            f"the post-hoc evaluator remained at {unresolved['evaluator_stage']!r}."
        )
        candidate = {
            "condition": (
                "SwingXtimes is active and the planner is considering a target-side, "
                "put-down, or stop-button phase transition."
            ),
            "lesson": (
                "Maintain a monotonic task-local ledger of confirmed right-then-left "
                "visits; increment a cycle only after a distinct confirmed right visit "
                "followed by a distinct confirmed left visit."
            ),
            "recommended_skill_or_policy": (
                "Allow only the ledger's next ordinal target skill. Permit put-down "
                "only after the requested cycle count and permit the stop button only "
                "after verified release."
            ),
            "success_evidence": [
                "The correct cube remains controlled while reaching the expected side.",
                "The expected target visibly activates or cube-target arrival is supported across current observations.",
                "A repeated view of the same visit is not counted as a new visit.",
            ],
            "forbidden_behavior": [
                "Do not advance to a later target ordinal, put-down, or button on status text alone.",
                "Do not count repeated contact with the same target as another cycle.",
            ],
            "scope": "task",
        }
        confidence = 0.96
        experience_scope = "swing_ordered_cycle_ledger"
    elif summary.get("result") in {"timeout", "stagnation_early_stop"} and stagnant_round is not None:
        failure_type = "recovery_failure"
        responsible = "recovery_policy"
        divergence = stagnant_round
        eligibility = "eligible_candidate"
        analysis_summary = (
            "The planner remained on one SwingXtimes phase through a long no-progress "
            "streak without changing recovery geometry."
        )
        expected = "Change approach/contact geometry after bounded same-phase stagnation."
        observed = (
            f"A no-progress streak began at round {stagnant_round} and continued "
            f"toward terminal round {terminal_round}."
        )
        candidate = {
            "condition": (
                "The expected SwingXtimes target phase remains visually incomplete "
                "after three semantically equivalent attempts."
            ),
            "lesson": "Coordinate jitter around one ineffective target approach is not recovery.",
            "recommended_skill_or_policy": (
                "Re-establish cube control or select a meaningfully different approach "
                "to the same expected target, then retry within a finite budget."
            ),
            "success_evidence": [
                "The cube remains held and moves closer to the expected target.",
                "The expected target visit becomes visually confirmed.",
            ],
            "forbidden_behavior": [
                "Do not repeat near-identical grounding points until timeout."
            ],
            "scope": "task",
        }
        confidence = 0.92
        experience_scope = "swing_same_phase_recovery"
    else:
        failure_type = (
            "executor_lapse" if summary.get("result") == "fail" else "unknown"
        )
        responsible = "executor" if failure_type == "executor_lapse" else "unknown"
        divergence = terminal_round
        eligibility = "needs_more_evidence"
        analysis_summary = (
            "The deterministic sequence pass found no unresolved semantic advance; "
            "visual review is required to distinguish grounding from executor failure."
        )
        expected = f"Complete the current phase {terminal_evaluator_stage!r}."
        observed = f"The episode ended after {terminal_subtask!r}."
        candidate = empty_candidate()
        confidence = 0.45
        experience_scope = "swing_visual_review_required"

    selected = evidence_rounds(history, divergence, stagnant_round)
    trace = compact_trace(history, stages, selected)
    causal_evidence = [
        {
            "source": "planner_trace",
            "reference": f"round_{row['round']:04d}",
            "observation": f"{row['recent_status']}: {row['recent_outcome']}",
        }
        for row in trace[:4]
    ]
    if terminal_evaluator_stage != "unknown":
        causal_evidence.append(
            {
                "source": "evaluator_diagnostic",
                "reference": "env_diagnostics:terminal_stage",
                "observation": (
                    f"Post-hoc evaluator stage is {terminal_evaluator_stage}; this "
                    "field is excluded from deployable experience content."
                ),
            }
        )
    return {
        "failure_type": failure_type,
        "responsible_module": responsible,
        "first_divergence_round": divergence,
        "analysis_summary": analysis_summary,
        "expected_result": expected,
        "observed_result": observed,
        "causal_evidence": causal_evidence,
        "alternative_causes": [],
        "experience_eligibility": eligibility,
        "candidate_experience": candidate,
        "confidence": confidence,
        "derived_context": {
            "evaluator_method": "ordered_target_cycles",
            "guidance_method": "ordered_target_cycles",
            "selected_method": "ordered_target_cycles",
            "experience_scope": experience_scope,
            "required_cycles": required_cycles,
            "terminal_env_stage_evaluator_only": terminal_evaluator_stage,
            "first_unresolved_advance": unresolved,
            "first_no_progress_streak_round": stagnant_round,
            "evidence_rounds": selected,
            "compact_trace": trace,
        },
    }


def heuristic_analysis(
    summary: dict[str, Any],
    history: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    if summary.get("task") == "SwingXtimes":
        return swing_heuristic_analysis(summary, history, diagnostics)
    stages = env_stage_by_round(diagnostics)
    initial_env_stage = stages.get(0, "unknown")
    evaluator_method = method_from_env_stage(initial_env_stage)
    selected_method = method_from_subtask(
        history[0].get("next_grounded_subtask") if history else None
    )
    guidance_method = method_from_guidance(summary)
    premature_round = first_premature_peg_transition(history, stages)
    stagnant_round = first_stagnant_streak(history)
    terminal_round = int(history[-1]["round_index"]) if history else None

    if evaluator_method != "unknown" and selected_method != evaluator_method:
        failure_type = "task_understanding_error"
        responsible = "init_processor"
        divergence = 0
        eligibility = "needs_more_evidence"
        analysis_summary = (
            f"The rollout selected {selected_method}, while post-hoc evaluator diagnostics "
            f"identify the demonstrated method as {evaluator_method}. The error therefore "
            "predates motor execution."
        )
        expected = f"Resolve and preserve the demonstrated {evaluator_method} method before rollout."
        observed = f"Init/planner guidance selected {guidance_method}; round 0 issued {selected_method}."
        candidate = {
            "condition": "The initialization video does not provide decisive evidence for the manipulation method.",
            "lesson": "Do not silently collapse method uncertainty into a single irreversible rollout method.",
            "recommended_skill_or_policy": "Run a dedicated method-evidence check or abstain for review before starting the rollout.",
            "success_evidence": [
                "The chosen method is supported by an observable lift, held peg, or continuous tabletop contact cue."
            ],
            "forbidden_behavior": [
                "Do not infer direct pushing solely from the absence of a clearly visible lift."
            ],
            "scope": "task_family",
        }
        confidence = 0.99
    elif premature_round is not None:
        failure_type = "completion_error"
        responsible = "completion_judge"
        divergence = premature_round
        eligibility = "eligible_candidate"
        analysis_summary = (
            "The planner advanced from peg pickup to peg-mediated cube pushing while the "
            "post-hoc evaluator still considered peg pickup incomplete."
        )
        expected = "Keep executing peg pickup until secure held-together motion is confirmed."
        observed = f"Round {premature_round} issued a hook-cube skill before pickup completion."
        candidate = {
            "condition": "Peg pushing is required and the planner is considering transition from peg pickup to cube contact.",
            "lesson": "Treat peg pickup as complete only after the peg is visibly retained and moves with the gripper across separated frames.",
            "recommended_skill_or_policy": "Retry the canonical peg-pickup skill when held-together motion is not confirmed.",
            "success_evidence": [
                "The peg remains between the fingers.",
                "The peg translates with the gripper after lift or lateral motion."
            ],
            "forbidden_behavior": [
                "Do not issue hook-cube merely because the gripper occludes or touches the peg."
            ],
            "scope": "task_family",
        }
        confidence = 0.96
    elif summary.get("result") in {
        "timeout",
        "stagnation_early_stop",
        "planner_round_limit",
    } and stagnant_round is not None:
        failure_type = "recovery_failure"
        responsible = "recovery_policy"
        divergence = stagnant_round
        eligibility = "eligible_candidate"
        analysis_summary = (
            "The method and semantic skill remained compatible with the evaluator stage, "
            "but repeated no-progress did not cause an effective recovery; near-equivalent "
            "commands continued until timeout."
        )
        expected = "After confirmed stagnation, change contact geometry or recover a prerequisite."
        observed = (
            f"A no-progress streak began at round {stagnant_round}; the episode continued "
            f"to round {terminal_round} without completion."
        )
        last_skill = history[-1].get("next_grounded_subtask", "") if history else ""
        candidate = {
            "condition": "The same canonical manipulation skill has three confirmed stagnant observations with unchanged object-target relation.",
            "lesson": "Repeated coordinate jitter of the same ineffective command is not recovery.",
            "recommended_skill_or_policy": "Re-establish the prerequisite/contact geometry, then retry once; if no supported recovery exists, mark a capability failure instead of looping.",
            "success_evidence": [
                "The manipulated object shows displacement toward the target after recovery.",
                "The required held/contact state is visibly re-established."
            ],
            "forbidden_behavior": [
                f"Do not repeat near-equivalent `{last_skill}` commands indefinitely without observable displacement."
            ],
            "scope": "task_family",
        }
        confidence = 0.94
    else:
        failure_type = "unknown"
        responsible = "unknown"
        divergence = stagnant_round
        eligibility = "needs_more_evidence"
        analysis_summary = "The deterministic evidence pass cannot isolate one causal failure."
        expected = "Obtain visual and state evidence around the first divergence."
        observed = str(summary.get("final_progress", "No final progress record."))
        candidate = empty_candidate()
        confidence = 0.35

    selected = evidence_rounds(history, divergence, stagnant_round)
    trace = compact_trace(history, stages, selected)
    causal_evidence = [
        {
            "source": "planner_trace",
            "reference": f"round_{row['round']:04d}",
            "observation": f"{row['recent_status']}: {row['recent_outcome']}",
        }
        for row in trace[:4]
    ]
    if initial_env_stage != "unknown":
        causal_evidence.append(
            {
                "source": "evaluator_diagnostic",
                "reference": "env_diagnostics:first_stage",
                "observation": f"Post-hoc evaluator stage is {initial_env_stage}; this is not deployment-visible memory content.",
            }
        )

    return {
        "failure_type": failure_type,
        "responsible_module": responsible,
        "first_divergence_round": divergence,
        "analysis_summary": analysis_summary,
        "expected_result": expected,
        "observed_result": observed,
        "causal_evidence": causal_evidence,
        "alternative_causes": [],
        "experience_eligibility": eligibility,
        "candidate_experience": candidate,
        "confidence": confidence,
        "derived_context": {
            "evaluator_method": evaluator_method,
            "guidance_method": guidance_method,
            "selected_method": selected_method,
            "initial_env_stage_evaluator_only": initial_env_stage,
            "premature_peg_transition_round": premature_round,
            "first_no_progress_streak_round": stagnant_round,
            "evidence_rounds": selected,
            "compact_trace": trace,
        },
    }


def validate_model_analysis(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Model analysis must be a JSON object")
    if value.get("failure_type") not in FAILURE_TYPES:
        raise ValueError(f"Invalid failure_type: {value.get('failure_type')}")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between zero and one")


def adjudicate_analysis(
    heuristic: dict[str, Any], model_analysis: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select an analysis without silently converting disagreement into memory."""
    if model_analysis is None:
        return heuristic, {
            "status": "rules_only",
            "selected_source": "heuristic",
            "agrees_on_failure_type": None,
            "agrees_on_responsible_module": None,
            "candidate_allowed": True,
        }
    type_agreement = heuristic["failure_type"] == model_analysis["failure_type"]
    module_agreement = (
        heuristic["responsible_module"] == model_analysis["responsible_module"]
    )
    if type_agreement and module_agreement:
        return model_analysis, {
            "status": "consensus",
            "selected_source": "model",
            "agrees_on_failure_type": True,
            "agrees_on_responsible_module": True,
            "candidate_allowed": True,
        }
    return heuristic, {
        "status": "disputed",
        "selected_source": "heuristic_pending_review",
        "agrees_on_failure_type": type_agreement,
        "agrees_on_responsible_module": module_agreement,
        "candidate_allowed": False,
    }


def render_model_prompt(
    *,
    system_prompt: str,
    summary: dict[str, Any],
    heuristic: dict[str, Any],
    ground_truth_reference_attached: bool = False,
) -> str:
    compact_summary = {
        key: summary.get(key)
        for key in (
            "task",
            "episode",
            "result",
            "steps_executed",
            "rounds",
            "detailed_task_description_for_planner",
            "planner_guidance_for_planner",
            "final_progress",
            "terminal_after_grounded_subtask",
        )
    }
    payload = {
        "episode_summary_with_init_processor_hypotheses": compact_summary,
        "deterministic_evidence_pass": heuristic,
        "image_note": (
            "Attached images, when present, are chronological side-by-side front/wrist "
            "rollout frames. Files prefixed gt_demo_ are chronological frames from "
            "the train/val expert demonstration and are offline diagnostic evidence "
            "only; any deployable lesson must remain executable without privileged "
            "expert actions."
            if ground_truth_reference_attached
            else "Attached images, when present, are chronological side-by-side "
            "front/wrist rollout frames."
        ),
    }
    return system_prompt.rstrip() + "\n\nEPISODE EVIDENCE\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def extract_evidence_images(
    episode_dir: Path, summary: dict[str, Any], evidence_round_indices: list[int], output_dir: Path
) -> list[Path]:
    try:
        import imageio.v2 as imageio
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Model review image extraction needs imageio and Pillow; run this command "
            "inside the RoboMME environment."
        ) from exc

    task = str(summary["task"])
    episode = int(summary["episode"])
    horizon = int(summary.get("obs_horizon", 16))
    video = episode_dir / "videos" / f"{task}_ep{episode:02d}.mp4"
    if not video.is_file():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video)
    frame_count = reader.count_frames()
    frame_indices = {0, max(0, frame_count - 1)}
    for round_index in evidence_round_indices:
        frame_indices.add(min(frame_count - 1, max(0, round_index * horizon)))
    paths = []
    try:
        for frame_index in sorted(frame_indices)[:10]:
            path = output_dir / f"rollout_frame_{frame_index:04d}.png"
            Image.fromarray(reader.get_data(frame_index)).save(path)
            paths.append(path)
    finally:
        reader.close()
    return paths


def extract_ground_truth_images(video: Path, output_dir: Path) -> list[Path]:
    import imageio.v2 as imageio
    from PIL import Image

    if not video.is_file():
        raise FileNotFoundError(f"Ground-truth demonstration is missing: {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video)
    try:
        frame_count = reader.count_frames()
        indices = sorted(
            {max(0, round((frame_count - 1) * fraction)) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)}
        )
        paths = []
        for index in indices:
            path = output_dir / f"gt_demo_{index:04d}.png"
            Image.fromarray(reader.get_data(index)).save(path)
            paths.append(path)
        return paths
    finally:
        reader.close()


def analyze_episode(
    episode_dir: Path,
    output_root: Path,
    *,
    run_codex: bool,
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    timeout_seconds: float,
    system_prompt: str,
    output_schema: Path,
    environment: dict[str, str],
    reuse_model_analysis: bool,
    dataset_split: str,
    ground_truth_video: Path | None,
) -> dict[str, Any]:
    summary = load_json(episode_dir / "summary.json")
    assert_analysis_split(summary, dataset_split)
    history = load_json(episode_dir / "codex_planner" / "progress_history.json")
    diagnostics = load_jsonl(episode_dir / "env_diagnostics.jsonl")
    heuristic = heuristic_analysis(summary, history, diagnostics)
    episode_output = output_root / f"episode_{int(summary['episode']):02d}"
    episode_output.mkdir(parents=True, exist_ok=True)
    write_json(episode_output / "heuristic_analysis.json", heuristic)

    existing_path = episode_output / "analysis.json"
    existing = load_json(existing_path) if existing_path.is_file() else {}
    model_analysis = None
    model_metadata = None
    model_attempts = list(existing.get("model_attempts") or [])
    ground_truth_sha256 = (
        sha256_file(ground_truth_video) if ground_truth_video is not None else None
    )
    if not model_attempts and existing.get("model_metadata"):
        model_attempts.append(existing["model_metadata"])
    if run_codex:
        can_reuse = (
            reuse_model_analysis
            and existing.get("model_analysis")
            and (existing.get("model_metadata") or {}).get(
                "ground_truth_video_sha256"
            )
            == ground_truth_sha256
        )
        if can_reuse:
            model_analysis = existing["model_analysis"]
            model_metadata = existing.get("model_metadata")
        else:
            evidence = extract_evidence_images(
                episode_dir,
                summary,
                heuristic["derived_context"]["evidence_rounds"],
                episode_output / "evidence_frames",
            )
            if not evidence:
                raise FileNotFoundError(f"No evidence images for {episode_dir}")
            ground_truth_evidence = (
                extract_ground_truth_images(
                    ground_truth_video, episode_output / "ground_truth_frames"
                )
                if ground_truth_video is not None
                else []
            )
            prompt = render_model_prompt(
                system_prompt=system_prompt,
                summary=summary,
                heuristic=heuristic,
                ground_truth_reference_attached=bool(ground_truth_evidence),
            )
            write_text(episode_output / "prompt.txt", prompt)
            result = run_codex_inference(
                codex_bin=codex_bin,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=prompt,
                output_schema=output_schema,
                image_paths=[*evidence, *ground_truth_evidence],
                destination=episode_output / "codex",
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
            model_analysis = result["response"]
            validate_model_analysis(model_analysis)
            model_metadata = {
                key: result.get(key)
                for key in (
                    "thread_id",
                    "usage",
                    "started_at",
                    "wall_seconds",
                    "exit_code",
                    "capacity_retry_count",
                )
            }
            model_metadata["prompt_sha256"] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            model_metadata["ground_truth_video_sha256"] = ground_truth_sha256
            model_attempts.append(model_metadata)
            write_json(episode_output / "model_analysis.json", model_analysis)

    final_analysis, adjudication = adjudicate_analysis(heuristic, model_analysis)

    final = {
        "analyzer_version": ANALYZER_VERSION,
        "task": summary["task"],
        "dataset_split": dataset_split,
        "ground_truth_reference_sha256": ground_truth_sha256,
        "episode": int(summary["episode"]),
        "source_result": summary.get("result"),
        "source_episode_dir": episode_dir.name,
        "heuristic_analysis": heuristic,
        "model_analysis": model_analysis,
        "model_metadata": model_metadata,
        "model_attempts": model_attempts,
        "final_analysis": final_analysis,
        "adjudication": adjudication,
        "promotion_status": "candidate_only",
    }
    write_json(episode_output / "analysis.json", final)
    return final


def write_aggregate(output_root: Path, analyses: list[dict[str, Any]]) -> None:
    active_token_totals: Counter[str] = Counter()
    all_attempt_token_totals: Counter[str] = Counter()
    rows = []
    for analysis in analyses:
        final, adjudication = adjudicate_analysis(
            analysis["heuristic_analysis"], analysis.get("model_analysis")
        )
        derived = analysis["heuristic_analysis"]["derived_context"]
        usage = (analysis.get("model_metadata") or {}).get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                active_token_totals[str(key)] += value
        attempts = analysis.get("model_attempts") or []
        if not attempts and analysis.get("model_metadata"):
            attempts = [analysis["model_metadata"]]
        cumulative_usage: Counter[str] = Counter()
        for attempt in attempts:
            for key, value in (attempt.get("usage") or {}).items():
                if isinstance(value, int):
                    all_attempt_token_totals[str(key)] += value
                    cumulative_usage[str(key)] += value
        rows.append(
            {
                "episode": analysis["episode"],
                "source_result": analysis["source_result"],
                "failure_type": final["failure_type"],
                "responsible_module": final["responsible_module"],
                "first_divergence_round": final["first_divergence_round"],
                "experience_eligibility": final["experience_eligibility"],
                "confidence": final["confidence"],
                "adjudication_status": adjudication["status"],
                "selected_source": adjudication["selected_source"],
                "evaluator_method": derived["evaluator_method"],
                "guidance_method": derived["guidance_method"],
                "selected_method": derived["selected_method"],
                "input_tokens": usage.get("input_tokens", 0),
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
                "model_attempt_count": len(attempts),
                "cumulative_input_tokens": cumulative_usage.get("input_tokens", 0),
                "cumulative_output_tokens": cumulative_usage.get("output_tokens", 0),
                "analysis_summary": final["analysis_summary"],
            }
        )
    with (output_root / "episode_analysis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    failure_counts = Counter(row["failure_type"] for row in rows)
    module_counts = Counter(row["responsible_module"] for row in rows)
    model_failure_counts = Counter(
        str(analysis["model_analysis"]["failure_type"])
        for analysis in analyses
        if analysis.get("model_analysis")
    )
    model_module_counts = Counter(
        str(analysis["model_analysis"]["responsible_module"])
        for analysis in analyses
        if analysis.get("model_analysis")
    )
    model_eligibility_counts = Counter(
        str(analysis["model_analysis"]["experience_eligibility"])
        for analysis in analyses
        if analysis.get("model_analysis")
    )
    aggregate = {
        "analyzer_version": ANALYZER_VERSION,
        "episode_count": len(rows),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "responsible_module_counts": dict(sorted(module_counts.items())),
        "model_review_failure_type_counts": dict(sorted(model_failure_counts.items())),
        "model_review_responsible_module_counts": dict(
            sorted(model_module_counts.items())
        ),
        "model_review_experience_eligibility_counts": dict(
            sorted(model_eligibility_counts.items())
        ),
        "model_token_usage_total": dict(sorted(all_attempt_token_totals.items())),
        "active_model_token_usage": dict(sorted(active_token_totals.items())),
        "adjudication_counts": dict(
            sorted(Counter(row["adjudication_status"] for row in rows).items())
        ),
        "episodes": rows,
    }
    write_json(output_root / "aggregate.json", aggregate)

    candidate_groups: dict[
        tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for analysis in analyses:
        final, adjudication = adjudicate_analysis(
            analysis["heuristic_analysis"], analysis.get("model_analysis")
        )
        if not adjudication["candidate_allowed"]:
            continue
        candidate = final.get("candidate_experience") or {}
        if candidate.get("scope") == "none" or not candidate.get("lesson"):
            continue
        method_scope = analysis["heuristic_analysis"]["derived_context"].get(
            "evaluator_method", "unknown"
        )
        candidate_groups[
            (final["failure_type"], final["responsible_module"], method_scope)
        ].append((analysis, final))
    experience_candidates = []
    for (
        failure_type,
        responsible_module,
        method_scope,
    ), group in sorted(candidate_groups.items()):
        representative_analysis, representative = max(
            group, key=lambda pair: float(pair[1].get("confidence", 0.0))
        )
        candidate = representative["candidate_experience"]
        identity_text = (
            f"{failure_type}|{responsible_module}|{method_scope}|{candidate['scope']}"
        )
        experience_id = "experience_" + hashlib.sha256(
            identity_text.encode("utf-8")
        ).hexdigest()[:12]
        support_refs = sorted(
            f"{item['task']}:{int(item['episode']):02d}" for item, _ in group
        )
        eligibility_counts = Counter(
            item[1]["experience_eligibility"] for item in group
        )
        experience_candidates.append(
            {
                "experience_id": experience_id,
                "revision": 1,
                "status": (
                    "candidate"
                    if eligibility_counts.get("eligible_candidate", 0)
                    else "needs_evidence"
                ),
                "failure_type": failure_type,
                "responsible_module": responsible_module,
                "method_scope": method_scope,
                "condition": candidate["condition"],
                "lesson": candidate["lesson"],
                "recommended_skill_or_policy": candidate[
                    "recommended_skill_or_policy"
                ],
                "success_evidence": candidate["success_evidence"],
                "forbidden_behavior": candidate["forbidden_behavior"],
                "scope": candidate["scope"],
                "supporting_episode_refs": support_refs,
                "contradicting_episode_refs": [],
                "source_analysis_count": len(group),
                "eligibility_counts": dict(sorted(eligibility_counts.items())),
                "validation": {
                    "attempts": 0,
                    "improved": 0,
                    "unchanged_success": 0,
                    "unchanged_failure": 0,
                    "regressed": 0,
                    "records": [],
                },
                "promotion_note": (
                    "Candidate only. Validate on predeclared held-out episodes before "
                    "planner-memory use."
                ),
            }
        )
    experience_memory = {
        "schema_version": "failure-experience-memory-v1",
        "revision": 1,
        "status": "candidate_only",
        "source_analyzer_version": ANALYZER_VERSION,
        "model_token_usage_total": dict(sorted(all_attempt_token_totals.items())),
        "active_model_token_usage": dict(sorted(active_token_totals.items())),
        "experiences": experience_candidates,
    }
    write_json(output_root / "experience_candidates.json", experience_memory)

    lines = [
        "# Rollout failure analysis v1",
        "",
        f"Episodes analyzed: {len(rows)}.",
        "",
        "All generated lessons have `candidate_only` status; this command does not update deployed memory.",
        "",
        "Disputed rule/model attributions are excluded from candidate experience support.",
        "",
        "## Model review distribution",
        "",
        "This table reports visual-model attribution even when the deterministic rule disagrees; disputed rows still cannot promote an experience.",
        "",
        "| Model-reviewed failure type | Count |",
        "| --- | ---: |",
    ]
    for failure_type, count in sorted(model_failure_counts.items()):
        lines.append(f"| `{failure_type}` | {count} |")
    lines.extend([
        "",
        "## Adjudicated failure groups",
        "",
        "| Failure type | Count | Episodes |",
        "| --- | ---: | --- |",
    ])
    for failure_type, count in sorted(failure_counts.items()):
        episodes = ", ".join(
            f"{int(row['episode']):02d}" for row in rows if row["failure_type"] == failure_type
        )
        lines.append(f"| `{failure_type}` | {count} | {episodes} |")
    lines.extend(["", "## Candidate experience suggestions", ""])
    for candidate in experience_candidates:
        lines.extend(
            [
                f"### `{candidate['experience_id']}` — {candidate['failure_type']}",
                "",
                f"- Status: `{candidate['status']}`; support: {candidate['source_analysis_count']} episodes.",
                f"- Condition: {candidate['condition']}",
                f"- Lesson: {candidate['lesson']}",
                f"- Recommendation: {candidate['recommended_skill_or_policy']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Model token usage",
            "",
            f"- Total input tokens consumed across all attempts: {all_attempt_token_totals.get('input_tokens', 0)}",
            f"- Total cached input tokens: {all_attempt_token_totals.get('cached_input_tokens', 0)}",
            f"- Total output tokens consumed: {all_attempt_token_totals.get('output_tokens', 0)}",
            f"- Total reasoning output tokens: {all_attempt_token_totals.get('reasoning_output_tokens', 0)}",
            f"- Active-analysis input/output tokens: {active_token_totals.get('input_tokens', 0)} / {active_token_totals.get('output_tokens', 0)}",
        ]
    )
    lines.extend(
        [
            "",
            "## Episode details",
            "",
            "| Episode | Result | Failure type | Module | First divergence | Review | Candidate status |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in rows:
        divergence = row["first_divergence_round"]
        lines.append(
            f"| {int(row['episode']):02d} | {row['source_result']} | "
            f"`{row['failure_type']}` | `{row['responsible_module']}` | "
            f"{divergence if divergence is not None else '-'} | "
            f"`{row['adjudication_status']}` | "
            f"`{row['experience_eligibility']}` |"
        )
    write_text(output_root / "REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "val"),
        required=True,
        help="Audited source split. Test trajectories are intentionally unsupported.",
    )
    parser.add_argument("--include-success", action="store_true")
    parser.add_argument("--episode-ids", help="Comma-separated episode ids")
    parser.add_argument("--run-codex", action="store_true")
    parser.add_argument(
        "--ground-truth-manifest",
        type=Path,
        help="Optional split-qualified init-demo JSONL used only by the offline analyzer.",
    )
    parser.add_argument(
        "--reuse-model-analysis",
        action="store_true",
        help="Reuse an existing per-episode model analysis while refreshing rules.",
    )
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Only rebuild aggregate files from existing episode analyses.",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-schema", type=Path, default=DEFAULT_OUTPUT_SCHEMA)
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        analyses = [
            load_json(path)
            for path in sorted(output_dir.glob("episode_*/analysis.json"))
        ]
        if not analyses:
            raise FileNotFoundError(f"No existing episode analyses under {output_dir}")
        bad_splits = sorted(
            {
                str(value.get("dataset_split"))
                for value in analyses
                if value.get("dataset_split") != args.dataset_split
            }
        )
        if bad_splits:
            raise ValueError(
                f"Existing analysis split mismatch: expected {args.dataset_split}, "
                f"found {bad_splits}"
            )
        write_aggregate(output_dir, analyses)
        print(f"Aggregated {len(analyses)} existing analyses in {output_dir}")
        return
    episode_dirs = discover_episode_dirs(task_dir, args.include_success)
    if args.episode_ids:
        requested = {int(value) for value in args.episode_ids.split(",") if value.strip()}
        episode_dirs = [path for path in episode_dirs if episode_number(path) in requested]
    if not episode_dirs:
        raise FileNotFoundError(f"No matching episodes under {task_dir}")

    environment = sanitized_codex_environment()
    if args.run_codex:
        require_chatgpt_login(args.codex_bin, environment)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    ground_truth_by_identity: dict[tuple[str, int], Path] = {}
    if args.ground_truth_manifest is not None:
        for row in load_jsonl(args.ground_truth_manifest.resolve()):
            if row.get("dataset_split") != args.dataset_split:
                raise ValueError("Ground-truth manifest split does not match analysis split")
            video_value = row.get("video_path")
            if not video_value:
                continue
            video_path = Path(str(video_value))
            if not video_path.is_absolute():
                video_path = args.ground_truth_manifest.resolve().parent / video_path
            ground_truth_by_identity[(str(row["task"]), int(row["episode"]))] = (
                video_path.resolve()
            )
    analyses = []
    for episode_dir in episode_dirs:
        print(f"Analyzing {episode_dir.name} ...", flush=True)
        analyses.append(
            analyze_episode(
                episode_dir,
                output_dir,
                run_codex=args.run_codex,
                codex_bin=args.codex_bin,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=args.timeout_seconds,
                system_prompt=system_prompt,
                output_schema=args.output_schema,
                environment=environment,
                reuse_model_analysis=args.reuse_model_analysis,
                dataset_split=args.dataset_split,
                ground_truth_video=ground_truth_by_identity.get(
                    (
                        str(load_json(episode_dir / "summary.json")["task"]),
                        int(load_json(episode_dir / "summary.json")["episode"]),
                    )
                ),
            )
        )
    write_aggregate(output_dir, analyses)
    print(f"Wrote {len(analyses)} analyses to {output_dir}")


if __name__ == "__main__":
    main()
