#!/usr/bin/env python3
"""Codex rollout planner using canonical text progress as its only memory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from planner import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    get_codex_version,
    normalize_response,
    render_grounded_subtask,
    require_chatgpt_login,
    run_codex_inference,
    sanitized_codex_environment,
    utc_now,
    write_json,
    write_text,
)
from planner_backends import (
    get_openai_compatible_version,
    run_openai_compatible_inference,
)
from skill_retrieval import SCHEMES as RETRIEVAL_SCHEMES
from skill_retrieval import ExecutorSkillRetriever
from interventions import canonical_hash as intervention_canonical_hash
from interventions import load_runtime_intervention_bundle


HARNESS_VERSION = "codex-text-progress-planner-v2.1"
OUTPUT_VALIDATION_RETRY_LIMIT = 2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = SCRIPT_DIR / "prompts" / "planner_system_text_progress_v2.txt"
DEFAULT_USER_PROMPT = SCRIPT_DIR / "prompts" / "planner_user_text_progress_v2.txt"
DEFAULT_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_output_v2.schema.json"
)
DEFAULT_PROMPT_PROFILE = "text_progress_v2"
RETHINKING_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_rethinking_v3.txt"
)
RETHINKING_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_rethinking_v3.txt"
)
RETHINKING_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_rethinking_output_v3.schema.json"
)
LESSON_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_lesson_v4.txt"
)
LESSON_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_lesson_v4.txt"
)
LESSON_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_lesson_output_v4.schema.json"
)
ORACLE_SKILLS_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_oracle_skills_v5.txt"
)
ORACLE_SKILLS_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_oracle_skills_v5.txt"
)
ORACLE_SKILLS_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_oracle_skills_output_v5.schema.json"
)
ORACLE_SKILL_CATALOG_PATH = SCRIPT_DIR / "oracle_skills" / "move_cube_v1.json"
GLOBAL_RETRIEVAL_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_retrieval_v6.txt"
)
GLOBAL_RETRIEVAL_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_retrieval_v6.txt"
)
GLOBAL_RETRIEVAL_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_retrieval_output_v6.schema.json"
)
ACTION_SUCCESS_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_action_success_v8.txt"
)
ACTION_SUCCESS_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_action_success_v8.txt"
)
ACTION_SUCCESS_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_action_success_output_v8.schema.json"
)
ACTION_SUCCESS_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v1.json"
)
ACTION_SUCCESS_V8_1_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_action_success_v8_1.txt"
)
ACTION_SUCCESS_V8_1_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v2.json"
)
ACTION_SUCCESS_V8_2_V6_PICKUP_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_action_success_v8_2_v6_pickup.txt"
)
ACTION_SUCCESS_V8_2_V6_PICKUP_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v3_v6_pickup_probe.json"
)
ACTION_SUCCESS_V8_2_V6_BUTTON_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_action_success_v8_2_v6_button.txt"
)
ACTION_SUCCESS_V8_2_V6_BUTTON_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v3_v6_button.json"
)
ACTION_SUCCESS_V8_3_PICKUP_LATCH_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_action_success_v8_3_pickup_latch.txt"
)
ACTION_SUCCESS_V8_3_PICKUP_LATCH_USER_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_user_text_progress_action_success_v8_3_pickup_latch.txt"
)
ACTION_SUCCESS_V8_3_PICKUP_LATCH_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v4_pickup_latch.json"
)
ACTION_GATES_V8_4_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_action_gates_v8_4.txt"
)
ACTION_GATES_V8_4_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_action_gates_v8_4.txt"
)
ACTION_GATES_V8_4_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_action_gates_output_v8_4.schema.json"
)
ACTION_GATES_V8_4_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v5_identity_pickup_release.json"
)
ACTION_GATES_V8_4_1_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_action_gates_v8_4_1.txt"
)
ACTION_GATES_V8_4_1_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v6_task_scoped_patches.json"
)
ACTION_GATES_V8_4_2_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_action_gates_v8_4_2.txt"
)
ACTION_GATES_V8_4_2_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_action_gates_v8_4_2.txt"
)
ACTION_GATES_V8_4_2_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_action_gates_output_v8_4_2.schema.json"
)
ACTION_GATES_V8_4_2_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v7_strict_button_binding_updates.json"
)
ACTION_GATES_V8_4_3_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_action_gates_v8_4_3.txt"
)
ACTION_GATES_V8_4_3_CRITERIA_PATH = (
    SCRIPT_DIR / "knowledge" / "action_success_criteria_v8_visual_downstroke.json"
)
ACTION_GATES_V8_4_4_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_action_gates_v8_4_4.txt"
)
ACTION_GATES_V8_4_5_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_action_gates_v8_4_5.txt"
)
ACTION_GATES_V8_4_5_CRITERIA_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "action_success_criteria_v9_pickhighlight_button.json"
)
ACTION_GATES_V8_4_6_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_action_gates_v8_4_6.txt"
)
ACTION_GATES_V8_4_6_CRITERIA_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "action_success_criteria_v10_pickhighlight_instances.json"
)
REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_system_text_progress_refine_then_lock_v8_5.txt"
)
REFINE_THEN_LOCK_V8_5_USER_PROMPT = (
    SCRIPT_DIR / "prompts" / "planner_user_text_progress_refine_then_lock_v8_5.txt"
)
REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "planner_text_progress_refine_then_lock_v8_5.schema.json"
)
SELF_EVOLVE_V1_CRITERIA_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "action_success_criteria_v11_self_evolve_movecube.json"
)
SELF_EVOLVE_V1_RUNTIME_MEMORY_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "self_evolve_v1_runtime_experiences.json"
)
SELF_EVOLVE_V2_CRITERIA_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "action_success_criteria_v12_self_evolve_swing.json"
)
SELF_EVOLVE_V2_RUNTIME_MEMORY_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "self_evolve_v2_runtime_experiences.json"
)
SELF_EVOLVE_V3_CRITERIA_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "action_success_criteria_v13_self_evolve_swing_binding.json"
)
SELF_EVOLVE_V3_RUNTIME_MEMORY_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "self_evolve_v3_runtime_experiences.json"
)
SELF_EVOLVE_V4_CRITERIA_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "action_success_criteria_v14_self_evolve_counting_ledger.json"
)
SELF_EVOLVE_V4_RUNTIME_MEMORY_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "self_evolve_v4_runtime_experiences.json"
)
SELF_EVOLVE_V5_RUNTIME_MEMORY_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "self_evolve_v5_runtime_experiences.json"
)
SELF_EVOLVE_V6_RUNTIME_MEMORY_PATH = (
    SCRIPT_DIR
    / "knowledge"
    / "self_evolve_v6_runtime_experiences.json"
)
IDENTITY_ISOLATED_PROMPT_PROFILES = frozenset(
    {
        "self_evolve_v2_train_active",
        "self_evolve_v3_train30_val20",
        "self_evolve_v3_multiproblem_auto_hard20",
        "self_evolve_v4_fullstack_trainval",
        "retrieval_v8_5_self_evolve_v1",
        "retrieval_v8_5_self_evolve_v2",
        "retrieval_v8_5_self_evolve_v3",
        "retrieval_v8_5_self_evolve_v4",
        "retrieval_v8_5_self_evolve_v5",
        "retrieval_v8_5_self_evolve_v6",
    }
)
TARGET_LOCAL_RETRIEVAL_SYSTEM_PROMPT = (
    SCRIPT_DIR
    / "prompts"
    / "planner_system_text_progress_retrieval_v7_target_local.txt"
)
GLOBAL_VOCABULARY_PATH = (
    SCRIPT_DIR / "knowledge" / "robomme_executor_vocabulary_v1.json"
)
PROMPT_PROFILES = {
    "text_progress_v2": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
        "output_schema": DEFAULT_OUTPUT_SCHEMA,
        "require_recent_status": False,
        "harness_version": HARNESS_VERSION,
    },
    "rethinking_v3": {
        "system_prompt": RETHINKING_SYSTEM_PROMPT,
        "user_prompt": RETHINKING_USER_PROMPT,
        "output_schema": RETHINKING_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "harness_version": "codex-text-progress-rethinking-v3",
    },
    "lesson_v4": {
        "system_prompt": LESSON_SYSTEM_PROMPT,
        "user_prompt": LESSON_USER_PROMPT,
        "output_schema": LESSON_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "harness_version": "codex-text-progress-lesson-v4",
    },
    "oracle_skills_v5": {
        "system_prompt": ORACLE_SKILLS_SYSTEM_PROMPT,
        "user_prompt": ORACLE_SKILLS_USER_PROMPT,
        "output_schema": ORACLE_SKILLS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_oracle_skills": True,
        "harness_version": "codex-text-progress-oracle-skills-v5",
    },
    "retrieval_v6": {
        "system_prompt": GLOBAL_RETRIEVAL_SYSTEM_PROMPT,
        "user_prompt": GLOBAL_RETRIEVAL_USER_PROMPT,
        "output_schema": GLOBAL_RETRIEVAL_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "harness_version": "codex-text-progress-retrieval-v6",
    },
    "retrieval_v7_target_local": {
        "system_prompt": TARGET_LOCAL_RETRIEVAL_SYSTEM_PROMPT,
        "user_prompt": GLOBAL_RETRIEVAL_USER_PROMPT,
        "output_schema": GLOBAL_RETRIEVAL_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "harness_version": "codex-text-progress-retrieval-v7-target-local",
    },
    "retrieval_v8_action_success": {
        "system_prompt": ACTION_SUCCESS_SYSTEM_PROMPT,
        "user_prompt": ACTION_SUCCESS_USER_PROMPT,
        "output_schema": ACTION_SUCCESS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "action_success_criteria_path": ACTION_SUCCESS_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v8-action-success",
    },
    "retrieval_v8_1_action_success": {
        "system_prompt": ACTION_SUCCESS_V8_1_SYSTEM_PROMPT,
        "user_prompt": ACTION_SUCCESS_USER_PROMPT,
        "output_schema": ACTION_SUCCESS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "action_success_criteria_path": ACTION_SUCCESS_V8_1_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.1-action-success",
    },
    "retrieval_v8_2_v6_pickup_probe": {
        "system_prompt": ACTION_SUCCESS_V8_2_V6_PICKUP_SYSTEM_PROMPT,
        "user_prompt": ACTION_SUCCESS_USER_PROMPT,
        "output_schema": ACTION_SUCCESS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "action_success_criteria_path": ACTION_SUCCESS_V8_2_V6_PICKUP_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.2-v6-pickup-probe",
    },
    "retrieval_v8_2_v6_button_confirm": {
        "system_prompt": ACTION_SUCCESS_V8_2_V6_BUTTON_SYSTEM_PROMPT,
        "user_prompt": ACTION_SUCCESS_USER_PROMPT,
        "output_schema": ACTION_SUCCESS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "action_success_criteria_path": ACTION_SUCCESS_V8_2_V6_BUTTON_CRITERIA_PATH,
        "completion_confirmation_retries": {"press_button_v6_visual_event": 1},
        "harness_version": "codex-text-progress-retrieval-v8.2-v6-button-confirm",
    },
    "retrieval_v8_3_pickup_latch": {
        "system_prompt": ACTION_SUCCESS_V8_3_PICKUP_LATCH_SYSTEM_PROMPT,
        "user_prompt": ACTION_SUCCESS_V8_3_PICKUP_LATCH_USER_PROMPT,
        "output_schema": ACTION_SUCCESS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_completion_evidence_latch": True,
        "action_success_criteria_path": ACTION_SUCCESS_V8_3_PICKUP_LATCH_CRITERIA_PATH,
        "completion_confirmation_retries": {"press_button_v6_visual_event": 1},
        "harness_version": "codex-text-progress-retrieval-v8.3-pickup-latch",
    },
    "retrieval_v8_4_identity_pickup_release": {
        "system_prompt": ACTION_GATES_V8_4_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_CRITERIA_PATH,
        "completion_confirmation_retries": {"press_button_v6_visual_event": 1},
        "harness_version": "codex-text-progress-retrieval-v8.4-identity-pickup-release",
    },
    "retrieval_v8_4_1_task_scoped_patches": {
        "system_prompt": ACTION_GATES_V8_4_1_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_1_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.4.1-task-scoped-patches",
    },
    "retrieval_v8_4_2_strict_button_bindings": {
        "system_prompt": ACTION_GATES_V8_4_2_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_2_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_2_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "require_binding_updates": True,
        "enable_action_completion_latches": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_2_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.4.2-strict-button-bindings",
    },
    "retrieval_v8_4_3_visual_downstroke": {
        "system_prompt": ACTION_GATES_V8_4_3_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_2_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_2_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "require_binding_updates": True,
        "enable_action_completion_latches": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_3_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.4.3-visual-downstroke",
    },
    "retrieval_v8_4_4_one_shot_reinforcement": {
        "system_prompt": ACTION_GATES_V8_4_4_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_2_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_2_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "require_binding_updates": True,
        "enable_action_completion_latches": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_3_CRITERIA_PATH,
        # criterion_id -> K, where K is the maximum number of *additional
        # executor-command dispatches* for one semantic action instance.  One
        # dispatch sends the same grounded text command to pi0.5 once and pi0.5
        # produces its normal action chunk.  K is not a number of simulator
        # steps and does not configure the length of that action chunk.
        "one_shot_completion_reinforcement": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
        },
        "harness_version": (
            "codex-text-progress-retrieval-v8.4.4-one-shot-reinforcement"
        ),
    },
    "retrieval_v8_4_5_pickhighlight_button": {
        "system_prompt": ACTION_GATES_V8_4_5_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_2_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_2_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "require_binding_updates": True,
        "enable_action_completion_latches": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_5_CRITERIA_PATH,
        "one_shot_completion_reinforcement": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
        },
        "harness_version": (
            "codex-text-progress-retrieval-v8.4.5-pickhighlight-button"
        ),
    },
    "retrieval_v8_4_6_pickhighlight_instances": {
        "system_prompt": ACTION_GATES_V8_4_6_SYSTEM_PROMPT,
        "user_prompt": ACTION_GATES_V8_4_2_USER_PROMPT,
        "output_schema": ACTION_GATES_V8_4_2_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "enable_target_identity_gate": True,
        "require_binding_updates": True,
        "enable_action_completion_latches": True,
        "enable_highlight_target_ledger": True,
        "action_success_criteria_path": ACTION_GATES_V8_4_6_CRITERIA_PATH,
        "one_shot_completion_reinforcement": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "harness_version": (
            "codex-text-progress-retrieval-v8.4.6-pickhighlight-instances"
        ),
    },
    "retrieval_v8_5_refine_then_lock": {
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        # The criterion is an advisory, structured self-check inside the single
        # planner call.  It has no Python-side veto authority.
        "enable_refine_check": True,
        "refine_check_criteria_path": ACTION_GATES_V8_4_6_CRITERIA_PATH,
        # criterion_id -> number of additional submissions of the previous
        # grounded command to pi0.5.  This is not an action-step count and does
        # not alter the normal executor action-chunk length.
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "harness_version": "codex-text-progress-retrieval-v8.5-refine-then-lock",
    },
    "self_evolve_v2_train_active": {
        # Train/val-active evolution starts byte-for-byte from the mature v8.5
        # decision policy. A candidate runtime memory and criterion catalog may
        # be supplied explicitly by the server only after offline verification.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": ACTION_GATES_V8_4_6_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "harness_version": "codex-text-progress-self-evolve-v2-train-active",
    },
    "self_evolve_v3_train30_val20": {
        # Isolated train/val-active campaign. Runtime behavior starts from the
        # mature v8.5 policy; only verifier-approved task lessons may be added.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": ACTION_GATES_V8_4_6_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "harness_version": "codex-text-progress-self-evolve-v3-train30-val20",
    },
    "self_evolve_v3_multiproblem_auto_hard20": {
        # Self-evolve v3 changes only the offline evolution controller. Runtime
        # planner behavior remains the mature v8.5 policy plus an explicitly
        # supplied task lesson bundle.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": ACTION_GATES_V8_4_6_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "harness_version": "codex-text-progress-self-evolve-v3-multiproblem-auto-hard20",
    },
    "self_evolve_v4_fullstack_trainval": {
        # V4 preserves the v3 train/val workflow but restores typed full-stack
        # intervention bundles.  With no bundle this is behaviorally the mature
        # v8.5 policy; v8.5 itself remains unchanged.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": ACTION_GATES_V8_4_6_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "harness_version": "codex-text-progress-self-evolve-v4-fullstack-trainval",
    },
    "retrieval_v8_5_self_evolve_v1": {
        # Experimental branch-only profile. Keep the mature v8.5 profile above
        # byte-for-byte behaviorally unchanged and test one learned experience.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": SELF_EVOLVE_V1_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "runtime_experience_memory_path": SELF_EVOLVE_V1_RUNTIME_MEMORY_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.5-self-evolve-v1",
    },
    "retrieval_v8_5_self_evolve_v2": {
        # Revision 2 keeps v8.5 and v1 immutable. It adds one task-scoped,
        # bounded SwingXtimes transition experience to the v1 MoveCube pilot.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": SELF_EVOLVE_V2_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "runtime_experience_memory_path": SELF_EVOLVE_V2_RUNTIME_MEMORY_PATH,
        "harness_version": "codex-text-progress-retrieval-v8.5-self-evolve-v2",
    },
    "retrieval_v8_5_self_evolve_v3": {
        # Revision 3 keeps the effective MoveCube v1 experience and replaces
        # the unreliable prose-only Swing ledger with a task-level visual
        # side-binding experience.  Proposed Swing target points are checked
        # against the fixed front-camera geometry before any action executes.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": SELF_EVOLVE_V3_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
        },
        "runtime_experience_memory_path": SELF_EVOLVE_V3_RUNTIME_MEMORY_PATH,
        "enable_swing_side_grounding_guard": True,
        "harness_version": "codex-text-progress-retrieval-v8.5-self-evolve-v3",
    },
    "retrieval_v8_5_self_evolve_v4": {
        # Revision 4 preserves v3 and adds a runtime-visible ordered-event
        # ledger for the two counting tasks whose baseline commonly advances
        # to the stop button before the physical repetitions are complete.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": SELF_EVOLVE_V4_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
            # One additional normal pickup command gives PickXtimes enough
            # lift margin before the release half of a counted repetition.
            "counting_requested_cube_pickup": 1,
        },
        "runtime_experience_memory_path": SELF_EVOLVE_V4_RUNTIME_MEMORY_PATH,
        "enable_swing_side_grounding_guard": True,
        "enable_counting_event_ledger_guard": True,
        "harness_version": "codex-text-progress-retrieval-v8.5-self-evolve-v4",
    },
    "retrieval_v8_5_self_evolve_v5": {
        # Revision 5 keeps v4 immutable. A failed PickXtimes pilot showed that
        # the initial composite place skill can terminate before the ledger has
        # any pickup evidence. Inject the requested color's released first-pick
        # template and require that explicit event before the first placement.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": SELF_EVOLVE_V4_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
            "counting_requested_cube_pickup": 1,
        },
        "runtime_experience_memory_path": SELF_EVOLVE_V5_RUNTIME_MEMORY_PATH,
        "enable_swing_side_grounding_guard": True,
        "enable_counting_event_ledger_guard": True,
        "enable_counting_explicit_first_pickup": True,
        "harness_version": "codex-text-progress-retrieval-v8.5-self-evolve-v5",
    },
    "retrieval_v8_5_self_evolve_v6": {
        # Revision 6 preserves v5 and binds every pickup command's ordinal to
        # the runtime event ledger. The exact task-scoped released template for
        # completed[color] + 1 is injected before proposal validation.
        "system_prompt": REFINE_THEN_LOCK_V8_5_SYSTEM_PROMPT,
        "user_prompt": REFINE_THEN_LOCK_V8_5_USER_PROMPT,
        "output_schema": REFINE_THEN_LOCK_V8_5_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_refine_check": True,
        "refine_check_criteria_path": SELF_EVOLVE_V4_CRITERIA_PATH,
        "reinforcement_lock": {
            "button_unmask_press_local_event": 1,
            "unmask_pickup_identity_and_separation": 1,
            "unmask_put_down_stable_release": 1,
            "pickhighlight_put_down_stable_release": 1,
            "counting_requested_cube_pickup": 1,
        },
        "runtime_experience_memory_path": SELF_EVOLVE_V6_RUNTIME_MEMORY_PATH,
        "enable_swing_side_grounding_guard": True,
        "enable_counting_event_ledger_guard": True,
        "enable_counting_explicit_first_pickup": True,
        "enable_counting_exact_pickup_ordinal": True,
        "harness_version": "codex-text-progress-retrieval-v8.5-self-evolve-v6",
    },
    # Compatibility alias for artifacts produced before the v8.x naming cleanup.
    # New experiments should use retrieval_v8_1_action_success.
    "retrieval_v9_action_success": {
        "system_prompt": ACTION_SUCCESS_V8_1_SYSTEM_PROMPT,
        "user_prompt": ACTION_SUCCESS_USER_PROMPT,
        "output_schema": ACTION_SUCCESS_OUTPUT_SCHEMA,
        "require_recent_status": True,
        "enable_lessons": True,
        "enable_global_retrieval": True,
        "catalog_retrieval_only": True,
        "enable_action_success_criteria": True,
        "action_success_criteria_path": ACTION_SUCCESS_V8_1_CRITERIA_PATH,
        "harness_version": "codex-text-progress-retrieval-v9-action-success-legacy-alias",
    },
}
V85_EMPTY_INTERVENTION_BEHAVIOR_ID = (
    "v8.5-refine-then-lock-empty-intervention-v1"
)
V85_EMPTY_INTERVENTION_PROFILES = frozenset({
    "self_evolve_v2_train_active",
    "self_evolve_v3_train30_val20",
    "self_evolve_v3_multiproblem_auto_hard20",
    "self_evolve_v4_fullstack_trainval",
})
RECENT_STATUSES = {
    "initial",
    "progressing",
    "completed",
    "uncertain",
    "stagnant",
    "regressed",
}
NO_PROGRESS_STATUSES = {"uncertain", "stagnant"}
RETHINKING_THRESHOLD = 3
EXPLORATION_PATIENCE = 2
CANDIDATE_STAGNANT_LIMIT = 2
MAX_EXPLORATION_CANDIDATES = 3
INITIAL_PROGRESS = "No rollout progress has been established yet."
INITIAL_PREVIOUS_SUBTASK = "None. These are the initial rollout observations."

LESSON_STATUSES = {
    "successful",
    "promising",
    "ineffective",
    "regressive",
    "unsafe",
    "inconclusive",
}
INSTRUCTION_STRATEGIES = {
    "normal",
    "paraphrase",
    "simplify",
    "explicit_state",
    "atomic_breakdown",
    "reuse_lesson",
    "reground",
    "retry_canonical",
    "prerequisite",
    "advance_skill",
    "catalog_backoff",
}
HELD_OBJECT_VALUES = {"none", "cube", "container", "peg", "tool", "other", "unknown"}
OBJECT_SUPPORT_VALUES = {
    "airborne",
    "table",
    "target",
    "held_unknown",
    "not_applicable",
    "unknown",
}
GRIPPER_STATE_VALUES = {"open", "closed", "unknown"}
COMPLETION_CHECK_VERDICTS = {"yes", "no", "uncertain"}
COMPLETION_CHECK_DECISIONS = {
    "satisfied",
    "not_satisfied",
    "uncertain",
    "not_applicable",
}
TARGET_COLORS = {"red", "green", "blue", "none", "unknown"}
TARGET_BINDING_STATUSES = {
    "not_applicable",
    "unresolved",
    "candidate",
    "locked",
    "verified",
    "mismatch",
}
TARGET_BINDING_BASES = {
    "not_applicable",
    "observed_cover_event",
    "init_demonstration",
    "revealed_color",
    "inference_only",
}
TARGET_IDENTITY_RELATIONS = {"yes", "no", "uncertain"}
SAFE_TARGET_BINDING_BASES = {
    "observed_cover_event",
    "init_demonstration",
    "revealed_color",
}
BINDING_UPDATE_STATUSES = {"locked", "verified"}
BINDING_UPDATE_BASES = SAFE_TARGET_BINDING_BASES
TARGETED_CONTAINER_PICKUP_RE = re.compile(
    r"^pick up the container(?: at <[0-9]+, [0-9]+>)? that hides the "
    r"(?P<color>red|green|blue) cube$",
    flags=re.IGNORECASE,
)
HIGHLIGHTED_CUBE_PICKUP_RE = re.compile(
    r"^pick up the (?:(?P<ordinal>first|second|third) )?highlighted cube"
    r"(?: at <[0-9]+, [0-9]+>)?, which is (?P<color>red|green|blue)$",
    flags=re.IGNORECASE,
)
PLACE_CUBE_ON_TABLE_RE = re.compile(
    r"^place the cube onto the table$", flags=re.IGNORECASE
)


def requested_pickup_color(subtask: str | None) -> str | None:
    if not isinstance(subtask, str):
        return None
    match = TARGETED_CONTAINER_PICKUP_RE.fullmatch(
        " ".join(subtask.strip().lower().split())
    )
    return match.group("color") if match is not None else None


def parse_highlighted_cube_pickup(
    subtask: str | None,
) -> tuple[str, str] | None:
    """Return the oracle action slot and color for a PickHighlight pickup."""
    if not isinstance(subtask, str):
        return None
    match = HIGHLIGHTED_CUBE_PICKUP_RE.fullmatch(
        " ".join(subtask.strip().lower().split())
    )
    if match is None:
        return None
    return (match.group("ordinal") or "single", match.group("color"))


def pickhighlight_pickup_signature(subtask: str | None) -> str | None:
    """Return an episode-local, coordinate-free highlighted target descriptor."""
    parsed = parse_highlighted_cube_pickup(subtask)
    if parsed is None:
        return None
    ordinal, color = parsed
    return f"{ordinal}::{color}"


def highlight_target_instance_id(
    skill_key: str | None,
    subtask_template: str | None,
    subtask: str | None,
) -> str | None:
    """Build a stable episode-local target ID from the frozen oracle action slot.

    The skill key separates first/second/third targets even when their colors are
    identical.  Coordinates are deliberately excluded because objects move.
    """
    parsed = parse_highlighted_cube_pickup(subtask_template)
    if parsed is None:
        parsed = parse_highlighted_cube_pickup(subtask)
    if parsed is None:
        return None
    ordinal, color = parsed
    stable_slot = skill_key or f"{ordinal}_{color}"
    return f"pickhighlight::{stable_slot}::{ordinal}::{color}"


def is_place_cube_on_table_subtask(subtask: str | None) -> bool:
    if not isinstance(subtask, str):
        return False
    return PLACE_CUBE_ON_TABLE_RE.fullmatch(
        " ".join(subtask.strip().lower().split())
    ) is not None


def action_instance_signature(
    criterion_id: str,
    skill_key: str | None,
    subtask_template: str | None,
    subtask: str,
) -> str:
    """Identify one semantic action attempt without depending on moving pixels."""
    return "::".join(
        (
            criterion_id,
            str(skill_key or subtask_template or subtask),
            str(requested_pickup_color(subtask) or "no_named_color"),
        )
    )


def is_button_press_subtask(subtask: str | None) -> bool:
    if not isinstance(subtask, str):
        return False
    return re.fullmatch(
        r"press (?:the )?(?:(?:first|second) )?button(?: at <[0-9]+, [0-9]+>)?",
        " ".join(subtask.strip().lower().split()),
    ) is not None


def target_binding_is_safe_for_pickup(
    target_identity: dict[str, Any], requested_color: str
) -> tuple[bool, str]:
    if target_identity.get("requested_color") != requested_color:
        return False, "requested_color_mismatch"
    if target_identity.get("binding_status") not in {"locked", "verified"}:
        return False, "binding_not_locked"
    if target_identity.get("binding_basis") not in SAFE_TARGET_BINDING_BASES:
        return False, "binding_basis_not_direct"
    if (
        target_identity.get("binding_basis")
        in {"observed_cover_event", "revealed_color"}
        and not target_identity.get("evidence_frame_ids")
    ):
        return False, "missing_visual_evidence"
    revealed_color = target_identity.get("revealed_color")
    if revealed_color not in {"unknown", requested_color}:
        return False, "revealed_color_mismatch"
    return True, "direct_binding_verified"


def sanitize_binding_updates(
    binding_updates: Any, *, allowed_front_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only directly evidenced, well-formed episode-local target bindings.

    Unlike ``target_identity``, these updates describe any color observed while a
    different semantic action (for example pressing a button) is active. Invalid
    entries are dropped instead of being allowed to authorize a hazardous pickup.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not isinstance(binding_updates, list):
        return [], {"changed": True, "accepted": 0, "rejected": [{"reason": "not_a_list"}]}
    seen_colors: set[str] = set()
    expected_fields = {
        "color",
        "binding_status",
        "binding_basis",
        "target_description",
        "evidence_frame_ids",
        "grounding_point",
        "revealed_color",
        "reason",
    }
    point_fields = {
        "object_description",
        "source_frame_id",
        "x_norm",
        "y_norm",
    }
    for index, item in enumerate(binding_updates[:3]):
        reason = None
        if not isinstance(item, dict) or set(item) != expected_fields:
            reason = "invalid_fields"
        else:
            color = item.get("color")
            basis = item.get("binding_basis")
            status = item.get("binding_status")
            frame_ids = item.get("evidence_frame_ids")
            point = item.get("grounding_point")
            if color not in {"red", "green", "blue"} or color in seen_colors:
                reason = "invalid_or_duplicate_color"
            elif status not in BINDING_UPDATE_STATUSES:
                reason = "binding_not_locked"
            elif basis not in BINDING_UPDATE_BASES:
                reason = "binding_basis_not_direct"
            elif (
                not isinstance(frame_ids, list)
                or not frame_ids
                or len(frame_ids) > 4
                or any(value not in allowed_front_ids for value in frame_ids)
            ):
                reason = "invalid_visual_evidence"
            elif not isinstance(point, dict) or set(point) != point_fields:
                reason = "invalid_grounding_point"
            elif point.get("source_frame_id") not in allowed_front_ids:
                reason = "invalid_grounding_source"
            elif not all(
                isinstance(point.get(axis), int) and 0 <= point[axis] <= 999
                for axis in ("x_norm", "y_norm")
            ):
                reason = "invalid_grounding_coordinates"
            elif not isinstance(item.get("target_description"), str) or not item[
                "target_description"
            ].strip():
                reason = "empty_target_description"
            elif not isinstance(item.get("reason"), str) or not item["reason"].strip():
                reason = "empty_reason"
            elif item.get("revealed_color") not in {"unknown", color}:
                reason = "revealed_color_mismatch"
        if reason is not None:
            rejected.append({"index": index, "reason": reason})
            continue
        normalized = dict(item)
        normalized["evidence_frame_ids"] = list(dict.fromkeys(item["evidence_frame_ids"]))
        normalized["grounding_point"] = dict(item["grounding_point"])
        accepted.append(normalized)
        seen_colors.add(str(item["color"]))
    if len(binding_updates) > 3:
        rejected.append({"reason": "more_than_three_updates"})
    return accepted, {
        "changed": accepted != binding_updates,
        "accepted": len(accepted),
        "rejected": rejected,
    }


def merge_target_binding_updates(
    bindings: dict[str, dict[str, Any]], updates: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Merge immutable direct observations without degrading an existing binding."""
    merged = {key: dict(value) for key, value in bindings.items()}
    decisions: list[dict[str, Any]] = []
    rank = {"locked": 1, "verified": 2}
    for update in updates:
        color = str(update["color"])
        existing = merged.get(color)
        if existing is not None and rank.get(str(existing.get("binding_status")), 0) >= rank[
            str(update["binding_status"])
        ]:
            decisions.append({"color": color, "decision": "preserved_existing"})
            continue
        binding = {
            "requested_color": color,
            "binding_status": update["binding_status"],
            "binding_basis": update["binding_basis"],
            "target_description": update["target_description"],
            "evidence_frame_ids": list(update["evidence_frame_ids"]),
            "revealed_color": update["revealed_color"],
            "same_as_bound_target": "yes",
            "reason": update["reason"],
            "grounding_point": dict(update["grounding_point"]),
        }
        merged[color] = binding
        decisions.append({"color": color, "decision": "accepted"})
    return merged, decisions


def resolve_pickup_binding(
    current_identity: dict[str, Any],
    requested_color: str,
    stored_binding: dict[str, Any] | None,
) -> tuple[bool, str, str]:
    """Prefer current direct evidence, then an immutable stored direct binding."""
    current_safe, current_reason = target_binding_is_safe_for_pickup(
        current_identity, requested_color
    )
    if current_safe:
        return True, current_reason, "current_response"
    if current_identity.get("same_as_bound_target") != "yes":
        return False, current_reason, "none"
    if stored_binding is not None:
        stored_safe, stored_reason = target_binding_is_safe_for_pickup(
            stored_binding, requested_color
        )
        if stored_safe:
            return True, stored_reason, "stored_binding"
    return False, current_reason, "none"


def completion_check_verdict(
    completion_check: dict[str, Any], check_id: str
) -> str | None:
    for check in completion_check.get("checks", []):
        if check.get("check_id") == check_id:
            return str(check.get("verdict"))
    return None


def validate_action_success_catalog(
    catalog: dict[str, Any], *, source: str
) -> dict[str, Any]:
    """Validate a materialized criterion catalog without requiring a file."""
    criteria = catalog.get("criteria")
    if not isinstance(catalog.get("version"), str) or not isinstance(criteria, list):
        raise ValueError(f"Invalid action success criteria catalog: {source}")
    seen: set[str] = set()
    for criterion in criteria:
        criterion_id = criterion.get("criterion_id")
        patterns = criterion.get("previous_subtask_patterns")
        checks = criterion.get("required_checks")
        applies_to_tasks = criterion.get("applies_to_tasks")
        if (
            not isinstance(criterion_id, str)
            or criterion_id in seen
            or not isinstance(patterns, list)
            or not patterns
            or not isinstance(checks, list)
            or not checks
        ):
            raise ValueError(f"Invalid action success criterion: {criterion!r}")
        if applies_to_tasks is not None and (
            not isinstance(applies_to_tasks, list)
            or not applies_to_tasks
            or any(not isinstance(task, str) or not task for task in applies_to_tasks)
            or len(set(applies_to_tasks)) != len(applies_to_tasks)
        ):
            raise ValueError(
                f"Invalid applies_to_tasks for action success criterion {criterion_id}"
            )
        seen.add(criterion_id)
        for pattern in patterns:
            re.compile(pattern, flags=re.IGNORECASE)
        check_ids = [check.get("check_id") for check in checks]
        if any(not isinstance(value, str) for value in check_ids) or len(
            set(check_ids)
        ) != len(check_ids):
            raise ValueError(f"Invalid required checks for {criterion_id}")
    return catalog


def load_action_success_criteria(
    path: Path = ACTION_SUCCESS_CRITERIA_PATH,
) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in catalog:
        base_name = catalog.get("extends")
        replacements = catalog.get("replace_criteria", [])
        additions = catalog.get("add_criteria", [])
        if (
            not isinstance(base_name, str)
            or Path(base_name).name != base_name
            or not isinstance(replacements, list)
            or not isinstance(additions, list)
        ):
            raise ValueError(f"Invalid criteria overlay: {path}")
        base = load_action_success_criteria(path.parent / base_name)
        replacement_by_id = {
            str(item.get("criterion_id")): item
            for item in replacements
            if isinstance(item, dict)
        }
        if len(replacement_by_id) != len(replacements):
            raise ValueError(f"Duplicate or invalid replacement criteria: {path}")
        base_ids = {item["criterion_id"] for item in base["criteria"]}
        unknown = set(replacement_by_id) - base_ids
        if unknown:
            raise ValueError(f"Overlay replaces unknown criteria {sorted(unknown)}: {path}")
        addition_ids = [
            item.get("criterion_id") for item in additions if isinstance(item, dict)
        ]
        if (
            len(addition_ids) != len(additions)
            or len(set(addition_ids)) != len(addition_ids)
            or set(addition_ids) & base_ids
        ):
            raise ValueError(f"Invalid or duplicate added criteria: {path}")
        catalog = {
            "version": catalog.get("version"),
            "criteria": [
                replacement_by_id.get(item["criterion_id"], item)
                for item in base["criteria"]
            ] + additions,
            "extends": base_name,
        }
    return validate_action_success_catalog(catalog, source=str(path))


def apply_action_success_criteria_overlays(
    base: dict[str, Any], overlay_paths: list[Path]
) -> dict[str, Any]:
    """Apply multiple v1-style add/replace overlays to one v8.5 catalog.

    Overlay ``extends`` filenames are provenance only here: every overlay is
    applied to the current working catalog, which lets independently accepted
    v4 candidates accumulate without rewriting a synthetic source file.
    """
    catalog = json.loads(json.dumps(base))
    for path in overlay_paths:
        overlay = json.loads(path.read_text(encoding="utf-8"))
        replacements = overlay.get("replace_criteria", [])
        additions = overlay.get("add_criteria", [])
        if (
            "extends" not in overlay
            or not isinstance(replacements, list)
            or not isinstance(additions, list)
            or not isinstance(overlay.get("version"), str)
        ):
            raise ValueError(f"V4 criterion component must be an overlay: {path}")
        replacement_by_id = {
            str(item.get("criterion_id")): item
            for item in replacements
            if isinstance(item, dict)
        }
        if len(replacement_by_id) != len(replacements):
            raise ValueError(f"Duplicate or invalid replacement criteria: {path}")
        base_ids = {item["criterion_id"] for item in catalog["criteria"]}
        unknown = set(replacement_by_id) - base_ids
        if unknown:
            raise ValueError(f"Overlay replaces unknown criteria {sorted(unknown)}: {path}")
        addition_ids = [
            item.get("criterion_id") for item in additions if isinstance(item, dict)
        ]
        if (
            len(addition_ids) != len(additions)
            or len(set(addition_ids)) != len(addition_ids)
            or set(addition_ids) & base_ids
        ):
            raise ValueError(f"Invalid or duplicate added criteria: {path}")
        catalog = {
            "version": overlay["version"],
            "criteria": [
                replacement_by_id.get(item["criterion_id"], item)
                for item in catalog["criteria"]
            ]
            + additions,
            "v4_overlay_chain": [
                *catalog.get("v4_overlay_chain", []),
                {"path": path.name, "sha256": sha256_file(path)},
            ],
        }
        validate_action_success_catalog(catalog, source=str(path))
    return catalog


FORBIDDEN_RUNTIME_EXPERIENCE_KEYS = frozenset(
    {
        "episode",
        "episode_id",
        "init_case_id",
        "seed",
        "source_episode_refs",
        "supporting_episode_refs",
        "evaluator_state",
        "future_frames",
        "absolute_coordinates",
    }
)
EPISODE_IDENTITY_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[^a-z0-9])(?:episode(?:_id)?|ep)[\s:=/_-]*0*[0-9]+(?:[^0-9]|$)"
    ),
    re.compile(r"(?i)\binit_case_id\b"),
    re.compile(r"(?i)\b(?:environment_)?seed\s*[:=]\s*[0-9]+\b"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_runtime_experience_has_no_episode_identity(
    value: Any, *, source: str
) -> None:
    """Reject per-episode or privileged fields from decision-time memory."""
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in FORBIDDEN_RUNTIME_EXPERIENCE_KEYS:
                raise ValueError(
                    f"Runtime experience memory {source} contains forbidden key "
                    f"{key!r}"
                )
            assert_runtime_experience_has_no_episode_identity(
                nested, source=f"{source}.{key}"
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_runtime_experience_has_no_episode_identity(
                nested, source=f"{source}[{index}]"
            )


def assert_prompt_has_no_episode_identity(text: str, *, source: str) -> None:
    for pattern in EPISODE_IDENTITY_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            raise ValueError(
                f"Identity-isolated planner input {source} contains episode-specific "
                f"text {match.group(0)!r}"
            )


def load_runtime_experience_memory(
    path: Path, *, refine_catalog: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    memory = json.loads(path.read_text(encoding="utf-8"))
    assert_runtime_experience_has_no_episode_identity(memory, source=path.name)
    if (
        set(memory)
        != {
            "schema_version",
            "memory_id",
            "revision",
            "base_profile",
            "experiences",
        }
        or memory.get("schema_version") != "self-evolve-runtime-memory-v1"
        or not isinstance(memory.get("memory_id"), str)
        or not memory["memory_id"]
        or not isinstance(memory.get("revision"), int)
        or memory["revision"] < 1
        or memory.get("base_profile") != "retrieval_v8_5_refine_then_lock"
        or not isinstance(memory.get("experiences"), list)
        or not memory["experiences"]
    ):
        raise ValueError(f"Invalid self-evolve runtime memory: {path}")
    criteria_by_id = {
        str(item["criterion_id"]): item for item in refine_catalog["criteria"]
    }
    gates: dict[str, dict[str, Any]] = {}
    experience_ids: set[str] = set()
    for experience in memory["experiences"]:
        if not isinstance(experience, dict):
            raise ValueError(f"Invalid runtime experience entry: {experience!r}")
        experience_id = experience.get("experience_id")
        criterion_id = experience.get("criterion_id")
        task_scope = experience.get("task_scope")
        intervention = experience.get("intervention")
        if (
            set(experience)
            != {
                "experience_id",
                "revision",
                "status",
                "task_scope",
                "situation_class",
                "criterion_id",
                "intervention",
            }
            or not isinstance(experience_id, str)
            or not experience_id
            or experience_id in experience_ids
            or not isinstance(experience.get("revision"), int)
            or experience["revision"] < 1
            or experience.get("status") not in {"candidate", "approved"}
            or not isinstance(experience.get("situation_class"), str)
            or not experience["situation_class"]
            or not isinstance(criterion_id, str)
            or criterion_id not in criteria_by_id
            or not isinstance(task_scope, list)
            or not task_scope
            or any(not isinstance(task, str) or not task for task in task_scope)
            or len(set(task_scope)) != len(task_scope)
            or not isinstance(intervention, dict)
            or set(intervention) != {"type", "max_blocked_transitions"}
            or intervention.get("type") != "bounded_transition_gate"
            or not isinstance(intervention.get("max_blocked_transitions"), int)
            or intervention["max_blocked_transitions"] < 1
        ):
            raise ValueError(f"Invalid runtime experience entry: {experience!r}")
        criterion_tasks = criteria_by_id[criterion_id].get("applies_to_tasks")
        if criterion_tasks is None or not set(task_scope).issubset(criterion_tasks):
            raise ValueError(
                f"Runtime experience {experience_id} task scope does not match "
                f"criterion {criterion_id}"
            )
        if criterion_id in gates:
            raise ValueError(
                f"Multiple runtime experiences target criterion {criterion_id}"
            )
        experience_ids.add(experience_id)
        gates[criterion_id] = {
            "experience_id": experience_id,
            "revision": experience["revision"],
            "task_scope": list(task_scope),
            "situation_class": experience["situation_class"],
            "max_blocked_transitions": intervention["max_blocked_transitions"],
        }
    return memory, gates


def load_runtime_lesson_memory(path: Path) -> dict[str, Any]:
    """Load task-scoped prompt experience with strict identity isolation."""
    memory = json.loads(path.read_text(encoding="utf-8"))
    assert_runtime_experience_has_no_episode_identity(memory, source=path.name)
    if (
        set(memory)
        != {"schema_version", "memory_id", "revision", "base_profile", "experiences"}
        or memory.get("schema_version") != "self-evolve-runtime-lessons-v2"
        or not isinstance(memory.get("memory_id"), str)
        or not memory["memory_id"]
        or not isinstance(memory.get("revision"), int)
        or memory["revision"] < 1
        or memory.get("base_profile") != "retrieval_v8_5_refine_then_lock"
        or not isinstance(memory.get("experiences"), list)
    ):
        raise ValueError(f"Invalid v2 runtime lesson memory: {path}")
    seen: set[str] = set()
    required = {
        "experience_id",
        "revision",
        "status",
        "task_scope",
        "condition",
        "lesson",
        "recommended_skill_or_policy",
        "success_evidence",
        "forbidden_behavior",
    }
    for experience in memory["experiences"]:
        experience_id = experience.get("experience_id") if isinstance(experience, dict) else None
        if (
            not isinstance(experience, dict)
            or set(experience) != required
            or not isinstance(experience_id, str)
            or not experience_id
            or experience_id in seen
            or not isinstance(experience.get("revision"), int)
            or experience["revision"] < 1
            or experience.get("status") not in {"candidate", "approved"}
            or not isinstance(experience.get("task_scope"), list)
            or not experience["task_scope"]
            or any(not isinstance(task, str) or not task for task in experience["task_scope"])
            or not isinstance(experience.get("condition"), str)
            or not experience["condition"].strip()
            or not isinstance(experience.get("lesson"), str)
            or not experience["lesson"].strip()
            or not isinstance(experience.get("recommended_skill_or_policy"), str)
            or not isinstance(experience.get("success_evidence"), list)
            or not isinstance(experience.get("forbidden_behavior"), list)
        ):
            raise ValueError(f"Invalid v2 runtime lesson entry: {experience!r}")
        for field in ("condition", "lesson", "recommended_skill_or_policy"):
            assert_prompt_has_no_episode_identity(
                experience[field], source=f"{path.name}.{experience_id}.{field}"
            )
        seen.add(experience_id)
    return memory


def match_action_success_criterion(
    previous_subtask: str,
    catalog: dict[str, Any],
    task: str | None = None,
) -> dict[str, Any] | None:
    normalized = " ".join(previous_subtask.strip().lower().split())
    for criterion in catalog["criteria"]:
        applies_to_tasks = criterion.get("applies_to_tasks")
        if task is not None and applies_to_tasks is not None and task not in applies_to_tasks:
            continue
        if any(
            re.fullmatch(pattern, normalized, flags=re.IGNORECASE)
            for pattern in criterion["previous_subtask_patterns"]
        ):
            return criterion
    return None


SWING_TARGET_SIDE_PATTERN = re.compile(
    r"move to the top of the (right|left)-side target\b",
    flags=re.IGNORECASE,
)


def validate_swing_side_grounding(
    response: dict[str, Any],
    rendered_points: list[dict[str, Any]],
    *,
    experience_id: str,
    experience_revision: int,
) -> dict[str, Any]:
    """Reject a visually swapped Swing target before it reaches the executor.

    RoboMME's fixed front camera maps the physical right-side target to the
    image-left region and the physical left-side target to image-right.  This
    task-level invariant is visible in every current front observation and does
    not depend on an episode identity, seed, evaluator state, or absolute saved
    coordinate.  The model still selects the concrete target point from the
    current images; the harness only checks its side consistency.
    """

    template = str(response.get("current_subtask_template", ""))
    match = SWING_TARGET_SIDE_PATTERN.search(template)
    audit = {
        "enabled": True,
        "applicable": match is not None,
        "experience_id": experience_id,
        "experience_revision": experience_revision,
        "task_level_camera_rule": (
            "right target is image-left; left target is image-right"
        ),
        "proposed_side": match.group(1).lower() if match is not None else None,
        "proposed_x_norm": None,
        "decision": "not_applicable" if match is None else "pending",
    }
    if match is None:
        return audit
    if len(rendered_points) != 1:
        raise ValueError(
            "Swing side-binding experience requires exactly one grounded target "
            f"point, got {len(rendered_points)}"
        )
    point = rendered_points[0]
    x_norm = point.get("x_norm")
    if isinstance(x_norm, bool) or not isinstance(x_norm, int):
        raise ValueError("Swing target point must have an integer x_norm")
    side = match.group(1).lower()
    side_consistent = (side == "right" and x_norm < 500) or (
        side == "left" and x_norm > 500
    )
    if not side_consistent:
        expected_region = "image-left (x_norm < 500)" if side == "right" else (
            "image-right (x_norm > 500)"
        )
        raise ValueError(
            "Swing task-level side-binding experience rejected a target swap: "
            f"the {side}-side target must be grounded in {expected_region}, "
            f"but the proposed point has x_norm={x_norm}. Re-inspect the current "
            "front images and select the gray/red circular target on the correct "
            "image side; do not reuse the opposite target's point."
        )
    audit["proposed_x_norm"] = x_norm
    audit["decision"] = "accepted"
    return audit


COUNTING_COLORS = ("red", "green", "blue")
COUNTING_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
COUNTING_ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth")
COUNTING_NUMBER_PATTERN = r"(?:0|1|2|3|4|5|zero|one|two|three|four|five)"
COUNTING_PICKUP_PATTERN = re.compile(
    r"^pick up the (?:(?:first|second|third|fourth|fifth) )?"
    r"(red|green|blue) cube(?: at <[0-9]+, [0-9]+>)?"
    r"(?: for the (?:first|second|third|fourth|fifth) time)?$",
    flags=re.IGNORECASE,
)
PICK_X_TIMES_PLACE_PATTERN = re.compile(
    r"^place the (red|green|blue) cube onto the target"
    r"(?: at <[0-9]+, [0-9]+>)?$",
    flags=re.IGNORECASE,
)
BIN_FILL_PLACE_PATTERN = re.compile(
    r"^put it into the bin(?: at <[0-9]+, [0-9]+>)?$",
    flags=re.IGNORECASE,
)
COUNTING_BUTTON_PATTERN = re.compile(
    r"^press (?:the )?button(?: at <[0-9]+, [0-9]+>)?(?: to stop)?$",
    flags=re.IGNORECASE,
)
PICK_X_TIMES_PICKUP_ORDINAL_PATTERN = re.compile(
    r"^pick up the (red|green|blue) cube"
    r"(?: at <[0-9]+, [0-9]+>)? for the "
    r"(first|second|third|fourth|fifth) time$",
    flags=re.IGNORECASE,
)
BIN_FILL_PICKUP_ORDINAL_PATTERN = re.compile(
    r"^pick up the (first|second|third|fourth|fifth) "
    r"(red|green|blue) cube(?: at <[0-9]+, [0-9]+>)?$",
    flags=re.IGNORECASE,
)


def _counting_number(value: str) -> int:
    normalized = value.lower()
    return int(normalized) if normalized.isdigit() else COUNTING_NUMBER_WORDS[normalized]


def counting_requirements_from_guidance(
    task: str, planner_guidance: dict[str, Any]
) -> dict[str, int]:
    """Parse a task-level counting contract from planner-visible guidance."""

    facts = planner_guidance.get("fixed_facts", [])
    if not isinstance(facts, list) or any(not isinstance(item, str) for item in facts):
        raise ValueError("Counting guard requires string planner_guidance.fixed_facts")
    sentences = [item.lower().replace("–", "-") for item in facts]
    text = " ".join(sentences)
    if task == "PickXtimes":
        color = next(
            (value for value in COUNTING_COLORS if re.search(rf"\b{value}\b", text)),
            None,
        )
        count = None
        patterns = (
            rf"exactly\s+({COUNTING_NUMBER_PATTERN})\s+"
            r"(?:valid\s+)?(?:pick-and-place\s+)?repetitions?",
            rf"(?:complete|place[^.]*?)\s+exactly\s+({COUNTING_NUMBER_PATTERN})",
            rf"required repetitions:\s*exactly\s+({COUNTING_NUMBER_PATTERN})",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match is not None:
                count = _counting_number(match.group(1))
                break
        if color is None or count is None or count < 1:
            raise ValueError(
                "Unable to derive PickXtimes color/count from planner-visible guidance"
            )
        return {color: count}

    if task != "BinFill":
        return {}
    requested: dict[str, int] = {}
    for sentence in sentences:
        for match in re.finditer(
            rf"(?P<count>{COUNTING_NUMBER_PATTERN})\s+"
            r"(?P<color>red|green|blue)(?:-cube)?(?:\s+cubes?)?",
            sentence,
        ):
            requested[match.group("color")] = _counting_number(match.group("count"))
        for match in re.finditer(
            r"(?P<color>red|green|blue)(?:-cube)?(?:\s+cube)?\s+counts?"
            r"(?:\s+in\s+the\s+bin)?\s*(?::|is|are)?\s*"
            rf"(?P<count>{COUNTING_NUMBER_PATTERN})",
            sentence,
        ):
            requested[match.group("color")] = _counting_number(match.group("count"))
        for match in re.finditer(
            r"(?P<color>red|green|blue)\s+"
            rf"(?P<count>{COUNTING_NUMBER_PATTERN})(?:\b|,)",
            sentence,
        ):
            requested[match.group("color")] = _counting_number(match.group("count"))
        mentioned = [
            color for color in COUNTING_COLORS if re.search(rf"\b{color}\b", sentence)
        ]
        zero_context = (
            (
                re.search(r"\b(?:zero|0)\b", sentence) is not None
                and ("count" in sentence or "requested" in sentence)
            )
            or sentence.startswith("no ")
            or any(
                phrase in sentence
                for phrase in (
                    "do not place",
                    "may be placed",
                    "not requested",
                    "cubes only",
                    "cube only",
                )
            )
        )
        if zero_context:
            for color in mentioned:
                requested.setdefault(color, 0)
    if len([count for count in requested.values() if count > 0]) == 1 and (
        "cube only" in text or "cubes only" in text
    ):
        for color in COUNTING_COLORS:
            requested.setdefault(color, 0)
    if set(requested) != set(COUNTING_COLORS) or not any(requested.values()):
        raise ValueError(
            "Unable to derive complete BinFill color quotas from planner-visible guidance"
        )
    return requested


def counting_pickup_color(subtask: str) -> str | None:
    match = COUNTING_PICKUP_PATTERN.fullmatch(" ".join(subtask.strip().split()))
    return match.group(1).lower() if match is not None else None


def counting_pickup_ordinal(task: str, subtask: str) -> int | None:
    normalized = " ".join(subtask.strip().split())
    if task == "PickXtimes":
        match = PICK_X_TIMES_PICKUP_ORDINAL_PATTERN.fullmatch(normalized)
        ordinal = match.group(2).lower() if match is not None else None
    elif task == "BinFill":
        match = BIN_FILL_PICKUP_ORDINAL_PATTERN.fullmatch(normalized)
        ordinal = match.group(1).lower() if match is not None else None
    else:
        return None
    return COUNTING_ORDINAL_WORDS.index(ordinal) + 1 if ordinal is not None else None


def pickxtimes_placement_color(subtask: str) -> str | None:
    match = PICK_X_TIMES_PLACE_PATTERN.fullmatch(" ".join(subtask.strip().split()))
    return match.group(1).lower() if match is not None else None


def counting_proposal_kind(task: str, subtask: str) -> str:
    if counting_pickup_color(subtask) is not None:
        return "pickup"
    if task == "PickXtimes" and PICK_X_TIMES_PLACE_PATTERN.fullmatch(subtask):
        return "placement"
    if task == "BinFill" and BIN_FILL_PLACE_PATTERN.fullmatch(subtask):
        return "placement"
    if COUNTING_BUTTON_PATTERN.fullmatch(" ".join(subtask.strip().split())):
        return "button"
    return "other"


def ensure_pickxtimes_first_pickup_candidate(
    candidates: list[dict[str, Any]],
    *,
    retriever: ExecutorSkillRetriever,
    state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Expose the requested color's canonical first-pick skill when required.

    R0's frequency top-k can omit the requested red/green first-pick template
    while retaining a composite place template.  The injected entry is still a
    released pi0.5 training label from the task-scoped frozen vocabulary; the
    runtime color comes only from the task contract stored in ``state``.
    """

    result = [dict(candidate) for candidate in candidates]
    if state is None or not state.get("explicit_first_pickup_required"):
        return result
    requested = state.get("requested")
    completed = state.get("completed")
    if (
        not isinstance(requested, dict)
        or len(requested) != 1
        or not isinstance(completed, dict)
        or sum(int(value) for value in completed.values()) != 0
        or state.get("pickup_verified")
    ):
        return result
    color = next(iter(requested))
    canonical = f"pick up the {color} cube at <point_1> for the first time"
    matching = [
        entry
        for entry in retriever.entries
        if entry.get("canonical_template") == canonical
        and "PickXtimes" in entry.get("tasks", [])
    ]
    if len(matching) != 1:
        raise ValueError(
            "Counting experience expected exactly one task-scoped canonical "
            f"first-pick skill for {color}, got {len(matching)}"
        )
    template_id = matching[0]["template_id"]
    if any(candidate.get("template_id") == template_id for candidate in result):
        return result
    result.append(
        {
            "rank": len(result) + 1,
            "retrieval_score": 0.0,
            "injected_by_experience": "pickxtimes_explicit_first_pickup",
            **matching[0],
        }
    )
    return result


def ensure_counting_next_pickup_candidates(
    candidates: list[dict[str, Any]],
    *,
    retriever: ExecutorSkillRetriever,
    task: str,
    state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Inject exact next-ordinal pickup labels derived from the live ledger."""

    result = [dict(candidate) for candidate in candidates]
    if state is None or not state.get("exact_pickup_ordinal_required"):
        return result
    requested = state.get("requested")
    completed = state.get("completed")
    if not isinstance(requested, dict) or not isinstance(completed, dict):
        return result
    active_color = state.get("active_color")
    colors = [active_color] if active_color in requested else list(requested)
    for color in colors:
        done = int(completed.get(color, 0))
        quota = int(requested[color])
        if done >= quota:
            continue
        next_index = done + 1
        if not 1 <= next_index <= len(COUNTING_ORDINAL_WORDS):
            raise ValueError(f"Unsupported counting pickup ordinal {next_index}")
        ordinal = COUNTING_ORDINAL_WORDS[next_index - 1]
        if task == "PickXtimes":
            canonical = (
                f"pick up the {color} cube at <point_1> for the {ordinal} time"
            )
        elif task == "BinFill":
            canonical = f"pick up the {ordinal} {color} cube at <point_1>"
        else:
            continue
        matching = [
            entry
            for entry in retriever.entries
            if entry.get("canonical_template") == canonical
            and task in entry.get("tasks", [])
        ]
        if len(matching) != 1:
            raise ValueError(
                "Counting experience expected exactly one task-scoped canonical "
                f"pickup skill {canonical!r}, got {len(matching)}"
            )
        template_id = matching[0]["template_id"]
        if any(candidate.get("template_id") == template_id for candidate in result):
            continue
        result.append(
            {
                "rank": len(result) + 1,
                "retrieval_score": 0.0,
                "injected_by_experience": "counting_exact_next_pickup_ordinal",
                **matching[0],
            }
        )
    return result


PATTERNLOCK_CANONICAL_DIRECTIONS = (
    "move forward",
    "move backward",
    "move left",
    "move right",
    "move forward-left",
    "move forward-right",
    "move backward-left",
    "move backward-right",
)


def ensure_patternlock_direction_candidates(
    candidates: list[dict[str, Any]],
    *,
    retriever: ExecutorSkillRetriever,
) -> list[dict[str, Any]]:
    """Expose every released PatternLock direction without inventing skills.

    Frequency-based top-k retrieval can omit the exact cardinal direction needed
    by the next visible grid transition.  This overlay only appends canonical
    labels already present exactly once in the frozen, task-scoped vocabulary.
    It uses no episode identity, target coordinates, or simulator state.
    """

    result = [dict(candidate) for candidate in candidates]
    present = {candidate.get("template_id") for candidate in result}
    for canonical in PATTERNLOCK_CANONICAL_DIRECTIONS:
        matching = [
            entry
            for entry in retriever.entries
            if entry.get("canonical_template") == canonical
            and "PatternLock" in entry.get("tasks", [])
        ]
        if len(matching) != 1:
            raise ValueError(
                "PatternLock experience expected exactly one task-scoped "
                f"canonical direction {canonical!r}, got {len(matching)}"
            )
        template_id = matching[0]["template_id"]
        if template_id in present:
            continue
        result.append(
            {
                "rank": len(result) + 1,
                "retrieval_score": 0.0,
                "injected_by_experience": "patternlock_all_directions",
                **matching[0],
            }
        )
        present.add(template_id)
    return result


PICKHIGHLIGHT_POINT_PICKUP_TEMPLATES = tuple(
    f"pick up the {ordinal}highlighted cube at <point_1>, which is {color}"
    for ordinal in ("", "first ", "second ", "third ")
    for color in ("blue", "green", "red")
)


def ensure_pickhighlight_pickup_candidates(
    candidates: list[dict[str, Any]],
    *,
    retriever: ExecutorSkillRetriever,
) -> list[dict[str, Any]]:
    """Expose every point-grounded PickHighlight ordinal/color skill.

    The temporary cue may reveal any color and up to three ordered instances,
    while frequency top-k can omit the exact color/ordinal template. This
    overlay only appends frozen task-scoped vocabulary entries; it does not
    choose an instance, a coordinate, or use episode identity.
    """

    result = [dict(candidate) for candidate in candidates]
    present = {candidate.get("template_id") for candidate in result}
    for canonical in PICKHIGHLIGHT_POINT_PICKUP_TEMPLATES:
        matching = [
            entry
            for entry in retriever.entries
            if entry.get("canonical_template") == canonical
            and "PickHighlight" in entry.get("tasks", [])
            and int(entry.get("point_arity", 0)) == 1
        ]
        if len(matching) != 1:
            raise ValueError(
                "PickHighlight experience expected exactly one task-scoped "
                f"point pickup {canonical!r}, got {len(matching)}"
            )
        template_id = matching[0]["template_id"]
        if template_id in present:
            continue
        result.append(
            {
                **matching[0],
                "rank": len(result) + 1,
                "retrieval_score": 0.0,
                "injected_by_experience": "pickhighlight_all_pickup_templates",
            }
        )
        present.add(template_id)
    return result


def canonicalize_counting_pickup_skill(
    skill_id: str,
    *,
    allowed_skills: dict[str, dict[str, Any]],
    task: str,
    state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Map a stale pickup ordinal to the ledger-required canonical skill."""

    audit = {
        "enabled": bool(
            state is not None and state.get("exact_pickup_ordinal_required")
        ),
        "applicable": False,
        "canonicalized": False,
        "original_skill_id": skill_id,
        "replacement_skill_id": skill_id,
        "replacement_injected_after_previous_event": False,
        "color": None,
        "proposed_ordinal": None,
        "expected_ordinal": None,
    }
    if not audit["enabled"] or skill_id not in allowed_skills:
        return skill_id, audit
    selected = allowed_skills[skill_id]
    canonical = re.sub(
        r"<point_[0-9]+>", "<1, 1>", str(selected["canonical_template"])
    )
    color = counting_pickup_color(canonical)
    if color is None:
        return skill_id, audit
    proposed_ordinal = counting_pickup_ordinal(task, canonical)
    requested = state.get("requested", {}) if state is not None else {}
    completed = state.get("completed", {}) if state is not None else {}
    if color not in requested or int(completed.get(color, 0)) >= int(requested[color]):
        return skill_id, audit
    expected_ordinal = int(completed.get(color, 0)) + 1
    audit.update(
        {
            "applicable": True,
            "color": color,
            "proposed_ordinal": proposed_ordinal,
            "expected_ordinal": expected_ordinal,
        }
    )
    if proposed_ordinal == expected_ordinal:
        return skill_id, audit
    point_arity = int(selected.get("point_arity", -1))
    replacements = []
    for candidate_id, candidate in allowed_skills.items():
        candidate_template = re.sub(
            r"<point_[0-9]+>",
            "<1, 1>",
            str(candidate["canonical_template"]),
        )
        if (
            counting_pickup_color(candidate_template) == color
            and counting_pickup_ordinal(task, candidate_template) == expected_ordinal
            and int(candidate.get("point_arity", -1)) == point_arity
        ):
            replacements.append(candidate_id)
    if len(replacements) != 1:
        raise ValueError(
            "Counting experience could not canonicalize stale pickup ordinal: "
            f"expected one same-arity skill for {color} ordinal {expected_ordinal}, "
            f"got {replacements}"
        )
    replacement = replacements[0]
    audit["canonicalized"] = True
    audit["replacement_skill_id"] = replacement
    return replacement, audit


def advance_counting_previous_event(
    *,
    task: str,
    state: dict[str, Any],
    previous_subtask: str,
    response: dict[str, Any],
    refine_criterion: dict[str, Any] | None,
    action_instance_id: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Commit evidence for the previously executed counting action."""

    updated = json.loads(json.dumps(state))
    decision = response.get("refine_check", {}).get("decision")
    criterion_id = (
        refine_criterion.get("criterion_id") if refine_criterion is not None else None
    )
    execution = response.get("execution_state", {})
    completed_evidence = response.get("recent_status") == "completed" and decision == "satisfied"
    previous_kind = counting_proposal_kind(task, previous_subtask)
    event = None
    if (
        previous_kind == "pickup"
        and criterion_id == "counting_requested_cube_pickup"
        and completed_evidence
        and execution.get("held_object") == "cube"
        and execution.get("object_support") in {"airborne", "held_unknown"}
    ):
        color = counting_pickup_color(previous_subtask)
        updated["active_color"] = color
        updated["pickup_verified"] = True
        event = "pickup_verified"
    elif (
        previous_kind == "placement"
        and criterion_id
        in {"pickxtimes_verified_target_release", "binfill_verified_bin_release"}
        and completed_evidence
        and (
            updated.get("pickup_verified")
            or (
                task == "PickXtimes"
                and updated.get("integrated_first_pending")
            )
        )
        and updated.get("active_color") in updated["requested"]
        and execution.get("held_object") == "none"
        and action_instance_id not in updated["counted_action_instances"]
    ):
        color = str(updated["active_color"])
        updated["completed"][color] += 1
        updated["counted_action_instances"].append(action_instance_id)
        updated["active_color"] = None
        updated["pickup_verified"] = False
        updated["integrated_first_pending"] = False
        event = "placement_verified"
    elif (
        previous_kind == "placement"
        and criterion_id
        in {"pickxtimes_verified_target_release", "binfill_verified_bin_release"}
        and decision == "not_satisfied"
        and updated.get("pickup_verified")
        and updated.get("active_color") in updated["requested"]
        and execution.get("held_object") == "none"
        and execution.get("object_support") == "table"
    ):
        # A release outside the destination ends the old held-object event but
        # must not increment the quota.  Keep the active color so the next
        # command can recover that same ordinal, and require a fresh verified
        # pickup before another placement attempt.
        updated["pickup_verified"] = False
        event = "placement_recovery_required"
    return updated, event


def evaluate_counting_event_ledger(
    *,
    task: str,
    state: dict[str, Any],
    previous_subtask: str,
    proposed_subtask: str,
    response: dict[str, Any],
    refine_criterion: dict[str, Any] | None,
    action_instance_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance and validate a runtime-visible pickup/release event ledger.

    The state uses only the task contract, prior executed command, and the
    planner's structured judgment over current frames. It never consumes an
    evaluator subgoal, environment diagnostic, episode index, or seed.
    """

    updated, event = advance_counting_previous_event(
        task=task,
        state=state,
        previous_subtask=previous_subtask,
        response=response,
        refine_criterion=refine_criterion,
        action_instance_id=action_instance_id,
    )
    audit = {
        "enabled": True,
        "task": task,
        "previous_kind": counting_proposal_kind(task, previous_subtask),
        "proposed_kind": counting_proposal_kind(task, proposed_subtask),
        "event": event,
        "decision": "accepted",
        "state_before": json.loads(json.dumps(state)),
        "state_after": None,
    }
    previous_kind = audit["previous_kind"]

    proposed_kind = audit["proposed_kind"]
    proposed_color = counting_pickup_color(proposed_subtask)
    if proposed_kind == "pickup":
        if task == "PickXtimes" and updated.get("integrated_first_pending"):
            raise ValueError(
                "Counting experience requires finishing the current integrated "
                "first pick-place skill before beginning another pickup"
            )
        if proposed_color not in updated["requested"]:
            raise ValueError(
                f"Counting experience rejected pickup of unrequested color {proposed_color}"
            )
        if (
            event == "placement_recovery_required"
            and proposed_color != updated.get("active_color")
        ):
            raise ValueError(
                "Counting experience requires recovering the same dropped "
                f"{updated.get('active_color')} cube before changing colors"
            )
        if updated["completed"].get(proposed_color, 0) >= updated["requested"][proposed_color]:
            raise ValueError(
                "Counting experience rejected an over-count pickup: the requested "
                f"quota for {proposed_color} is already complete"
            )
        if updated.get("exact_pickup_ordinal_required"):
            proposed_ordinal = counting_pickup_ordinal(task, proposed_subtask)
            expected_ordinal = updated["completed"].get(proposed_color, 0) + 1
            if proposed_ordinal != expected_ordinal:
                expected_word = COUNTING_ORDINAL_WORDS[expected_ordinal - 1]
                raise ValueError(
                    "Counting experience rejected a stale pickup ordinal: "
                    f"{proposed_color} requires the {expected_word} pickup "
                    f"(ordinal {expected_ordinal}), got {proposed_ordinal!r}. "
                    "Select the injected exact next-ordinal canonical skill."
                )
        if updated.get("pickup_verified") and previous_kind != "pickup":
            raise ValueError(
                "Counting experience requires releasing the already verified held "
                "cube before beginning another pickup"
            )
    elif proposed_kind == "placement" and not updated.get("pickup_verified"):
        placement_color = pickxtimes_placement_color(proposed_subtask)
        integrated_first_allowed = (
            task == "PickXtimes"
            and not updated.get("explicit_first_pickup_required", False)
            and sum(updated["completed"].values()) == 0
            and (
                (
                    updated.get("integrated_first_pending")
                    and previous_kind == "placement"
                )
                or (
                    not updated.get("integrated_first_consumed")
                    and previous_kind != "placement"
                )
            )
            and placement_color in updated["requested"]
        )
        if integrated_first_allowed:
            updated["active_color"] = placement_color
            updated["integrated_first_pending"] = True
            updated["integrated_first_consumed"] = True
            audit["event"] = audit["event"] or "integrated_first_started"
        else:
            raise ValueError(
                "Counting experience rejected placement without a verified current "
                "pickup/lift event; select the requested cube pickup first"
            )
    elif proposed_kind == "button":
        if updated.get("pickup_verified") or updated["completed"] != updated["requested"]:
            raise ValueError(
                "Counting experience rejected an early stop-button proposal: "
                f"verified={updated['completed']}, requested={updated['requested']}"
            )
    if (
        audit["event"] == "placement_verified"
        and proposed_kind == "placement"
    ):
        raise ValueError(
            "Counting experience rejected repeating a placement after its release "
            "event was verified; advance to the next required pickup or final button"
        )
    audit["state_after"] = json.loads(json.dumps(updated))
    return updated, audit


BUTTON_SWAP_PRESS_PATTERN = re.compile(
    r"^press the (first|second) button(?: at <\s*[0-9]+\s*,\s*[0-9]+\s*>)?$",
    flags=re.IGNORECASE,
)
BUTTON_SWAP_PRESS_ORDINALS = {"first": 1, "second": 2}


def button_swap_press_ordinal(subtask: str | None) -> int | None:
    match = BUTTON_SWAP_PRESS_PATTERN.fullmatch(
        " ".join(str(subtask or "").strip().lower().split())
    )
    return BUTTON_SWAP_PRESS_ORDINALS[match.group(1)] if match else None


def evaluate_button_swap_press_grounding_recovery(
    *,
    state: dict[str, Any],
    previous_subtask: str,
    proposed_subtask: str,
    proposed_points: list[dict[str, Any]],
    response: dict[str, Any],
    refine_criterion: dict[str, Any] | None,
    action_instance_id: str | None,
    config: dict[str, int],
    allow_retry_budget_bypass: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Track distinct button events and force bounded contact re-grounding.

    The ledger observes only the current task contract, planner judgments over
    current images, executed subtask text, and normalized points emitted in the
    current episode.  It neither selects a saved coordinate nor reads an
    evaluator, seed, simulator state, or episode identity.
    """

    updated = json.loads(json.dumps(state))
    previous_ordinal = button_swap_press_ordinal(previous_subtask)
    proposed_ordinal = button_swap_press_ordinal(proposed_subtask)
    criterion_id = (
        refine_criterion.get("criterion_id") if refine_criterion is not None else None
    )
    decision = response.get("refine_check", {}).get("decision")
    completed_evidence = (
        previous_ordinal is not None
        and criterion_id == "button_unmask_press_local_event"
        and response.get("recent_status") == "completed"
        and decision == "satisfied"
    )
    failed_evidence = (
        previous_ordinal is not None
        and criterion_id == "button_unmask_press_local_event"
        and decision == "not_satisfied"
    )
    event = None
    previous_point = updated.get("last_dispatched_point_norm")
    completed_points = list(updated.get("completed_points_norm", []))
    counted_instances = list(updated.get("counted_action_instances", []))

    if (
        completed_evidence
        and previous_point is not None
        and previous_ordinal == len(completed_points) + 1
        and (action_instance_id is None or action_instance_id not in counted_instances)
    ):
        completed_points.append(list(previous_point))
        if action_instance_id is not None:
            counted_instances.append(action_instance_id)
        updated.update(
            {
                "completed_points_norm": completed_points,
                "counted_action_instances": counted_instances,
                "active_ordinal": None,
                "last_dispatched_point_norm": None,
                "recent_failed_points_norm": [],
                "consecutive_failed_presses": 0,
                "forced_regrounds": 0,
            }
        )
        event = "button_event_verified"
    elif failed_evidence and updated.get("active_ordinal") == previous_ordinal:
        recent = list(updated.get("recent_failed_points_norm", []))
        if previous_point is not None and previous_point not in recent:
            recent.append(list(previous_point))
        updated["recent_failed_points_norm"] = recent[-4:]
        updated["consecutive_failed_presses"] = int(
            updated.get("consecutive_failed_presses", 0)
        ) + 1
        event = "button_press_failed"

    audit = {
        "enabled": True,
        "applicable": proposed_ordinal is not None,
        "previous_ordinal": previous_ordinal,
        "proposed_ordinal": proposed_ordinal,
        "event": event,
        "decision": "not_applicable" if proposed_ordinal is None else "accepted",
        "completed_button_delta_norm": None,
        "nearest_failed_point_delta_norm": None,
        "state_before": json.loads(json.dumps(state)),
        "state_after_observation": json.loads(json.dumps(updated)),
        "state_after": None,
    }
    if proposed_ordinal is None:
        if (
            requested_pickup_color(proposed_subtask) is not None
            and len(updated.get("completed_points_norm", [])) < 2
        ):
            if not allow_retry_budget_bypass:
                raise ValueError(
                    "ButtonUnmaskSwap press ledger rejected container pickup before "
                    "two distinct button events were visually verified. Continue the "
                    "pending button and do not infer completion from command issuance."
                )
            audit["decision"] = (
                "untrusted_early_pickup_after_validation_retry_budget"
            )
            audit["untrusted_dispatch"] = True
        audit["state_after"] = json.loads(json.dumps(updated))
        return updated, audit

    expected_ordinal = len(updated.get("completed_points_norm", [])) + 1
    if expected_ordinal > 2:
        if not allow_retry_budget_bypass:
            raise ValueError(
                "ButtonUnmaskSwap press ledger rejected an extra button press after "
                "two distinct button events were verified; advance to the requested "
                "container pickup."
            )
        audit["decision"] = (
            "untrusted_extra_press_after_validation_retry_budget"
        )
        audit["untrusted_dispatch"] = True
        audit["state_after"] = json.loads(json.dumps(updated))
        return updated, audit
    if proposed_ordinal != expected_ordinal:
        expected_word = "first" if expected_ordinal == 1 else "second"
        if not allow_retry_budget_bypass:
            raise ValueError(
                "ButtonUnmaskSwap press ledger rejected a stale button ordinal: "
                f"the next runtime-visible event is the {expected_word} button, "
                f"but the proposal selected ordinal {proposed_ordinal}."
            )
        audit["decision"] = (
            "untrusted_stale_ordinal_after_validation_retry_budget"
        )
        audit["untrusted_dispatch"] = True
        audit["state_after"] = json.loads(json.dumps(updated))
        return updated, audit
    if len(proposed_points) != 1:
        raise ValueError(
            "ButtonUnmaskSwap press recovery requires exactly one current-image "
            f"button point, got {len(proposed_points)}"
        )
    point = proposed_points[0]
    current_point = [point.get("x_norm"), point.get("y_norm")]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in current_point
    ):
        raise ValueError(
            "ButtonUnmaskSwap press point must have integer normalized coordinates"
        )

    completed_points = list(updated.get("completed_points_norm", []))
    if completed_points:
        completed_delta = min(
            max(
                abs(int(current_point[0]) - int(completed[0])),
                abs(int(current_point[1]) - int(completed[1])),
            )
            for completed in completed_points
        )
        audit["completed_button_delta_norm"] = completed_delta
        if completed_delta < int(config["minimum_completed_button_delta_norm"]):
            if not allow_retry_budget_bypass:
                raise ValueError(
                    "ButtonUnmaskSwap press ledger rejected a point on or too near an "
                    "already completed button. Re-inspect the current front image and "
                    "ground the other visible button; do not reuse the completed "
                    f"button location (normalized delta {completed_delta})."
                )
            audit["decision"] = (
                "untrusted_completed_button_point_after_validation_retry_budget"
            )
            audit["untrusted_dispatch"] = True
            audit["state_after"] = json.loads(json.dumps(updated))
            return updated, audit

    if updated.get("active_ordinal") not in {None, proposed_ordinal}:
        updated.update(
            {
                "active_ordinal": proposed_ordinal,
                "last_dispatched_point_norm": None,
                "recent_failed_points_norm": [],
                "consecutive_failed_presses": 0,
                "forced_regrounds": 0,
            }
        )
    updated["active_ordinal"] = proposed_ordinal
    recent_failed = list(updated.get("recent_failed_points_norm", []))
    deltas = [
        max(
            abs(int(current_point[0]) - int(failed[0])),
            abs(int(current_point[1]) - int(failed[1])),
        )
        for failed in recent_failed
    ]
    nearest = min(deltas) if deltas else None
    audit["nearest_failed_point_delta_norm"] = nearest
    failures = int(updated.get("consecutive_failed_presses", 0))
    forced = int(updated.get("forced_regrounds", 0))
    trigger = int(config["trigger_after_failed_presses"])
    maximum = int(config["max_forced_regrounds_per_button"])
    if (
        nearest is not None
        and failures >= trigger
        and forced < maximum
        and nearest < int(config["minimum_point_delta_norm"])
    ):
        if not allow_retry_budget_bypass:
            raise ValueError(
                "ButtonUnmaskSwap recovery rejected another near-identical button "
                f"point after {failures} visually verified misses. Retract, "
                "re-inspect the current front image, keep the pending button "
                "distinct from the completed one, and choose a materially "
                "different contact point on that same pending button."
            )
        updated["forced_regrounds"] = forced + 1
        audit["decision"] = "accepted_after_validation_retry_budget"
    elif (
        nearest is not None
        and failures >= trigger
        and forced < maximum
        and nearest >= int(config["minimum_point_delta_norm"])
    ):
        updated["forced_regrounds"] = forced + 1
        audit["decision"] = "accepted_reground"
    updated["last_dispatched_point_norm"] = current_point
    audit["state_after"] = json.loads(json.dumps(updated))
    return updated, audit


def reconcile_button_swap_press_state_after_runtime_overrides(
    *,
    proposed_state: dict[str, Any],
    guard: dict[str, Any],
    planner_proposed_subtask: str,
    dispatched_subtask: str,
    override_source: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Commit the press ledger against the command actually sent to the VLA.

    The candidate gate validates the planner proposal before the mature completion,
    experience, and refinement locks run. When one of those locks intentionally
    replays the prior command, committing the proposal would make the ledger claim
    that an action was dispatched when it was not. Preserve only the observation
    update in that case; the next unlocked round can commit the new proposal.
    """

    if planner_proposed_subtask == dispatched_subtask:
        return proposed_state, guard
    observed = guard.get("state_after_observation")
    if not isinstance(observed, dict):
        raise ValueError(
            "ButtonUnmaskSwap runtime override requires an observation-only state"
        )
    reconciled_guard = json.loads(json.dumps(guard))
    reconciled_state = json.loads(json.dumps(observed))
    reconciled_guard["decision_before_runtime_override"] = guard.get("decision")
    reconciled_guard["decision"] = "deferred_by_runtime_override"
    reconciled_guard["runtime_override"] = {
        "source": override_source or "unknown_runtime_override",
        "planner_proposed_subtask": planner_proposed_subtask,
        "dispatched_subtask": dispatched_subtask,
    }
    reconciled_guard["state_after"] = json.loads(json.dumps(reconciled_state))
    return reconciled_state, reconciled_guard


VIDEO_REPICK_PICKUP_PATTERN = re.compile(
    r"^pick up the correct cube(?: at <[0-9]+, [0-9]+>)? for the "
    r"(first|second|third) time$",
    flags=re.IGNORECASE,
)
VIDEO_REPICK_ORDINALS = {"first": 1, "second": 2, "third": 3}


def video_repick_required_repetitions(
    description: str, guidance: dict[str, Any]
) -> int:
    """Read the task-level repetition contract without episode metadata."""

    texts = [description]
    for value in guidance.values():
        if isinstance(value, list):
            texts.extend(str(item) for item in value)
        elif isinstance(value, str):
            texts.append(value)
    joined = "\n".join(texts).lower()
    words = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3}
    patterns = (
        r"exactly\s+(one|once|two|twice|three|[123])\s+",
        r"(?:perform|complete)\s+(one|once|two|twice|three|[123])\s+",
        r"(?:count|repetitions?)[^\n]{0,24}\b(?:to|0\s*(?:through|-))\s*([123])\b",
    )
    values: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, joined):
            token = match.group(1)
            values.append(int(token) if token.isdigit() else words[token])
    if not values:
        raise ValueError("VideoRepick ledger could not parse required repetitions")
    # The task contract is redundantly stated. Conflicting values are unsafe.
    if len(set(values)) != 1:
        raise ValueError(
            f"VideoRepick ledger found conflicting repetition counts {sorted(set(values))}"
        )
    return values[0]


def video_repick_proposal_kind(subtask: str) -> str | None:
    normalized = " ".join(subtask.strip().lower().split())
    if VIDEO_REPICK_PICKUP_PATTERN.fullmatch(normalized):
        return "pickup"
    if normalized == "put it down":
        return "placement"
    if re.fullmatch(
        r"press the button(?: at <[0-9]+, [0-9]+>)? to finish", normalized
    ):
        return "button"
    return None


def video_repick_pickup_ordinal(subtask: str) -> int | None:
    match = VIDEO_REPICK_PICKUP_PATTERN.fullmatch(
        " ".join(subtask.strip().lower().split())
    )
    return VIDEO_REPICK_ORDINALS[match.group(1).lower()] if match else None


def evaluate_video_repick_event_ledger(
    *,
    state: dict[str, Any],
    previous_subtask: str,
    proposed_subtask: str,
    response: dict[str, Any],
    refine_criterion: dict[str, Any] | None,
    action_instance_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gate VideoRepick progress with ordered, runtime-visible physical events."""

    updated = json.loads(json.dumps(state))
    previous_kind = video_repick_proposal_kind(previous_subtask)
    proposed_kind = video_repick_proposal_kind(proposed_subtask)
    criterion_id = (
        refine_criterion.get("criterion_id") if refine_criterion is not None else None
    )
    decision = response.get("refine_check", {}).get("decision")
    execution = response.get("execution_state", {})
    completed_evidence = (
        response.get("recent_status") == "completed" and decision == "satisfied"
    )
    event = None
    if (
        previous_kind == "pickup"
        and criterion_id == "videorepick_verified_target_pickup"
        and completed_evidence
        and execution.get("held_object") == "cube"
        and execution.get("object_support") in {"airborne", "held_unknown"}
    ):
        ordinal = video_repick_pickup_ordinal(previous_subtask)
        if ordinal == int(updated["completed"]) + 1:
            updated["active_ordinal"] = ordinal
            updated["pickup_verified"] = True
            event = "pickup_verified"
    elif (
        previous_kind == "placement"
        and criterion_id == "videorepick_verified_table_release"
        and completed_evidence
        and updated.get("pickup_verified")
        and updated.get("active_ordinal") == int(updated["completed"]) + 1
        and execution.get("held_object") == "none"
        and execution.get("object_support") == "table"
        and action_instance_id not in updated["counted_release_instances"]
    ):
        updated["completed"] += 1
        updated["counted_release_instances"].append(action_instance_id)
        updated["active_ordinal"] = None
        updated["pickup_verified"] = False
        event = "cycle_verified"

    audit = {
        "enabled": True,
        "applicable": True,
        "task": "VideoRepick",
        "previous_kind": previous_kind,
        "proposed_kind": proposed_kind,
        "event": event,
        "decision": "accepted",
        "state_before": json.loads(json.dumps(state)),
        "state_after": None,
    }
    if proposed_kind == "pickup":
        ordinal = video_repick_pickup_ordinal(proposed_subtask)
        expected = int(updated["completed"]) + 1
        if int(updated["completed"]) >= int(updated["requested"]):
            raise ValueError("VideoRepick ledger rejected an excess pickup cycle")
        if ordinal != expected:
            raise ValueError(
                "VideoRepick ledger rejected a stale pickup ordinal: "
                f"expected {expected}, got {ordinal}"
            )
        if updated.get("pickup_verified") and previous_kind != "pickup":
            raise ValueError(
                "VideoRepick ledger requires releasing the verified held cube "
                "before another pickup"
            )
    elif proposed_kind == "placement" and not updated.get("pickup_verified"):
        raise ValueError(
            "VideoRepick ledger rejected put-down without a verified current pickup"
        )
    elif proposed_kind == "button":
        if updated.get("pickup_verified") or int(updated["completed"]) != int(
            updated["requested"]
        ):
            raise ValueError(
                "VideoRepick ledger rejected an early finish-button proposal: "
                f"verified={updated['completed']}, requested={updated['requested']}"
            )
    if event == "cycle_verified" and proposed_kind == "placement":
        raise ValueError(
            "VideoRepick ledger rejected repeating put-down after its release "
            "was verified"
        )
    audit["state_after"] = json.loads(json.dumps(updated))
    return updated, audit


VIDEO_REPICK_RENDERED_POINT_PATTERN = re.compile(
    r"<\s*([0-9]+)\s*,\s*([0-9]+)\s*>"
)


def evaluate_video_repick_grounding_recovery(
    *,
    state: dict[str, Any],
    previous_subtask: str,
    proposed_subtask: str,
    proposed_points: list[dict[str, Any]],
    response: dict[str, Any],
    refine_criterion: dict[str, Any] | None,
    config: dict[str, int],
    allow_retry_budget_bypass: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Force a real visual re-ground after repeated verified pickup misses.

    The gate never chooses an object or stores an episode coordinate.  It only
    detects that the last accepted pickup point has repeatedly produced an
    empty gripper and table-supported target, then rejects another nearly
    identical point.  The model must re-inspect the current image and provide a
    materially different point on the task-described instance.  A finite
    per-ordinal budget prevents this recovery rule from trapping execution.
    """

    updated = json.loads(json.dumps(state))
    previous_kind = video_repick_proposal_kind(previous_subtask)
    proposed_kind = video_repick_proposal_kind(proposed_subtask)
    previous_ordinal = video_repick_pickup_ordinal(previous_subtask)
    proposed_ordinal = video_repick_pickup_ordinal(proposed_subtask)
    criterion_id = (
        refine_criterion.get("criterion_id") if refine_criterion is not None else None
    )
    decision = response.get("refine_check", {}).get("decision")
    execution = response.get("execution_state", {})
    verified_pickup_failure = (
        previous_kind == "pickup"
        and criterion_id == "videorepick_verified_target_pickup"
        and decision == "not_satisfied"
        and execution.get("held_object") == "none"
        and execution.get("object_support") == "table"
    )
    verified_pickup_success = (
        previous_kind == "pickup"
        and criterion_id == "videorepick_verified_target_pickup"
        and decision == "satisfied"
        and execution.get("held_object") == "cube"
    )
    if verified_pickup_success:
        updated = {
            "ordinal": previous_ordinal,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": None,
            "forced_regrounds": 0,
        }
    elif verified_pickup_failure:
        if updated.get("ordinal") != previous_ordinal:
            updated = {
                "ordinal": previous_ordinal,
                "consecutive_failed_pickups": 0,
                "last_dispatched_point_norm": updated.get(
                    "last_dispatched_point_norm"
                ),
                "forced_regrounds": 0,
            }
        updated["consecutive_failed_pickups"] = int(
            updated.get("consecutive_failed_pickups", 0)
        ) + 1

    audit = {
        "enabled": True,
        "applicable": proposed_kind == "pickup",
        "previous_ordinal": previous_ordinal,
        "proposed_ordinal": proposed_ordinal,
        "verified_pickup_failure": verified_pickup_failure,
        "decision": "not_applicable" if proposed_kind != "pickup" else "accepted",
        "point_delta_norm": None,
        "state_before": json.loads(json.dumps(state)),
        "state_after": None,
    }
    if proposed_kind != "pickup":
        audit["state_after"] = json.loads(json.dumps(updated))
        return updated, audit
    if len(proposed_points) != 1:
        raise ValueError(
            "VideoRepick grounding recovery requires exactly one current-image "
            f"pickup point, got {len(proposed_points)}"
        )
    point = proposed_points[0]
    current_point = [point.get("x_norm"), point.get("y_norm")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in current_point):
        raise ValueError("VideoRepick pickup point must have integer normalized coordinates")
    if updated.get("ordinal") not in {None, proposed_ordinal}:
        updated = {
            "ordinal": proposed_ordinal,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": None,
            "forced_regrounds": 0,
        }
    updated["ordinal"] = proposed_ordinal
    previous_point = updated.get("last_dispatched_point_norm")
    failures = int(updated.get("consecutive_failed_pickups", 0))
    forced = int(updated.get("forced_regrounds", 0))
    trigger = int(config["trigger_after_failed_pickups"])
    maximum = int(config["max_forced_regrounds_per_ordinal"])
    if previous_point is not None:
        delta = max(
            abs(int(current_point[0]) - int(previous_point[0])),
            abs(int(current_point[1]) - int(previous_point[1])),
        )
        audit["point_delta_norm"] = delta
        if (
            failures >= trigger
            and forced < maximum
            and delta < int(config["minimum_point_delta_norm"])
        ):
            if not allow_retry_budget_bypass:
                raise ValueError(
                    "VideoRepick recovery rejected another nearly identical pickup "
                    f"point after {failures} visually verified misses. Retract, "
                    "re-inspect the current front image, preserve the demonstrated "
                    "instance relation from the task guidance, and choose a materially "
                    "different top-down point on that same cube. Do not sweep to a "
                    "different same-colored cube."
                )
            updated["forced_regrounds"] = forced + 1
            audit["decision"] = "accepted_after_validation_retry_budget"
        if failures >= trigger and forced < maximum and delta >= int(
            config["minimum_point_delta_norm"]
        ):
            updated["forced_regrounds"] = forced + 1
            audit["decision"] = "accepted_reground"
    updated["last_dispatched_point_norm"] = current_point
    audit["state_after"] = json.loads(json.dumps(updated))
    return updated, audit


def evaluate_unmask_pickup_grounding_recovery(
    *,
    state: dict[str, Any],
    previous_subtask: str,
    proposed_subtask: str,
    proposed_points: list[dict[str, Any]],
    response: dict[str, Any],
    refine_criterion: dict[str, Any] | None,
    config: dict[str, int],
    allow_retry_budget_bypass: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject stale unmask pickup points after repeated visible misses.

    This task-scoped gate uses only planner-visible execution evidence and
    normalized points emitted during the current episode. It does not select a
    container, read simulator state, or persist episode coordinates. A bounded
    retry budget keeps the intervention from trapping execution.
    """

    updated = json.loads(json.dumps(state))
    previous_color = requested_pickup_color(previous_subtask)
    proposed_color = requested_pickup_color(proposed_subtask)
    criterion_id = (
        refine_criterion.get("criterion_id") if refine_criterion is not None else None
    )
    decision = response.get("refine_check", {}).get("decision")
    execution = response.get("execution_state", {})
    verified_pickup_failure = (
        previous_color is not None
        and criterion_id == "unmask_pickup_identity_and_separation"
        and decision == "not_satisfied"
        and execution.get("held_object") == "none"
        and execution.get("object_support") == "table"
    )
    verified_pickup_success = (
        previous_color is not None
        and criterion_id == "unmask_pickup_identity_and_separation"
        and decision == "satisfied"
        and execution.get("held_object") == "container"
    )
    if verified_pickup_success:
        updated = {
            "target_color": previous_color,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "forced_regrounds": 0,
        }
    elif verified_pickup_failure:
        if updated.get("target_color") != previous_color:
            updated = {
                "target_color": previous_color,
                "consecutive_failed_pickups": 0,
                "last_dispatched_point_norm": updated.get(
                    "last_dispatched_point_norm"
                ),
                "recent_failed_points_norm": [],
                "forced_regrounds": 0,
            }
        last_point = updated.get("last_dispatched_point_norm")
        recent = list(updated.get("recent_failed_points_norm", []))
        if last_point is not None and last_point not in recent:
            recent.append(last_point)
        updated["recent_failed_points_norm"] = recent[-4:]
        updated["consecutive_failed_pickups"] = int(
            updated.get("consecutive_failed_pickups", 0)
        ) + 1

    audit = {
        "enabled": True,
        "applicable": proposed_color is not None,
        "previous_target_color": previous_color,
        "proposed_target_color": proposed_color,
        "verified_pickup_failure": verified_pickup_failure,
        "decision": "not_applicable" if proposed_color is None else "accepted",
        "nearest_failed_point_delta_norm": None,
        "state_before": json.loads(json.dumps(state)),
        "state_after": None,
    }
    if proposed_color is None:
        audit["state_after"] = json.loads(json.dumps(updated))
        return updated, audit
    if len(proposed_points) != 1:
        raise ValueError(
            "Unmask pickup grounding recovery requires exactly one current-image "
            f"pickup point, got {len(proposed_points)}"
        )
    point = proposed_points[0]
    current_point = [point.get("x_norm"), point.get("y_norm")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in current_point):
        raise ValueError("Unmask pickup point must have integer normalized coordinates")
    if updated.get("target_color") not in {None, proposed_color}:
        updated = {
            "target_color": proposed_color,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "forced_regrounds": 0,
        }
    updated["target_color"] = proposed_color
    recent = list(updated.get("recent_failed_points_norm", []))
    deltas = [
        max(
            abs(int(current_point[0]) - int(previous[0])),
            abs(int(current_point[1]) - int(previous[1])),
        )
        for previous in recent
    ]
    nearest_delta = min(deltas) if deltas else None
    audit["nearest_failed_point_delta_norm"] = nearest_delta
    failures = int(updated.get("consecutive_failed_pickups", 0))
    forced = int(updated.get("forced_regrounds", 0))
    trigger = int(config["trigger_after_failed_pickups"])
    maximum = int(config["max_forced_regrounds_per_target"])
    stale = (
        nearest_delta is not None
        and nearest_delta < int(config["minimum_point_delta_norm"])
    )
    if failures >= trigger and forced < maximum and stale:
        if not allow_retry_budget_bypass:
            raise ValueError(
                "Unmask pickup recovery rejected a point that repeats a recent "
                f"failed location after {failures} visually verified misses. "
                "Re-inspect the current front image and choose a materially "
                "different top-down point on the same task-described container; "
                "do not switch to a different container or reuse an earlier point."
            )
        updated["forced_regrounds"] = forced + 1
        audit["decision"] = "accepted_after_validation_retry_budget"
    elif failures >= trigger and forced < maximum:
        updated["forced_regrounds"] = forced + 1
        audit["decision"] = "accepted_reground"
    updated["last_dispatched_point_norm"] = current_point
    audit["state_after"] = json.loads(json.dumps(updated))
    return updated, audit


def evaluate_pickhighlight_pickup_grounding_recovery(
    *,
    state: dict[str, Any],
    previous_subtask: str,
    proposed_subtask: str,
    proposed_points: list[dict[str, Any]],
    response: dict[str, Any],
    config: dict[str, int],
    allow_retry_budget_bypass: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject near-duplicate PickHighlight points after visible pickup misses.

    The target descriptor comes only from the task-scoped canonical action, and
    failure evidence comes only from the current episode's planner-visible status
    and execution state. The gate never selects a point or reads evaluator,
    simulator, seed, or episode identity.
    """

    updated = json.loads(json.dumps(state))
    previous_target = pickhighlight_pickup_signature(previous_subtask)
    proposed_target = pickhighlight_pickup_signature(proposed_subtask)
    execution = response.get("execution_state", {})
    status = response.get("recent_status")
    verified_pickup_failure = (
        previous_target is not None
        and status == "stagnant"
        and execution.get("held_object") == "none"
        and execution.get("object_support") == "table"
    )
    verified_pickup_success = (
        previous_target is not None
        and status == "completed"
        and execution.get("held_object") == "cube"
        and execution.get("object_support") in {"airborne", "held_unknown"}
    )
    if verified_pickup_success:
        updated = {
            "target_signature": previous_target,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "forced_regrounds": 0,
        }
    elif verified_pickup_failure:
        if updated.get("target_signature") != previous_target:
            updated = {
                "target_signature": previous_target,
                "consecutive_failed_pickups": 0,
                "last_dispatched_point_norm": None,
                "recent_failed_points_norm": [],
                "forced_regrounds": 0,
            }
        last_point = updated.get("last_dispatched_point_norm")
        recent = list(updated.get("recent_failed_points_norm", []))
        if last_point is not None and last_point not in recent:
            recent.append(last_point)
        updated["recent_failed_points_norm"] = recent[-4:]
        updated["consecutive_failed_pickups"] = int(
            updated.get("consecutive_failed_pickups", 0)
        ) + 1

    audit = {
        "enabled": True,
        "applicable": proposed_target is not None,
        "previous_target_signature": previous_target,
        "proposed_target_signature": proposed_target,
        "verified_pickup_failure": verified_pickup_failure,
        "verified_pickup_success": verified_pickup_success,
        "decision": "not_applicable" if proposed_target is None else "accepted",
        "nearest_failed_point_delta_norm": None,
        "state_before": json.loads(json.dumps(state)),
        "state_after": None,
    }
    if proposed_target is None:
        audit["state_after"] = json.loads(json.dumps(updated))
        return updated, audit
    if len(proposed_points) != 1:
        raise ValueError(
            "PickHighlight pickup grounding recovery requires exactly one "
            f"current-image pickup point, got {len(proposed_points)}"
        )
    point = proposed_points[0]
    current_point = [point.get("x_norm"), point.get("y_norm")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in current_point):
        raise ValueError(
            "PickHighlight pickup point must have integer normalized coordinates"
        )
    if updated.get("target_signature") not in {None, proposed_target}:
        updated = {
            "target_signature": proposed_target,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "forced_regrounds": 0,
        }
    updated["target_signature"] = proposed_target
    recent = list(updated.get("recent_failed_points_norm", []))
    deltas = [
        max(
            abs(int(current_point[0]) - int(previous[0])),
            abs(int(current_point[1]) - int(previous[1])),
        )
        for previous in recent
    ]
    nearest_delta = min(deltas) if deltas else None
    audit["nearest_failed_point_delta_norm"] = nearest_delta
    failures = int(updated.get("consecutive_failed_pickups", 0))
    forced = int(updated.get("forced_regrounds", 0))
    trigger = int(config["trigger_after_failed_pickups"])
    maximum = int(config["max_forced_regrounds_per_target"])
    stale = (
        nearest_delta is not None
        and nearest_delta < int(config["minimum_point_delta_norm"])
    )
    if failures >= trigger and forced < maximum and stale:
        if not allow_retry_budget_bypass:
            raise ValueError(
                "PickHighlight pickup recovery rejected a point that repeats a "
                f"recent failed location after {failures} visibly verified misses. "
                "Re-inspect the current front image and choose a materially "
                "different top-down point on the same pending highlighted target; "
                "do not return to the button or another same-colored cube."
            )
        updated["forced_regrounds"] = forced + 1
        audit["decision"] = "accepted_after_validation_retry_budget"
    elif failures >= trigger and forced < maximum:
        updated["forced_regrounds"] = forced + 1
        audit["decision"] = "accepted_reground"
    updated["last_dispatched_point_norm"] = current_point
    audit["state_after"] = json.loads(json.dumps(updated))
    return updated, audit


def sanitize_target_identity_frame_ids(
    target_identity: dict[str, Any], *, allowed_front_ids: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop fabricated frame references without failing the rollout.

    Only references to front images attached in the current planner turn are accepted.
    Text memory may retain an older frame identifier as provenance, but text-progress
    mode does not retain that image and therefore cannot use it as current visual
    evidence or as a grounding source. If all current visual references are invalid,
    a claimed direct visual binding is downgraded to an inference so it cannot
    authorize a hazardous pickup.
    """
    sanitized = dict(target_identity)
    raw_ids = sanitized.get("evidence_frame_ids")
    if not isinstance(raw_ids, list):
        return sanitized, {"changed": False, "reason": "non_list_left_for_validation"}
    accepted = [
        value
        for value in raw_ids
        if isinstance(value, str) and value in allowed_front_ids
    ][:4]
    dropped = [value for value in raw_ids if value not in accepted]
    sanitized["evidence_frame_ids"] = accepted
    downgraded = False
    if (
        sanitized.get("binding_basis") in {"observed_cover_event", "revealed_color"}
        and not accepted
    ):
        sanitized["binding_basis"] = "inference_only"
        if sanitized.get("binding_status") in {"locked", "verified"}:
            sanitized["binding_status"] = "candidate"
        downgraded = True
    return sanitized, {
        "changed": sanitized != target_identity,
        "accepted_frame_ids": accepted,
        "dropped_frame_ids": dropped,
        "direct_binding_downgraded": downgraded,
    }


def completion_evidence_candidate(
    completion_check: dict[str, Any], criterion: dict[str, Any]
) -> tuple[bool, bool]:
    """Return (candidate, hard_reset) for an optional accumulation policy."""
    policy = criterion.get("evidence_accumulation_policy")
    if not isinstance(policy, dict):
        return False, False
    verdicts = {
        str(item["check_id"]): str(item["verdict"])
        for item in completion_check.get("checks", [])
    }
    candidate_requirements = policy.get("candidate_requirements", [])
    candidate = bool(candidate_requirements) and all(
        verdicts.get(str(item.get("check_id")))
        in {str(value) for value in item.get("allowed_verdicts", [])}
        for item in candidate_requirements
    )
    reset_requirements = policy.get("hard_reset_if_all", [])
    hard_reset = bool(reset_requirements) and all(
        verdicts.get(str(item.get("check_id"))) == str(item.get("verdict"))
        for item in reset_requirements
    )
    return candidate, hard_reset


def update_completion_evidence_state(
    previous: dict[str, Any] | None,
    completion_check: dict[str, Any],
    criterion: dict[str, Any],
) -> dict[str, Any]:
    """Accumulate conservative cross-chunk evidence for one semantic action."""
    policy = criterion.get("evidence_accumulation_policy")
    if not isinstance(policy, dict):
        return {}
    required = int(policy.get("required_consecutive_candidate_rounds", 2))
    if required < 1:
        raise ValueError("required_consecutive_candidate_rounds must be positive")
    state = {
        "candidate_round_count": int((previous or {}).get("candidate_round_count", 0)),
        "completion_latched": bool((previous or {}).get("completion_latched", False)),
        "required_candidate_rounds": required,
        "last_update": "retained",
    }
    candidate, hard_reset = completion_evidence_candidate(
        completion_check, criterion
    )
    if hard_reset:
        state["candidate_round_count"] = 0
        state["completion_latched"] = False
        state["last_update"] = "hard_reset"
    elif completion_check.get("decision") == "satisfied":
        state["candidate_round_count"] = max(required, state["candidate_round_count"])
        state["completion_latched"] = True
        state["last_update"] = "raw_satisfied"
    elif candidate:
        state["candidate_round_count"] += 1
        state["completion_latched"] = state["candidate_round_count"] >= required
        state["last_update"] = "candidate_increment"
    return state


def validate_completion_check(
    completion_check: Any,
    active_criterion: dict[str, Any] | None,
) -> None:
    expected_fields = {"criterion_id", "decision", "checks", "reason"}
    if not isinstance(completion_check, dict) or set(completion_check) != expected_fields:
        raise ValueError(
            f"completion_check fields must be exactly {sorted(expected_fields)}"
        )
    decision = completion_check["decision"]
    checks = completion_check["checks"]
    reason = completion_check["reason"]
    if decision not in COMPLETION_CHECK_DECISIONS:
        raise ValueError("completion_check.decision is invalid")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 320:
        raise ValueError("completion_check.reason must be 1-320 characters")
    if not isinstance(checks, list):
        raise ValueError("completion_check.checks must be a list")
    if active_criterion is None:
        if (
            completion_check["criterion_id"] != "none"
            or decision != "not_applicable"
            or checks
        ):
            raise ValueError(
                "No action-specific criterion applies: expected criterion_id=none, "
                "decision=not_applicable, and empty checks"
            )
        return

    criterion_id = active_criterion["criterion_id"]
    if completion_check["criterion_id"] != criterion_id:
        raise ValueError(
            f"completion_check.criterion_id must be {criterion_id!r}"
        )
    expected_ids = [item["check_id"] for item in active_criterion["required_checks"]]
    observed_ids: list[str] = []
    verdicts: list[str] = []
    for index, check in enumerate(checks):
        fields = {"check_id", "verdict", "evidence"}
        if not isinstance(check, dict) or set(check) != fields:
            raise ValueError(f"completion_check.checks[{index}] has invalid fields")
        observed_ids.append(check["check_id"])
        verdict = check["verdict"]
        verdicts.append(verdict)
        if verdict not in COMPLETION_CHECK_VERDICTS:
            raise ValueError(f"completion_check.checks[{index}].verdict is invalid")
        evidence = check["evidence"]
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 240:
            raise ValueError(
                f"completion_check.checks[{index}].evidence must be 1-240 characters"
            )
    if observed_ids != expected_ids:
        raise ValueError(
            "completion_check checks must match the criterion order exactly: "
            f"expected={expected_ids}, observed={observed_ids}"
        )
    expected_decision = (
        "satisfied"
        if all(value == "yes" for value in verdicts)
        else "not_satisfied"
        if any(value == "no" for value in verdicts)
        else "uncertain"
    )
    if decision != expected_decision:
        raise ValueError(
            f"completion_check.decision must be {expected_decision!r} for verdicts "
            f"{verdicts!r}"
        )


def sanitize_duplicate_completion_checks(
    completion_check: Any,
    active_criterion: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Remove semantically harmless duplicate checks before strict validation.

    Codex can occasionally emit the same required check more than once.  We repair
    only repetitions of a check known to the active criterion whose verdict agrees
    with the first occurrence.  Unknown checks, malformed entries, conflicting
    verdicts, missing checks, and incorrect ordering remain untouched so that the
    strict validator still rejects them.
    """
    metadata: dict[str, Any] = {
        "changed": False,
        "removed_duplicates": [],
    }
    if not isinstance(completion_check, dict) or active_criterion is None:
        return completion_check, metadata
    checks = completion_check.get("checks")
    if not isinstance(checks, list):
        return completion_check, metadata

    expected_ids = {
        item["check_id"] for item in active_criterion.get("required_checks", [])
    }
    first_by_id: dict[str, dict[str, Any]] = {}
    sanitized: list[Any] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            return completion_check, metadata
        check_id = check.get("check_id")
        if not isinstance(check_id, str) or check_id not in expected_ids:
            return completion_check, metadata
        first = first_by_id.get(check_id)
        if first is None:
            first_by_id[check_id] = check
            sanitized.append(check)
            continue
        if check.get("verdict") != first.get("verdict"):
            return completion_check, metadata
        metadata["removed_duplicates"].append(
            {
                "check_id": check_id,
                "index": index,
                "same_evidence": check.get("evidence") == first.get("evidence"),
            }
        )

    if not metadata["removed_duplicates"]:
        return completion_check, metadata
    repaired = dict(completion_check)
    repaired["checks"] = sanitized
    metadata["changed"] = True
    return repaired, metadata


def validate_target_identity(
    target_identity: Any, *, allowed_recent_front_ids: set[str]
) -> None:
    expected_fields = {
        "requested_color",
        "binding_status",
        "binding_basis",
        "target_description",
        "evidence_frame_ids",
        "revealed_color",
        "same_as_bound_target",
        "reason",
    }
    if not isinstance(target_identity, dict) or set(target_identity) != expected_fields:
        raise ValueError(
            f"target_identity fields must be exactly {sorted(expected_fields)}"
        )
    if target_identity["requested_color"] not in TARGET_COLORS:
        raise ValueError("target_identity.requested_color is invalid")
    if target_identity["binding_status"] not in TARGET_BINDING_STATUSES:
        raise ValueError("target_identity.binding_status is invalid")
    if target_identity["binding_basis"] not in TARGET_BINDING_BASES:
        raise ValueError("target_identity.binding_basis is invalid")
    if target_identity["revealed_color"] not in TARGET_COLORS:
        raise ValueError("target_identity.revealed_color is invalid")
    if target_identity["same_as_bound_target"] not in TARGET_IDENTITY_RELATIONS:
        raise ValueError("target_identity.same_as_bound_target is invalid")
    for field, limit in (("target_description", 180), ("reason", 320)):
        value = target_identity[field]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"target_identity.{field} must be 1-{limit} characters")
    frame_ids = target_identity["evidence_frame_ids"]
    if (
        not isinstance(frame_ids, list)
        or len(frame_ids) > 4
        or any(value not in allowed_recent_front_ids for value in frame_ids)
    ):
        raise ValueError(
            "target_identity.evidence_frame_ids must reference at most four current "
            "front observations"
        )
    if (
        target_identity["binding_basis"]
        in {"observed_cover_event", "revealed_color"}
        and not frame_ids
    ):
        raise ValueError(
            "Visual target binding bases require at least one evidence frame"
        )


def load_oracle_skill_catalog(path: Path = ORACLE_SKILL_CATALOG_PATH) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("task") != "MoveCube" or not isinstance(catalog.get("methods"), dict):
        raise ValueError(f"Invalid MoveCube oracle skill catalog: {path}")
    return catalog


def select_move_cube_method(
    detailed_task_description: str, planner_guidance: dict[str, Any]
) -> str:
    evidence = (
        detailed_task_description
        + "\n"
        + json.dumps(planner_guidance, ensure_ascii=False)
    ).lower()
    markers = {
        "direct_gripper_push": (
            "direct-gripper-pushing",
            "direct gripper pushing",
            "use the closed gripper to directly push",
            "push with the closed gripper directly contacting",
        ),
        "peg_push": (
            "peg-pushing method",
            "peg pushing method",
            "use the held peg",
            "reproduce the demonstrated peg",
        ),
        "grasp_putdown": (
            "pick-and-place method",
            "pick and place method",
            "directly grasp the red cube",
            "lift it visibly clear of the tabletop",
        ),
    }
    matches = [
        method
        for method, values in markers.items()
        if any(value in evidence for value in values)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Could not determine exactly one MoveCube demonstrated method from init "
            f"semantics; matches={matches}"
        )
    return matches[0]


def validate_response(
    response: Any,
    *,
    allowed_recent_front_ids: set[str],
    require_recent_status: bool = False,
    require_lesson_fields: bool = False,
    require_completion_check: bool = False,
    require_refine_check: bool = False,
    require_target_identity: bool = False,
    require_binding_updates: bool = False,
    active_completion_criterion: dict[str, Any] | None = None,
    active_refine_criterion: dict[str, Any] | None = None,
) -> None:
    if not isinstance(response, dict):
        raise ValueError("Planner response must be a JSON object")
    expected = {
        "recent_outcome",
        "updated_progress",
        "current_subtask_template",
        "points",
    }
    if require_recent_status:
        expected.add("recent_status")
    if require_lesson_fields:
        expected.update({"skill_key", "execution_state", "instruction_strategy"})
    if require_completion_check:
        expected.add("completion_check")
    if require_refine_check:
        expected.add("refine_check")
    if require_target_identity:
        expected.add("target_identity")
    if require_binding_updates:
        expected.add("binding_updates")
    if set(response) != expected:
        raise ValueError(f"Planner response fields must be exactly {sorted(expected)}")
    if require_recent_status and (
        not isinstance(response["recent_status"], str)
        or response["recent_status"] not in RECENT_STATUSES
    ):
        raise ValueError(
            f"recent_status must be one of {sorted(RECENT_STATUSES)}; "
            f"got {response['recent_status']!r}"
        )
    if require_lesson_fields:
        skill_key = response["skill_key"]
        if not isinstance(skill_key, str) or re.fullmatch(
            r"[a-z][a-z0-9_]{0,63}", skill_key
        ) is None:
            raise ValueError(f"skill_key must be stable snake_case; got {skill_key!r}")
        strategy = response["instruction_strategy"]
        if strategy not in INSTRUCTION_STRATEGIES:
            raise ValueError(
                f"instruction_strategy must be one of {sorted(INSTRUCTION_STRATEGIES)}; "
                f"got {strategy!r}"
            )
        execution_state = response["execution_state"]
        expected_state_fields = {"held_object", "object_support", "gripper_state"}
        if not isinstance(execution_state, dict) or set(execution_state) != expected_state_fields:
            raise ValueError(
                "execution_state fields must be exactly "
                f"{sorted(expected_state_fields)}"
            )
        if execution_state["held_object"] not in HELD_OBJECT_VALUES:
            raise ValueError("execution_state.held_object is invalid")
        if execution_state["object_support"] not in OBJECT_SUPPORT_VALUES:
            raise ValueError("execution_state.object_support is invalid")
        if execution_state["gripper_state"] not in GRIPPER_STATE_VALUES:
            raise ValueError("execution_state.gripper_state is invalid")
    if require_completion_check:
        validate_completion_check(
            response["completion_check"], active_completion_criterion
        )
    if require_refine_check:
        validate_completion_check(response["refine_check"], active_refine_criterion)
    if require_target_identity:
        validate_target_identity(
            response["target_identity"],
            allowed_recent_front_ids=allowed_recent_front_ids,
        )
    if require_binding_updates:
        updates, metadata = sanitize_binding_updates(
            response["binding_updates"],
            allowed_front_ids=allowed_recent_front_ids,
        )
        if metadata["rejected"] or updates != response["binding_updates"]:
            raise ValueError(
                "binding_updates must contain at most three unique, directly evidenced "
                f"bindings; details={metadata}"
            )
    for key, limit in (
        ("recent_outcome", 320),
        ("updated_progress", 1600 if require_recent_status else 1200),
        ("current_subtask_template", 240),
    ):
        if not isinstance(response[key], str) or not response[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
        if len(response[key]) > limit:
            raise ValueError(f"{key} exceeds {limit} characters")

    points = response["points"]
    if not isinstance(points, list) or len(points) > 2:
        raise ValueError("points must be a list containing at most two entries")
    point_ids: set[str] = set()
    point_fields = {"id", "object_description", "source_frame_id", "x_norm", "y_norm"}
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != point_fields:
            raise ValueError(f"points[{index}] has invalid fields")
        point_id = point["id"]
        if point_id not in {"point_1", "point_2"} or point_id in point_ids:
            raise ValueError(f"points[{index}].id is invalid or duplicated: {point_id}")
        point_ids.add(point_id)
        if point["source_frame_id"] not in allowed_recent_front_ids:
            raise ValueError(
                f"points[{index}].source_frame_id is not an attached recent front frame"
            )
        if not isinstance(point["object_description"], str) or not point[
            "object_description"
        ].strip():
            raise ValueError(f"points[{index}].object_description must be non-empty")
        for coordinate in ("x_norm", "y_norm"):
            value = point[coordinate]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
                raise ValueError(
                    f"points[{index}].{coordinate} must be an integer in [0, 999]"
                )

    template_ids = set(
        re.findall(r"<(point_[0-9]+)>", response["current_subtask_template"])
    )
    if template_ids != point_ids:
        raise ValueError(
            "current_subtask_template placeholders must match points exactly: "
            f"template={sorted(template_ids)}, points={sorted(point_ids)}"
        )


class CodexTextProgressPlanner:
    """Maintain one compact text state and attach only the newest image chunk."""

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        prompt_profile: str = DEFAULT_PROMPT_PROFILE,
        system_prompt_path: Path | None = None,
        user_prompt_path: Path | None = None,
        output_schema_path: Path | None = None,
        rethinking_threshold: int = RETHINKING_THRESHOLD,
        exploration_patience: int = EXPLORATION_PATIENCE,
        candidate_stagnant_limit: int = CANDIDATE_STAGNANT_LIMIT,
        max_exploration_candidates: int = MAX_EXPLORATION_CANDIDATES,
        retrieval_scheme: str = "R0",
        retrieval_top_k: int = 5,
        timeout_seconds: float = 900.0,
        keep_recent_inputs: bool = False,
        verify_login: bool = True,
        inference_provider: str = "codex",
        openai_base_url: str | None = None,
        openai_api_key: str | None = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        thinking_budget_tokens: int | None = None,
        thinking_finalizer_max_output_tokens: int = 4096,
        runtime_experience_memory_path: Path | None = None,
        runtime_lesson_memory_path: Path | None = None,
        refine_check_criteria_path: Path | None = None,
        runtime_intervention_bundle_path: Path | None = None,
    ) -> None:
        if prompt_profile not in PROMPT_PROFILES:
            raise ValueError(
                f"Unknown prompt_profile {prompt_profile!r}; expected {sorted(PROMPT_PROFILES)}"
            )
        profile = PROMPT_PROFILES[prompt_profile]
        self.runtime_intervention_bundle_path = (
            runtime_intervention_bundle_path.resolve()
            if runtime_intervention_bundle_path is not None
            else None
        )
        self.runtime_intervention_plan = (
            load_runtime_intervention_bundle(self.runtime_intervention_bundle_path)
            if self.runtime_intervention_bundle_path is not None
            else None
        )
        if (
            self.runtime_intervention_plan is not None
            and prompt_profile != "self_evolve_v4_fullstack_trainval"
        ):
            raise ValueError(
                "Runtime intervention bundles require the isolated "
                "self_evolve_v4_fullstack_trainval profile"
            )
        if self.runtime_intervention_plan is not None and (
            self.runtime_intervention_plan.python_plugins
        ):
            raise ValueError(
                "Python intervention plugins must be registered and tested in the "
                "v4 runtime before their manifest can be executed"
            )
        self.codex_bin = codex_bin
        self.model = model
        self.reasoning_effort = reasoning_effort
        if inference_provider not in {"codex", "local_openai"}:
            raise ValueError(
                "inference_provider must be 'codex' or 'local_openai'; "
                f"got {inference_provider!r}"
            )
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if thinking_budget_tokens is not None and thinking_budget_tokens < 1:
            raise ValueError("thinking_budget_tokens must be positive when set")
        if thinking_finalizer_max_output_tokens < 1:
            raise ValueError("thinking_finalizer_max_output_tokens must be positive")
        self.inference_provider = inference_provider
        self.openai_base_url = openai_base_url
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self.thinking_finalizer_max_output_tokens = (
            thinking_finalizer_max_output_tokens
        )
        self.prompt_profile = prompt_profile
        self.decision_identity_isolation = (
            prompt_profile in IDENTITY_ISOLATED_PROMPT_PROFILES
        )
        self.harness_version = str(profile["harness_version"])
        self.require_recent_status = bool(profile["require_recent_status"])
        self.enable_lessons = bool(profile.get("enable_lessons", False))
        self.enable_oracle_skills = bool(profile.get("enable_oracle_skills", False))
        self.enable_global_retrieval = bool(
            profile.get("enable_global_retrieval", False)
        )
        self.catalog_retrieval_only = bool(
            profile.get("catalog_retrieval_only", False)
        )
        self.enable_action_success_criteria = bool(
            profile.get("enable_action_success_criteria", False)
        )
        self.enable_refine_check = bool(profile.get("enable_refine_check", False))
        self.enable_completion_evidence_latch = bool(
            profile.get("enable_completion_evidence_latch", False)
        )
        self.enable_target_identity_gate = bool(
            profile.get("enable_target_identity_gate", False)
        )
        self.require_binding_updates = bool(
            profile.get("require_binding_updates", False)
        )
        self.enable_action_completion_latches = bool(
            profile.get("enable_action_completion_latches", False)
        )
        self.enable_highlight_target_ledger = bool(
            profile.get("enable_highlight_target_ledger", False)
        )
        self.enable_swing_side_grounding_guard = bool(
            profile.get("enable_swing_side_grounding_guard", False)
        )
        self.enable_counting_event_ledger_guard = bool(
            profile.get("enable_counting_event_ledger_guard", False)
        )
        self.enable_counting_explicit_first_pickup = bool(
            profile.get("enable_counting_explicit_first_pickup", False)
        )
        self.enable_counting_exact_pickup_ordinal = bool(
            profile.get("enable_counting_exact_pickup_ordinal", False)
        )
        self.enable_patternlock_all_directions = False
        self.enable_pickhighlight_all_pickup_templates = False
        self.enable_video_repick_event_ledger_guard = False
        self.video_repick_grounding_recovery_config: dict[str, int] | None = None
        self.unmask_pickup_grounding_recovery_config: dict[str, int] | None = None
        self.button_swap_press_grounding_recovery_config: (
            dict[str, int] | None
        ) = None
        self.pickhighlight_pickup_grounding_recovery_config: (
            dict[str, int] | None
        ) = None
        if self.runtime_intervention_plan is not None:
            self.enable_swing_side_grounding_guard |= (
                self.runtime_intervention_plan.swing_gate_experience is not None
            )
            self.enable_counting_event_ledger_guard |= (
                self.runtime_intervention_plan.counting_ledger
            )
            self.enable_counting_explicit_first_pickup |= (
                self.runtime_intervention_plan.counting_explicit_first_pickup
            )
            self.enable_counting_exact_pickup_ordinal |= (
                self.runtime_intervention_plan.counting_exact_pickup_ordinal
            )
            self.enable_patternlock_all_directions |= (
                self.runtime_intervention_plan.patternlock_all_directions
            )
            self.enable_pickhighlight_all_pickup_templates |= (
                self.runtime_intervention_plan.pickhighlight_all_pickup_templates
            )
            self.enable_video_repick_event_ledger_guard |= (
                self.runtime_intervention_plan.video_repick_ledger
            )
            self.video_repick_grounding_recovery_config = (
                self.runtime_intervention_plan.video_repick_grounding_recovery
            )
            self.unmask_pickup_grounding_recovery_config = (
                self.runtime_intervention_plan.unmask_pickup_grounding_recovery
            )
            self.button_swap_press_grounding_recovery_config = (
                self.runtime_intervention_plan.button_swap_press_grounding_recovery
            )
            self.pickhighlight_pickup_grounding_recovery_config = (
                self.runtime_intervention_plan.pickhighlight_pickup_grounding_recovery
            )
        self.action_success_criteria_path = Path(
            profile.get("action_success_criteria_path", ACTION_SUCCESS_CRITERIA_PATH)
        ).resolve()
        self.action_success_catalog = (
            load_action_success_criteria(self.action_success_criteria_path)
            if self.enable_action_success_criteria
            else None
        )
        self.refine_check_criteria_path = Path(
            refine_check_criteria_path
            or profile.get("refine_check_criteria_path", ACTION_GATES_V8_4_6_CRITERIA_PATH)
        ).resolve()
        self.refine_check_catalog = (
            load_action_success_criteria(self.refine_check_criteria_path)
            if self.enable_refine_check
            else None
        )
        if (
            self.runtime_intervention_plan is not None
            and self.runtime_intervention_plan.criterion_overlay_paths
        ):
            if self.refine_check_catalog is None:
                raise ValueError("V4 criterion overlays require refine checks")
            self.refine_check_catalog = apply_action_success_criteria_overlays(
                self.refine_check_catalog,
                self.runtime_intervention_plan.criterion_overlay_paths,
            )
        if self.runtime_intervention_plan is not None:
            criteria_by_id = {
                row["criterion_id"]: row
                for row in (self.refine_check_catalog or {}).get("criteria", [])
            }
            for criterion_id, gate in (
                self.runtime_intervention_plan.transition_gates.items()
            ):
                criterion = criteria_by_id.get(criterion_id)
                if criterion is None:
                    raise ValueError(
                        f"V4 transition gate references missing criterion {criterion_id}"
                    )
                criterion_tasks = set(criterion.get("applies_to_tasks") or [])
                if not set(gate["task_scope"]).issubset(criterion_tasks):
                    raise ValueError(
                        f"V4 transition gate task scope conflicts with {criterion_id}"
                    )
            if (
                self.runtime_intervention_plan.swing_gate_experience is not None
                and "swing_target_side_visual_binding" not in criteria_by_id
            ):
                raise ValueError(
                    "V4 Swing visual binding gate requires its criterion overlay"
                )
            if self.runtime_intervention_plan.counting_ledger:
                bundle_tasks = set(
                    self.runtime_intervention_plan.bundle["task_scope"]
                )
                required_counting_criteria = {"counting_requested_cube_pickup"}
                if "PickXtimes" in bundle_tasks:
                    required_counting_criteria.add("pickxtimes_verified_target_release")
                if "BinFill" in bundle_tasks:
                    required_counting_criteria.add("binfill_verified_bin_release")
                missing = required_counting_criteria - set(criteria_by_id)
                if missing:
                    raise ValueError(
                        "V4 counting ledger requires criterion overlays "
                        f"{sorted(missing)}"
                    )
            if self.runtime_intervention_plan.video_repick_ledger:
                required_repick_criteria = {
                    "videorepick_verified_target_pickup",
                    "videorepick_verified_table_release",
                    "videorepick_verified_finish_button",
                }
                missing = required_repick_criteria - set(criteria_by_id)
                if missing:
                    raise ValueError(
                        "VideoRepick ledger requires criterion overlays "
                        f"{sorted(missing)}"
                    )
        self.reinforcement_lock_config = {
            str(key): int(value)
            for key, value in profile.get("reinforcement_lock", {}).items()
        }
        if self.runtime_intervention_plan is not None:
            for key, value in (
                self.runtime_intervention_plan.reinforcement_dispatch_counts.items()
            ):
                if key in self.reinforcement_lock_config:
                    raise ValueError(
                        f"V4 reinforcement overlay conflicts with base key {key!r}"
                    )
                self.reinforcement_lock_config[key] = value
        if any(value < 1 for value in self.reinforcement_lock_config.values()):
            raise ValueError(
                "reinforcement lock command-dispatch counts must be positive"
            )
        runtime_memory_path_value = (
            runtime_experience_memory_path
            if runtime_experience_memory_path is not None
            else profile.get("runtime_experience_memory_path")
        )
        self.runtime_experience_memory_path = (
            Path(runtime_memory_path_value).resolve()
            if runtime_memory_path_value is not None
            else None
        )
        self.runtime_experience_memory: dict[str, Any] | None = None
        if self.runtime_experience_memory_path is not None:
            if self.refine_check_catalog is None:
                raise ValueError("Runtime experience memory requires refine checks")
            (
                self.runtime_experience_memory,
                self.experience_transition_gate_config,
            ) = load_runtime_experience_memory(
                self.runtime_experience_memory_path,
                refine_catalog=self.refine_check_catalog,
            )
        else:
            self.experience_transition_gate_config = {
                str(key): dict(value)
                for key, value in profile.get(
                    "experience_transition_gate", {}
                ).items()
            }
        if self.runtime_intervention_plan is not None:
            for criterion_id, gate in (
                self.runtime_intervention_plan.transition_gates.items()
            ):
                if criterion_id in self.experience_transition_gate_config:
                    raise ValueError(
                        f"V4 transition gate conflicts on criterion {criterion_id}"
                    )
                self.experience_transition_gate_config[criterion_id] = dict(gate)
            swing = self.runtime_intervention_plan.swing_gate_experience
            if swing is not None:
                swing_key = "swing_target_side_visual_binding"
                if swing_key in self.experience_transition_gate_config:
                    raise ValueError("V4 Swing visual binding gate is duplicated")
                self.experience_transition_gate_config[swing_key] = {
                    **swing,
                    "task_scope": ["SwingXtimes"],
                    "situation_class": "swing_target_side_visual_binding",
                    "max_blocked_transitions": 1,
                }
        self.runtime_lesson_memory_path = (
            Path(runtime_lesson_memory_path).resolve()
            if runtime_lesson_memory_path is not None
            else None
        )
        self.runtime_lesson_memory = (
            load_runtime_lesson_memory(self.runtime_lesson_memory_path)
            if self.runtime_lesson_memory_path is not None
            else None
        )
        if (
            self.runtime_intervention_plan is not None
            and self.runtime_intervention_plan.prompt_experiences
        ):
            if self.runtime_lesson_memory is not None:
                raise ValueError(
                    "Use either a v4 intervention bundle or legacy runtime lesson memory"
                )
            self.runtime_lesson_memory = {
                "schema_version": "self-evolve-runtime-lessons-v2",
                "memory_id": self.runtime_intervention_plan.bundle["bundle_id"],
                "revision": self.runtime_intervention_plan.bundle["revision"],
                "base_profile": "retrieval_v8_5_refine_then_lock",
                "experiences": self.runtime_intervention_plan.prompt_experiences,
            }
        for criterion_id, config in self.experience_transition_gate_config.items():
            if (
                not isinstance(config.get("experience_id"), str)
                or not config["experience_id"]
                or not isinstance(config.get("revision"), int)
                or config["revision"] < 1
                or not isinstance(config.get("max_blocked_transitions"), int)
                or config["max_blocked_transitions"] < 1
            ):
                raise ValueError(
                    f"Invalid experience transition gate for {criterion_id}: {config}"
                )
        self.completion_confirmation_retries = {
            str(key): int(value)
            for key, value in profile.get("completion_confirmation_retries", {}).items()
        }
        if any(value < 1 for value in self.completion_confirmation_retries.values()):
            raise ValueError("completion confirmation retry counts must be positive")
        self.one_shot_completion_reinforcement = {
            str(key): int(value)
            for key, value in profile.get(
                "one_shot_completion_reinforcement", {}
            ).items()
        }
        if any(value < 1 for value in self.one_shot_completion_reinforcement.values()):
            raise ValueError(
                "completion reinforcement command-dispatch budgets must be positive"
            )
        self.system_prompt_path = (system_prompt_path or profile["system_prompt"]).resolve()
        self.user_prompt_path = (user_prompt_path or profile["user_prompt"]).resolve()
        self.output_schema_path = (output_schema_path or profile["output_schema"]).resolve()
        if rethinking_threshold < 1:
            raise ValueError("rethinking_threshold must be positive")
        if exploration_patience < 1:
            raise ValueError("exploration_patience must be positive")
        if candidate_stagnant_limit < 1:
            raise ValueError("candidate_stagnant_limit must be positive")
        if max_exploration_candidates < 1:
            raise ValueError("max_exploration_candidates must be positive")
        self.rethinking_threshold = rethinking_threshold
        self.exploration_patience = exploration_patience
        self.candidate_stagnant_limit = candidate_stagnant_limit
        self.max_exploration_candidates = max_exploration_candidates
        if retrieval_scheme not in RETRIEVAL_SCHEMES:
            raise ValueError(
                f"retrieval_scheme must be one of {RETRIEVAL_SCHEMES}; "
                f"got {retrieval_scheme!r}"
            )
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        self.retrieval_scheme = retrieval_scheme
        self.retrieval_top_k = retrieval_top_k
        self.skill_retriever = (
            ExecutorSkillRetriever(GLOBAL_VOCABULARY_PATH)
            if self.enable_global_retrieval
            else None
        )
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        self.user_prompt_template = self.user_prompt_path.read_text(encoding="utf-8")
        self.rollout_behavior_id = None
        if (
            self.prompt_profile in V85_EMPTY_INTERVENTION_PROFILES
            and self.runtime_intervention_plan is None
            and self.runtime_experience_memory is None
            and self.runtime_lesson_memory is None
            and not self.experience_transition_gate_config
            and self.system_prompt_path == Path(profile["system_prompt"]).resolve()
            and self.user_prompt_path == Path(profile["user_prompt"]).resolve()
            and self.output_schema_path == Path(profile["output_schema"]).resolve()
            and (
                not self.enable_refine_check
                or self.refine_check_criteria_path
                == Path(profile["refine_check_criteria_path"]).resolve()
            )
        ):
            self.rollout_behavior_id = V85_EMPTY_INTERVENTION_BEHAVIOR_ID
        self.timeout_seconds = timeout_seconds
        self.keep_recent_inputs = keep_recent_inputs
        if self.inference_provider == "codex":
            self.environment = sanitized_codex_environment()
            self.codex_version = get_codex_version(codex_bin, self.environment)
            self.provider_version = self.codex_version
            self.login_status = (
                require_chatgpt_login(codex_bin, self.environment)
                if verify_login
                else "not checked"
            )
            self.inference_runner = run_codex_inference
            self.provider_authentication = "local ChatGPT login"
        else:
            if not openai_base_url:
                raise ValueError(
                    "openai_base_url is required for inference_provider='local_openai'"
                )
            self.environment = {}
            self.provider_version = get_openai_compatible_version(
                base_url=openai_base_url,
                api_key=openai_api_key,
                timeout_seconds=min(timeout_seconds, 30.0),
            )
            self.codex_version = None
            self.login_status = "not applicable"
            self.inference_runner = partial(
                run_openai_compatible_inference,
                base_url=openai_base_url,
                api_key=openai_api_key,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                thinking_finalizer_max_output_tokens=(
                    thinking_finalizer_max_output_tokens
                ),
            )
            self.provider_authentication = (
                "bearer token" if openai_api_key else "none"
            )
        self.run_dir: Path | None = None
        self.detailed_task_description = ""
        self.task = ""
        self.round_index = 0
        self.progress = INITIAL_PROGRESS
        self.previous_subtask = INITIAL_PREVIOUS_SUBTASK
        self.previous_subtask_template: str | None = None
        self.previous_skill_key: str | None = None
        self.previous_execution_state: dict[str, str] | None = None
        self.planner_guidance: dict[str, Any] = {}
        self.oracle_skill_method: str | None = None
        self.allowed_executor_skills: dict[str, dict[str, Any]] = {}
        self.current_retrieval_candidates: list[dict[str, Any]] = []
        self.current_action_success_criterion: dict[str, Any] | None = None
        self.current_refine_criterion: dict[str, Any] | None = None
        self.completion_confirmation_counts: dict[str, int] = {}
        self.completion_evidence_states: dict[str, dict[str, Any]] = {}
        self.active_completion_evidence_key: str | None = None
        self.target_identity_bindings: dict[str, dict[str, Any]] = {}
        self.action_completion_latches: dict[str, dict[str, Any]] = {}
        self.action_instance_sequence = 0
        self.active_action_instance_signature: str | None = None
        self.active_action_instance_id: str | None = None
        self.action_reinforcement_states: dict[str, dict[str, Any]] = {}
        self.refine_action_instance_sequence = 0
        self.active_refine_action_instance_signature: str | None = None
        self.active_refine_action_instance_id: str | None = None
        self.refinement_lock_states: dict[str, dict[str, Any]] = {}
        self.experience_gate_states: dict[str, dict[str, Any]] = {}
        self.highlight_target_instances: dict[str, dict[str, Any]] = {}
        self.active_highlight_target_instance_id: str | None = None
        self.highlight_recovery_target_instance_id: str | None = None
        self.highlight_wrong_target_events: list[dict[str, Any]] = []
        self.counting_event_ledger_state: dict[str, Any] | None = None
        self.video_repick_event_ledger_state: dict[str, Any] | None = None
        self.video_repick_grounding_recovery_state: dict[str, Any] | None = None
        self.unmask_pickup_grounding_recovery_state: dict[str, Any] | None = None
        self.button_swap_press_grounding_recovery_state: (
            dict[str, Any] | None
        ) = None
        self.pickhighlight_pickup_grounding_recovery_state: (
            dict[str, Any] | None
        ) = None
        self.observed_front_frame_ids: set[str] = set()
        self.previous_recent_status = "initial"
        self.consecutive_no_progress_rounds = 0
        self.consecutive_stagnant_rounds = 0
        self.exploration_active = False
        self.exploration_skill_key: str | None = None
        self.active_candidate: dict[str, Any] | None = None
        self.lesson_memory: list[dict[str, Any]] = []
        self.exploration_events: list[dict[str, Any]] = []
        self.progress_history: list[dict[str, Any]] = []

    def start_episode(
        self,
        *,
        run_dir: Path,
        detailed_task_description: str,
        task: str,
        episode: int | None = None,
        init_case_id: str | None = None,
        planner_guidance: dict[str, Any] | None = None,
    ) -> None:
        description = detailed_task_description.strip()
        if not description:
            raise ValueError("detailed_task_description must be non-empty")
        if planner_guidance is not None and not isinstance(planner_guidance, dict):
            raise ValueError("planner_guidance must be an object when provided")
        guidance = planner_guidance or {}
        if self.decision_identity_isolation:
            if episode is not None or init_case_id is not None:
                raise ValueError(
                    "The self-evolve decision planner must not receive episode or "
                    "init_case_id"
                )
            assert_prompt_has_no_episode_identity(
                description, source="detailed_task_description"
            )
            assert_runtime_experience_has_no_episode_identity(
                guidance, source="planner_guidance"
            )
            assert_prompt_has_no_episode_identity(
                json.dumps(guidance, ensure_ascii=False),
                source="planner_guidance",
            )
        if self.runtime_intervention_plan is not None and task not in set(
            self.runtime_intervention_plan.bundle["task_scope"]
        ):
            raise ValueError(
                f"Runtime intervention bundle does not apply to task {task!r}"
            )
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty planner run directory: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "rounds").mkdir()
        self.run_dir = run_dir
        self.task = task
        self.detailed_task_description = description
        self.planner_guidance = guidance
        self.oracle_skill_method = None
        self.allowed_executor_skills = {}
        self.current_retrieval_candidates = []
        self.current_action_success_criterion = None
        self.current_refine_criterion = None
        self.completion_confirmation_counts = {}
        self.completion_evidence_states = {}
        self.active_completion_evidence_key = None
        self.target_identity_bindings = {}
        self.action_completion_latches = {}
        self.action_instance_sequence = 0
        self.active_action_instance_signature = None
        self.active_action_instance_id = None
        self.action_reinforcement_states = {}
        self.refine_action_instance_sequence = 0
        self.active_refine_action_instance_signature = None
        self.active_refine_action_instance_id = None
        self.refinement_lock_states = {}
        self.experience_gate_states = {}
        self.highlight_target_instances = {}
        self.active_highlight_target_instance_id = None
        self.highlight_recovery_target_instance_id = None
        self.highlight_wrong_target_events = []
        self.counting_event_ledger_state = None
        self.video_repick_event_ledger_state = None
        self.video_repick_grounding_recovery_state = None
        self.unmask_pickup_grounding_recovery_state = None
        self.button_swap_press_grounding_recovery_state = None
        self.pickhighlight_pickup_grounding_recovery_state = None
        self.observed_front_frame_ids = set()
        self.previous_recent_status = "initial"
        if self.enable_counting_event_ledger_guard and task in {
            "PickXtimes",
            "BinFill",
        }:
            requested = counting_requirements_from_guidance(task, guidance)
            self.counting_event_ledger_state = {
                "requested": requested,
                "completed": {color: 0 for color in requested},
                "active_color": None,
                "pickup_verified": False,
                "integrated_first_pending": False,
                "integrated_first_consumed": False,
                "explicit_first_pickup_required": (
                    self.enable_counting_explicit_first_pickup
                    and task == "PickXtimes"
                ),
                "exact_pickup_ordinal_required": (
                    self.enable_counting_exact_pickup_ordinal
                ),
                "counted_action_instances": [],
            }
        if self.enable_video_repick_event_ledger_guard and task == "VideoRepick":
            self.video_repick_event_ledger_state = {
                "requested": video_repick_required_repetitions(description, guidance),
                "completed": 0,
                "active_ordinal": None,
                "pickup_verified": False,
                "counted_release_instances": [],
            }
        if self.video_repick_grounding_recovery_config is not None:
            if task != "VideoRepick":
                raise ValueError("VideoRepick grounding recovery escaped task scope")
            self.video_repick_grounding_recovery_state = {
                "ordinal": None,
                "consecutive_failed_pickups": 0,
                "last_dispatched_point_norm": None,
                "forced_regrounds": 0,
            }
        if self.unmask_pickup_grounding_recovery_config is not None:
            if task not in {"ButtonUnmask", "ButtonUnmaskSwap"}:
                raise ValueError("Unmask pickup grounding recovery escaped task scope")
            self.unmask_pickup_grounding_recovery_state = {
                "target_color": None,
                "consecutive_failed_pickups": 0,
                "last_dispatched_point_norm": None,
                "recent_failed_points_norm": [],
                "forced_regrounds": 0,
            }
        if self.button_swap_press_grounding_recovery_config is not None:
            if task != "ButtonUnmaskSwap":
                raise ValueError(
                    "ButtonUnmaskSwap press grounding recovery escaped task scope"
                )
            self.button_swap_press_grounding_recovery_state = {
                "completed_points_norm": [],
                "counted_action_instances": [],
                "active_ordinal": None,
                "last_dispatched_point_norm": None,
                "recent_failed_points_norm": [],
                "consecutive_failed_presses": 0,
                "forced_regrounds": 0,
            }
        if self.pickhighlight_pickup_grounding_recovery_config is not None:
            if task != "PickHighlight":
                raise ValueError(
                    "PickHighlight pickup grounding recovery escaped task scope"
                )
            self.pickhighlight_pickup_grounding_recovery_state = {
                "target_signature": None,
                "consecutive_failed_pickups": 0,
                "last_dispatched_point_norm": None,
                "recent_failed_points_norm": [],
                "forced_regrounds": 0,
            }
        if self.enable_oracle_skills:
            if task != "MoveCube":
                raise ValueError(
                    "oracle_skills_v5 is currently a MoveCube pilot profile; "
                    f"got task={task!r}"
                )
            catalog = load_oracle_skill_catalog()
            self.oracle_skill_method = select_move_cube_method(
                description, self.planner_guidance
            )
            method_config = catalog["methods"][self.oracle_skill_method]
            skills = method_config.get("skills")
            if not isinstance(skills, list) or not skills:
                raise ValueError(
                    f"No oracle skills for MoveCube method {self.oracle_skill_method}"
                )
            self.allowed_executor_skills = {
                str(skill["skill_id"]): dict(skill) for skill in skills
            }
        self.round_index = 0
        self.progress = INITIAL_PROGRESS
        self.previous_subtask = INITIAL_PREVIOUS_SUBTASK
        self.previous_subtask_template = None
        self.previous_skill_key = None
        self.previous_execution_state = None
        self.consecutive_no_progress_rounds = 0
        self.consecutive_stagnant_rounds = 0
        self.exploration_active = False
        self.exploration_skill_key = None
        self.active_candidate = None
        self.lesson_memory = []
        self.exploration_events = []
        self.progress_history = []
        episode_config = {
                "harness_version": self.harness_version,
                "prompt_profile": self.prompt_profile,
                "rollout_behavior_id": self.rollout_behavior_id,
                "memory_mode": "text_progress",
                "task": task,
                "decision_identity_isolation": self.decision_identity_isolation,
                "detailed_task_description": description,
                "planner_guidance": self.planner_guidance,
                "rethinking_threshold": self.rethinking_threshold,
                "exploration_patience": self.exploration_patience,
                "candidate_stagnant_limit": self.candidate_stagnant_limit,
                "max_exploration_candidates": self.max_exploration_candidates,
                "episode_local_lessons_enabled": self.enable_lessons,
                "oracle_skill_catalog_enabled": self.enable_oracle_skills,
                "oracle_skill_catalog_version": (
                    load_oracle_skill_catalog().get("version")
                    if self.enable_oracle_skills
                    else None
                ),
                "oracle_skill_method": self.oracle_skill_method,
                "allowed_executor_skills": list(
                    self.allowed_executor_skills.values()
                ),
                "global_retrieval_enabled": self.enable_global_retrieval,
                "retrieval_scheme": (
                    self.retrieval_scheme if self.enable_global_retrieval else None
                ),
                "retrieval_top_k": (
                    self.retrieval_top_k if self.enable_global_retrieval else None
                ),
                "vocabulary_version": (
                    self.skill_retriever.vocabulary.get("version")
                    if self.skill_retriever is not None
                    else None
                ),
                "vocabulary_entry_count": (
                    len(self.skill_retriever.entries)
                    if self.skill_retriever is not None
                    else None
                ),
                "novel_instruction_exploration_enabled": not self.catalog_retrieval_only,
                "action_success_criteria_enabled": self.enable_action_success_criteria,
                "action_success_criteria_version": (
                    self.action_success_catalog.get("version")
                    if self.action_success_catalog is not None
                    else None
                ),
                "action_success_criterion_ids": (
                    [
                        item["criterion_id"]
                        for item in self.action_success_catalog["criteria"]
                    ]
                    if self.action_success_catalog is not None
                    else []
                ),
                "completion_confirmation_retries": self.completion_confirmation_retries,
                "one_shot_completion_reinforcement": (
                    self.one_shot_completion_reinforcement
                ),
                "refine_check_enabled": self.enable_refine_check,
                "refine_check_criteria_version": (
                    self.refine_check_catalog.get("version")
                    if self.refine_check_catalog is not None
                    else None
                ),
                "reinforcement_lock": self.reinforcement_lock_config,
                "experience_transition_gate": self.experience_transition_gate_config,
                "runtime_experience_memory": (
                    {
                        "schema_version": self.runtime_experience_memory[
                            "schema_version"
                        ],
                        "memory_id": self.runtime_experience_memory["memory_id"],
                        "revision": self.runtime_experience_memory["revision"],
                        "sha256": sha256_file(self.runtime_experience_memory_path),
                        "artifact": "runtime_experience_memory.json",
                    }
                    if self.runtime_experience_memory is not None
                    and self.runtime_experience_memory_path is not None
                    else None
                ),
                "runtime_intervention_bundle": (
                    {
                        "schema_version": self.runtime_intervention_plan.bundle[
                            "schema_version"
                        ],
                        "bundle_id": self.runtime_intervention_plan.bundle["bundle_id"],
                        "revision": self.runtime_intervention_plan.bundle["revision"],
                        "bundle_sha256": self.runtime_intervention_plan.bundle[
                            "bundle_sha256"
                        ],
                        "artifact": "runtime_intervention_bundle/runtime_intervention_bundle.json",
                    }
                    if self.runtime_intervention_plan is not None
                    else None
                ),
                "refine_check_criteria_sha256": (
                    intervention_canonical_hash(self.refine_check_catalog)
                    if self.runtime_intervention_plan is not None
                    and self.runtime_intervention_plan.criterion_overlay_paths
                    else sha256_file(self.refine_check_criteria_path)
                    if self.enable_refine_check
                    else None
                ),
                "refine_check_has_command_veto_authority": False,
                "completion_evidence_latch_enabled": self.enable_completion_evidence_latch,
                "target_identity_gate_enabled": self.enable_target_identity_gate,
                "binding_updates_required": self.require_binding_updates,
                "action_completion_latches_enabled": (
                    self.enable_action_completion_latches
                ),
                "counting_event_ledger_guard_enabled": (
                    self.enable_counting_event_ledger_guard
                ),
                "counting_event_ledger_initial_state": self.counting_event_ledger_state,
                "video_repick_event_ledger_enabled": (
                    self.enable_video_repick_event_ledger_guard
                ),
                "video_repick_event_ledger_initial_state": (
                    self.video_repick_event_ledger_state
                ),
                "video_repick_grounding_recovery": (
                    self.video_repick_grounding_recovery_config
                ),
                "unmask_pickup_grounding_recovery": (
                    self.unmask_pickup_grounding_recovery_config
                ),
                "button_swap_press_grounding_recovery": (
                    self.button_swap_press_grounding_recovery_config
                ),
                "pickhighlight_pickup_grounding_recovery": (
                    self.pickhighlight_pickup_grounding_recovery_config
                ),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "planner_provider": self.inference_provider,
                "provider_version": self.provider_version,
                "provider_base_url": self.openai_base_url,
                "codex_version": self.codex_version,
                "authentication": self.provider_authentication,
                "ephemeral_codex_turns": self.inference_provider == "codex",
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "enable_thinking": self.enable_thinking,
                "thinking_budget_tokens": self.thinking_budget_tokens,
                "thinking_finalizer_max_output_tokens": (
                    self.thinking_finalizer_max_output_tokens
                ),
                "initial_demo_images_sent_to_planner": False,
                "historical_images_sent_to_planner": False,
                "recent_frame_filtering": False,
                "created_at": utc_now(),
            }
        if (
            self.prompt_profile
            in {
                "self_evolve_v2_train_active",
                "self_evolve_v3_train30_val20",
                "self_evolve_v3_multiproblem_auto_hard20",
                "self_evolve_v4_fullstack_trainval",
            }
            or self.runtime_lesson_memory is not None
        ):
            episode_config["runtime_lesson_memory"] = (
                {
                    "schema_version": self.runtime_lesson_memory["schema_version"],
                    "memory_id": self.runtime_lesson_memory["memory_id"],
                    "revision": self.runtime_lesson_memory["revision"],
                    "sha256": (
                        sha256_file(self.runtime_lesson_memory_path)
                        if self.runtime_lesson_memory_path is not None
                        else intervention_canonical_hash(self.runtime_lesson_memory)
                    ),
                    "artifact": (
                        "runtime_lesson_memory.json"
                        if self.runtime_lesson_memory_path is not None
                        else "runtime_intervention_bundle/runtime_intervention_bundle.json"
                    ),
                }
                if self.runtime_lesson_memory is not None
                else None
            )
            episode_config["cross_episode_lesson_loading"] = (
                self.runtime_lesson_memory is not None
            )
        if not self.decision_identity_isolation:
            episode_config.update(
                {"episode": episode, "init_case_id": init_case_id}
            )
        write_json(run_dir / "episode_config.json", episode_config)
        shutil.copy2(self.system_prompt_path, run_dir / "planner_system_prompt.txt")
        shutil.copy2(self.user_prompt_path, run_dir / "planner_user_prompt_template.txt")
        shutil.copy2(self.output_schema_path, run_dir / "planner_output_schema.json")
        if self.enable_refine_check:
            if (
                self.runtime_intervention_plan is not None
                and self.runtime_intervention_plan.criterion_overlay_paths
            ):
                write_json(run_dir / "refine_check_criteria.json", self.refine_check_catalog)
            else:
                shutil.copy2(
                    self.refine_check_criteria_path,
                    run_dir / "refine_check_criteria.json",
                )
        if self.runtime_experience_memory_path is not None:
            shutil.copy2(
                self.runtime_experience_memory_path,
                run_dir / "runtime_experience_memory.json",
            )
        if self.runtime_lesson_memory_path is not None:
            shutil.copy2(
                self.runtime_lesson_memory_path,
                run_dir / "runtime_lesson_memory.json",
            )
        if self.runtime_intervention_plan is not None:
            bundle_output = run_dir / "runtime_intervention_bundle"
            bundle_output.mkdir()
            shutil.copy2(
                self.runtime_intervention_bundle_path,
                bundle_output / "runtime_intervention_bundle.json",
            )
            for component in self.runtime_intervention_plan.bundle["components"]:
                for descriptor in component["artifacts"].values():
                    relative = Path(descriptor["path"])
                    destination = bundle_output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        self.runtime_intervention_bundle_path.parent / relative,
                        destination,
                    )
        self._write_progress_history()
        if self.enable_lessons:
            self._write_lesson_state()

    def _require_started(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("start_episode must be called before plan")
        return self.run_dir

    def _write_progress_history(self) -> None:
        write_json(self._require_started() / "progress_history.json", self.progress_history)

    @staticmethod
    def _sum_inference_usage(results: list[dict[str, Any]]) -> dict[str, int] | None:
        totals: dict[str, int] = {}
        for result in results:
            usage = result.get("usage")
            if not isinstance(usage, dict):
                continue
            for key, value in usage.items():
                if isinstance(value, int):
                    totals[str(key)] = totals.get(str(key), 0) + value
        return totals or None

    @staticmethod
    def _copy_final_inference_artifacts(source: Path, destination: Path) -> None:
        for name in ("codex_events.jsonl", "codex_stderr.log", "response.txt"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)

    def _prepare_inference_response(
        self,
        *,
        raw_response: Any,
        current_front_ids: set[str],
        front_shapes: dict[str, tuple[int, int]],
        round_index: int,
        prior_requested_color: str | None,
        validation_attempt: int = 0,
    ) -> dict[str, Any]:
        """Normalize and validate one model response without mutating episode state."""
        proposed_subtask_template = (
            raw_response.get("current_subtask_template")
            if isinstance(raw_response, dict)
            else None
        )
        template_canonicalized = False
        counting_pickup_ordinal_canonicalization = {
            "enabled": self.enable_counting_exact_pickup_ordinal,
            "applicable": False,
            "canonicalized": False,
        }
        if self.enable_oracle_skills or self.enable_global_retrieval:
            if not isinstance(raw_response, dict):
                raise ValueError("Planner response must be a JSON object")
            response_for_normalization = dict(raw_response)
            skill_id = response_for_normalization.get("skill_key")
            if skill_id not in self.allowed_executor_skills:
                raise ValueError(
                    "Planner selected a skill outside the demonstrated method's oracle "
                    f"catalog: {skill_id!r}; allowed={sorted(self.allowed_executor_skills)}"
                )
            if (
                self.enable_counting_exact_pickup_ordinal
                and self.task in {"PickXtimes", "BinFill"}
            ):
                canonicalization_state = self.counting_event_ledger_state
                if canonicalization_state is not None:
                    canonicalization_state, _ = advance_counting_previous_event(
                        task=self.task,
                        state=canonicalization_state,
                        previous_subtask=self.previous_subtask,
                        response=response_for_normalization,
                        refine_criterion=self.current_refine_criterion,
                        action_instance_id=self.active_refine_action_instance_id,
                    )
                canonicalization_skills = dict(self.allowed_executor_skills)
                if canonicalization_state is not None:
                    post_event_candidates = ensure_counting_next_pickup_candidates(
                        self.current_retrieval_candidates,
                        retriever=self.skill_retriever,
                        task=self.task,
                        state=canonicalization_state,
                    )
                    canonicalization_skills.update(
                        {
                            candidate["template_id"]: candidate
                            for candidate in post_event_candidates
                        }
                    )
                skill_id, counting_pickup_ordinal_canonicalization = (
                    canonicalize_counting_pickup_skill(
                        str(skill_id),
                        allowed_skills=canonicalization_skills,
                        task=self.task,
                        state=canonicalization_state,
                    )
                )
            canonical_template = self.allowed_executor_skills[skill_id][
                "canonical_template"
            ]
            template_canonicalized = proposed_subtask_template != canonical_template
            response_for_normalization["current_subtask_template"] = canonical_template
        else:
            response_for_normalization = raw_response

        response, response_normalization = normalize_response(
            response_for_normalization
        )
        target_identity_frame_sanitization = None
        binding_updates_sanitization = None
        completion_check_sanitization = None
        if self.enable_action_success_criteria and isinstance(response, dict):
            response = dict(response)
            (
                response["completion_check"],
                completion_check_sanitization,
            ) = sanitize_duplicate_completion_checks(
                response.get("completion_check"),
                self.current_action_success_criterion,
            )
        refine_check_sanitization = None
        if self.enable_refine_check and isinstance(response, dict):
            response = dict(response)
            (
                response["refine_check"],
                refine_check_sanitization,
            ) = sanitize_duplicate_completion_checks(
                response.get("refine_check"),
                self.current_refine_criterion,
            )
        if self.enable_target_identity_gate and isinstance(response, dict):
            response = dict(response)
            (
                response["target_identity"],
                target_identity_frame_sanitization,
            ) = sanitize_target_identity_frame_ids(
                response.get("target_identity", {}),
                allowed_front_ids=current_front_ids,
            )
        if self.require_binding_updates and isinstance(response, dict):
            response = dict(response)
            (
                response["binding_updates"],
                binding_updates_sanitization,
            ) = sanitize_binding_updates(
                response.get("binding_updates", []),
                allowed_front_ids=current_front_ids,
            )

        validate_response(
            response,
            allowed_recent_front_ids=current_front_ids,
            require_recent_status=self.require_recent_status,
            require_lesson_fields=self.enable_lessons,
            require_completion_check=self.enable_action_success_criteria,
            require_target_identity=self.enable_target_identity_gate,
            require_binding_updates=self.require_binding_updates,
            active_completion_criterion=self.current_action_success_criterion,
            require_refine_check=self.enable_refine_check,
            active_refine_criterion=self.current_refine_criterion,
        )

        target_identity_color_canonicalized = False
        if self.enable_target_identity_gate and prior_requested_color is not None:
            identity = response["target_identity"]
            if identity["requested_color"] in {"none", "unknown"}:
                response = dict(response)
                identity = dict(identity)
                identity["requested_color"] = prior_requested_color
                if (
                    identity["revealed_color"] == prior_requested_color
                    and identity["same_as_bound_target"] == "yes"
                    and identity["evidence_frame_ids"]
                ):
                    identity["binding_status"] = "verified"
                    identity["binding_basis"] = "revealed_color"
                response["target_identity"] = identity
                target_identity_color_canonicalized = True

        if self.require_recent_status:
            status = response["recent_status"]
            if round_index == 0 and status != "initial":
                raise ValueError("recent_status must be 'initial' on planner round 0")
            if round_index > 0 and status == "initial":
                raise ValueError("recent_status may only be 'initial' on planner round 0")
        else:
            status = None

        rendered_subtask, rendered_points = render_grounded_subtask(
            response, front_shapes
        )
        swing_side_grounding_guard = {
            "enabled": self.enable_swing_side_grounding_guard,
            "applicable": False,
            "experience_id": None,
            "experience_revision": None,
            "decision": "disabled",
        }
        if self.enable_swing_side_grounding_guard and self.task == "SwingXtimes":
            experience = self.experience_transition_gate_config.get(
                "swing_target_side_visual_binding"
            )
            if experience is None:
                raise ValueError(
                    "Swing side grounding guard requires the retrieved "
                    "swing_target_side_visual_binding experience"
                )
            swing_side_grounding_guard = validate_swing_side_grounding(
                response,
                rendered_points,
                experience_id=str(experience["experience_id"]),
                experience_revision=int(experience["revision"]),
            )
        counting_event_ledger_guard = {
            "enabled": self.enable_counting_event_ledger_guard,
            "applicable": False,
            "decision": "disabled",
        }
        proposed_counting_state = self.counting_event_ledger_state
        if (
            self.enable_counting_event_ledger_guard
            and self.task in {"PickXtimes", "BinFill"}
        ):
            if self.counting_event_ledger_state is None:
                raise ValueError("Counting event ledger was not initialized")
            proposed_counting_state, counting_event_ledger_guard = (
                evaluate_counting_event_ledger(
                    task=self.task,
                    state=self.counting_event_ledger_state,
                    previous_subtask=self.previous_subtask,
                    proposed_subtask=rendered_subtask,
                    response=response,
                    refine_criterion=self.current_refine_criterion,
                    action_instance_id=self.active_refine_action_instance_id,
                )
            )
            counting_event_ledger_guard["applicable"] = True
        video_repick_event_ledger_guard = {
            "enabled": self.enable_video_repick_event_ledger_guard,
            "applicable": False,
            "decision": "disabled",
        }
        proposed_video_repick_state = self.video_repick_event_ledger_state
        if self.enable_video_repick_event_ledger_guard and self.task == "VideoRepick":
            if self.video_repick_event_ledger_state is None:
                raise ValueError("VideoRepick event ledger was not initialized")
            (
                proposed_video_repick_state,
                video_repick_event_ledger_guard,
            ) = evaluate_video_repick_event_ledger(
                state=self.video_repick_event_ledger_state,
                previous_subtask=self.previous_subtask,
                proposed_subtask=rendered_subtask,
                response=response,
                refine_criterion=self.current_refine_criterion,
                action_instance_id=self.active_refine_action_instance_id,
            )
        video_repick_grounding_recovery_guard = {
            "enabled": self.video_repick_grounding_recovery_config is not None,
            "applicable": False,
            "decision": "disabled",
        }
        proposed_video_repick_grounding_state = (
            self.video_repick_grounding_recovery_state
        )
        if self.video_repick_grounding_recovery_config is not None:
            if self.video_repick_grounding_recovery_state is None:
                raise ValueError("VideoRepick grounding recovery was not initialized")
            (
                proposed_video_repick_grounding_state,
                video_repick_grounding_recovery_guard,
            ) = evaluate_video_repick_grounding_recovery(
                state=self.video_repick_grounding_recovery_state,
                previous_subtask=self.previous_subtask,
                proposed_subtask=rendered_subtask,
                proposed_points=rendered_points,
                response=response,
                refine_criterion=self.current_refine_criterion,
                config=self.video_repick_grounding_recovery_config,
                allow_retry_budget_bypass=(
                    validation_attempt >= OUTPUT_VALIDATION_RETRY_LIMIT
                ),
            )
        pickhighlight_pickup_grounding_recovery_guard = {
            "enabled": (
                self.pickhighlight_pickup_grounding_recovery_config is not None
            ),
            "applicable": False,
            "decision": "disabled",
        }
        proposed_pickhighlight_pickup_grounding_state = (
            self.pickhighlight_pickup_grounding_recovery_state
        )
        if self.pickhighlight_pickup_grounding_recovery_config is not None:
            if self.pickhighlight_pickup_grounding_recovery_state is None:
                raise ValueError(
                    "PickHighlight pickup grounding recovery was not initialized"
                )
            (
                proposed_pickhighlight_pickup_grounding_state,
                pickhighlight_pickup_grounding_recovery_guard,
            ) = evaluate_pickhighlight_pickup_grounding_recovery(
                state=self.pickhighlight_pickup_grounding_recovery_state,
                previous_subtask=self.previous_subtask,
                proposed_subtask=rendered_subtask,
                proposed_points=rendered_points,
                response=response,
                config=self.pickhighlight_pickup_grounding_recovery_config,
                allow_retry_budget_bypass=(
                    validation_attempt >= OUTPUT_VALIDATION_RETRY_LIMIT
                ),
            )
        unmask_pickup_grounding_recovery_guard = {
            "enabled": self.unmask_pickup_grounding_recovery_config is not None,
            "applicable": False,
            "decision": "disabled",
        }
        proposed_unmask_pickup_grounding_state = (
            self.unmask_pickup_grounding_recovery_state
        )
        if self.unmask_pickup_grounding_recovery_config is not None:
            if self.unmask_pickup_grounding_recovery_state is None:
                raise ValueError("Unmask pickup grounding recovery was not initialized")
            (
                proposed_unmask_pickup_grounding_state,
                unmask_pickup_grounding_recovery_guard,
            ) = evaluate_unmask_pickup_grounding_recovery(
                state=self.unmask_pickup_grounding_recovery_state,
                previous_subtask=self.previous_subtask,
                proposed_subtask=rendered_subtask,
                proposed_points=rendered_points,
                response=response,
                refine_criterion=self.current_refine_criterion,
                config=self.unmask_pickup_grounding_recovery_config,
                allow_retry_budget_bypass=(
                    validation_attempt >= OUTPUT_VALIDATION_RETRY_LIMIT
                ),
            )
        button_swap_press_grounding_recovery_guard = {
            "enabled": (
                self.button_swap_press_grounding_recovery_config is not None
            ),
            "applicable": False,
            "decision": "disabled",
        }
        proposed_button_swap_press_grounding_state = (
            self.button_swap_press_grounding_recovery_state
        )
        if self.button_swap_press_grounding_recovery_config is not None:
            if self.button_swap_press_grounding_recovery_state is None:
                raise ValueError(
                    "ButtonUnmaskSwap press grounding recovery was not initialized"
                )
            (
                proposed_button_swap_press_grounding_state,
                button_swap_press_grounding_recovery_guard,
            ) = evaluate_button_swap_press_grounding_recovery(
                state=self.button_swap_press_grounding_recovery_state,
                previous_subtask=self.previous_subtask,
                proposed_subtask=rendered_subtask,
                proposed_points=rendered_points,
                response=response,
                refine_criterion=self.current_refine_criterion,
                action_instance_id=self.active_refine_action_instance_id,
                config=self.button_swap_press_grounding_recovery_config,
                allow_retry_budget_bypass=(
                    validation_attempt >= OUTPUT_VALIDATION_RETRY_LIMIT
                ),
            )
        return {
            "response": response,
            "proposed_subtask_template": proposed_subtask_template,
            "template_canonicalized": template_canonicalized,
            "counting_pickup_ordinal_canonicalization": (
                counting_pickup_ordinal_canonicalization
            ),
            "response_normalization": response_normalization,
            "target_identity_frame_sanitization": (
                target_identity_frame_sanitization
            ),
            "binding_updates_sanitization": binding_updates_sanitization,
            "completion_check_sanitization": completion_check_sanitization,
            "refine_check_sanitization": refine_check_sanitization,
            "target_identity_color_canonicalized": (
                target_identity_color_canonicalized
            ),
            "status": status,
            "rendered_subtask": rendered_subtask,
            "rendered_points": rendered_points,
            "swing_side_grounding_guard": swing_side_grounding_guard,
            "counting_event_ledger_guard": counting_event_ledger_guard,
            "proposed_counting_event_ledger_state": proposed_counting_state,
            "video_repick_event_ledger_guard": video_repick_event_ledger_guard,
            "proposed_video_repick_event_ledger_state": proposed_video_repick_state,
            "video_repick_grounding_recovery_guard": (
                video_repick_grounding_recovery_guard
            ),
            "proposed_video_repick_grounding_recovery_state": (
                proposed_video_repick_grounding_state
            ),
            "unmask_pickup_grounding_recovery_guard": (
                unmask_pickup_grounding_recovery_guard
            ),
            "proposed_unmask_pickup_grounding_recovery_state": (
                proposed_unmask_pickup_grounding_state
            ),
            "button_swap_press_grounding_recovery_guard": (
                button_swap_press_grounding_recovery_guard
            ),
            "proposed_button_swap_press_grounding_recovery_state": (
                proposed_button_swap_press_grounding_state
            ),
            "pickhighlight_pickup_grounding_recovery_guard": (
                pickhighlight_pickup_grounding_recovery_guard
            ),
            "proposed_pickhighlight_pickup_grounding_recovery_state": (
                proposed_pickhighlight_pickup_grounding_state
            ),
        }

    @staticmethod
    def _normalize_instruction(value: str) -> str:
        normalized = value.strip().lower()
        normalized = re.sub(r"<point_[0-9]+>", "<point>", normalized)
        normalized = re.sub(r"<\s*[0-9]+\s*,\s*[0-9]+\s*>", "<point>", normalized)
        normalized = re.sub(r"[^a-z0-9_<>]+", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _state_key(execution_state: dict[str, str] | None) -> str:
        if not execution_state:
            return "unknown|unknown|unknown"
        return "|".join(
            (
                execution_state.get("held_object", "unknown"),
                execution_state.get("object_support", "unknown"),
                execution_state.get("gripper_state", "unknown"),
            )
        )

    def _write_lesson_state(self) -> None:
        if not self.enable_lessons:
            return
        run_dir = self._require_started()
        write_json(
            run_dir / "lesson_memory.json",
            {
                "scope": "episode_local",
                "cross_episode_loading": False,
                "lessons": self.lesson_memory,
            },
        )
        event_text = "".join(
            json.dumps(event, ensure_ascii=False) + "\n"
            for event in self.exploration_events
        )
        write_text(run_dir / "exploration_events.jsonl", event_text)

    def _append_exploration_event(self, event: dict[str, Any]) -> None:
        self.exploration_events.append({"round_index": self.round_index, **event})
        self._write_lesson_state()

    def _upsert_lesson(
        self,
        *,
        candidate: dict[str, Any],
        effect: str,
        effect_summary: str,
        source: str,
    ) -> None:
        if effect not in LESSON_STATUSES:
            raise ValueError(f"Unknown lesson effect: {effect}")
        state_key = self._state_key(candidate.get("execution_state"))
        identity = (
            candidate["skill_key"],
            state_key,
            candidate["normalized_instruction"],
        )
        lesson = next(
            (
                item
                for item in self.lesson_memory
                if (
                    item["skill_key"],
                    item["state_key"],
                    item["normalized_instruction"],
                )
                == identity
            ),
            None,
        )
        if lesson is None:
            lesson = {
                "lesson_id": f"lesson_{len(self.lesson_memory):03d}",
                "skill_key": candidate["skill_key"],
                "state_key": state_key,
                "execution_state": candidate.get("execution_state"),
                "candidate_instruction": candidate["template"],
                "normalized_instruction": candidate["normalized_instruction"],
                "instruction_strategy": candidate["strategy"],
                "attempt_chunks": 0,
                "effect_counts": {name: 0 for name in sorted(LESSON_STATUSES)},
                "lesson_status": "inconclusive",
                "effect_summary": "",
                "first_round": candidate["selected_round"],
                "last_round": self.round_index,
                "source": source,
            }
            self.lesson_memory.append(lesson)
        lesson["attempt_chunks"] = max(
            int(lesson["attempt_chunks"]), int(candidate["attempt_chunks"])
        )
        lesson["effect_counts"][effect] += 1
        lesson["effect_summary"] = effect_summary
        lesson["last_round"] = self.round_index
        if lesson["effect_counts"]["successful"]:
            lesson["lesson_status"] = "successful"
        elif lesson["effect_counts"]["unsafe"]:
            lesson["lesson_status"] = "unsafe"
        elif lesson["effect_counts"]["regressive"]:
            lesson["lesson_status"] = "regressive"
        elif lesson["effect_counts"]["promising"]:
            lesson["lesson_status"] = "promising"
        elif lesson["effect_counts"]["ineffective"]:
            lesson["lesson_status"] = "ineffective"
        else:
            lesson["lesson_status"] = "inconclusive"
        self._append_exploration_event(
            {
                "event": "lesson_update",
                "lesson_id": lesson["lesson_id"],
                "skill_key": lesson["skill_key"],
                "state_key": state_key,
                "candidate_instruction": lesson["candidate_instruction"],
                "effect": effect,
                "lesson_status": lesson["lesson_status"],
                "effect_summary": effect_summary,
                "source": source,
            }
        )

    def _update_active_candidate(self, status: str, recent_outcome: str) -> None:
        candidate = self.active_candidate
        if candidate is None or status == "initial":
            return
        candidate["attempt_chunks"] += 1
        candidate["last_status"] = status
        candidate["status_counts"][status] = candidate["status_counts"].get(status, 0) + 1
        if status == "stagnant":
            candidate["consecutive_stagnant_chunks"] += 1
        else:
            candidate["consecutive_stagnant_chunks"] = 0
        if status == "uncertain":
            candidate["consecutive_uncertain_chunks"] += 1
        else:
            candidate["consecutive_uncertain_chunks"] = 0

        if not candidate["is_exploration_candidate"]:
            return
        effect: str | None = None
        if status == "completed":
            effect = "successful"
        elif status == "progressing":
            effect = "promising"
        elif status == "regressed":
            effect = "regressive"
        elif (
            status == "stagnant"
            and candidate["consecutive_stagnant_chunks"] >= self.candidate_stagnant_limit
        ):
            effect = "ineffective"
        elif (
            status == "uncertain"
            and candidate["consecutive_uncertain_chunks"] >= self.candidate_stagnant_limit
        ):
            effect = "inconclusive"
        if effect is not None:
            self._upsert_lesson(
                candidate=candidate,
                effect=effect,
                effect_summary=recent_outcome,
                source="exploration_candidate",
            )

    def _promote_stalled_baseline(self) -> None:
        candidate = self.active_candidate
        if candidate is None or candidate["is_exploration_candidate"]:
            return
        # The compact executor state may change while the semantic intent stays
        # stalled, so use the intent-level counter for the exploration gate.
        if self.consecutive_stagnant_rounds < (
            self.rethinking_threshold + self.exploration_patience
        ):
            return
        self._upsert_lesson(
            candidate=candidate,
            effect="ineffective",
            effect_summary=(
                "The baseline instruction remained visually stagnant after normal "
                "rethinking and post-rethinking patience."
            ),
            source="stalled_baseline",
        )

    def _relevant_lessons(self) -> dict[str, Any]:
        skill_key = self.previous_skill_key
        state_key = self._state_key(self.previous_execution_state)
        matching = [
            lesson for lesson in self.lesson_memory if lesson["skill_key"] == skill_key
        ]
        exact = [lesson for lesson in matching if lesson["state_key"] == state_key]
        pool = exact or matching
        by_status: dict[str, list[dict[str, Any]]] = {
            "successful": [],
            "promising": [],
            "do_not_retry": [],
        }
        for lesson in sorted(pool, key=lambda item: int(item["last_round"]), reverse=True):
            compact = {
                "lesson_id": lesson["lesson_id"],
                "state_key": lesson["state_key"],
                "candidate_instruction": lesson["candidate_instruction"],
                "instruction_strategy": lesson["instruction_strategy"],
                "lesson_status": lesson["lesson_status"],
                "attempt_chunks": lesson["attempt_chunks"],
                "effect_summary": lesson["effect_summary"],
            }
            if lesson["lesson_status"] == "successful" and len(by_status["successful"]) < 1:
                by_status["successful"].append(compact)
            elif lesson["lesson_status"] == "promising" and len(by_status["promising"]) < 2:
                by_status["promising"].append(compact)
            elif lesson["lesson_status"] in {"ineffective", "regressive", "unsafe"} and len(
                by_status["do_not_retry"]
            ) < 5:
                by_status["do_not_retry"].append(compact)
        cross_episode = []
        if self.runtime_lesson_memory is not None:
            cross_episode = [
                {
                    "experience_id": experience["experience_id"],
                    "condition": experience["condition"],
                    "lesson": experience["lesson"],
                    "recommended_skill_or_policy": experience[
                        "recommended_skill_or_policy"
                    ],
                    "success_evidence": experience["success_evidence"],
                    "forbidden_behavior": experience["forbidden_behavior"],
                }
                for experience in self.runtime_lesson_memory["experiences"]
                if self.task in experience["task_scope"]
            ]
        result = {
            "query_skill_key": skill_key,
            "query_state_key": state_key,
            "exact_state_match_used": bool(exact),
            **by_status,
        }
        if self.runtime_lesson_memory is not None:
            result["cross_episode"] = cross_episode
        return result

    def _exploration_candidate_count(self) -> int:
        skill_key = self.exploration_skill_key or self.previous_skill_key
        normalized = {
            event["normalized_instruction"]
            for event in self.exploration_events
            if event.get("event") == "candidate_selected"
            and event.get("skill_key") == skill_key
        }
        if (
            self.active_candidate is not None
            and self.active_candidate["skill_key"] == skill_key
            and self.active_candidate["is_exploration_candidate"]
        ):
            normalized.add(self.active_candidate["normalized_instruction"])
        return len(normalized)

    def _register_next_candidate(
        self,
        *,
        response: dict[str, Any],
        rendered_subtask: str,
        mode: str,
        prior_skill_key: str | None,
    ) -> None:
        if not self.enable_lessons:
            return
        skill_key = response["skill_key"]
        execution_state = dict(response["execution_state"])
        template = response["current_subtask_template"].strip()
        normalized = self._normalize_instruction(template)
        is_exploration_candidate = (
            mode == "explore_instruction" and skill_key == prior_skill_key
        )
        same_candidate = (
            self.active_candidate is not None
            and self.active_candidate["skill_key"] == skill_key
            and self.active_candidate["normalized_instruction"] == normalized
            and self._state_key(self.active_candidate.get("execution_state"))
            == self._state_key(execution_state)
        )
        if same_candidate:
            self.active_candidate["template"] = template
            self.active_candidate["rendered_instruction"] = rendered_subtask
            self.active_candidate["strategy"] = response["instruction_strategy"]
            self.active_candidate["is_exploration_candidate"] = (
                self.active_candidate["is_exploration_candidate"]
                or is_exploration_candidate
            )
            return
        if (
            self.active_candidate is not None
            and self.active_candidate["is_exploration_candidate"]
        ):
            previous = self.active_candidate
            decisive = (
                previous.get("last_status") in {"completed", "progressing", "regressed"}
                or previous["consecutive_stagnant_chunks"]
                >= self.candidate_stagnant_limit
                or previous["consecutive_uncertain_chunks"]
                >= self.candidate_stagnant_limit
            )
            if not decisive:
                self._upsert_lesson(
                    candidate=previous,
                    effect="inconclusive",
                    effect_summary=(
                        "The planner replaced this exploration candidate before "
                        "collecting decisive visual evidence."
                    ),
                    source="exploration_candidate",
                )
        self.active_candidate = {
            "selected_round": self.round_index,
            "skill_key": skill_key,
            "execution_state": execution_state,
            "template": template,
            "rendered_instruction": rendered_subtask,
            "normalized_instruction": normalized,
            "strategy": response["instruction_strategy"],
            "is_exploration_candidate": is_exploration_candidate,
            "attempt_chunks": 0,
            "consecutive_stagnant_chunks": 0,
            "consecutive_uncertain_chunks": 0,
            "status_counts": {},
            "last_status": None,
        }
        if is_exploration_candidate:
            self._append_exploration_event(
                {
                    "event": "candidate_selected",
                    "skill_key": skill_key,
                    "state_key": self._state_key(execution_state),
                    "candidate_instruction": template,
                    "normalized_instruction": normalized,
                    "instruction_strategy": response["instruction_strategy"],
                }
            )

    def _current_mode(self) -> str:
        if not self.enable_lessons:
            return (
                "rethink_verify"
                if self.consecutive_no_progress_rounds >= self.rethinking_threshold
                else "normal"
            )
        if self.catalog_retrieval_only:
            if self.consecutive_stagnant_rounds >= (
                self.rethinking_threshold + self.exploration_patience
            ):
                return "retrieve_known"
            if self.consecutive_no_progress_rounds >= self.rethinking_threshold:
                return "rethink_verify"
            return "normal"
        if (
            not self.exploration_active
            and self.previous_skill_key is not None
            and self.consecutive_stagnant_rounds
            >= self.rethinking_threshold + self.exploration_patience
        ):
            self.exploration_active = True
            self.exploration_skill_key = self.previous_skill_key
            self._promote_stalled_baseline()
            self._append_exploration_event(
                {
                    "event": "exploration_started",
                    "skill_key": self.exploration_skill_key,
                    "reason": "normal_rethinking_unresolved_after_confirmed_stagnation",
                    "consecutive_stagnant_rounds": self.consecutive_stagnant_rounds,
                }
            )
        if self.exploration_active:
            return "explore_instruction"
        if self.consecutive_no_progress_rounds >= self.rethinking_threshold:
            return "rethink_verify"
        return "normal"

    def _render_prompt(self, image_manifest: str, *, mode: str) -> str:
        if self.enable_global_retrieval:
            assert self.skill_retriever is not None
            retrieval_query = (
                f"Fixed task description: {self.detailed_task_description}\n"
                f"Current progress: {self.progress}\n"
                f"Previous executor skill: {self.previous_subtask}\n"
                f"Previous status: {self.previous_recent_status}"
            )
            self.current_retrieval_candidates = self.skill_retriever.retrieve(
                query=retrieval_query,
                task=self.task,
                previous_template_id=self.previous_skill_key,
                recent_status=self.previous_recent_status,
                top_k=self.retrieval_top_k,
                scheme=self.retrieval_scheme,
            )
            if (
                self.enable_counting_explicit_first_pickup
                and self.task == "PickXtimes"
            ):
                self.current_retrieval_candidates = (
                    ensure_pickxtimes_first_pickup_candidate(
                        self.current_retrieval_candidates,
                        retriever=self.skill_retriever,
                        state=self.counting_event_ledger_state,
                    )
                )
            if (
                self.enable_counting_exact_pickup_ordinal
                and self.task in {"PickXtimes", "BinFill"}
            ):
                self.current_retrieval_candidates = (
                    ensure_counting_next_pickup_candidates(
                        self.current_retrieval_candidates,
                        retriever=self.skill_retriever,
                        task=self.task,
                        state=self.counting_event_ledger_state,
                    )
                )
            if self.enable_patternlock_all_directions and self.task == "PatternLock":
                self.current_retrieval_candidates = (
                    ensure_patternlock_direction_candidates(
                        self.current_retrieval_candidates,
                        retriever=self.skill_retriever,
                    )
                )
            if (
                self.enable_pickhighlight_all_pickup_templates
                and self.task == "PickHighlight"
            ):
                self.current_retrieval_candidates = (
                    ensure_pickhighlight_pickup_candidates(
                        self.current_retrieval_candidates,
                        retriever=self.skill_retriever,
                    )
                )
            self.allowed_executor_skills = {
                candidate["template_id"]: candidate
                for candidate in self.current_retrieval_candidates
            }
        loop_control_state = json.dumps(
            {
                "consecutive_no_progress_rounds_before_this_turn": (
                    self.consecutive_no_progress_rounds
                ),
                "rethinking_required_this_turn": (
                    self.consecutive_no_progress_rounds >= self.rethinking_threshold
                ),
                "rethinking_threshold": self.rethinking_threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
        candidate_count = self._exploration_candidate_count() if self.enable_lessons else 0
        active_candidate_state = None
        if self.enable_lessons and self.active_candidate is not None:
            active_candidate_state = {
                "instruction": self.active_candidate["template"],
                "strategy": self.active_candidate["strategy"],
                "is_exploration_candidate": self.active_candidate[
                    "is_exploration_candidate"
                ],
                "attempt_chunks": self.active_candidate["attempt_chunks"],
                "consecutive_stagnant_chunks": self.active_candidate[
                    "consecutive_stagnant_chunks"
                ],
                "candidate_stagnant_limit": self.candidate_stagnant_limit,
            }
        exploration_control_state = json.dumps(
            {
                "mode": mode,
                "consecutive_no_progress_rounds_before_this_turn": (
                    self.consecutive_no_progress_rounds
                ),
                "consecutive_stagnant_rounds_before_this_turn": (
                    self.consecutive_stagnant_rounds
                ),
                "rethinking_threshold": self.rethinking_threshold,
                "exploration_threshold": (
                    self.rethinking_threshold + self.exploration_patience
                ),
                "active_skill_key": self.previous_skill_key,
                "active_candidate": active_candidate_state,
                "exploration_candidates_tried": candidate_count,
                "max_exploration_candidates": self.max_exploration_candidates,
                "remaining_new_candidate_budget": max(
                    0, self.max_exploration_candidates - candidate_count
                ),
                "cross_episode_lessons_enabled": False,
                "retrieval_scheme": (
                    self.retrieval_scheme if self.enable_global_retrieval else None
                ),
                "retrieval_top_k": (
                    self.retrieval_top_k if self.enable_global_retrieval else None
                ),
                "novel_instruction_exploration_enabled": not self.catalog_retrieval_only,
            },
            ensure_ascii=False,
            indent=2,
        )
        relevant_lessons = json.dumps(
            self._relevant_lessons() if self.enable_lessons else {},
            ensure_ascii=False,
            indent=2,
        )
        planner_guidance_payload = dict(self.planner_guidance)
        if self.counting_event_ledger_state is not None:
            planner_guidance_payload["runtime_counting_event_ledger"] = {
                **json.loads(json.dumps(self.counting_event_ledger_state)),
                "authority": (
                    "Use this harness-maintained visible-event ledger instead of "
                    "a repetition count inferred only from prose memory."
                ),
            }
        if self.video_repick_event_ledger_state is not None:
            planner_guidance_payload["runtime_video_repick_event_ledger"] = {
                **json.loads(json.dumps(self.video_repick_event_ledger_state)),
                "authority": (
                    "This harness-maintained ledger is authoritative. Count a "
                    "cycle only after its visually verified pickup and supported "
                    "release; do not infer counts from prose or command issuance."
                ),
            }
        if self.video_repick_grounding_recovery_state is not None:
            planner_guidance_payload["runtime_video_repick_grounding_recovery"] = {
                **json.loads(
                    json.dumps(self.video_repick_grounding_recovery_state)
                ),
                "authority": (
                    "After repeated visually verified pickup misses, do not issue "
                    "another near-identical point. Re-inspect the current front "
                    "image, retain the demonstrated instance relation, and choose "
                    "a materially corrected top-down point on that same cube."
                ),
            }
        if self.unmask_pickup_grounding_recovery_state is not None:
            planner_guidance_payload["runtime_unmask_pickup_grounding_recovery"] = {
                **json.loads(
                    json.dumps(self.unmask_pickup_grounding_recovery_state)
                ),
                "authority": (
                    "After repeated visually verified empty pickup misses, do not "
                    "return to a recent failed point. Re-inspect the current front "
                    "image and choose a materially different top-down point on the "
                    "same task-described container."
                ),
            }
        if self.button_swap_press_grounding_recovery_state is not None:
            planner_guidance_payload[
                "runtime_button_swap_press_grounding_recovery"
            ] = {
                **json.loads(
                    json.dumps(self.button_swap_press_grounding_recovery_state)
                ),
                "authority": (
                    "Count a button only after a visually verified completed "
                    "local press event. Keep the pending button spatially "
                    "distinct from completed points. After repeated verified "
                    "misses, re-inspect the current image and choose a materially "
                    "different contact point on that same pending button."
                ),
            }
        if self.pickhighlight_pickup_grounding_recovery_state is not None:
            planner_guidance_payload[
                "runtime_pickhighlight_pickup_grounding_recovery"
            ] = {
                **json.loads(
                    json.dumps(self.pickhighlight_pickup_grounding_recovery_state)
                ),
                "authority": (
                    "After repeated visibly verified empty pickup misses, do not "
                    "return to a recent failed point. Re-inspect the current front "
                    "image and choose a materially different top-down point on the "
                    "same pending highlighted target; never return to the button "
                    "or substitute another same-colored target."
                ),
            }
        try:
            user_prompt = self.user_prompt_template.format(
                detailed_task_description=self.detailed_task_description,
                planner_guidance=json.dumps(
                    planner_guidance_payload, ensure_ascii=False, indent=2
                ),
                previous_progress=self.progress,
                previous_subtask=self.previous_subtask,
                target_identity_state=json.dumps(
                    {
                        "enabled": self.enable_target_identity_gate,
                        "bindings": self.target_identity_bindings,
                        "completed_action_latches": self.action_completion_latches,
                        "active_action_instance_id": self.active_action_instance_id,
                        "active_action_reinforcement": (
                            self.action_reinforcement_states.get(
                                self.active_action_instance_id
                            )
                            if self.active_action_instance_id is not None
                            else None
                        ),
                        "highlight_target_ledger_enabled": (
                            self.enable_highlight_target_ledger
                        ),
                        "highlight_target_instances": self.highlight_target_instances,
                        "active_highlight_target_instance_id": (
                            self.active_highlight_target_instance_id
                        ),
                        "highlight_recovery_target_instance_id": (
                            self.highlight_recovery_target_instance_id
                        ),
                        "recent_highlight_wrong_target_events": (
                            self.highlight_wrong_target_events[-4:]
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                action_success_criterion=json.dumps(
                    self.current_action_success_criterion
                    if self.current_action_success_criterion is not None
                    else {
                        "criterion_id": "none",
                        "applies": False,
                        "instruction": (
                            "No action-specific success criterion matches the previous "
                            "skill. Use the baseline visual completion judgment."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                refine_check_criterion=json.dumps(
                    self.current_refine_criterion
                    if self.current_refine_criterion is not None
                    else {
                        "criterion_id": "none",
                        "applies": False,
                        "instruction": (
                            "No action-specific refine check matches the previous "
                            "skill. Return criterion_id=none, decision=not_applicable, "
                            "and an empty checks list."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                loop_control_state=loop_control_state,
                exploration_control_state=exploration_control_state,
                relevant_lessons=relevant_lessons,
                allowed_executor_skills=json.dumps(
                    {
                        "source": "automatically extracted released pi0.5 training labels",
                        "retrieval_scheme": self.retrieval_scheme,
                        "task_tag_used_as_automatic_metadata": self.task,
                        "skills": [
                            {
                                "rank": candidate["rank"],
                                "template_id": candidate["template_id"],
                                "canonical_template": candidate["canonical_template"],
                                "point_arity": candidate["point_arity"],
                                "simple_labels": candidate["simple_labels"],
                                "training_frame_count": candidate[
                                    "training_frame_count"
                                ],
                            }
                            for candidate in self.current_retrieval_candidates
                        ],
                    }
                    if self.enable_global_retrieval
                    else {
                        "task": "MoveCube",
                        "demonstrated_method": self.oracle_skill_method,
                        "skills": list(self.allowed_executor_skills.values()),
                    }
                    if self.enable_oracle_skills
                    else {},
                    ensure_ascii=False,
                    indent=2,
                ),
                image_manifest=image_manifest,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported placeholder in rollout user prompt: {exc}") from exc
        prompt = f"{self.system_prompt.strip()}\n\n{user_prompt.strip()}\n"
        if self.decision_identity_isolation:
            assert_prompt_has_no_episode_identity(prompt, source="rendered_prompt")
        return prompt

    def plan(
        self,
        *,
        front_frames: list[np.ndarray],
        wrist_frames: list[np.ndarray],
        frame_ids: list[int],
    ) -> tuple[str, dict[str, Any]]:
        run_dir = self._require_started()
        if (
            not front_frames
            or len(front_frames) != len(wrist_frames)
            or len(front_frames) != len(frame_ids)
        ):
            raise ValueError(
                "front_frames, wrist_frames, and frame_ids must have the same non-zero length"
            )
        if len(set(frame_ids)) != len(frame_ids) or any(value < 0 for value in frame_ids):
            raise ValueError("frame_ids must be unique non-negative integers")
        if frame_ids != sorted(frame_ids):
            raise ValueError("frame_ids must be chronological")

        round_index = self.round_index
        round_dir = run_dir / "rounds" / f"round_{round_index:04d}"
        if round_dir.exists():
            raise FileExistsError(f"Planner round directory already exists: {round_dir}")
        inputs_dir = round_dir / "recent_inputs"
        inputs_dir.mkdir(parents=True)

        recent_paths: dict[str, Path] = {}
        front_shapes: dict[str, tuple[int, int]] = {}
        attachment_rows: list[dict[str, Any]] = []
        image_paths: list[Path] = []
        manifest_lines: list[str] = []
        for rollout_frame, front, wrist in zip(
            frame_ids, front_frames, wrist_frames, strict=True
        ):
            for view, short, value in (
                ("front", "f", front),
                ("wrist", "w", wrist),
            ):
                array = np.asarray(value, dtype=np.uint8)
                if array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError(f"{view.title()} frame {rollout_frame} is not RGB")
                frame_id = f"r_{short}{rollout_frame:04d}"
                path = inputs_dir / f"{frame_id}.png"
                Image.fromarray(array).save(path)
                recent_paths[frame_id] = path
                image_paths.append(path)
                manifest_lines.append(
                    f"{frame_id}: recent {view} observation at rollout frame {rollout_frame}"
                )
                attachment_rows.append(
                    {
                        "frame_id": frame_id,
                        "kind": "recent_observation",
                        "view": view,
                        "rollout_frame": rollout_frame,
                        "file": str(path.relative_to(run_dir)),
                    }
                )
                if view == "front":
                    front_shapes[frame_id] = (int(array.shape[0]), int(array.shape[1]))

        self.observed_front_frame_ids.update(front_shapes)

        prior_progress = self.progress
        prior_subtask = self.previous_subtask
        prior_subtask_template = self.previous_subtask_template
        prior_skill_key = self.previous_skill_key
        target_bindings_before = {
            color: dict(binding)
            for color, binding in self.target_identity_bindings.items()
        }
        action_completion_latches_before = {
            key: dict(value) for key, value in self.action_completion_latches.items()
        }
        highlight_target_instances_before = {
            key: dict(value) for key, value in self.highlight_target_instances.items()
        }
        self.current_action_success_criterion = (
            match_action_success_criterion(
                prior_subtask, self.action_success_catalog, task=self.task
            )
            if self.action_success_catalog is not None
            else None
        )
        self.current_refine_criterion = (
            match_action_success_criterion(
                prior_subtask, self.refine_check_catalog, task=self.task
            )
            if self.refine_check_catalog is not None
            else None
        )
        if self.current_refine_criterion is not None:
            refine_signature = action_instance_signature(
                str(self.current_refine_criterion["criterion_id"]),
                prior_skill_key,
                prior_subtask_template,
                prior_subtask,
            )
            if refine_signature != self.active_refine_action_instance_signature:
                self.refine_action_instance_sequence += 1
                self.active_refine_action_instance_signature = refine_signature
                self.active_refine_action_instance_id = (
                    f"refine_action_{self.refine_action_instance_sequence:03d}::"
                    f"{refine_signature}"
                )
                refine_criterion_id = str(
                    self.current_refine_criterion["criterion_id"]
                )
                if refine_criterion_id in self.reinforcement_lock_config:
                    self.refinement_lock_states[
                        self.active_refine_action_instance_id
                    ] = {
                        "action_instance_id": self.active_refine_action_instance_id,
                        "criterion_id": refine_criterion_id,
                        "skill_key": prior_skill_key,
                        "requested_color": requested_pickup_color(prior_subtask),
                        "status": "available",
                        "extra_command_budget": self.reinforcement_lock_config[
                            refine_criterion_id
                        ],
                        "extra_commands_consumed": 0,
                        "first_blocked_proposal": None,
                        "latest_planner_proposal": None,
                        "planner_proposal_history": [],
                    }
                experience_config = self.experience_transition_gate_config.get(
                    refine_criterion_id
                )
                if experience_config is not None:
                    self.experience_gate_states[
                        self.active_refine_action_instance_id
                    ] = {
                        "action_instance_id": self.active_refine_action_instance_id,
                        "criterion_id": refine_criterion_id,
                        "experience_id": experience_config["experience_id"],
                        "experience_revision": experience_config["revision"],
                        "situation_class": experience_config["situation_class"],
                        "status": "active",
                        "max_blocked_transitions": experience_config[
                            "max_blocked_transitions"
                        ],
                        "blocked_transition_count": 0,
                        "proposal_history": [],
                    }
        else:
            # A different command has actually reached the executor. Closing
            # the prior instance here is what permits a later return to the same
            # semantic action to receive a fresh, independent allowance.
            self.active_refine_action_instance_signature = None
            self.active_refine_action_instance_id = None
        prior_requested_color = requested_pickup_color(prior_subtask)
        if self.current_action_success_criterion is not None:
            current_action_instance_signature = action_instance_signature(
                str(self.current_action_success_criterion["criterion_id"]),
                prior_skill_key,
                prior_subtask_template,
                prior_subtask,
            )
            if (
                current_action_instance_signature
                != self.active_action_instance_signature
            ):
                self.action_instance_sequence += 1
                self.active_action_instance_signature = (
                    current_action_instance_signature
                )
                self.active_action_instance_id = (
                    f"action_{self.action_instance_sequence:03d}::"
                    f"{current_action_instance_signature}"
                )
                criterion_id = str(
                    self.current_action_success_criterion["criterion_id"]
                )
                if criterion_id in self.one_shot_completion_reinforcement:
                    # This value counts repeated command submissions to the
                    # downstream executor.  It deliberately does not control
                    # the number of low-level actions returned by one policy
                    # call; run_robomme_batch.py owns that chunk size through
                    # args.obs_horizon.
                    extra_execution_budget = self.one_shot_completion_reinforcement[
                        criterion_id
                    ]
                    self.action_reinforcement_states[self.active_action_instance_id] = {
                        "action_instance_id": self.active_action_instance_id,
                        "criterion_id": criterion_id,
                        "skill_key": prior_skill_key,
                        "requested_color": prior_requested_color,
                        "status": "available",
                        "extra_execution_budget": extra_execution_budget,
                        "extra_execution_consumed": 0,
                        # Compatibility-only legacy field.  The old state
                        # machine froze the first proposed successor here and
                        # could later force that stale command.  New code keeps
                        # it null and tracks the planner's latest proposal.
                        "deferred_successor": None,
                        "blocked_transition_proposal": None,
                        "latest_planner_proposal": None,
                        "planner_proposal_history": [],
                    }
        else:
            self.active_action_instance_signature = None
            self.active_action_instance_id = None
        if (
            self.enable_target_identity_gate
            and self.current_action_success_criterion is not None
            and prior_requested_color is not None
        ):
            self.current_action_success_criterion = dict(
                self.current_action_success_criterion
            )
            self.current_action_success_criterion["active_target_binding"] = {
                "requested_color": prior_requested_color,
                "binding": self.target_identity_bindings.get(prior_requested_color),
            }
        completion_evidence_key = None
        if (
            self.enable_completion_evidence_latch
            and self.current_action_success_criterion is not None
            and isinstance(
                self.current_action_success_criterion.get(
                    "evidence_accumulation_policy"
                ),
                dict,
            )
        ):
            completion_evidence_key = "::".join(
                (
                    str(self.current_action_success_criterion["criterion_id"]),
                    str(prior_skill_key or prior_subtask_template or prior_subtask),
                )
            )
            if self.active_completion_evidence_key != completion_evidence_key:
                self.active_completion_evidence_key = completion_evidence_key
                self.completion_evidence_states[completion_evidence_key] = {
                    "candidate_round_count": 0,
                    "completion_latched": False,
                    "required_candidate_rounds": int(
                        self.current_action_success_criterion[
                            "evidence_accumulation_policy"
                        ].get("required_consecutive_candidate_rounds", 2)
                    ),
                    "last_update": "initialized",
                }
            self.current_action_success_criterion = dict(
                self.current_action_success_criterion
            )
            self.current_action_success_criterion["harness_evidence_state"] = dict(
                self.completion_evidence_states[completion_evidence_key]
            )
        elif self.active_completion_evidence_key is not None:
            self.active_completion_evidence_key = None
        no_progress_before = self.consecutive_no_progress_rounds
        stagnant_before = self.consecutive_stagnant_rounds
        mode = self._current_mode()
        rethinking_required = mode in {
            "rethink_verify",
            "retrieve_known",
            "explore_instruction",
        }
        prompt = self._render_prompt("\n".join(manifest_lines), mode=mode)
        write_text(round_dir / "prompt.txt", prompt)
        write_json(
            round_dir / "input_manifest.json",
            {
                "round_index": round_index,
                "recent_rollout_frame_ids": frame_ids,
                "previous_progress": prior_progress,
                "previous_subtask": prior_subtask,
                "previous_subtask_template": prior_subtask_template,
                "previous_skill_key": prior_skill_key,
                "action_success_criterion": self.current_action_success_criterion,
                "refine_check_criterion": self.current_refine_criterion,
                "active_refine_action_instance_id": (
                    self.active_refine_action_instance_id
                ),
                "refinement_lock_states": self.refinement_lock_states,
                "experience_gate_states": self.experience_gate_states,
                "counting_event_ledger_state": self.counting_event_ledger_state,
                "video_repick_event_ledger_state": (
                    self.video_repick_event_ledger_state
                ),
                "video_repick_grounding_recovery_state": (
                    self.video_repick_grounding_recovery_state
                ),
                "unmask_pickup_grounding_recovery_state": (
                    self.unmask_pickup_grounding_recovery_state
                ),
                "button_swap_press_grounding_recovery_state": (
                    self.button_swap_press_grounding_recovery_state
                ),
                "pickhighlight_pickup_grounding_recovery_state": (
                    self.pickhighlight_pickup_grounding_recovery_state
                ),
                "target_identity_bindings": self.target_identity_bindings,
                "action_completion_latches": self.action_completion_latches,
                "active_action_instance_id": self.active_action_instance_id,
                "action_reinforcement_states": self.action_reinforcement_states,
                "completion_evidence_key": completion_evidence_key,
                "consecutive_no_progress_rounds_before": no_progress_before,
                "consecutive_stagnant_rounds_before": stagnant_before,
                "rethinking_required_this_turn": rethinking_required,
                "planner_mode": mode,
                "episode_local_lesson_count": len(self.lesson_memory),
                "historical_image_count": 0,
                "recent_pair_count": len(frame_ids),
                "attachment_count": len(image_paths),
                "attachments": attachment_rows,
            },
        )

        current_front_ids = set(front_shapes)
        inference_results: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        prepared: dict[str, Any] | None = None
        retry_prompt = prompt
        for attempt in range(OUTPUT_VALIDATION_RETRY_LIMIT + 1):
            destination = (
                round_dir
                if attempt == 0
                else round_dir / f"validation_retry_{attempt:02d}"
            )
            try:
                result = self.inference_runner(
                    codex_bin=self.codex_bin,
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    prompt=retry_prompt,
                    output_schema=self.output_schema_path,
                    image_paths=image_paths,
                    destination=destination,
                    environment=self.environment,
                    timeout_seconds=self.timeout_seconds,
                )
            except Exception as exc:
                # A provider/parse failure happens before this round is accepted,
                # so planner memory and round_index intentionally remain unchanged.
                # Archive the complete input and raw provider response, allowing a
                # later call with newer observations to retry the same logical round
                # without colliding with this directory.
                write_json(
                    round_dir / "inference_error.json",
                    {
                        "round_index": round_index,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                failed_root = run_dir / "failed_inference_rounds"
                failed_root.mkdir(parents=True, exist_ok=True)
                suffix = 1
                failed_round = failed_root / (
                    f"round_{round_index:04d}_attempt_{suffix:03d}"
                )
                while failed_round.exists():
                    suffix += 1
                    failed_round = failed_root / (
                        f"round_{round_index:04d}_attempt_{suffix:03d}"
                    )
                shutil.move(str(round_dir), str(failed_round))
                raise
            inference_results.append(result)
            raw_response = result["response"]
            try:
                prepared = self._prepare_inference_response(
                    raw_response=raw_response,
                    current_front_ids=current_front_ids,
                    front_shapes=front_shapes,
                    round_index=round_index,
                    prior_requested_color=prior_requested_color,
                    validation_attempt=attempt,
                )
                if attempt:
                    self._copy_final_inference_artifacts(destination, round_dir)
                break
            except (KeyError, ValueError) as exc:
                validation_errors.append(f"{type(exc).__name__}: {exc}")
                write_json(destination / "invalid_response.json", raw_response)
                if attempt >= OUTPUT_VALIDATION_RETRY_LIMIT:
                    raise
                retry_prompt = (
                    prompt
                    + "\n\nThe previous response was rejected by the harness without "
                    "executing any environment action. Correct only the invalid "
                    "structured output and re-evaluate the same attached images. "
                    f"Harness error: {type(exc).__name__}: {exc}. "
                    "Every points[].source_frame_id and every newly emitted visual "
                    "grounding source must use a front frame ID from the current "
                    "image manifest. Historical frame IDs in text memory are "
                    "provenance only. recent_status='initial' is allowed only on "
                    "planner round 0. Return the complete schema object."
                )
        assert prepared is not None
        response = prepared["response"]
        proposed_subtask_template = prepared["proposed_subtask_template"]
        template_canonicalized = prepared["template_canonicalized"]
        counting_pickup_ordinal_canonicalization = prepared[
            "counting_pickup_ordinal_canonicalization"
        ]
        response_normalization = prepared["response_normalization"]
        target_identity_frame_sanitization = prepared[
            "target_identity_frame_sanitization"
        ]
        binding_updates_sanitization = prepared["binding_updates_sanitization"]
        completion_check_sanitization = prepared[
            "completion_check_sanitization"
        ]
        refine_check_sanitization = prepared["refine_check_sanitization"]
        target_identity_color_canonicalized = prepared[
            "target_identity_color_canonicalized"
        ]
        status = prepared["status"]
        rendered_subtask = prepared["rendered_subtask"]
        rendered_points = prepared["rendered_points"]
        swing_side_grounding_guard = prepared["swing_side_grounding_guard"]
        counting_event_ledger_guard = prepared["counting_event_ledger_guard"]
        self.counting_event_ledger_state = prepared[
            "proposed_counting_event_ledger_state"
        ]
        video_repick_event_ledger_guard = prepared[
            "video_repick_event_ledger_guard"
        ]
        self.video_repick_event_ledger_state = prepared[
            "proposed_video_repick_event_ledger_state"
        ]
        video_repick_grounding_recovery_guard = prepared[
            "video_repick_grounding_recovery_guard"
        ]
        self.video_repick_grounding_recovery_state = prepared[
            "proposed_video_repick_grounding_recovery_state"
        ]
        unmask_pickup_grounding_recovery_guard = prepared[
            "unmask_pickup_grounding_recovery_guard"
        ]
        self.unmask_pickup_grounding_recovery_state = prepared[
            "proposed_unmask_pickup_grounding_recovery_state"
        ]
        button_swap_press_grounding_recovery_guard = prepared[
            "button_swap_press_grounding_recovery_guard"
        ]
        proposed_button_swap_press_grounding_recovery_state = prepared[
            "proposed_button_swap_press_grounding_recovery_state"
        ]
        pickhighlight_pickup_grounding_recovery_guard = prepared[
            "pickhighlight_pickup_grounding_recovery_guard"
        ]
        self.pickhighlight_pickup_grounding_recovery_state = prepared[
            "proposed_pickhighlight_pickup_grounding_recovery_state"
        ]
        raw_response = inference_results[-1]["response"]
        result = dict(inference_results[-1])
        result["usage"] = self._sum_inference_usage(inference_results)
        result["wall_seconds"] = sum(
            float(value["wall_seconds"]) for value in inference_results
        )
        result["capacity_retry_count"] = sum(
            int(value.get("capacity_retry_count", 0)) for value in inference_results
        )
        if (
            response_normalization
            or template_canonicalized
            or validation_errors
            or (
                completion_check_sanitization is not None
                and completion_check_sanitization.get("changed")
            )
            or (
                refine_check_sanitization is not None
                and refine_check_sanitization.get("changed")
            )
            or (
                target_identity_frame_sanitization is not None
                and target_identity_frame_sanitization.get("changed")
            )
            or (
                binding_updates_sanitization is not None
                and binding_updates_sanitization.get("changed")
            )
        ):
            write_json(round_dir / "raw_response.json", raw_response)
        if validation_errors:
            write_json(
                round_dir / "validation_retries.json",
                {
                    "retry_count": len(validation_errors),
                    "errors": validation_errors,
                    "environment_advanced_between_attempts": False,
                },
            )
        binding_update_merge: list[dict[str, Any]] = []
        if self.require_binding_updates:
            (
                self.target_identity_bindings,
                binding_update_merge,
            ) = merge_target_binding_updates(
                self.target_identity_bindings,
                response["binding_updates"],
            )
        planner_proposed_grounded_subtask = rendered_subtask
        planner_response_before_completion_gate: dict[str, Any] | None = None
        planner_response_before_experience_gate: dict[str, Any] | None = None
        planner_response_before_refinement_lock: dict[str, Any] | None = None
        completion_gate = {
            "applicable": self.current_action_success_criterion is not None,
            "criterion_id": (
                self.current_action_success_criterion["criterion_id"]
                if self.current_action_success_criterion is not None
                else None
            ),
            "decision": (
                response["completion_check"]["decision"]
                if self.enable_action_success_criteria
                else None
            ),
            "semantic_switch_blocked": False,
            "completed_status_blocked": False,
            "confirmation_retry_required": False,
            "confirmation_count_before": 0,
            "confirmation_count_after": 0,
            "raw_decision": (
                response["completion_check"]["decision"]
                if self.enable_action_success_criteria
                else None
            ),
            "effective_decision": (
                response["completion_check"]["decision"]
                if self.enable_action_success_criteria
                else None
            ),
            "completion_evidence_key": completion_evidence_key,
            "completion_evidence_state_before": None,
            "completion_evidence_state_after": None,
            "completion_latch_key": None,
            "completion_latch_before": None,
            "completion_latch_after": None,
            "completion_latch_applied": False,
            "reopened_action_blocked": False,
            "action_instance_id": self.active_action_instance_id,
            "one_shot_reinforcement_configured": False,
            "one_shot_reinforcement_triggered": False,
            "one_shot_reinforcement_pending_before": False,
            "one_shot_reinforcement_completed": False,
            "one_shot_reinforcement_forced_successor": False,
            "action_transition_lock_active_before": False,
            "action_transition_lock_active_after": False,
            "action_transition_lock_blocked_proposal": None,
            "action_transition_lock_released": False,
            "one_shot_reinforcement_state_before": None,
            "one_shot_reinforcement_state_after": None,
        }
        effective_next_subtask_template = " ".join(
            response["current_subtask_template"].strip().lower().split()
        )
        effective_next_skill_key = (
            response.get("skill_key") if self.enable_lessons else None
        )
        target_gate = {
            "enabled": self.enable_target_identity_gate,
            "proposed_requested_color": requested_pickup_color(rendered_subtask),
            "prior_requested_color": prior_requested_color,
            "binding_accepted": False,
            "pickup_blocked": False,
            "wrong_target_recovery_allowed": False,
            "reason": "not_applicable",
            "binding_source": "none",
            "binding_updates": (
                response.get("binding_updates", [])
                if self.require_binding_updates
                else []
            ),
            "binding_update_merge": binding_update_merge,
            "state_before": target_bindings_before,
            "state_after": None,
        }
        if self.enable_target_identity_gate and prior_requested_color is not None:
            identity = response["target_identity"]
            revealed_color = identity["revealed_color"]
            identity_disagrees = (
                identity["requested_color"]
                not in {"none", "unknown", prior_requested_color}
                or identity["same_as_bound_target"] == "no"
                or revealed_color
                not in {"unknown", "none", prior_requested_color}
            )
            if identity_disagrees:
                completion_gate["effective_decision"] = "not_satisfied"
                target_gate["reason"] = "active_pickup_identity_not_verified"
        recovery_config = (
            self.current_action_success_criterion.get("wrong_target_recovery")
            if self.current_action_success_criterion is not None
            else None
        )
        if isinstance(recovery_config, dict):
            recovery_triggered = completion_check_verdict(
                response["completion_check"],
                str(recovery_config.get("trigger_check_id")),
            ) == str(recovery_config.get("trigger_verdict"))
            identity = response.get("target_identity", {})
            verified_color_mismatch = (
                prior_requested_color is not None
                and identity.get("binding_status") == "mismatch"
                and identity.get("revealed_color")
                in (TARGET_COLORS - {"none", "unknown", prior_requested_color})
                and response.get("execution_state", {}).get("held_object")
                == "container"
                and response.get("execution_state", {}).get("object_support")
                in {"airborne", "held_unknown"}
            )
            recovery_target_allowed = any(
                re.fullmatch(pattern, rendered_subtask, flags=re.IGNORECASE)
                for pattern in recovery_config.get("allowed_successor_patterns", [])
            )
            if (
                recovery_triggered
                and verified_color_mismatch
                and recovery_target_allowed
            ):
                target_gate["wrong_target_recovery_allowed"] = True
                target_gate["reason"] = "verified_wrong_target_requires_safe_release"
                status = "regressed"
                response = dict(response)
                response["recent_status"] = "regressed"
                response["updated_progress"] = (
                    response["updated_progress"].rstrip()
                    + "\nTarget identity gate: the lifted container exposed a color "
                    "that does not match the requested target. This is not completed "
                    "task progress; safely put down the wrong container before "
                    "rebinding and retrying the requested color."
                )[-1600:]
                if prior_requested_color is not None:
                    mismatch_binding = dict(response["target_identity"])
                    mismatch_binding["binding_status"] = "mismatch"
                    self.target_identity_bindings[prior_requested_color] = mismatch_binding
        criterion_id = completion_gate["criterion_id"]
        required_confirmation_retries = self.completion_confirmation_retries.get(
            criterion_id, 0
        )
        confirmation_count_before = self.completion_confirmation_counts.get(
            criterion_id, 0
        )
        completion_gate["confirmation_count_before"] = confirmation_count_before
        if completion_evidence_key is not None:
            evidence_before = dict(
                self.completion_evidence_states[completion_evidence_key]
            )
            evidence_after = update_completion_evidence_state(
                evidence_before,
                response["completion_check"],
                self.current_action_success_criterion,
            )
            self.completion_evidence_states[completion_evidence_key] = evidence_after
            completion_gate["completion_evidence_state_before"] = evidence_before
            completion_gate["completion_evidence_state_after"] = dict(evidence_after)
            if evidence_after.get("completion_latched"):
                completion_gate["effective_decision"] = "satisfied"
        completion_latch_key = (
            self.current_action_success_criterion.get("completion_latch_key")
            if self.enable_action_completion_latches
            and self.current_action_success_criterion is not None
            else None
        )
        if isinstance(completion_latch_key, str) and completion_latch_key:
            completion_gate["completion_latch_key"] = completion_latch_key
            completion_gate["completion_latch_before"] = action_completion_latches_before.get(
                completion_latch_key
            )
            if completion_latch_key in self.action_completion_latches:
                completion_gate["effective_decision"] = "satisfied"
                completion_gate["completion_latch_applied"] = True
            elif response["completion_check"]["decision"] == "satisfied":
                self.action_completion_latches[completion_latch_key] = {
                    "criterion_id": criterion_id,
                    "completed_round": round_index,
                    "evidence": response["completion_check"]["reason"].strip(),
                }
                completion_gate["completion_latch_applied"] = True
            completion_gate["completion_latch_after"] = self.action_completion_latches.get(
                completion_latch_key
            )
        reinforcement_state = (
            self.action_reinforcement_states.get(self.active_action_instance_id)
            if self.active_action_instance_id is not None
            else None
        )
        if reinforcement_state is not None:
            completion_gate["one_shot_reinforcement_configured"] = True
            completion_gate["one_shot_reinforcement_state_before"] = dict(
                reinforcement_state
            )
            reinforcement_pending = (
                reinforcement_state.get("status")
                == "pending_one_shot_extra_execution"
            )
            completion_gate["action_transition_lock_active_before"] = (
                reinforcement_pending
            )
            completion_gate["one_shot_reinforcement_pending_before"] = (
                reinforcement_pending
            )
            if reinforcement_pending:
                latest_planner_proposal = {
                    "grounded_subtask": rendered_subtask,
                    "subtask_template": effective_next_subtask_template,
                    "skill_key": effective_next_skill_key,
                    "rendered_points": [dict(point) for point in rendered_points],
                    "proposed_round": round_index,
                }
                # The cache represents the planner's newest intent under the
                # lock, not the first transition that happened to be blocked.
                # Thus B is replaced by A in a B->A sequence, and by C in a
                # B->C sequence.  The current value is what will naturally be
                # accepted when the finite lock expires.
                reinforcement_state["latest_planner_proposal"] = (
                    latest_planner_proposal
                )
                reinforcement_state.setdefault("planner_proposal_history", []).append(
                    latest_planner_proposal
                )
                # These observations result from one harness-requested repeat of
                # the same grounded command.  That command was submitted to pi0.5
                # once and produced one normal action chunk.  Consume exactly one
                # command-dispatch unit; do not interpret this value as low-level
                # action steps or as the policy action-chunk length.
                # While budget remains, the harness may keep overriding the planner;
                # once exhausted, it must yield control and must never turn a missing
                # saved successor into an exception.
                extra_execution_budget = int(
                    reinforcement_state.get("extra_execution_budget", 1)
                )
                extra_execution_consumed = min(
                    int(reinforcement_state.get("extra_execution_consumed", 0)) + 1,
                    extra_execution_budget,
                )
                reinforcement_state["extra_execution_consumed"] = (
                    extra_execution_consumed
                )
                completion_gate["effective_decision"] = "satisfied"
                if extra_execution_consumed < extra_execution_budget:
                    planner_response_before_completion_gate = dict(response)
                    response = dict(response)
                    response["recent_status"] = "progressing"
                    status = "progressing"
                    response["updated_progress"] = (
                        prior_progress.rstrip()
                        + "\nBounded completion reinforcement: execute another "
                        "normal executor command for physical margin. The harness "
                        "has used "
                        f"{extra_execution_consumed} of {extra_execution_budget} "
                        "allowed extra command dispatches for this action instance."
                    )[-1600:]
                    rendered_subtask = prior_subtask
                    effective_next_subtask_template = prior_subtask_template
                    effective_next_skill_key = prior_skill_key
                    completion_gate["semantic_switch_blocked"] = True
                    completion_gate["completed_status_blocked"] = True
                    completion_gate["action_transition_lock_active_after"] = True
                    completion_gate["action_transition_lock_blocked_proposal"] = {
                        **latest_planner_proposal,
                    }
                else:
                    # The lock has consumed all K repeated command dispatches.
                    # Yield to the planner's *current* proposal.  In particular,
                    # never restore the proposal that was blocked when the lock
                    # was first acquired: it was never executed, may now be stale,
                    # and therefore cannot delimit action instances.
                    reinforcement_state["status"] = "exhausted"
                    reinforcement_state["completed_round"] = round_index
                    completion_gate["one_shot_reinforcement_completed"] = True
                    completion_gate["action_transition_lock_released"] = True
                    response = dict(response)
                    response["updated_progress"] = (
                        response["updated_progress"].rstrip()
                        + "\nBounded completion reinforcement: the configured "
                        f"budget of {extra_execution_budget} extra executor-command "
                        "dispatch(es) is "
                        "exhausted. The harness now yields to the planner and cannot "
                        "override this action instance again."
                    )[-1600:]
            elif (
                reinforcement_state.get("status") == "available"
                and completion_gate["effective_decision"] == "satisfied"
                # Acquire the lock only when the planner actually proposes a
                # semantic transition away from the command that was submitted
                # in the preceding executor round. Repeating the same command is
                # already continued execution and must not spend reinforcement.
                and effective_next_skill_key != prior_skill_key
                and rendered_subtask != prior_subtask
            ):
                blocked_transition_proposal = {
                    "grounded_subtask": rendered_subtask,
                    "subtask_template": effective_next_subtask_template,
                    "skill_key": effective_next_skill_key,
                    "rendered_points": [dict(point) for point in rendered_points],
                    "proposed_round": round_index,
                }
                reinforcement_state["status"] = (
                    "pending_one_shot_extra_execution"
                )
                reinforcement_state["extra_execution_consumed"] = 0
                reinforcement_state["triggered_round"] = round_index
                # Audit only. This proposal is deliberately not a deferred
                # command: the lock prevented it from reaching the executor, so
                # it cannot later be forced after observations have changed.
                reinforcement_state["blocked_transition_proposal"] = (
                    blocked_transition_proposal
                )
                reinforcement_state["latest_planner_proposal"] = (
                    blocked_transition_proposal
                )
                reinforcement_state["planner_proposal_history"] = [
                    blocked_transition_proposal
                ]
                reinforcement_state["deferred_successor"] = None
                planner_response_before_completion_gate = dict(response)
                response = dict(response)
                response["recent_status"] = "progressing"
                status = "progressing"
                response["updated_progress"] = (
                    prior_progress.rstrip()
                    + "\nBounded completion reinforcement: this action instance "
                    "first met its completion criterion. Execute "
                    f"{int(reinforcement_state['extra_execution_budget'])} additional "
                    "submission(s) of the same canonical command to the executor "
                    "for physical margin. Each submission uses the executor's normal "
                    "action-chunk size. When this finite budget "
                    "is exhausted, the harness must yield control to the planner."
                )[-1600:]
                rendered_subtask = prior_subtask
                effective_next_subtask_template = prior_subtask_template
                effective_next_skill_key = prior_skill_key
                completion_gate["semantic_switch_blocked"] = True
                completion_gate["completed_status_blocked"] = True
                completion_gate["one_shot_reinforcement_triggered"] = True
                completion_gate["action_transition_lock_active_after"] = True
                completion_gate["action_transition_lock_blocked_proposal"] = (
                    blocked_transition_proposal
                )
            completion_gate["one_shot_reinforcement_state_after"] = dict(
                reinforcement_state
            )
        if (
            criterion_id is not None
            and required_confirmation_retries > 0
            and response["completion_check"]["decision"] == "satisfied"
            and confirmation_count_before < required_confirmation_retries
        ):
            planner_response_before_completion_gate = dict(response)
            response = dict(response)
            response["recent_status"] = "progressing"
            status = "progressing"
            confirmation_count_after = confirmation_count_before + 1
            self.completion_confirmation_counts[criterion_id] = confirmation_count_after
            response["updated_progress"] = (
                prior_progress.rstrip()
                + "\nCompletion confirmation: a plausible button-local press event was "
                "observed. The harness requires one additional canonical press chunk "
                "before allowing a successor skill."
            )[-1600:]
            rendered_subtask = prior_subtask
            effective_next_subtask_template = prior_subtask_template
            effective_next_skill_key = prior_skill_key
            completion_gate["semantic_switch_blocked"] = True
            completion_gate["completed_status_blocked"] = True
            completion_gate["confirmation_retry_required"] = True
            completion_gate["confirmation_count_after"] = confirmation_count_after
        elif criterion_id is not None and required_confirmation_retries > 0:
            if response["completion_check"]["decision"] == "satisfied":
                self.completion_confirmation_counts.pop(criterion_id, None)
                completion_gate["confirmation_count_after"] = 0
            else:
                self.completion_confirmation_counts.pop(criterion_id, None)
                completion_gate["confirmation_count_after"] = 0
        if (
            self.current_action_success_criterion is not None
            and completion_gate["effective_decision"] != "satisfied"
            and not target_gate["wrong_target_recovery_allowed"]
        ):
            if status == "completed":
                planner_response_before_completion_gate = dict(response)
                response = dict(response)
                response["recent_status"] = "uncertain"
                status = "uncertain"
                gate_reason = response["completion_check"]["reason"].strip()
                response["updated_progress"] = (
                    prior_progress.rstrip()
                    + "\nCompletion gate: the previous semantic action is not "
                    "confirmed. Keep its phase active and continue the same "
                    f"canonical executor skill. Evidence: {gate_reason}"
                )[-1600:]
                completion_gate["completed_status_blocked"] = True
            if (
                self.enable_lessons
                and prior_skill_key is not None
                and effective_next_skill_key != prior_skill_key
            ):
                if planner_response_before_completion_gate is None:
                    planner_response_before_completion_gate = dict(response)
                    response = dict(response)
                gate_reason = response["completion_check"]["reason"].strip()
                response["updated_progress"] = (
                    prior_progress.rstrip()
                    + "\nCompletion gate: the previous semantic action is not "
                    "confirmed. Keep its phase active and continue the same "
                    f"canonical executor skill. Evidence: {gate_reason}"
                )[-1600:]
                rendered_subtask = prior_subtask
                effective_next_subtask_template = prior_subtask_template
                effective_next_skill_key = prior_skill_key
                completion_gate["semantic_switch_blocked"] = True

        if (
            self.enable_action_completion_latches
            and "button_prerequisite" in self.action_completion_latches
            and is_button_press_subtask(rendered_subtask)
            and not is_button_press_subtask(prior_subtask)
        ):
            if planner_response_before_completion_gate is None:
                planner_response_before_completion_gate = dict(response)
                response = dict(response)
            rendered_subtask = prior_subtask
            effective_next_subtask_template = prior_subtask_template
            effective_next_skill_key = prior_skill_key
            status = "uncertain"
            response["recent_status"] = "uncertain"
            response["updated_progress"] = (
                response["updated_progress"].rstrip()
                + "\nCompletion latch: the button prerequisite was already strictly "
                "verified. Do not reopen it; continue the current post-button phase."
            )[-1600:]
            completion_gate["reopened_action_blocked"] = True

        proposed_requested_color = requested_pickup_color(rendered_subtask)
        target_gate["proposed_requested_color"] = proposed_requested_color
        if (
            self.enable_target_identity_gate
            and proposed_requested_color is not None
            and proposed_requested_color != prior_requested_color
        ):
            identity = response["target_identity"]
            stored_binding = self.target_identity_bindings.get(proposed_requested_color)
            binding_safe, binding_reason, binding_source = resolve_pickup_binding(
                identity,
                proposed_requested_color,
                stored_binding,
            )
            if binding_safe:
                binding = dict(
                    stored_binding
                    if binding_source == "stored_binding" and stored_binding is not None
                    else identity
                )
                binding["grounding_points"] = rendered_points
                binding["bound_from_subtask"] = rendered_subtask
                self.target_identity_bindings[proposed_requested_color] = binding
                target_gate["binding_accepted"] = True
                target_gate["reason"] = binding_reason
                target_gate["binding_source"] = binding_source
            else:
                if planner_response_before_completion_gate is None:
                    planner_response_before_completion_gate = dict(response)
                    response = dict(response)
                rendered_subtask = prior_subtask
                effective_next_subtask_template = prior_subtask_template
                effective_next_skill_key = prior_skill_key
                status = "uncertain"
                response["recent_status"] = "uncertain"
                response["updated_progress"] = (
                    response["updated_progress"].rstrip()
                    + "\nTarget identity gate: hazardous pickup blocked because the "
                    f"{proposed_requested_color} target binding is not supported by "
                    f"direct evidence ({binding_reason}). Preserve the observation "
                    "phase and obtain a direct cover-event or demonstration binding."
                )[-1600:]
                target_gate["pickup_blocked"] = True
                target_gate["reason"] = binding_reason
        target_gate["state_after"] = dict(self.target_identity_bindings)

        highlight_target_gate = {
            "enabled": self.enable_highlight_target_ledger,
            "prior_target_instance_id": None,
            "proposed_target_instance_id": None,
            "completed_instance_recorded": False,
            "released_instance_recorded": False,
            "wrong_completed_instance_detected": None,
            "recovery_forced": False,
            "reopened_completed_instance_blocked": False,
            "state_before": highlight_target_instances_before,
            "state_after": None,
        }
        if self.enable_highlight_target_ledger and self.task == "PickHighlight":
            prior_highlight_id = highlight_target_instance_id(
                prior_skill_key,
                prior_subtask_template,
                prior_subtask,
            )
            highlight_target_gate["prior_target_instance_id"] = prior_highlight_id
            if prior_highlight_id is not None:
                ordinal, color = parse_highlighted_cube_pickup(
                    prior_subtask_template
                ) or parse_highlighted_cube_pickup(prior_subtask) or ("unknown", "unknown")
                prior_record = self.highlight_target_instances.setdefault(
                    prior_highlight_id,
                    {
                        "instance_id": prior_highlight_id,
                        "skill_key": prior_skill_key,
                        "ordinal": ordinal,
                        "color": color,
                        "state": "pending",
                        "first_selected_round": max(0, round_index - 1),
                    },
                )
                prior_record["last_grounded_subtask"] = prior_subtask
                identity = response.get("target_identity", {})
                if isinstance(identity, dict):
                    prior_record["last_target_description"] = identity.get(
                        "target_description"
                    )
                    prior_record["last_evidence_frame_ids"] = list(
                        identity.get("evidence_frame_ids", [])
                    )
                if (
                    status == "completed"
                    and response.get("execution_state", {}).get("held_object") == "cube"
                    and response.get("execution_state", {}).get("object_support")
                    in {"airborne", "held_unknown"}
                ):
                    prior_record["state"] = "picked"
                    prior_record["picked_round"] = round_index
                    self.active_highlight_target_instance_id = prior_highlight_id
                    highlight_target_gate["completed_instance_recorded"] = True

                revealed_color = identity.get("revealed_color")
                mismatch_held = (
                    identity.get("binding_status") == "mismatch"
                    and revealed_color in {"red", "green", "blue"}
                    and response.get("execution_state", {}).get("held_object") == "cube"
                    and response.get("execution_state", {}).get("object_support")
                    in {"airborne", "held_unknown"}
                )
                if mismatch_held:
                    completed_matches = [
                        instance_id
                        for instance_id, item in self.highlight_target_instances.items()
                        if instance_id != prior_highlight_id
                        and item.get("color") == revealed_color
                        and item.get("state") in {"picked", "retired"}
                    ]
                    if len(completed_matches) == 1:
                        wrong_id = completed_matches[0]
                        self.highlight_recovery_target_instance_id = wrong_id
                        event = {
                            "round_index": round_index,
                            "desired_target_instance_id": prior_highlight_id,
                            "wrong_completed_target_instance_id": wrong_id,
                            "revealed_color": revealed_color,
                            "grounded_subtask": prior_subtask,
                        }
                        self.highlight_wrong_target_events.append(event)
                        highlight_target_gate[
                            "wrong_completed_instance_detected"
                        ] = wrong_id
                        place_candidate = next(
                            (
                                candidate
                                for candidate in self.current_retrieval_candidates
                                if candidate.get("canonical_template")
                                == "place the cube onto the table"
                            ),
                            None,
                        )
                        if (
                            place_candidate is not None
                            and not is_place_cube_on_table_subtask(rendered_subtask)
                        ):
                            if planner_response_before_completion_gate is None:
                                planner_response_before_completion_gate = dict(response)
                                response = dict(response)
                            rendered_subtask = "place the cube onto the table"
                            rendered_points = []
                            effective_next_subtask_template = rendered_subtask
                            effective_next_skill_key = str(
                                place_candidate["template_id"]
                            )
                            status = "regressed"
                            response["recent_status"] = "regressed"
                            response["updated_progress"] = (
                                response["updated_progress"].rstrip()
                                + "\nCompleted-target exclusion: the executor lifted an "
                                "already retired highlighted target instead of the "
                                "requested pending target. Release it once before "
                                "regrounding; it remains retired and receives no new "
                                "task credit."
                            )[-1600:]
                            highlight_target_gate["recovery_forced"] = True

            if (
                is_place_cube_on_table_subtask(prior_subtask)
                and status == "completed"
            ):
                released_id = (
                    self.highlight_recovery_target_instance_id
                    or self.active_highlight_target_instance_id
                )
                if released_id in self.highlight_target_instances:
                    released = self.highlight_target_instances[released_id]
                    released["state"] = "retired"
                    released["released_round"] = round_index
                    highlight_target_gate["released_instance_recorded"] = True
                if released_id == self.highlight_recovery_target_instance_id:
                    self.highlight_recovery_target_instance_id = None
                if released_id == self.active_highlight_target_instance_id:
                    self.active_highlight_target_instance_id = None

            proposed_highlight_id = highlight_target_instance_id(
                effective_next_skill_key,
                effective_next_subtask_template,
                rendered_subtask,
            )
            highlight_target_gate[
                "proposed_target_instance_id"
            ] = proposed_highlight_id
            if proposed_highlight_id is not None:
                parsed = parse_highlighted_cube_pickup(
                    effective_next_subtask_template
                ) or parse_highlighted_cube_pickup(rendered_subtask)
                assert parsed is not None
                ordinal, color = parsed
                proposed_record = self.highlight_target_instances.setdefault(
                    proposed_highlight_id,
                    {
                        "instance_id": proposed_highlight_id,
                        "skill_key": effective_next_skill_key,
                        "ordinal": ordinal,
                        "color": color,
                        "state": "pending",
                        "first_selected_round": round_index,
                    },
                )
                if proposed_record.get("state") in {"picked", "retired"}:
                    if planner_response_before_completion_gate is None:
                        planner_response_before_completion_gate = dict(response)
                        response = dict(response)
                    # Repeating the just-finished release is a safe bounded fallback.
                    # The next prompt exposes the immutable ledger so the planner can
                    # select a genuinely pending action slot.
                    rendered_subtask = prior_subtask
                    effective_next_subtask_template = prior_subtask_template
                    effective_next_skill_key = prior_skill_key
                    rendered_points = []
                    status = "uncertain"
                    response["recent_status"] = "uncertain"
                    response["updated_progress"] = (
                        response["updated_progress"].rstrip()
                        + "\nCompleted-target exclusion: the proposed highlighted "
                        f"target instance {proposed_highlight_id} is already retired. "
                        "Do not reopen it; select a pending target instance next."
                    )[-1600:]
                    highlight_target_gate[
                        "reopened_completed_instance_blocked"
                    ] = True
                else:
                    proposed_record["last_grounded_subtask"] = rendered_subtask
                    proposed_record["last_selected_round"] = round_index

        highlight_target_gate["state_after"] = {
            key: dict(value) for key, value in self.highlight_target_instances.items()
        }

        # Experimental self-evolution path. A candidate experience may veto only
        # its explicitly scoped semantic transition, and only for a finite budget.
        # This leaves the mature v8.5 profile unchanged because it has no config.
        experience_gate = {
            "enabled": bool(self.experience_transition_gate_config),
            "applicable": False,
            "experience_id": None,
            "experience_revision": None,
            "criterion_id": (
                self.current_refine_criterion.get("criterion_id")
                if self.current_refine_criterion is not None
                else None
            ),
            "action_instance_id": self.active_refine_action_instance_id,
            "decision": (
                response.get("refine_check", {}).get("decision")
                if self.enable_refine_check
                else None
            ),
            "semantic_transition_proposed": False,
            "transition_blocked": False,
            "released_due_to_satisfied_evidence": False,
            "released_due_to_budget": False,
            "state_before": None,
            "state_after": None,
            "blocked_proposal": None,
        }
        experience_state = (
            self.experience_gate_states.get(self.active_refine_action_instance_id)
            if self.active_refine_action_instance_id is not None
            else None
        )
        if experience_state is not None:
            experience_gate["applicable"] = True
            experience_gate["experience_id"] = experience_state["experience_id"]
            experience_gate["experience_revision"] = experience_state[
                "experience_revision"
            ]
            experience_gate["state_before"] = json.loads(
                json.dumps(experience_state)
            )
            proposed_transition = (
                effective_next_skill_key != prior_skill_key
                and rendered_subtask != prior_subtask
            )
            experience_gate["semantic_transition_proposed"] = proposed_transition
            if proposed_transition:
                proposal = {
                    "grounded_subtask": rendered_subtask,
                    "subtask_template": effective_next_subtask_template,
                    "skill_key": effective_next_skill_key,
                    "proposed_round": round_index,
                    "refine_decision": experience_gate["decision"],
                }
                experience_state["proposal_history"].append(proposal)
                if experience_gate["decision"] == "satisfied":
                    experience_state["status"] = "satisfied"
                    experience_state["released_round"] = round_index
                    experience_gate["released_due_to_satisfied_evidence"] = True
                elif int(experience_state["blocked_transition_count"]) < int(
                    experience_state["max_blocked_transitions"]
                ):
                    planner_response_before_experience_gate = dict(response)
                    experience_state["blocked_transition_count"] += 1
                    experience_state["last_blocked_round"] = round_index
                    experience_gate["transition_blocked"] = True
                    experience_gate["blocked_proposal"] = proposal
                    rendered_subtask = prior_subtask
                    effective_next_subtask_template = prior_subtask_template
                    effective_next_skill_key = prior_skill_key
                    rendered_points = []
                    status = "uncertain"
                    response = dict(response)
                    response["recent_status"] = "uncertain"
                    response["updated_progress"] = (
                        prior_progress.rstrip()
                        + "\nCandidate-experience gate: the current action lacks "
                        "the task-scoped positive evidence required by "
                        f"{experience_state['situation_class']}, so this semantic "
                        "transition was blocked and the same canonical command "
                        "will be retried within a finite budget."
                    )[-1600:]
                else:
                    experience_state["status"] = "budget_exhausted"
                    experience_state["released_round"] = round_index
                    experience_gate["released_due_to_budget"] = True
            experience_gate["state_after"] = json.loads(
                json.dumps(experience_state)
            )

        # v8.5 clean path: the structured refine check has already influenced the
        # planner's final status and command inside the single model call.  Python
        # never reads its decision to veto that command.  This finite lock sees
        # only the refined final output and can repeat a configured previous
        # command for physical margin after a normal completed transition.
        refinement_lock = {
            "enabled": self.enable_refine_check,
            "criterion_id": (
                self.current_refine_criterion.get("criterion_id")
                if self.current_refine_criterion is not None
                else None
            ),
            "action_instance_id": self.active_refine_action_instance_id,
            "state_before": None,
            "state_after": None,
            "planner_transition_proposed": False,
            "triggered": False,
            "active_before": False,
            "active_after": False,
            "released": False,
            "blocked_current_proposal": None,
            "refine_check_has_veto_authority": False,
        }
        refine_lock_state = (
            self.refinement_lock_states.get(self.active_refine_action_instance_id)
            if self.active_refine_action_instance_id is not None
            else None
        )
        if refine_lock_state is not None:
            refinement_lock["state_before"] = dict(refine_lock_state)
            planner_proposal = {
                "grounded_subtask": rendered_subtask,
                "subtask_template": effective_next_subtask_template,
                "skill_key": effective_next_skill_key,
                "rendered_points": [dict(point) for point in rendered_points],
                "proposed_round": round_index,
                "recent_status": status,
                "instruction_strategy": response.get("instruction_strategy"),
            }
            semantic_transition = (
                effective_next_skill_key != prior_skill_key
                and rendered_subtask != prior_subtask
            )
            refinement_lock["planner_transition_proposed"] = semantic_transition
            lock_pending = refine_lock_state.get("status") == "locked"
            refinement_lock["active_before"] = lock_pending
            if lock_pending:
                refine_lock_state["latest_planner_proposal"] = planner_proposal
                refine_lock_state.setdefault("planner_proposal_history", []).append(
                    planner_proposal
                )
                consumed = min(
                    int(refine_lock_state.get("extra_commands_consumed", 0)) + 1,
                    int(refine_lock_state["extra_command_budget"]),
                )
                refine_lock_state["extra_commands_consumed"] = consumed
                if consumed < int(refine_lock_state["extra_command_budget"]):
                    planner_response_before_refinement_lock = dict(response)
                    refinement_lock["blocked_current_proposal"] = planner_proposal
                    rendered_subtask = prior_subtask
                    effective_next_subtask_template = prior_subtask_template
                    effective_next_skill_key = prior_skill_key
                    rendered_points = []
                    status = "progressing"
                    response = dict(response)
                    response["recent_status"] = "progressing"
                    response["updated_progress"] = (
                        prior_progress.rstrip()
                        + "\nReinforcement lock: repeat the previous command for "
                        "physical margin. The advisory refine check has no veto "
                        f"authority; {consumed} of "
                        f"{int(refine_lock_state['extra_command_budget'])} additional "
                        "command submissions have completed."
                    )[-1600:]
                    refinement_lock["active_after"] = True
                else:
                    refine_lock_state["status"] = "exhausted"
                    refine_lock_state["released_round"] = round_index
                    refinement_lock["released"] = True
                    # Accept this round's newest refined command exactly as-is.
                    # The first blocked proposal is audit data, not a queue.
            elif (
                refine_lock_state.get("status") == "available"
                and status == "completed"
                and semantic_transition
            ):
                refine_lock_state["status"] = "locked"
                refine_lock_state["triggered_round"] = round_index
                refine_lock_state["first_blocked_proposal"] = planner_proposal
                refine_lock_state["latest_planner_proposal"] = planner_proposal
                refine_lock_state["planner_proposal_history"] = [planner_proposal]
                planner_response_before_refinement_lock = dict(response)
                refinement_lock["triggered"] = True
                refinement_lock["active_after"] = True
                refinement_lock["blocked_current_proposal"] = planner_proposal
                rendered_subtask = prior_subtask
                effective_next_subtask_template = prior_subtask_template
                effective_next_skill_key = prior_skill_key
                rendered_points = []
                status = "progressing"
                response = dict(response)
                response["recent_status"] = "progressing"
                response["updated_progress"] = (
                    prior_progress.rstrip()
                    + "\nReinforcement lock: the refined planner marked the previous "
                    "configured action completed and proposed a semantic transition. "
                    f"Repeat the previous command for {int(refine_lock_state['extra_command_budget'])} "
                    "additional pi0.5 command submission(s), then yield to the newest "
                    "refined planner output."
                )[-1600:]
            refinement_lock["state_after"] = dict(refine_lock_state)

        if self.button_swap_press_grounding_recovery_config is not None:
            override_source = None
            if rendered_subtask != planner_proposed_grounded_subtask:
                if refinement_lock["active_after"]:
                    override_source = "refinement_lock"
                elif experience_gate["transition_blocked"]:
                    override_source = "experience_gate"
                elif completion_gate["semantic_switch_blocked"]:
                    override_source = "completion_gate"
                else:
                    override_source = "runtime_harness"
            (
                self.button_swap_press_grounding_recovery_state,
                button_swap_press_grounding_recovery_guard,
            ) = reconcile_button_swap_press_state_after_runtime_overrides(
                proposed_state=proposed_button_swap_press_grounding_recovery_state,
                guard=button_swap_press_grounding_recovery_guard,
                planner_proposed_subtask=planner_proposed_grounded_subtask,
                dispatched_subtask=rendered_subtask,
                override_source=override_source,
            )

        if self.enable_lessons:
            self._update_active_candidate(status, response["recent_outcome"].strip())

        self.progress = response["updated_progress"].strip()
        self.previous_recent_status = status or "initial"
        self.previous_subtask = rendered_subtask
        next_subtask_template = effective_next_subtask_template
        if self.require_recent_status and status in NO_PROGRESS_STATUSES:
            self.consecutive_no_progress_rounds += 1
        else:
            self.consecutive_no_progress_rounds = 0
        if self.require_recent_status and status == "stagnant":
            self.consecutive_stagnant_rounds += 1
        else:
            self.consecutive_stagnant_rounds = 0
        if not self.enable_lessons and (
            prior_subtask_template is not None
            and next_subtask_template != prior_subtask_template
        ):
            self.consecutive_no_progress_rounds = 0
        next_skill_key = effective_next_skill_key
        if (
            self.enable_lessons
            and prior_skill_key is not None
            and next_skill_key != prior_skill_key
        ):
            self.consecutive_no_progress_rounds = 0
            self.consecutive_stagnant_rounds = 0
            if self.exploration_active:
                self._append_exploration_event(
                    {
                        "event": "exploration_ended",
                        "skill_key": self.exploration_skill_key,
                        "reason": "semantic_intent_changed",
                        "next_skill_key": next_skill_key,
                    }
                )
            self.exploration_active = False
            self.exploration_skill_key = None
        self.previous_subtask_template = next_subtask_template
        if self.enable_lessons:
            self.previous_skill_key = next_skill_key
            self.previous_execution_state = dict(response["execution_state"])
            if (
                not completion_gate["semantic_switch_blocked"]
                and not target_gate["pickup_blocked"]
                and not experience_gate["transition_blocked"]
                and not refinement_lock["active_after"]
            ):
                self._register_next_candidate(
                    response=response,
                    rendered_subtask=rendered_subtask,
                    mode=mode,
                    prior_skill_key=prior_skill_key,
                )
        if (
            completion_evidence_key is not None
            and not completion_gate["semantic_switch_blocked"]
            and next_skill_key != prior_skill_key
        ):
            self.active_completion_evidence_key = None
        history_row = {
            "round_index": round_index,
            "recent_rollout_frame_ids": frame_ids,
            "previous_progress": prior_progress,
            "previous_subtask": prior_subtask,
            "recent_status": status,
            "recent_outcome": response["recent_outcome"].strip(),
            "updated_progress": self.progress,
            "next_grounded_subtask": rendered_subtask,
            "consecutive_no_progress_rounds_before": no_progress_before,
            "consecutive_no_progress_rounds_after": self.consecutive_no_progress_rounds,
            "consecutive_stagnant_rounds_before": stagnant_before,
            "consecutive_stagnant_rounds_after": self.consecutive_stagnant_rounds,
            "rethinking_required_this_turn": rethinking_required,
            "planner_mode": mode,
            "completion_check": (
                response.get("completion_check")
                if self.enable_action_success_criteria
                else None
            ),
            "refine_check": (
                response.get("refine_check") if self.enable_refine_check else None
            ),
            "refine_check_sanitization": refine_check_sanitization,
            "refine_check_criterion": self.current_refine_criterion,
            "refinement_lock": refinement_lock,
            "experience_gate": experience_gate,
            "swing_side_grounding_guard": swing_side_grounding_guard,
            "counting_event_ledger_guard": counting_event_ledger_guard,
            "video_repick_event_ledger_guard": video_repick_event_ledger_guard,
            "video_repick_grounding_recovery_guard": (
                video_repick_grounding_recovery_guard
            ),
            "unmask_pickup_grounding_recovery_guard": (
                unmask_pickup_grounding_recovery_guard
            ),
            "button_swap_press_grounding_recovery_guard": (
                button_swap_press_grounding_recovery_guard
            ),
            "pickhighlight_pickup_grounding_recovery_guard": (
                pickhighlight_pickup_grounding_recovery_guard
            ),
            "experience_gate_states": {
                key: dict(value) for key, value in self.experience_gate_states.items()
            },
            "active_refine_action_instance_id": (
                self.active_refine_action_instance_id
            ),
            "refinement_lock_states": {
                key: dict(value) for key, value in self.refinement_lock_states.items()
            },
            "completion_gate": completion_gate,
            "completion_check_sanitization": completion_check_sanitization,
            "target_identity": (
                response.get("target_identity")
                if self.enable_target_identity_gate
                else None
            ),
            "target_identity_gate": target_gate,
            "highlight_target_gate": highlight_target_gate,
            "highlight_target_instances": {
                key: dict(value)
                for key, value in self.highlight_target_instances.items()
            },
            "highlight_wrong_target_events": list(
                self.highlight_wrong_target_events
            ),
            "target_identity_frame_sanitization": target_identity_frame_sanitization,
            "binding_updates_sanitization": binding_updates_sanitization,
            "action_completion_latches": dict(self.action_completion_latches),
            "active_action_instance_id": self.active_action_instance_id,
            "action_reinforcement_states": {
                key: dict(value)
                for key, value in self.action_reinforcement_states.items()
            },
        }
        if self.enable_lessons:
            history_row.update(
                {
                    "skill_key": next_skill_key,
                    "execution_state": response["execution_state"],
                    "instruction_strategy": response["instruction_strategy"],
                    "lesson_count": len(self.lesson_memory),
                }
            )
        self.progress_history.append(history_row)
        self._write_progress_history()
        record = {
            "round_index": round_index,
            "planner_memory_mode": "text_progress",
            "prompt_profile": self.prompt_profile,
            "rollout_behavior_id": self.rollout_behavior_id,
            "rethinking_threshold": self.rethinking_threshold,
            "recent_rollout_frame_ids": frame_ids,
            "historical_image_count": 0,
            "previous_progress": prior_progress,
            "previous_subtask": prior_subtask,
            "consecutive_no_progress_rounds_before": no_progress_before,
            "consecutive_no_progress_rounds_after": self.consecutive_no_progress_rounds,
            "consecutive_stagnant_rounds_before": stagnant_before,
            "consecutive_stagnant_rounds_after": self.consecutive_stagnant_rounds,
            "rethinking_required_this_turn": rethinking_required,
            "planner_mode": mode,
            "response": response,
            "planner_proposed_subtask_template": proposed_subtask_template,
            "planner_proposed_grounded_subtask": planner_proposed_grounded_subtask,
            "template_canonicalized_by_harness": template_canonicalized,
            "counting_pickup_ordinal_canonicalization": (
                counting_pickup_ordinal_canonicalization
            ),
            "target_identity_color_canonicalized_by_harness": (
                target_identity_color_canonicalized
            ),
            "action_success_criterion": self.current_action_success_criterion,
            "refine_check_criterion": self.current_refine_criterion,
            "refinement_lock": refinement_lock,
            "experience_gate": experience_gate,
            "swing_side_grounding_guard": swing_side_grounding_guard,
            "counting_event_ledger_guard": counting_event_ledger_guard,
            "video_repick_event_ledger_guard": video_repick_event_ledger_guard,
            "button_swap_press_grounding_recovery_guard": (
                button_swap_press_grounding_recovery_guard
            ),
            "completion_gate": completion_gate,
            "target_identity_gate": target_gate,
            "highlight_target_gate": highlight_target_gate,
            "highlight_target_instances_before": highlight_target_instances_before,
            "highlight_target_instances_after": {
                key: dict(value)
                for key, value in self.highlight_target_instances.items()
            },
            "highlight_wrong_target_events": list(
                self.highlight_wrong_target_events
            ),
            "target_identity_frame_sanitization": target_identity_frame_sanitization,
            "binding_updates_sanitization": binding_updates_sanitization,
            "action_completion_latches_before": action_completion_latches_before,
            "action_completion_latches_after": dict(self.action_completion_latches),
            "active_action_instance_id": self.active_action_instance_id,
            "action_reinforcement_states": {
                key: dict(value)
                for key, value in self.action_reinforcement_states.items()
            },
            "planner_response_before_completion_gate": (
                planner_response_before_completion_gate
            ),
            "planner_response_before_refinement_lock": (
                planner_response_before_refinement_lock
            ),
            "planner_response_before_experience_gate": (
                planner_response_before_experience_gate
            ),
            "active_refine_action_instance_id": (
                self.active_refine_action_instance_id
            ),
            "refinement_lock_states": {
                key: dict(value) for key, value in self.refinement_lock_states.items()
            },
            "experience_gate_states": {
                key: dict(value) for key, value in self.experience_gate_states.items()
            },
            "oracle_skill_method": self.oracle_skill_method,
            "allowed_executor_skill_ids": sorted(self.allowed_executor_skills),
            "retrieval_scheme": (
                self.retrieval_scheme if self.enable_global_retrieval else None
            ),
            "retrieval_candidates": (
                [
                    {
                        "rank": candidate["rank"],
                        "template_id": candidate["template_id"],
                        "canonical_template": candidate["canonical_template"],
                        "retrieval_score": candidate["retrieval_score"],
                    }
                    for candidate in self.current_retrieval_candidates
                ]
                if self.enable_global_retrieval
                else []
            ),
            "response_normalization": response_normalization,
            "codex_output_validation_retry_count": len(validation_errors),
            "codex_output_validation_errors": validation_errors,
            "rendered_points": rendered_points,
            "grounded_subtask_for_vla": rendered_subtask,
            "planner_provider": self.inference_provider,
            "planner_request_id": result["thread_id"],
            "planner_usage": result["usage"],
            "planner_started_at": result["started_at"],
            "planner_wall_seconds": result["wall_seconds"],
            "thinking_finalizer_used": result.get("thinking_finalizer_used", False),
            "thinking_stage_finish_reason": result.get(
                "thinking_stage_finish_reason"
            ),
            "thinking_stage_usage": result.get("thinking_stage_usage"),
            "finalization_stage_usage": result.get("finalization_stage_usage"),
            "codex_thread_id": result["thread_id"],
            "codex_usage": result["usage"],
            "codex_capacity_retry_count": result.get("capacity_retry_count", 0),
            "codex_started_at": result["started_at"],
            "codex_wall_seconds": result["wall_seconds"],
            "initial_demo_images_sent": False,
        }
        if self.enable_lessons:
            record.update(
                {
                    "skill_key": next_skill_key,
                    "execution_state": response["execution_state"],
                    "instruction_strategy": response["instruction_strategy"],
                    "episode_local_lesson_count": len(self.lesson_memory),
                    "relevant_lessons": self._relevant_lessons(),
                }
            )
        write_json(round_dir / "response.json", response)
        write_json(round_dir / "record.json", record)
        if self.enable_lessons:
            self._write_lesson_state()
        self.round_index += 1
        if not self.keep_recent_inputs:
            shutil.rmtree(inputs_dir)
        return rendered_subtask, record
