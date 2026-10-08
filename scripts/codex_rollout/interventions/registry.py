"""Stable registry and hook contract for self-evolve runtime interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


HOOK_ORDER = (
    "after_observation",
    "after_retrieval",
    "before_prompt",
    "canonicalize_proposal",
    "validate_transition",
    "before_dispatch",
)


class RuntimeIntervention(Protocol):
    """Protocol for a future registered Python intervention implementation.

    V4's initial built-ins compile to the already-tested planner mechanisms.
    New Python implementations must conform to this protocol and be registered
    before a runtime bundle may execute them.
    """

    def initialize(self, context: dict[str, Any], config: dict[str, Any]) -> Any: ...

    def apply(
        self, hook: str, context: dict[str, Any], value: Any, state: Any
    ) -> Any: ...


@dataclass(frozen=True)
class PluginRegistration:
    component_type: str
    hooks: tuple[str, ...]
    stateful: bool
    runtime_adapter: str
    declarative_authoring: bool = True


PLUGIN_REGISTRY = {
    row.component_type: row
    for row in (
        PluginRegistration("prompt_lesson", ("before_prompt",), False, "lesson_memory"),
        PluginRegistration(
            "criterion_overlay", ("before_prompt",), False, "refine_catalog"
        ),
        PluginRegistration(
            "bounded_transition_gate",
            ("validate_transition",),
            True,
            "v1_experience_transition_gate",
        ),
        PluginRegistration(
            "swing_visual_binding_gate",
            ("validate_transition",),
            False,
            "v3_swing_side_grounding_guard",
        ),
        PluginRegistration(
            "counting_event_ledger",
            ("after_observation", "before_prompt", "validate_transition"),
            True,
            "v4_counting_event_ledger",
        ),
        PluginRegistration(
            "video_repick_event_ledger",
            ("after_observation", "before_prompt", "validate_transition"),
            True,
            "v5_video_repick_event_ledger",
        ),
        PluginRegistration(
            "video_repick_grounding_recovery_gate",
            ("after_observation", "before_prompt", "validate_transition"),
            True,
            "v5_video_repick_grounding_recovery",
        ),
        PluginRegistration(
            "unmask_pickup_grounding_recovery_gate",
            ("after_observation", "before_prompt", "validate_transition"),
            True,
            "v5_unmask_pickup_grounding_recovery",
        ),
        PluginRegistration(
            "button_swap_press_grounding_recovery_gate",
            ("after_observation", "before_prompt", "validate_transition"),
            True,
            "v5_button_swap_press_grounding_recovery",
        ),
        PluginRegistration(
            "pickhighlight_pickup_grounding_recovery_gate",
            ("after_observation", "before_prompt", "validate_transition"),
            True,
            "v5_pickhighlight_pickup_grounding_recovery",
        ),
        PluginRegistration(
            "retrieval_overlay",
            ("after_retrieval",),
            False,
            "v5_v6_counting_retrieval",
        ),
        PluginRegistration(
            "action_canonicalizer",
            ("canonicalize_proposal",),
            False,
            "v6_counting_ordinal_canonicalizer",
        ),
        PluginRegistration(
            "reinforcement_overlay",
            ("before_dispatch",),
            True,
            "v8_5_reinforcement_lock",
        ),
        PluginRegistration(
            "python_plugin", HOOK_ORDER, True, "registered_python_entrypoint", False
        ),
        PluginRegistration(
            "core_harness_patch", HOOK_ORDER, True, "materialized_source_commit", False
        ),
    )
}


def registration(component_type: str) -> PluginRegistration:
    try:
        return PLUGIN_REGISTRY[component_type]
    except KeyError as error:
        raise ValueError(
            f"Unregistered intervention component type {component_type!r}"
        ) from error
