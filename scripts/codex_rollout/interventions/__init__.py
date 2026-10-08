"""Typed runtime interventions for train/val self-evolution."""

from .bundle import (
    BUNDLE_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    RuntimeInterventionPlan,
    canonical_hash,
    compose_candidate_manifests,
    load_runtime_intervention_bundle,
    validate_candidate_manifest,
)
from .registry import HOOK_ORDER, PLUGIN_REGISTRY, RuntimeIntervention

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "RuntimeInterventionPlan",
    "canonical_hash",
    "compose_candidate_manifests",
    "load_runtime_intervention_bundle",
    "validate_candidate_manifest",
    "HOOK_ORDER",
    "PLUGIN_REGISTRY",
    "RuntimeIntervention",
]
