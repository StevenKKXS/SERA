# Agentic execution and self-evolution

The planner combines recent front/wrist observations with a textual progress
state and retrieved executor commands. It emits a grounded subtask for the
fixed RoboMME π0.5 policy. Structured refinement supplies advisory feedback;
runtime control determines the command issued to the executor.

| Component | Entry point |
| --- | --- |
| Planner service | `planner_server.py` |
| Task context and planning | `text_progress_planner.py` |
| Executor command retrieval | `skill_retrieval.py` |
| Simulator client | `run_robomme_batch.py` |
| HTTP client | `remote_planner.py` |
| Failure analysis | `analyze_rollout_failures.py` |
| Self-evolution controller | `self_evolve_v5.py` |
| Paired candidate verification | `verify_self_evolve_v5_candidate.py` |
| Runtime intervention components | `interventions/` |
| Result aggregation | `summarize_results.py` |

See the [root README](../../README.md) for execution and self-evolution entry
points. The base launcher selects `retrieval_v8_5_refine_then_lock` with R0
retrieval and five candidates. An `INTERVENTION_BUNDLE` selects the typed
intervention runtime. Prompts, schemas and knowledge assets are included.

Each episode has independent planner state. Outputs contain an episode summary,
planner requests/responses, timings and optional video. Evaluation resumes from
completed summaries. Infrastructure exceptions may be retried up to the
configured attempt limit; terminal simulator failures are not retried.
Optional privileged environment diagnostics are written for offline analysis
and are never supplied to the runtime planner or policy.
