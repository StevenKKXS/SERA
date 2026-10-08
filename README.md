# SERA: Self-Evolving Robotic Agent

## From VLM Planners to Robotic Agents: Self-Evolving Harnesses for History-Dependent Manipulation

**Tingwen Du, Haoyu Guo, Hanxue Zhang, Xuanxu Wang, Junting Dong, Chao Gao, Mingfei Han, Tao Lu, Mulin Yu, Chunhua Shen, Xiaojun Chang, Zhihui Li**

University of Science and Technology of China · Shanghai Artificial Intelligence Laboratory · Shanghai Jiao Tong University · Mohamed bin Zayed University of Artificial Intelligence · Zhejiang University

<!-- Add the public paper and project-page links here when available. -->

[Overview](#overview) · [Installation](#installation) · [Quick Start](#quick-start) · [Self-Evolution](#self-evolution) · [Citation](#citation)

## Overview

SERA organizes a vision-language model as a robotic agent through an editable
harness for history-dependent manipulation. The harness maintains task context,
checks grounded commands, and incorporates execution feedback while a fixed
vision-language-action policy provides low-level control. Execution-driven
self-evolution diagnoses rollout failures and proposes, verifies, and selects
harness revisions before held-out evaluation.

![Agentic execution and harness self-evolution](assets/method_overview.svg)

This repository provides the RoboMME implementation: episode initialization,
context-aware planning, executor command retrieval, typed harness interventions,
self-evolution controllers, and evaluation. The executor uses the released
RoboMME grounded-subgoal π0.5 checkpoint. Model weights and benchmark assets are
downloaded separately.

## Installation

Use Linux with Python 3.11, an NVIDIA CUDA-capable GPU, and a working Vulkan
renderer for RoboMME. The policy server and simulator use separate Python
environments. Running them on separate GPUs avoids memory contention; set
`POLICY_GPU` and `SIM_GPU` to choose devices.

From the repository root:

```bash
git submodule update --init --recursive
python3.11 -m pip install uv
uv venv --python 3.11 .venv-policy
```

Install the policy and simulator dependencies:

```bash
UV_PROJECT_ENVIRONMENT=.venv-policy uv sync --python 3.11 --no-dev
uv venv --python 3.11 .venv-sim
uv pip install --python .venv-sim/bin/python -r examples/robomme/requirements.txt
uv pip install --python .venv-sim/bin/python -e third_party/robomme_benchmark -e packages/openpi-client
.venv-sim/bin/python examples/robomme/simple_test.py
```

RoboMME is pinned to submodule commit
`856bc3a189d4172f3f47dbee4424d585f8d78db3`. Follow the
[benchmark installation and asset instructions](https://github.com/RoboMME/robomme_benchmark/tree/856bc3a189d4172f3f47dbee4424d585f8d78db3)
to obtain its simulator assets. Evaluation does not require downloading the
policy-training dataset.

The planner adapter invokes a locally installed, authenticated `codex` CLI.
Set `PLANNER_MODEL` explicitly to a model your CLI can access. Both episode
initialization and online planning use that model with high reasoning effort.
The adapter expects JSON event output, schema-constrained responses, and the
CLI flags assembled in `scripts/codex_rollout/planner.py`; use a compatible CLI.
The model service is an external dependency and can affect run-to-run behavior.

## Model checkpoint

Download the upstream grounded-subgoal checkpoint at a fixed revision:

```bash
uv pip install --python .venv-sim/bin/python huggingface_hub
.venv-sim/bin/hf download Yinpei/mme_vla_suite \
  symbolic-grounded-subgoal/79999.zip \
  symbolic-grounded-subgoal/history_config.txt \
  --revision 5db4d53ddb98c7f80cab08792dd53d985d712ab1 \
  --local-dir models/mme_vla_suite

(cd models/mme_vla_suite/symbolic-grounded-subgoal && sha256sum -c ../../../configs/checkpoint.sha256)
.venv-sim/bin/python scripts/unzip_ckpt.py models/mme_vla_suite
export CHECKPOINT="$PWD/models/mme_vla_suite/symbolic-grounded-subgoal/79999"
export OPENPI_DATA_HOME="$PWD/models/openpi_cache"
```

Keep `history_config.txt` alongside the `79999` directory. The policy loader
uses it to configure the executor's visual history.

## Quick Start

The following runs one `MoveCube` test episode. Set the same environment values
in each terminal, and run commands from the repository root:

```bash
export PLANNER_MODEL="<your-planner-model>"
export CHECKPOINT="$PWD/models/mme_vla_suite/symbolic-grounded-subgoal/79999"
export OPENPI_DATA_HOME="$PWD/models/openpi_cache"
export RUN_ROOT="$PWD/outputs/movecube"
export TASKS=MoveCube
export NUM_EPISODES=1
export POLICY_GPU=0
export SIM_GPU=1
```

First, capture and interpret initial demonstrations:

```bash
bash scripts/run_robomme.sh capture
bash scripts/run_robomme.sh init
```

Capture reads benchmark-provided reset observations without executing a policy.
For tasks without a demonstration, the manifest records a null video path.
Initialization produces a detailed task description and planner guidance for
each episode. Use a new `RUN_ROOT` when changing the task or episode selection.

Start the policy server in terminal 1:

```bash
bash scripts/run_robomme.sh policy
```

Start the planner service in terminal 2:

```bash
bash scripts/run_robomme.sh planner
```

After both services are ready, evaluate in terminal 3:

```bash
bash scripts/run_robomme.sh eval
bash scripts/run_robomme.sh summary
```

The base harness is used when `INTERVENTION_BUNDLE` is unset. Each episode
receives synchronized front/wrist observations, a compact textual task context,
and up to five retrieved executor commands. The action chunk length is 16;
the planner samples recent frames with stride 2. The launcher uses the default
1300-step episode limit.

Outputs are written under `RUN_ROOT`: initialization manifests and responses,
`evaluation/<Task>/episode_XX/summary.json`, planner traces, timings, and optional
videos. Restarting `eval` resumes completed work. Runtime outputs are ignored
by Git.

## All RoboMME tasks

Use the same capture, initialization, and evaluation commands with:

```bash
export RUN_ROOT="$PWD/outputs/robomme"
export NUM_EPISODES=50
export TASKS="BinFill,PickXtimes,SwingXtimes,StopCube,VideoUnmask,ButtonUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick"
```

The batch runner also supports explicit episode selection and sharding. Run
`python scripts/codex_rollout/run_robomme_batch.py --help` for the full interface.
Use independent policy sessions when running concurrent episodes.

## Self-Evolution

Self-evolution uses training and validation rollouts to diagnose failures,
revise a typed harness intervention, and compare it with the current harness
on paired episodes. The controller tracks candidate lineage, verification,
retrospectives, and selection. The selected bundle is fixed before testing;
task context continues to update within each test episode.

The implementation is organized around the following entry points:

| Stage | Code |
| --- | --- |
| Sample training/validation episodes | `episode_sampling.py`, `prepare_evolution_dataset.py` |
| Capture initialization context | `capture_robomme_init_demos.py`, `prepare_evolution_init_manifest.py` |
| Analyze execution and task-progress failures | `analyze_rollout_failures.py`, `prepare_v5_problem_records.py` |
| Prepare and compose typed repairs | `prepare_self_evolve_v5_repair_request.py`, `build_self_evolve_v4_candidate.py` |
| Manage revision and selection | `self_evolve_v5.py` |
| Verify paired candidate rollouts | `verify_self_evolve_v5_candidate.py` |
| Load selected revisions at runtime | `interventions/`, `text_progress_planner.py` |

These files are under `scripts/codex_rollout/`. Configuration is provided in
[`configs/self_evolution.json`](configs/self_evolution.json); JSON schemas and
analysis prompts are included alongside the implementation. Initialize a task
controller and inspect its supported transitions with:

```bash
.venv-sim/bin/python scripts/codex_rollout/self_evolve_v5.py init \
  --repo-root "$PWD" --config configs/self_evolution.json \
  --run-root outputs/evolution/MoveCube --task MoveCube
.venv-sim/bin/python scripts/codex_rollout/self_evolve_v5.py --help
```

The controller consumes rollout records, candidate manifests, paired-panel
verification reports, and retrospectives. Initializing it does not launch a
complete evolution run. Use each stage's `--help` and the schemas to supply
those inputs. Test episodes are reserved for evaluation and cannot be used as
evolution feedback.

To evaluate a selected runtime bundle, restart the planner service with:

```bash
export INTERVENTION_BUNDLE="/path/to/selected-runtime-bundle.json"
bash scripts/run_robomme.sh planner
```

Then run `eval` with the matching task selection and a fresh output directory
containing the corresponding initialization responses. Candidate manifests and
selected runtime bundles are generated by the evolution procedure; experiment
outputs are not bundled with the source code.

## Repository layout

```text
assets/                      Method illustration and normalization reference
configs/                     Resource checksums and self-evolution configuration
examples/robomme/            Simulator adapter
packages/openpi-client/      Executor client
scripts/codex_init_video/    Episode initialization
scripts/codex_rollout/       Planning, self-evolution, and evaluation
scripts/run_robomme.sh       Single-machine launcher
src/                        RoboMME/OpenPI executor implementation
third_party/robomme_benchmark/  Pinned benchmark submodule
```

## Tests

The planner and self-evolution unit tests use synthetic inputs and mock model
responses; they do not need model weights or a simulator GPU:

```bash
.venv-sim/bin/python -m unittest discover -s scripts/codex_init_video -p 'test_*.py'
.venv-sim/bin/python -m unittest discover -s scripts/codex_rollout -p 'test_*.py'
```

## Citation

```bibtex
@misc{du2026sera,
  title={From VLM Planners to Robotic Agents: Self-Evolving Harnesses for History-Dependent Manipulation},
  author={Tingwen Du and Haoyu Guo and Hanxue Zhang and Xuanxu Wang and Junting Dong and Chao Gao and Mingfei Han and Tao Lu and Mulin Yu and Chunhua Shen and Xiaojun Chang and Zhihui Li},
  year={2026},
  note={Preprint}
}
```

## Acknowledgments and license

We thank [RoboMME](https://github.com/RoboMME/robomme_policy_learning),
[OpenPI](https://github.com/Physical-Intelligence/openpi), and
[MemER](https://github.com/memer-policy/memer) for their open research and software.
See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md) for code attribution.
Model weights, data, and simulator assets retain their upstream terms.
