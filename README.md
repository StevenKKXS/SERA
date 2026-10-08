# SERA: Self-Evolving Robotic Agent

## From VLM Planners to Robotic Agents: Self-Evolving Harnesses for History-Dependent Manipulation

**Tingwen Du, Haoyu Guo, Hanxue Zhang, Xuanxu Wang, Junting Dong, Chao Gao, Mingfei Han, Tao Lu, Mulin Yu, Chunhua Shen, Xiaojun Chang, Zhihui Li**

University of Science and Technology of China · Shanghai Artificial Intelligence Laboratory · Shanghai Jiao Tong University · Mohamed bin Zayed University of Artificial Intelligence · Zhejiang University

<!-- Add the public paper and project-page links here when available. -->

## 🧭 Outline

- [📖 Overview](#overview)
- [🛠️ Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Model Preparation](#model-preparation)
- [🚀 Evaluation](#evaluation)
  - [Running the Base Agent](#running-the-base-agent)
  - [Self-Evolution](#self-evolution)
  - [Running with Evolved Experience](#running-with-evolved-experience)
  - [Results](#results)
- [📝 Citation](#citation)
- [🙏 Acknowledgments](#acknowledgments)
- [📄 License](#license)

<a id="overview"></a>
## 📖 Overview

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

<a id="installation"></a>
## 🛠️ Installation

### Environment Setup

Use Linux with Python 3.11, an NVIDIA CUDA-capable GPU, and a working Vulkan
renderer for RoboMME. The policy server and simulator use separate Python
environments. Running them on separate GPUs avoids memory contention; set
`POLICY_GPU` and `SIM_GPU` to choose devices.

Clone the repository, then run the remaining commands from its root:

```bash
git clone https://github.com/StevenKKXS/SERA.git
cd SERA
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
```

The planner adapter invokes a locally installed, authenticated `codex` CLI.
Set `PLANNER_MODEL` explicitly to a model your CLI can access. Both episode
initialization and online planning use that model with high reasoning effort.
The adapter expects JSON event output, schema-constrained responses, and the
CLI flags assembled in `scripts/codex_rollout/planner.py`; use a compatible CLI.
The model service is an external dependency and can affect run-to-run behavior.

### Model Preparation

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

RoboMME is pinned to submodule commit
`856bc3a189d4172f3f47dbee4424d585f8d78db3`. Follow the
[benchmark installation and asset instructions](https://github.com/RoboMME/robomme_benchmark/tree/856bc3a189d4172f3f47dbee4424d585f8d78db3)
to obtain its simulator assets. Evaluation does not require downloading the
policy-training dataset.

After preparing the simulator assets, check the environment:

```bash
.venv-sim/bin/python examples/robomme/simple_test.py
```

<a id="evaluation"></a>
## 🚀 Evaluation

The commands below evaluate all 16 RoboMME tasks with 50 test episodes per task.
Set the following environment variables in each terminal, from the repository
root:

```bash
export PLANNER_MODEL="<your-planner-model>"
export CHECKPOINT="$PWD/models/mme_vla_suite/symbolic-grounded-subgoal/79999"
export OPENPI_DATA_HOME="$PWD/models/openpi_cache"
export RUN_ROOT="$PWD/outputs/robomme_base"
export TASKS="BinFill,PickXtimes,SwingXtimes,StopCube,VideoUnmask,ButtonUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick"
export NUM_EPISODES=50
export POLICY_GPU=0
export SIM_GPU=1
unset INTERVENTION_BUNDLE
```

`PLANNER_MODEL` selects the planner model; `CHECKPOINT` points to the extracted
policy checkpoint. `RUN_ROOT` stores initialization and evaluation outputs;
`TASKS` and `NUM_EPISODES` select the tasks and episode count. `POLICY_GPU` and
`SIM_GPU` select the policy and simulator GPUs. For a single GPU, set both to
`0` and ensure sufficient memory is available. The launcher accepts configuration
through environment variables; relative paths are resolved from the repository
root.

Start the shared policy server in terminal 1 and leave it running:

```bash
bash scripts/run_robomme.sh policy
```

### Running the Base Agent

[▶ Prepare episode context](#running-the-base-agent)

In terminal 3, capture the benchmark demonstrations and generate the initial
task context:

```bash
bash scripts/run_robomme.sh capture
bash scripts/run_robomme.sh init
```

Capture reads the benchmark's reset observations without executing a policy.
Initialization interprets the demonstrations and prepares guidance for each
episode. Tasks without a demonstration use the available task description.

[▶ Start the planner](#running-the-base-agent)

In terminal 2:

```bash
bash scripts/run_robomme.sh planner
```

[▶ Run evaluation](#running-the-base-agent)

Once both services are ready, run in terminal 3:

```bash
bash scripts/run_robomme.sh eval
bash scripts/run_robomme.sh summary
```

Evaluation writes per-episode summaries under
`RUN_ROOT/evaluation/<Task>/episode_XX/summary.json`. Rerunning `eval` resumes
completed work. Use a fresh `RUN_ROOT` for a different configuration or episode
selection.

### Self-Evolution

Self-evolution improves the agent's harness using training and validation
experience while keeping the low-level policy fixed:

1. **Collect experience:** run training and validation episodes and record the
   agent's decisions and execution feedback.
2. **Analyze failures:** identify recurring errors and the earliest point where
   execution diverges from the task requirements.
3. **Propose revisions:** update planning guidance, action choices, or execution
   checks to address the identified problems.
4. **Verify and select:** compare revisions on paired episodes and retain those
   supported by validation feedback.
5. **Freeze for evaluation:** fix the selected experience before running held-out
   test episodes.

[▶ Initialize self-evolution](#self-evolution)

Initialize a controller for each task listed in the supplied configuration:

```bash
.venv-sim/bin/python - <<'PYTHON'
import json
import subprocess
import sys
from pathlib import Path

config = Path("configs/self_evolution.json")
for task in json.loads(config.read_text())["task_order"]:
    subprocess.run([
        sys.executable, "scripts/codex_rollout/self_evolve_v5.py", "init",
        "--repo-root", str(Path.cwd()),
        "--config", str(config),
        "--run-root", f"outputs/evolution/{task}",
        "--task", task,
    ], check=True)
PYTHON

.venv-sim/bin/python scripts/codex_rollout/self_evolve_v5.py --help
```

`--repo-root` specifies the repository, `--config` selects the evolution
configuration, `--run-root` stores each task's evolution state, and `--task`
selects the task. This command initializes the controllers; it does not execute
the full evolution loop. Subsequent stages consume the collected rollouts,
proposed revisions, and verification results through the controller's
subcommands. Test episodes are not used as evolution feedback.

### Running with Evolved Experience

Use the selected runtime bundle produced by self-evolution, keeping its referenced
files alongside it. Bundles and rollout outputs are generated locally and are
not included in this repository.

[▶ Load the selected experience](#running-with-evolved-experience)

Stop the base planner in terminal 2. In terminals 2 and 3, keep the shared
configuration above and update the following values:

```bash
export RUN_ROOT="$PWD/outputs/robomme_evolved"
export INTERVENTION_BUNDLE="<path-to-selected-runtime-bundle.json>"
export TASKS="<comma-separated-tasks-matching-the-bundle>"
```

`INTERVENTION_BUNDLE` selects the frozen experience. Set `TASKS` to match the
bundle's task scope and use a separate `RUN_ROOT` for each selected bundle.
For tasks without selected revisions, retain the base agent's results.

Prepare context in terminal 3:

```bash
bash scripts/run_robomme.sh capture
bash scripts/run_robomme.sh init
```

Restart the planner in terminal 2; the policy server can remain running:

```bash
bash scripts/run_robomme.sh planner
```

[▶ Evaluate with the selected experience](#running-with-evolved-experience)

In terminal 3:

```bash
bash scripts/run_robomme.sh eval
bash scripts/run_robomme.sh summary
```

Repeat for the remaining selected bundles to cover the benchmark.

### Results

**RoboMME results reported in the paper.** Values are task-averaged success rates
(%; higher is better). Baseline scores are taken from the official leaderboard
as reported in the paper. SERA uses autonomous harness selection before testing,
with 50 test episodes per task. **Bold** indicates the best score in each column.

| Method | Counting | Permanence | Reference | Imitation | Average ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| *VLA baselines* | | | | | |
| π₀.₅ | 28.78 | 17.00 | 17.17 | 8.78 | 17.93 |
| π₀.₅ w/ action history | 29.09 | 22.75 | 14.59 | 11.17 | 19.40 |
| *Memory-augmented policies* | | | | | |
| SAM2Act | 35.33 | 26.00 | 16.83 | 7.33 | 21.37 |
| FrameSamp+Modul | **65.22** | 25.11 | 36.33 | **51.39** | 44.51 |
| *Hierarchical policies* | | | | | |
| SimpleSG+QwenVL | 44.61 | 19.61 | 25.22 | 26.56 | 29.00 |
| GroundSG+Gemini | 12.75 | 15.75 | 11.25 | 6.50 | 11.56 |
| GroundSG+QwenVL | 38.00 | 39.34 | 31.56 | 21.89 | 32.70 |
| MemER | 48.83 | 53.17 | 38.00 | 29.50 | 42.38 |
| **SERA (Ours)** | 49.00 | **57.00** | **69.50** | 30.00 | **51.38** |

**Effect of self-evolution.** The base agent achieves **44.38%** overall success
(355/800); SERA with self-evolution achieves **51.38%** (411/800), an improvement
of **7.00 percentage points**.

<a id="citation"></a>
## 📝 Citation

```bibtex
@misc{du2026sera,
  title={From VLM Planners to Robotic Agents: Self-Evolving Harnesses for History-Dependent Manipulation},
  author={Tingwen Du and Haoyu Guo and Hanxue Zhang and Xuanxu Wang and Junting Dong and Chao Gao and Mingfei Han and Tao Lu and Mulin Yu and Chunhua Shen and Xiaojun Chang and Zhihui Li},
  year={2026},
  note={Preprint}
}
```

<a id="acknowledgments"></a>
## 🙏 Acknowledgments

We thank [RoboMME](https://github.com/RoboMME/robomme_policy_learning),
[OpenPI](https://github.com/Physical-Intelligence/openpi), and
[MemER](https://github.com/memer-policy/memer) for their open research and software.

<a id="license"></a>
## 📄 License

The code is released under the [Apache License 2.0](LICENSE).
See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party code attribution.
Model weights, data, and simulator assets retain their upstream terms.
