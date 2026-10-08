# Initial-demonstration interpretation

`capture_init.py` resets the selected RoboMME test episodes and writes a JSONL
manifest with `case_id`, `task`, `episode`, `task_description`, and `video_path`.
Only demonstration frames preceding the execution observation are saved.
Tasks without a demonstration have a null video path.

`run_init_video.py run` samples the demonstration at 10 FPS and combines it with
the task specification to generate a detailed task description and structured
planner guidance. The launcher selects `planner_guidance_v3`. Responses and
frame caches can be reused for the same episodes.

See the [Quick Start](../../README.md#quick-start). No existing baseline run or
experiment directory is needed.
