"""
Build VLM subgoal prediction dataset for QwenVL.

We duplicate keyframe training samples for balanced training data, which is
crucial for the VLM to predict correct subgoal changes.
"""

import json
import os

import cv2
import h5py
import imageio
import numpy as np

from mme_vla_suite.dataset_builder.vlm_subgoal_dataset_base import BaseVLMSubgoalDatasetBuilder


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

SIMPLE_SUBGOAL_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of language subgoals"
)
GROUNDED_SUBGOAL_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of grounded language subgoals"
)


# -----------------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------------


class DatasetBuilder(BaseVLMSubgoalDatasetBuilder):
    # -------------------------------------------------------------------------
    # Simple subgoal data
    # -------------------------------------------------------------------------

    def make_simple_subgoal_data(
        self,
        task_goal: str,
        subgoal: str,
        image_path: str,
        video_path: str | None = None,
    ) -> dict:
        video_prefix = "<video>" if video_path else ""
        if len(self.history_simple_subgoals) == 0:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                "This is the initial turn for prediction\n"
                "<image>What's the next language subgoal based on current observation?"
            )
        else:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                f"The history of previous predicted language subgoals are: {self._wrap_history_subgoals(self.history_simple_subgoals)}\n"
                "<image>What's the next language subgoal based on current observation?"
            )

        result = {
            "messages": [
                {"role": "system", "content": SIMPLE_SUBGOAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": subgoal},
            ],
            "images": [image_path],
        }
        if video_path:
            result["videos"] = [video_path]

        if self.history_simple_subgoals:
            if self.history_simple_subgoals[-1] != subgoal:
                self.history_simple_subgoals.append(subgoal)
        else:
            self.history_simple_subgoals.append(subgoal)

        return result

    # -------------------------------------------------------------------------
    # Grounded subgoal data
    # -------------------------------------------------------------------------

    def make_grounded_subgoal_data(
        self,
        task_goal: str,
        subgoal: str,
        image_path: str,
        video_path: str | None = None,
    ) -> dict:
        video_prefix = "<video>" if video_path else ""
        assistant_prompt, bbox = self._preprocess_grounded_subgoal(subgoal)

        if len(self.history_grounded_subgoals) == 0:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                "This is the initial turn for prediction\n"
                "<image>What's the next grounded language subgoal based on current observation?"
            )
        else:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                f"The history of previous predicted grounded language subgoals are: {self._wrap_history_subgoals(self.history_grounded_subgoals)}\n"
                "<image>What's the next grounded language subgoal based on current observation?"
            )

        result = {
            "messages": [
                {"role": "system", "content": GROUNDED_SUBGOAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt},
            ],
            "objects": {
                "ref": [],
                "bbox": self._add_noise_to_bbox(self.history_grounded_bboxes + bbox),
            },
            "images": [image_path],
        }
        if video_path:
            result["videos"] = [video_path]

        if self.history_grounded_subgoals:
            if self.history_grounded_subgoals[-1] != assistant_prompt:
                self.history_grounded_subgoals.append(assistant_prompt)
                self.history_grounded_bboxes.extend(bbox)
        else:
            self.history_grounded_subgoals.append(assistant_prompt)
            self.history_grounded_bboxes.extend(bbox)

        return result

    # -------------------------------------------------------------------------
    # Transition / selection (QwenVL-specific)
    # -------------------------------------------------------------------------

    def _compute_transition_idxs(
        self,
        episode_data: h5py.Group,
        env_id: str,
        exec_start_idx: int,
        timestep_indexs: list[int],
    ) -> list[int]:
        """Compute subgoal transition frame indices."""
        if "PatternLock" in env_id:
            return np.arange(
                exec_start_idx, len(timestep_indexs), 32
            ).astype(np.int32).tolist()

        transition_idxs = []
        last_simple_subgoal = None
        idx = exec_start_idx
        while idx < len(timestep_indexs):
            simple_subgoal = episode_data[f"timestep_{idx}"]["info"]["simple_subgoal"][()].decode().lower()
            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if simple_subgoal != last_simple_subgoal:
                transition_idxs.append(idx)
            last_simple_subgoal = simple_subgoal
            idx += 1
        transition_idxs.append(len(timestep_indexs) - 1)
        return transition_idxs

    def _compute_select_and_duplicate_idxs(
        self,
        transition_idxs: list[int],
        num_timesteps: int,
        env_id: str,
    ) -> tuple[list[int], dict]:
        """Compute selected frame indices and how many times to duplicate each keyframe."""
        stride = 32 if "StopCube" in env_id else 16
        select_idxs = []
        duplicate_idxs = {}

        for start_idx, end_idx in zip(transition_idxs[:-1], transition_idxs[1:]):
            mid_number = (end_idx - start_idx) // stride
            mid_idxs = np.linspace(
                start_idx, end_idx, mid_number, endpoint=False
            ).astype(np.int32).tolist()
            select_idxs.extend(mid_idxs)
            duplicate_idxs[end_idx] = max(mid_number - 1, 0)

        duplicate_idxs.pop(num_timesteps - 1, None)
        select_idxs.append(num_timesteps - 1)
        select_idxs = sorted(list(set(select_idxs)))
        return select_idxs, duplicate_idxs

    def _append_training_rows(
        self,
        simple_subgoal_data: dict,
        grounded_subgoal_data: dict,
        times: int = 1,
    ) -> None:
        """Append training rows to the JSONL files."""
        for _ in range(times):
            with open(self.simple_subgoal_train_data_path, "a") as f:
                f.write(json.dumps(simple_subgoal_data) + "\n")
            with open(self.grounded_subgoal_train_data_path, "a") as f:
                f.write(json.dumps(grounded_subgoal_data) + "\n")

    # -------------------------------------------------------------------------
    # Episode processing
    # -------------------------------------------------------------------------

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ) -> None:
        from mme_vla_suite.dataset_builder.robomme_h5_utils import get_timestep_indices, get_task_goal

        print(f"processing episode {episode_idx} of {env_id}...")
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []

        episode_data = env_dataset[f"episode_{episode_idx}"]
        task_goal = get_task_goal(episode_data, lower=True)
        timestep_indexs = get_timestep_indices(episode_data)
        exec_start_idx = self._first_execution_step(episode_data)

        transition_idxs = self._compute_transition_idxs(
            episode_data, env_id, exec_start_idx, timestep_indexs
        )
        if transition_idxs[-1] != len(timestep_indexs) - 1:
            transition_idxs.append(len(timestep_indexs) - 1)
        print("transition_idxs: ", transition_idxs)

        select_idxs, duplicate_idxs = self._compute_select_and_duplicate_idxs(
            transition_idxs, len(timestep_indexs), env_id
        )
        print("select_idxs: ", select_idxs)

        if exec_start_idx > 0:
            video_frames = [
                episode_data[f"timestep_{i}"]["obs"]["front_rgb"][()]
                for i in range(exec_start_idx)
            ]
            video_path = os.path.join(
                self.images_dir, f"{env_id}_ep{episode_idx}_video.mp4"
            )
            imageio.mimsave(video_path, video_frames, fps=30)
        else:
            video_path = None

        last_simple_subgoal = None
        last_grounded_subgoal = None
        if self.visualize:
            save_images = []
            visualization_video_path = os.path.join(
                os.path.dirname(self.images_dir), "visualization"
            )
            os.makedirs(visualization_video_path, exist_ok=True)

        for idx in select_idxs:
            image = episode_data[f"timestep_{idx}"]["obs"]["front_rgb"][()]
            simple_subgoal = episode_data[f"timestep_{idx}"]["info"]["simple_subgoal"][()].decode().lower()
            grounded_subgoal = episode_data[f"timestep_{idx}"]["info"]["grounded_subgoal"][()].decode().lower()

            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if "complete" in grounded_subgoal:
                grounded_subgoal = last_grounded_subgoal

            image_path = os.path.join(
                self.images_dir, f"{env_id}_ep{episode_idx}_step{idx}.png"
            )
            imageio.imwrite(image_path, image)

            simple_subgoal_data = self.make_simple_subgoal_data(
                task_goal, simple_subgoal, image_path, video_path
            )
            grounded_subgoal_data = self.make_grounded_subgoal_data(
                task_goal, grounded_subgoal, image_path, video_path
            )

            self._append_training_rows(simple_subgoal_data, grounded_subgoal_data)

            if self.visualize:
                vis_image = image.copy()
                vis_image = cv2.putText(
                    vis_image, f"Step {idx}: {simple_subgoal}", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                save_images.append(vis_image)

                dup_count = duplicate_idxs.get(idx, 0)
                if dup_count > 0:
                    print(f"duplicate {idx} for {dup_count} more times")
                    self._append_training_rows(
                        simple_subgoal_data, grounded_subgoal_data, times=dup_count
                    )
                    for _ in range(dup_count):
                        dup_image = image.copy()
                        dup_image = cv2.putText(
                            dup_image, f"Duplicate: {simple_subgoal}", (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )
                        save_images.append(dup_image)

            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal

        if self.visualize:
            out_path = os.path.join(
                visualization_video_path,
                f"{env_id}_ep{episode_idx}_save_images.mp4",
            )
            imageio.mimsave(out_path, save_images, fps=1)


# def _parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Preprocess raw HDF5 dataset for training")
#     parser.add_argument("--raw_data_path", type=str, default="data/robomme_h5_data", help="Raw HDF5 directory")
#     parser.add_argument("--preprocessed_data_path", type=str, default="data/vlm_subgoal_prediction_data/qwenvl", help="Output directory")
#     parser.add_argument("--max_episodes", type=int, default=None, help="Cap episodes per file (default: all)")
#     parser.add_argument("--visualize", action="store_true", help="Write visualization MP4s")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = _parse_args()
#     builder = DatasetBuilder(
#         raw_data_path=args.raw_data_path,
#         preprocessed_data_path=args.preprocessed_data_path,
#         max_episodes=args.max_episodes,
#         visualize=args.visualize,
#     )
#     builder.run()
