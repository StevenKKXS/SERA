"""
Build VLM subgoal prediction dataset for MemER
"""

import json
import os

import cv2
import h5py
import imageio
import numpy as np
import argparse

from mme_vla_suite.dataset_builder.robomme_h5_utils import get_task_goal, get_timestep_indices
from mme_vla_suite.dataset_builder.vlm_subgoal_dataset_base import BaseVLMSubgoalDatasetBuilder

np.set_printoptions(precision=4, suppress=True)


# -----------------------------------------------------------------------------
# Keyframe detection (MemER-specific)
# -----------------------------------------------------------------------------


def find_local_minima(episode_data, timestep_indexs, threshold=0.001, min_distance=5):
    delta_actions = []
    for idx in timestep_indexs:
        state = episode_data[f"timestep_{idx}"]["obs"]["joint_state"][()]
        action = episode_data[f"timestep_{idx}"]["action"]["joint_action"][()][:7]
        delta_action = action - state
        delta_actions.append(np.linalg.norm(delta_action))
    delta_actions = np.array(delta_actions)

    minima_indices = []
    n = len(delta_actions)

    for i in range(1, n - 1):
        if delta_actions[i] >= threshold:
            continue
        if delta_actions[i] >= delta_actions[i - 1] or delta_actions[i] >= delta_actions[i + 1]:
            continue
        if minima_indices and (i - minima_indices[-1] < min_distance):
            continue
        minima_indices.append(i)

    return minima_indices


def get_middle_point(frm_0, frm_1, num=1) -> list:
    return np.linspace(frm_0, frm_1, num + 2)[1:-1].astype(int).tolist()


def merge(transition_idxs: list, local_minima_idx: list, exec_start_idx: int) -> list:
    merged_idxs = transition_idxs + local_minima_idx
    merged_idxs = sorted(list(set(merged_idxs)))
    merged_idxs = [idx for idx in merged_idxs if idx >= exec_start_idx]
    output = [merged_idxs[0]]
    for i in range(1, len(merged_idxs)):
        if abs(merged_idxs[i] - output[-1]) < 10:
            output[-1] = (merged_idxs[i] + output[-1]) // 2
        else:
            output.append(merged_idxs[i])
    return output


# -----------------------------------------------------------------------------
# Text / image helpers (visualization)
# -----------------------------------------------------------------------------


def wrap_text_opencv(text, font, font_scale, max_width, thickness=1):
    """Wrap text to fit within max_width pixels."""
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def put_wrapped_text(image, text, org, font, font_scale, color, thickness=1, line_spacing=20):
    """Put wrapped text on image."""
    x, y = org
    max_width = image.shape[1] - x - 10
    lines = wrap_text_opencv(text, font, font_scale, max_width, thickness)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (x, y + i * line_spacing), font, font_scale, color, thickness)
    return image, y + len(lines) * line_spacing


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

SUBGOAL_SYSTEM_PROMPT = (
    "You are a robot program that predicts actions. The current input images from the front-view camera shows the most recent actions the robot has executed. "
    "The past keyframes are selected frames of particular importance from all the actions the robot has executed so far. Based on these, output the current subtask the robot should execute and nothing else. "
    "Some tasks may have a video input for initial setup, some may not.\n\n"
    "Return a JSON with:\n"
    "- current_subtask: the action that should be executed at the current timestep\n"
    "- keyframe_positions: list of frame positions (1-indexed) from the current input images where actions change"
)


# -----------------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------------


class DatasetBuilder(BaseVLMSubgoalDatasetBuilder):
    def run(self) -> list:
        results = super().run()
        if results:
            print(f"max_frame_len: {max(r or 0 for r in results)}")
        return results

    # -------------------------------------------------------------------------
    # Prompt / text helpers (MemER-specific)
    # -------------------------------------------------------------------------

    def _wrap_keyframes(self, key_frame_paths: list) -> str:
        return "; ".join(
            [f"Past Keyframe {i+1}: <image>" for i in range(len(key_frame_paths))]
        )

    def _wrap_execution_frames(self, execution_frame_paths: list) -> str:
        return "; ".join(
            [
                f"Executed Frame {i+1}: <image>"
                for i in range(len(execution_frame_paths))
            ]
        )

    def _wrap_images(self, image_paths: list) -> str:
        if len(image_paths) == 0:
            return "[]"
        return "[" + ", ".join(["<image>" for _ in image_paths]) + "]"

    # -------------------------------------------------------------------------
    # Simple subgoal data
    # -------------------------------------------------------------------------

    def make_simple_subgoal_data(
        self,
        task_goal: str,
        subgoal: str,
        execution_frame_paths: list,
        key_frame_paths: list,
        video_path: str | None = None,
        candidate_frame_idx: int | None = None,
    ) -> dict:
        video_prefix = (
            "The task has a video input for initial setup: <video>\n" if video_path else ""
        )
        user_prompt = (
            f"{video_prefix}The task goal is: {task_goal}\n"
            f"Here are the selected frames from the entirety of the full execution that are of particular importance:{self._wrap_images(key_frame_paths)}\n"
            f"Here is current input image list from the front-view camera: {self._wrap_images(execution_frame_paths)}\n\n"
            "What subtask should the robot execute and what is the keyframe position?"
        )
        all_image_paths = key_frame_paths + execution_frame_paths

        if candidate_frame_idx is not None:
            assistant_response = f"""{{"current_subtask": "{subgoal}", "keyframe_positions": [{candidate_frame_idx}]}}"""
        else:
            assistant_response = f"""{{"current_subtask":"{subgoal}", "keyframe_positions": []}}"""

        result = {
            "messages": [
                {"role": "system", "content": SUBGOAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "images": all_image_paths,
        }
        if video_path:
            result["videos"] = [video_path]
        return result

    # -------------------------------------------------------------------------
    # Grounded subgoal data
    # -------------------------------------------------------------------------

    def make_grounded_subgoal_data(
        self,
        task_goal: str,
        subgoal: str,
        execution_frame_paths: list,
        key_frame_paths: list,
        video_path: str | None = None,
        candidate_frame_idx: int | None = None,
    ) -> dict:
        video_prefix = (
            "The task has a video input for initial setup: <video>\n" if video_path else ""
        )
        user_prompt = (
            f"{video_prefix}The task goal is: {task_goal}\n"
            f"Here are the selected frames from the entirety of the full execution that are of particular importance:{self._wrap_images(key_frame_paths)}\n"
            f"Here is current input image list from the front-view camera: {self._wrap_images(execution_frame_paths)}\n\n"
            "What subtask should the robot execute and what is the keyframe position?"
        )
        all_image_paths = key_frame_paths + execution_frame_paths
        assistant_prompt, bbox = self._preprocess_grounded_subgoal(subgoal)

        if candidate_frame_idx is not None:
            assistant_response = f"""{{"current_subtask": "{assistant_prompt}", "keyframe_positions": [{candidate_frame_idx}]}}"""
        else:
            assistant_response = f"""{{"current_subtask":"{assistant_prompt}", "keyframe_positions": []}}"""

        result = {
            "messages": [
                {"role": "system", "content": SUBGOAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "objects": {
                "ref": [],
                "bbox": self._add_noise_to_bbox(self.history_grounded_bboxes + bbox),
            },
            "images": all_image_paths,
        }
        if video_path:
            result["videos"] = [video_path]
        return result

    # -------------------------------------------------------------------------
    # Visualization (MemER-specific)
    # -------------------------------------------------------------------------

    def _build_visualization_frame(
        self,
        grounded_subgoal_data: dict,
        memory_frames: list,
        execution_frames: list,
        candidate_frame_idx: int | None,
    ) -> np.ndarray:
        """Build a single visualization image (top text + keyframes + execution frames)."""
        top_image = np.zeros((100, 256 * 8, 3), dtype=np.uint8)
        top_image, _ = put_wrapped_text(
            top_image,
            grounded_subgoal_data["messages"][1]["content"],
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        top_image, _ = put_wrapped_text(
            top_image,
            grounded_subgoal_data["messages"][2]["content"],
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        middle_image = np.zeros((256, 256 * 8, 3), dtype=np.uint8)
        si = 0
        for i, key_image in enumerate(memory_frames):
            if i < len(memory_frames) - 8:
                continue
            key_image = imageio.imread(key_image).copy()
            key_image = cv2.putText(
                key_image, f"KeyFrame {i+1}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )
            middle_image[:, si * 256 : (si + 1) * 256, :] = key_image
            si += 1

        bottom_image = np.zeros((256, 256 * 8, 3), dtype=np.uint8)
        for i, execution_image in enumerate(execution_frames):
            execution_image = imageio.imread(execution_image).copy()
            execution_image = cv2.putText(
                execution_image, f"Frame {i+1}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )
            for bbox in grounded_subgoal_data["objects"]["bbox"]:
                x, y = int(bbox[1]), int(bbox[0])
                execution_image = cv2.circle(
                    execution_image, (x, y), 5, (255, 0, 255), -1
                )
            if i + 1 == candidate_frame_idx:
                execution_image = cv2.rectangle(
                    execution_image, (0, 0), (255, 255), (255, 0, 0), 10
                )
            bottom_image[:, i * 256 : (i + 1) * 256, :] = execution_image

        return np.concatenate([top_image, middle_image, bottom_image], axis=0)

    # -------------------------------------------------------------------------
    # Episode processing
    # -------------------------------------------------------------------------

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ) -> int:
        print(f"processing episode {episode_idx} of {env_id}...")
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []

        episode_data = env_dataset[f"episode_{episode_idx}"]
        task_goal = get_task_goal(episode_data, lower=True)
        timestep_indexs = get_timestep_indices(episode_data)
        exec_start_idx = self._first_execution_step(episode_data)

        # Subgoal transition frame indices
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
        print("transition_idxs: ", transition_idxs)

        if env_id in ["PatternLock", "RouteStick"]:
            local_minima_idx = find_local_minima(episode_data, timestep_indexs)
            key_frame_idx = merge(transition_idxs, local_minima_idx, exec_start_idx)
        else:
            key_frame_idx = transition_idxs.copy()

        if env_id == "PickHighlight":
            frm_0, frm_1 = key_frame_idx[0], key_frame_idx[1]
            key_frame_idx.extend(get_middle_point(frm_0, frm_1, 1))
            key_frame_idx = sorted(key_frame_idx)

        if env_id in "ButtonUnmaskSwap":
            frm_0, frm_1, frm_2 = key_frame_idx[0], key_frame_idx[1], key_frame_idx[2]
            key_frame_idx.extend(get_middle_point(frm_0, frm_1, 2))
            key_frame_idx.extend(get_middle_point(frm_1, frm_2, 2))
            key_frame_idx = sorted(key_frame_idx)

        key_frame_idx.pop()
        print("key_frame_idx: ", key_frame_idx)

        if exec_start_idx > 0:
            video_frames = [
                episode_data[f"timestep_{i}"]["obs"]["front_rgb"][()]
                for i in range(exec_start_idx)
            ]
            init_video_path = os.path.join(
                self.images_dir, f"{env_id}_ep{episode_idx}_video.mp4"
            )
            imageio.mimsave(init_video_path, video_frames, fps=30)
        else:
            init_video_path = None

        last_simple_subgoal = None
        last_grounded_subgoal = None
        if self.visualize:
            save_images = []
            visualization_video_path = os.path.join(
                os.path.dirname(self.images_dir), "visualization"
            )
            os.makedirs(visualization_video_path, exist_ok=True)

        memory_frames = []
        execution_frames = []
        candidate_frame_idx = None
        candidate_frame_image_path = None
        max_frame_len = 0

        for step, idx in enumerate(range(exec_start_idx, len(timestep_indexs), 2)):
            simple_subgoal = episode_data[f"timestep_{idx}"]["info"]["simple_subgoal"][()].decode().lower()
            grounded_subgoal = episode_data[f"timestep_{idx}"]["info"]["grounded_subgoal"][()].decode().lower()
            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if "complete" in grounded_subgoal:
                grounded_subgoal = last_grounded_subgoal

            image = episode_data[f"timestep_{idx}"]["obs"]["front_rgb"][()]
            image_path = os.path.join(
                self.images_dir, f"{env_id}_ep{episode_idx}_step{idx}.png"
            )
            imageio.imwrite(image_path, image)
            execution_frames.append(image_path)

            if candidate_frame_idx is None and key_frame_idx and abs(idx - key_frame_idx[0]) < 2:
                candidate_frame_idx = len(execution_frames)
                candidate_frame_image_path = image_path

            if step % 8 == 0 or idx in [len(timestep_indexs) - 1, len(timestep_indexs) - 2]:
                simple_subgoal_data = self.make_simple_subgoal_data(
                    task_goal, simple_subgoal, execution_frames, memory_frames,
                    init_video_path, candidate_frame_idx,
                )
                grounded_subgoal_data = self.make_grounded_subgoal_data(
                    task_goal, grounded_subgoal, execution_frames, memory_frames,
                    init_video_path, candidate_frame_idx,
                )
                max_frame_len = max(
                    len(memory_frames) + len(execution_frames), max_frame_len
                )

                with open(self.simple_subgoal_train_data_path, "a") as f:
                    f.write(json.dumps(simple_subgoal_data) + "\n")
                with open(self.grounded_subgoal_train_data_path, "a") as f:
                    f.write(json.dumps(grounded_subgoal_data) + "\n")

                if self.visualize:
                    big_image = self._build_visualization_frame(
                        grounded_subgoal_data,
                        memory_frames,
                        execution_frames,
                        candidate_frame_idx,
                    )
                    save_images.append(big_image)

                if candidate_frame_idx is not None:
                    memory_frames.append(candidate_frame_image_path)
                    key_frame_idx.pop(0)
                candidate_frame_idx = None
                execution_frames = []

            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal

        if self.visualize:
            out_path = os.path.join(
                visualization_video_path,
                f"{env_id}_ep{episode_idx}_save_images.mp4",
            )
            imageio.mimsave(out_path, save_images, fps=1)
        return max_frame_len


# def _parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Preprocess raw HDF5 dataset for training")
#     parser.add_argument("--raw_data_path", type=str, default="data/robomme_h5_data", help="Raw HDF5 directory")
#     parser.add_argument("--preprocessed_data_path", type=str, default="data/vlm_subgoal_prediction_data/memer", help="Output directory")
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
