"""Base class for VLM subgoal dataset builders (QwenVL, MemER).

Shared setup, H5 iteration, and grounded-subgoal/bbox helpers.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import cv2
import h5py
import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    add_noise_to_bbox,
    first_execution_step,
    get_env_id_from_filename,
    get_episode_indices,
    preprocess_grounded_subgoal,
    wrap_history_subgoals,
)

if TYPE_CHECKING:
    pass


class BaseVLMSubgoalDatasetBuilder:
    """Base for building VLM subgoal JSONL datasets from RoboMME HDF5."""

    def __init__(
        self,
        raw_data_path: str = "data/robomme_h5_data",
        preprocessed_data_path: str = "data/robomme_preprocessed_data",
        max_episodes: int | None = None,
        visualize: bool = False,
        vlm_dir_name: str = "vlm_subgoal",
    ) -> None:
        self.raw_data_path = raw_data_path
        self.preprocessed_data_path = preprocessed_data_path
        self.max_episodes = max_episodes
        self.visualize = visualize

        self.data_dir = os.path.join(preprocessed_data_path, vlm_dir_name)
        self.images_dir = os.path.join(self.data_dir, "images")
        self.simple_subgoal_train_data_path = os.path.join(
            self.data_dir, "simple_subgoal_train.jsonl"
        )
        self.grounded_subgoal_train_data_path = os.path.join(
            self.data_dir, "grounded_subgoal_train.jsonl"
        )
        self._setup_output_dirs()
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []

    def _setup_output_dirs(self) -> None:
        if os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
        os.makedirs(self.images_dir, exist_ok=True)
        if os.path.exists(self.simple_subgoal_train_data_path):
            os.remove(self.simple_subgoal_train_data_path)
        if os.path.exists(self.grounded_subgoal_train_data_path):
            os.remove(self.grounded_subgoal_train_data_path)

    def run(self) -> list:
        """Process all H5 files and episodes. Returns list of process_per_episode return values."""
        results: list = []
        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".h5"):
                continue
            print(f"\nprocessing file: {file}")
            with h5py.File(os.path.join(self.raw_data_path, file), "r") as data:
                env_id = get_env_id_from_filename(file)
                episode_indices = get_episode_indices(data, self.max_episodes)
                for episode_idx in episode_indices:
                    r = self.process_per_episode(data, env_id, episode_idx)
                    results.append(r)
        return results

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ):
        """Process one episode; subclasses must implement. Return value is builder-specific."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared helpers (delegate to robomme_h5_utils)
    # -------------------------------------------------------------------------

    def _first_execution_step(self, episode_data: h5py.Group) -> int:
        return first_execution_step(episode_data)

    def _preprocess_grounded_subgoal(self, subgoal: str) -> tuple[str, list]:
        return preprocess_grounded_subgoal(subgoal)

    def _add_noise_to_bbox(self, bbox: list) -> list:
        return add_noise_to_bbox(bbox)

    def _wrap_history_subgoals(self, subgoals: list) -> str:
        return wrap_history_subgoals(subgoals)

    def combine_image_and_wrist_image(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        simple_subgoal: str,
    ) -> np.ndarray:
        """Horizontal stack of image and wrist_image with subgoal text overlay."""
        output = np.concatenate([image, wrist_image], axis=1)
        output = cv2.putText(
            output,
            simple_subgoal,
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return output
