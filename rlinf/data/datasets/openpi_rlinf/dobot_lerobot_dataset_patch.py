# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patches for Dobot LeRobot v3 datasets with schema/metadata mismatches."""


def apply_dobot_lerobot_hf_dataset_patch() -> None:
    """Load frame parquet without strict HF casting and drop video path columns.

    Dobot NormalData stores:
    - ``observation.velocity`` as float64 lists while info.json declares float32.
    - Video paths as string columns in the data parquet (unlike Cobot exports).

    Loading without an explicit ``features`` schema avoids cast errors. Video path
    columns are then removed so ``__getitem__`` decoded frames are not overwritten
    by path strings.
    """
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    from lerobot.datasets.utils import hf_transform_to_torch, load_nested_dataset

    if getattr(lerobot_dataset.LeRobotDataset, "_rlinf_dobot_hf_dataset_patch", False):
        return

    def load_hf_dataset_patched(self):
        hf_dataset = load_nested_dataset(
            self.root / "data", features=None, episodes=self.episodes
        )
        drop_cols = [key for key in self.meta.video_keys if key in hf_dataset.column_names]
        if drop_cols:
            hf_dataset = hf_dataset.remove_columns(drop_cols)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    lerobot_dataset.LeRobotDataset.load_hf_dataset = load_hf_dataset_patched
    lerobot_dataset.LeRobotDataset._rlinf_dobot_hf_dataset_patch = True
