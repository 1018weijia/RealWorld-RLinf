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

"""Patch HuggingFace datasets to accept legacy ``List`` feature metadata.

Some Dobot LeRobot v3 exports embed ``_type: List`` in episode parquet
metadata; current ``datasets`` expects ``Sequence`` instead.
"""


def _rewrite_list_feature_types(obj):
    if isinstance(obj, dict):
        out = dict(obj)
        if out.get("_type") == "List":
            out["_type"] = "Sequence"
        return {key: _rewrite_list_feature_types(val) for key, val in out.items()}
    if isinstance(obj, list):
        return [_rewrite_list_feature_types(item) for item in obj]
    return obj


def apply_lerobot_list_feature_patch() -> None:
    """Rewrite legacy ``List`` dtypes to ``Sequence`` before HF feature parsing."""
    import datasets.features.features as features_mod

    if getattr(features_mod, "_rlinf_list_feature_patch", False):
        return

    original = features_mod.generate_from_dict

    def generate_from_dict_patched(obj):
        return original(_rewrite_list_feature_types(obj))

    generate_from_dict_patched._rlinf_list_feature_patch = True  # type: ignore[attr-defined]
    features_mod.generate_from_dict = generate_from_dict_patched
    features_mod._rlinf_list_feature_patch = True
