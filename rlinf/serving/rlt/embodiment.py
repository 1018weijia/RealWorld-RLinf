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

"""The per-robot contract a Stage 2 server serves, in one object.

Stage 2 itself is robot-agnostic: the learner, the replay buffer and the
WebSocket protocol are identical for every embodiment, and the joint-versus-
end-effector difference lives entirely in the frozen Stage 1 OpenPI transform
pipeline selected by ``openpi_config_name``. What actually differs per robot is
a small set of numbers -- tensor widths, the executed chunk horizon and the
camera layout.

Those numbers used to be spelled out in four separate config sections, with
``action_dim`` written three times and nothing checking that the three agreed.
A disagreement is not a loud failure: a Stage 2 head built for 14 actions on
top of a Stage 1 pipeline configured for 16 mostly runs, and produces wrong
robot units. :class:`EmbodimentProfile` gathers the contract into one place and
:meth:`EmbodimentProfile.check_config` makes a disagreement a startup error.

The config group under ``examples/embodiment/config/embodiment/`` stays the
source of truth -- this module reads it, never overwrites it -- so adding a
robot means adding one YAML file, not editing Python.

Deliberately excluded: the task prompt, the norm statistics and the Stage 1
checkpoint. Those vary per *task* on a fixed robot, and folding them in here
would force a new embodiment for every new skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["EmbodimentError", "EmbodimentProfile"]


class EmbodimentError(RuntimeError):
    """The embodiment contract is missing or disagrees with the config."""


def _require(cfg: Any, path: str) -> Any:
    """Read a dotted config path, reporting the full path when it is absent."""
    node = cfg
    for part in path.split("."):
        if node is None:
            break
        node = node.get(part) if hasattr(node, "get") else getattr(node, part, None)
    if node is None:
        raise EmbodimentError(f"Required config key {path!r} is not set")
    return node


@dataclass(frozen=True)
class EmbodimentProfile:
    """Everything that varies between robots on a Stage 2 server.

    Attributes:
        name: Identifier of the config group entry, e.g. ``x2robot``.
        robot_type: Embodiment reported in the handshake. Clients use it to
            refuse a server built for another robot; two embodiments can share
            every tensor width and still be incompatible.
        action_schema: Versioned meaning of the action vector, e.g.
            ``x2robot-ee14-v1``. Distinguishes 14 joint angles from a 14-wide
            dual-arm end-effector pose, which are otherwise indistinguishable.
        action_dim: Width of one robot-space action.
        proprio_dim: Width of the proprioceptive state the client sends.
        chunk_length: Actions the robot executes per chunk. One chunk is one
            replay transition, so this sets the unit of training data.
        ref_chunk_length: Horizon the frozen Stage 1 VLA proposes. May exceed
            ``chunk_length``: Cobot executes 16 of the 20 proposed actions.
        camera_keys: Client camera keys, main view first and wrist views after.
            Order is load-bearing -- the Aloha transform reads the wrist views
            as left then right, so swapping them mirrors the robot.
        openpi_config_name: Named OpenPI data config that supplies the whole
            robot-specific transform pipeline: camera repack, delta/absolute
            action masks, Aloha decode and un-normalization.
    """

    name: str
    robot_type: str
    action_schema: str
    action_dim: int
    proprio_dim: int
    chunk_length: int
    ref_chunk_length: int
    camera_keys: tuple[str, ...]
    openpi_config_name: str

    @property
    def main_camera(self) -> str:
        """The primary view fed to the first OpenPI image slot."""
        return self.camera_keys[0]

    @property
    def wrist_cameras(self) -> tuple[str, ...]:
        """The remaining views, in Aloha left-then-right order."""
        return self.camera_keys[1:]

    @classmethod
    def from_config(cls, cfg: Any) -> EmbodimentProfile:
        """Build a profile from the ``embodiment`` section of a server config.

        Args:
            cfg: Full server config, carrying a top-level ``embodiment`` group.

        Returns:
            The parsed profile.

        Raises:
            EmbodimentError: The section is absent, a key is missing, or the
                declared values are not self-consistent.
        """
        section = cfg.get("embodiment") if hasattr(cfg, "get") else None
        if section is None:
            raise EmbodimentError(
                "No 'embodiment' section in the server config. Select one with "
                "a Hydra default, e.g. 'defaults: [embodiment: x2robot]'."
            )
        profile = cls(
            name=str(_require(section, "name")),
            robot_type=str(_require(section, "robot_type")),
            action_schema=str(_require(section, "action_schema")),
            action_dim=int(_require(section, "action_dim")),
            proprio_dim=int(_require(section, "proprio_dim")),
            chunk_length=int(_require(section, "chunk_length")),
            ref_chunk_length=int(_require(section, "ref_chunk_length")),
            camera_keys=tuple(str(key) for key in _require(section, "camera_keys")),
            openpi_config_name=str(_require(section, "openpi_config_name")),
        )
        profile._check_self()
        return profile

    def _check_self(self) -> None:
        """Reject a profile that cannot describe a working robot."""
        for field, value in (
            ("action_dim", self.action_dim),
            ("proprio_dim", self.proprio_dim),
            ("chunk_length", self.chunk_length),
            ("ref_chunk_length", self.ref_chunk_length),
        ):
            if value <= 0:
                raise EmbodimentError(
                    f"embodiment.{field}={value} must be positive ({self.name})"
                )
        if self.ref_chunk_length < self.chunk_length:
            raise EmbodimentError(
                f"embodiment.ref_chunk_length={self.ref_chunk_length} is shorter "
                f"than chunk_length={self.chunk_length} ({self.name}): the robot "
                "cannot execute more actions than Stage 1 proposes."
            )
        if not self.camera_keys:
            raise EmbodimentError(f"embodiment.camera_keys is empty ({self.name})")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise EmbodimentError(
                f"Duplicate embodiment.camera_keys {list(self.camera_keys)} "
                f"({self.name}); each OpenPI image slot needs a distinct camera."
            )

    def check_config(self, cfg: Any) -> None:
        """Verify the rest of the config agrees with this profile.

        The learner and the Stage 1 wrapper read the embodiment numbers from
        their own config section, which is expected to interpolate from
        ``embodiment``. A hand-written override can quietly break that link,
        and the resulting mismatch degrades the policy instead of raising.

        Only sections that something actually consumes are checked. The
        handshake is not among them: it is built from this profile directly, so
        verifying a config copy of it would be a tautology.

        Args:
            cfg: Full server config.

        Raises:
            EmbodimentError: A section disagrees with the profile.
        """
        expected = {
            "actor.model.action_dim": self.action_dim,
            "actor.model.proprio_dim": self.proprio_dim,
            "actor.model.num_action_chunks": self.chunk_length,
            "actor.model.ref_num_action_chunks": self.ref_chunk_length,
            "rlt_feature_model.action_dim": self.action_dim,
            "rlt_feature_model.num_action_chunks": self.ref_chunk_length,
            "rlt_feature_model.openpi.num_images_in_input": len(self.camera_keys),
        }
        mismatches = [
            f"  {path}={actual!r} but embodiment {self.name!r} declares {value!r}"
            for path, value in expected.items()
            if (actual := int(_require(cfg, path))) != value
        ]

        config_name = str(_require(cfg, "rlt_feature_model.openpi.config_name"))
        if config_name != self.openpi_config_name:
            mismatches.append(
                f"  rlt_feature_model.openpi.config_name={config_name!r} but "
                f"embodiment {self.name!r} declares "
                f"{self.openpi_config_name!r}"
            )

        if mismatches:
            raise EmbodimentError(
                "Config disagrees with the selected embodiment:\n"
                + "\n".join(mismatches)
                + "\n\nThese keys are meant to interpolate from 'embodiment', "
                "e.g. 'action_dim: ${embodiment.action_dim}'. A mismatch is not "
                "a loud failure at runtime -- it produces wrong robot units."
            )
