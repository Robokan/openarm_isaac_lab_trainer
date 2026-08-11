#!/usr/bin/env python3
"""Convert OpenArm VR teleop recordings to native LeRobot v3.0 datasets.

Reads the fallback format written by teleop_bimanual.py / teleop_xr.sh:

    vla_teleop_data/
      metadata.json
      episodes/episode_N/
        data.parquet          # observation.state.0..N, action.0..N
        ego|left_wrist|right_wrist/frame_*.jpg   # optional
        metadata.json

And writes a LeRobot v3.0 dataset (compatible with /sparkpack/lerobot):

    <output>/
      meta/info.json
      meta/tasks.parquet
      meta/episodes/...
      data/chunk-*/file-*.parquet
      videos/observation.images.<cam>/chunk-*/file-*.mp4

Handles both 22-dim (legacy) and 16-dim episodes. 22-dim is remapped to
ALOHA-style 16-dim: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)].

Usage:
    # Convert default teleop data next to this repo:
    python scripts/convert_teleop_to_lerobot.py \\
        --input vla_teleop_data \\
        --output vla_teleop_data_lerobot

    # With the sibling lerobot checkout on PYTHONPATH (auto-detected):
    python scripts/convert_teleop_to_lerobot.py --input vla_teleop_data

    # Limit / skip for testing:
    python scripts/convert_teleop_to_lerobot.py --input vla_teleop_data --max-episodes 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import tqdm

# Prefer installed lerobot; fall back to sibling checkout at ../lerobot
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIBLING_LEROBOT_SRC = _REPO_ROOT.parent / "lerobot" / "src"
if str(_SIBLING_LEROBOT_SRC) not in sys.path and _SIBLING_LEROBOT_SRC.exists():
    sys.path.insert(0, str(_SIBLING_LEROBOT_SRC))

from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

# Default 22→16 index mapping (arms grouped, then grippers)
DEFAULT_LEFT_ARM_IDS = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_RIGHT_ARM_IDS = [7, 8, 9, 10, 11, 12, 13]
DEFAULT_LEFT_GRIP_ID = 14
DEFAULT_RIGHT_GRIP_ID = 17

MOTORS_16 = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
    "right_gripper",
]

CAMERAS = ["ego", "left_wrist", "right_wrist"]


@dataclasses.dataclass(frozen=True)
class JointMapping:
    """Maps 22-dim joint_pos indices to 16-dim ALOHA-style ordering."""

    left_arm_ids: list[int] = dataclasses.field(default_factory=lambda: list(DEFAULT_LEFT_ARM_IDS))
    right_arm_ids: list[int] = dataclasses.field(default_factory=lambda: list(DEFAULT_RIGHT_ARM_IDS))
    left_grip_id: int = DEFAULT_LEFT_GRIP_ID
    right_grip_id: int = DEFAULT_RIGHT_GRIP_ID

    @staticmethod
    def from_json(path: str) -> "JointMapping":
        with open(path) as f:
            data = json.load(f)
        m = data.get("aloha_16_mapping", data)
        return JointMapping(
            left_arm_ids=m.get("left_arm_indices", DEFAULT_LEFT_ARM_IDS),
            right_arm_ids=m.get("right_arm_indices", DEFAULT_RIGHT_ARM_IDS),
            left_grip_id=m.get("left_grip_index", DEFAULT_LEFT_GRIP_ID),
            right_grip_id=m.get("right_grip_index", DEFAULT_RIGHT_GRIP_ID),
        )

    def extract_16(self, data: np.ndarray) -> np.ndarray:
        """Extract 16-dim from 22-dim, or pass through if already 16-dim."""
        last_dim = data.shape[-1] if data.ndim > 0 else 0
        if last_dim == 16:
            return data.astype(np.float32)
        if last_dim < 18:
            raise ValueError(f"Unexpected state/action dim {last_dim}; expected 16 or 22")
        indices = self.left_arm_ids + [self.left_grip_id] + self.right_arm_ids + [self.right_grip_id]
        return data[..., indices].astype(np.float32)


def _list_episode_dirs(input_dir: Path, include_mirrored: bool) -> list[Path]:
    episodes_dir = input_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No episodes/ directory in {input_dir}")

    ep_dirs = sorted(
        [d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
        key=lambda x: int(x.name.split("_")[1]),
    )

    mirrored_dirs: list[Path] = []
    mirrored_dir = input_dir / "mirrored"
    if include_mirrored and mirrored_dir.exists():
        mirrored_dirs = sorted(
            [d for d in mirrored_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
            key=lambda x: int(x.name.split("_")[1]),
        )
    return mirrored_dirs + ep_dirs


def _camera_files(ep_dir: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for cam in CAMERAS:
        cam_dir = ep_dir / cam
        if not cam_dir.exists():
            continue
        frames = sorted(
            list(cam_dir.glob("frame_*.png")) + list(cam_dir.glob("frame_*.jpg")) + list(cam_dir.glob("frame_*.jpeg")),
            key=lambda f: f.stem,
        )
        if frames:
            files[cam] = frames
    return files


def _detect_image_size(ep_dirs: list[Path], image_size: int | None) -> tuple[int, int]:
    if image_size is not None:
        return image_size, image_size
    from PIL import Image

    for ep_dir in ep_dirs:
        cams = _camera_files(ep_dir)
        for frames in cams.values():
            with Image.open(frames[0]) as img:
                w, h = img.size
                return h, w
    return 480, 640


def _detect_cameras(ep_dirs: list[Path]) -> list[str]:
    present = set()
    for ep_dir in ep_dirs:
        present.update(_camera_files(ep_dir).keys())
    return [c for c in CAMERAS if c in present]


def create_dataset(
    repo_id: str,
    root: Path,
    fps: int,
    cameras: list[str],
    image_hw: tuple[int, int],
    mode: Literal["video", "image"],
) -> LeRobotDataset:
    h, w = image_hw
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": {"axes": MOTORS_16},
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": {"axes": MOTORS_16},
        },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }

    if root.exists():
        print(f"Removing existing output: {root}")
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="openarm_bimanual",
        features=features,
        root=root,
        use_videos=(mode == "video"),
        image_writer_processes=0,
        image_writer_threads=4,
    )


def convert_raw_teleop(
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    mapping: JointMapping,
    max_episodes: int | None = None,
    include_mirrored: bool = True,
    skip_episodes: int = 0,
    mode: Literal["video", "image"] = "video",
    image_size: int | None = None,
    require_cameras: bool = True,
) -> Path:
    import pandas as pd
    from PIL import Image

    meta_path = input_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    fps = int(metadata.get("fps", 60))
    default_task = metadata.get("task_text", "bimanual manipulation")

    all_dirs = _list_episode_dirs(input_dir, include_mirrored=include_mirrored)
    if skip_episodes > 0:
        print(f"Skipping first {skip_episodes} episodes")
        all_dirs = all_dirs[skip_episodes:]
    if max_episodes is not None:
        all_dirs = all_dirs[:max_episodes]
        print(f"Limiting to {max_episodes} episodes")

    cameras = _detect_cameras(all_dirs)
    if require_cameras and not cameras:
        raise RuntimeError(
            "No camera frames found under episodes/*/ego|left_wrist|right_wrist.\n"
            "Re-record with: ./scripts/teleop_xr.sh --collect-video\n"
            "Or regenerate images with: ./scripts/play_teleop_data.sh --collect-video\n"
            "Or pass --no-require-cameras to convert state/action only."
        )
    if not cameras:
        print("WARNING: No cameras found; writing state/action-only dataset")

    image_hw = _detect_image_size(all_dirs, image_size) if cameras else (480, 640)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"  Episodes queued: {len(all_dirs)}")
    print(f"  FPS: {fps}")
    print(f"  Cameras: {cameras or '(none)'}")
    if cameras:
        print(f"  Image size (H,W): {image_hw}, mode: {mode}")

    dataset = create_dataset(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        cameras=cameras,
        image_hw=image_hw,
        mode=mode,
    )

    converted = 0
    skipped = 0
    for ep_dir in tqdm.tqdm(all_dirs, desc="Converting episodes"):
        try:
            parquet_path = ep_dir / "data.parquet"
            if not parquet_path.exists():
                print(f"  Skipping {ep_dir.name}: no data.parquet")
                skipped += 1
                continue

            df = pd.read_parquet(parquet_path)
            state_cols = sorted(
                [c for c in df.columns if c.startswith("observation.state.")],
                key=lambda x: int(x.split(".")[-1]),
            )
            action_cols = sorted(
                [c for c in df.columns if c.startswith("action.")],
                key=lambda x: int(x.split(".")[-1]),
            )
            if not state_cols or not action_cols:
                print(f"  Skipping {ep_dir.name}: missing state/action columns")
                skipped += 1
                continue

            states = df[state_cols].values.astype(np.float32)
            actions = df[action_cols].values.astype(np.float32)

            ep_task = default_task
            ep_meta_path = ep_dir / "metadata.json"
            if ep_meta_path.exists():
                with open(ep_meta_path) as f:
                    ep_meta = json.load(f)
                ep_task = ep_meta.get("task_text", default_task)

            cam_files = _camera_files(ep_dir)
            if cameras:
                missing = [c for c in cameras if c not in cam_files]
                if missing:
                    print(f"  Skipping {ep_dir.name}: missing cameras {missing}")
                    skipped += 1
                    continue
                num_frames = min(len(states), min(len(cam_files[c]) for c in cameras))
            else:
                num_frames = len(states)

            if num_frames == 0:
                print(f"  Skipping {ep_dir.name}: 0 frames")
                skipped += 1
                continue

            for i in range(num_frames):
                state_16 = mapping.extract_16(states[i])
                action_16 = mapping.extract_16(actions[i])
                frame = {
                    "observation.state": state_16,
                    "action": action_16,
                    "task": ep_task,
                }
                for cam in cameras:
                    img = Image.open(cam_files[cam][i])
                    if image_size is not None:
                        img = img.resize((image_size, image_size), Image.LANCZOS)
                    img_array = np.asarray(img)
                    if img_array.ndim == 3 and img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                    frame[f"observation.images.{cam}"] = img_array
                dataset.add_frame(frame)

            dataset.save_episode()
            converted += 1
        except Exception as e:
            print(f"\n  ERROR processing {ep_dir.name}: {e}")
            import traceback

            traceback.print_exc()
            skipped += 1
            continue

    dataset.finalize()
    print(f"\nDone! Native LeRobot v3 dataset at: {output_dir}")
    print(f"  Converted: {converted}, skipped: {skipped}")
    print(f"  State/action dim: 16")
    print(f"  Load with: LeRobotDataset(repo_id='{repo_id}', root='{output_dir}')")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenArm teleop fallback data to native LeRobot v3.0"
    )
    parser.add_argument(
        "--input",
        default="vla_teleop_data",
        help="Path to teleop data dir (contains episodes/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset root (default: <input>_lerobot)",
    )
    parser.add_argument("--repo-id", default="local/openarm-teleop", help="Dataset repo id in metadata")
    parser.add_argument("--joint-map", default=None, help="Optional joint_mapping.json for 22→16 remap")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert at most N episodes")
    parser.add_argument(
        "--no-include-mirrored",
        action="store_true",
        help="Do not include mirrored/ episodes",
    )
    parser.add_argument("--skip-episodes", type=int, default=0, help="Skip N episodes from the start")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resize images to NxN (default: keep native size)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Store cameras as images instead of MP4 videos",
    )
    parser.add_argument(
        "--no-require-cameras",
        action="store_true",
        help="Allow state/action-only conversion when no camera frames exist",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    if not input_dir.is_absolute():
        input_dir = (_REPO_ROOT / input_dir).resolve()
    else:
        input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")

    if args.output is None:
        output_dir = input_dir.parent / f"{input_dir.name}_lerobot"
    else:
        output_dir = Path(args.output).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_REPO_ROOT / output_dir).resolve()
        else:
            output_dir = output_dir.resolve()

    if args.joint_map:
        mapping = JointMapping.from_json(args.joint_map)
        print(f"Using joint mapping from: {args.joint_map}")
    else:
        mapping = JointMapping()
        print("Using DEFAULT 22→16 joint mapping (pass-through if already 16-dim)")
    print(f"  Left arm:   {mapping.left_arm_ids}")
    print(f"  Left grip:  {mapping.left_grip_id}")
    print(f"  Right arm:  {mapping.right_arm_ids}")
    print(f"  Right grip: {mapping.right_grip_id}")
    print(f"  HF_LEROBOT_HOME fallback would be: {HF_LEROBOT_HOME}")
    print()

    mode: Literal["video", "image"] = "image" if args.no_video else "video"
    convert_raw_teleop(
        input_dir=input_dir,
        output_dir=output_dir,
        repo_id=args.repo_id,
        mapping=mapping,
        max_episodes=args.max_episodes,
        include_mirrored=not args.no_include_mirrored,
        skip_episodes=args.skip_episodes,
        mode=mode,
        image_size=args.image_size,
        require_cameras=not args.no_require_cameras,
    )


if __name__ == "__main__":
    main()
