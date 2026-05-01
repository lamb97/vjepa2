#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a flattened Cartesian episode dataset into the V-JEPA DROID trajectory format."
    )
    parser.add_argument("--input-root", required=True, help="Source dataset root containing states/actions/seq_lengths/obses_npy.")
    parser.add_argument("--output-root", required=True, help="Output directory for traj_xxxxxx folders and CSV manifests.")
    parser.add_argument("--fps", type=float, default=4.0, help="FPS used when encoding left.mp4.")
    parser.add_argument("--camera-name", default="left", help="Camera basename used for metadata.json and camera_extrinsics key.")
    parser.add_argument("--workers", type=int, default=1, help="Number of processes to use for episode conversion.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of episodes to convert.")
    parser.add_argument(
        "--source-label",
        default=None,
        help="Optional source label stored in metadata. Defaults to the input episode .npy path.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_video(frames: np.ndarray, output_path: Path, fps: float):
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shaped [T, H, W, 3], got {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_trajectory_h5(states: np.ndarray, output_path: Path, camera_name: str):
    cartesian = np.asarray(states[:, :6], dtype=np.float32)
    gripper = np.asarray(states[:, 6], dtype=np.float32)
    extrinsics = np.zeros((len(states), 6), dtype=np.float32)
    with h5py.File(output_path, "w") as f:
        obs = f.create_group("observation")
        cam = obs.create_group("camera_extrinsics")
        robot = obs.create_group("robot_state")
        cam.create_dataset(f"{camera_name}_left", data=extrinsics)
        robot.create_dataset("cartesian_position", data=cartesian)
        robot.create_dataset("gripper_position", data=gripper)


def write_metadata(output_path: Path, camera_name: str, num_frames: int, source_video: str):
    metadata = {
        f"{camera_name}_mp4_path": f"recordings/MP4/{camera_name}.mp4",
        "source_video": source_video,
        "num_frames": int(num_frames),
    }
    with output_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def convert_one_episode(job):
    (
        episode_idx,
        output_root,
        input_root,
        seq_len,
        states_slice,
        camera_name,
        fps,
        overwrite,
        source_label,
    ) = job

    traj_dir = Path(output_root) / f"traj_{episode_idx:06d}"
    mp4_dir = traj_dir / "recordings" / "MP4"
    ensure_dir(mp4_dir)

    episode_file = Path(input_root) / "obses_npy" / f"episode_{episode_idx:06d}.npy"
    frames = np.load(episode_file)
    if len(frames) != seq_len:
        raise ValueError(f"Episode {episode_idx} frame count mismatch: {len(frames)} vs {seq_len}")

    states = np.asarray(states_slice[:seq_len], dtype=np.float32)
    if states.shape != (seq_len, 7):
        raise ValueError(f"Episode {episode_idx} state shape mismatch: {states.shape}")

    mp4_path = mp4_dir / f"{camera_name}.mp4"
    h5_path = traj_dir / "trajectory.h5"
    metadata_path = traj_dir / "metadata.json"

    if overwrite or not mp4_path.exists():
        write_video(frames, mp4_path, fps=fps)
    if overwrite or not h5_path.exists():
        write_trajectory_h5(states, h5_path, camera_name=camera_name)
    if overwrite or not metadata_path.exists():
        source_video = source_label or str(episode_file)
        write_metadata(metadata_path, camera_name=camera_name, num_frames=seq_len, source_video=source_video)

    return {
        "episode_idx": episode_idx,
        "traj_dir": str(traj_dir.resolve()),
        "num_frames": int(seq_len),
        "mp4_path": str(mp4_path.resolve()),
    }


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    seq_lengths = np.load(input_root / "seq_lengths.npy")
    states = np.load(input_root / "states.npy", mmap_mode="r")

    num_episodes = len(seq_lengths)
    if states.shape[0] != num_episodes:
        raise ValueError(f"states first dimension {states.shape[0]} does not match seq_lengths {num_episodes}")

    episode_count = num_episodes if args.limit is None else min(num_episodes, args.limit)
    jobs = [
        (
            episode_idx,
            str(output_root),
            str(input_root),
            int(seq_lengths[episode_idx]),
            states[episode_idx],
            args.camera_name,
            args.fps,
            args.overwrite,
            args.source_label,
        )
        for episode_idx in range(episode_count)
    ]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                tqdm(
                    executor.map(convert_one_episode, jobs),
                    total=len(jobs),
                    desc="Converting episodes",
                )
            )
    else:
        results = [convert_one_episode(job) for job in tqdm(jobs, total=len(jobs), desc="Converting episodes")]

    traj_dirs = [item["traj_dir"] for item in sorted(results, key=lambda x: x["episode_idx"])]
    for manifest_name in ("all_paths.csv", "train_paths.csv"):
        with (output_root / manifest_name).open("w") as f:
            for traj_dir in traj_dirs:
                f.write(f"{traj_dir}\n")

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_episodes": len(traj_dirs),
        "fps": args.fps,
        "camera_name": args.camera_name,
        "rotation_mode": "rotvec",
        "notes": [
            "trajectory.h5 stores cartesian_position as xyz + rotvec",
            "camera_extrinsics are filled with zeros because the source dataset does not provide them",
            "Set data.action_from_state_rotation_mode=rotvec during training",
        ],
    }
    with (output_root / "conversion_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
