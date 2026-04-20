#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("/home/yang/vjepa2/configs/train/vitg16/ur5e-256px-8f.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/yang/ur5_vjepa_0409"))
    parser.add_argument("--paths-file", type=Path, default=Path("/home/yang/ur5_vjepa_0409/train_paths.csv"))
    parser.add_argument("--traj-index", type=int, default=0)
    parser.add_argument("--camera-view", type=str, default="left_mp4_path")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--rollout", type=int, default=3)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--state-mode", choices=["gt", "pred"], default="gt")
    return parser.parse_args()


class InferenceTransform:
    def __init__(self, crop_size: int):
        self.crop_size = int(crop_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32) * 255.0
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32) * 255.0

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        if frames.ndim == 3:
            frames = frames[None, ...]
        tensor = torch.as_tensor(frames, dtype=torch.float32)
        tensor = tensor.permute(0, 3, 1, 2)
        height, width = tensor.shape[-2:]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        tensor = tensor[:, :, top : top + side, left : left + side]
        tensor = F.interpolate(tensor, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3)
        flat = tensor.reshape(3, -1).transpose(0, 1)
        flat = (flat - self.mean) / self.std
        return flat.transpose(0, 1).reshape(3, tensor.shape[1], self.crop_size, self.crop_size)


def normalize_traj_path(raw_path: str, dataset_root: Path) -> Path:
    raw = Path(raw_path.strip())
    if raw.exists():
        return raw
    return dataset_root / raw.name


def load_traj_paths(paths_file: Path, dataset_root: Path) -> List[Path]:
    paths = []
    for line in paths_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        paths.append(normalize_traj_path(line, dataset_root))
    return paths


def poses_to_diffs(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    xyz = poses[:, :3]
    rot = poses[:, 3:6]
    xyz_diff = xyz[1:] - xyz[:-1]
    rotations = [R.from_rotvec(theta) for theta in rot]
    rot_diff = [(rotations[t].inv() * rotations[t + 1]).as_rotvec() for t in range(len(rotations) - 1)]
    rot_diff = np.stack(rot_diff, axis=0).astype(np.float32)
    gripper = poses[:, -1:]
    gripper_diff = gripper[1:] - gripper[:-1]
    return np.concatenate([xyz_diff, rot_diff, gripper_diff], axis=1)


def compute_new_pose(pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    device, dtype = pose.device, pose.dtype
    pose_np = pose[:, 0].detach().cpu().numpy()
    action_np = action[:, 0].detach().cpu().numpy()
    new_xyz = pose_np[:, :3] + action_np[:, :3]
    current_rot = [R.from_rotvec(theta) for theta in pose_np[:, 3:6]]
    delta_rot = [R.from_rotvec(theta) for theta in action_np[:, 3:6]]
    new_rot = np.stack([(current_rot[i] * delta_rot[i]).as_rotvec() for i in range(len(current_rot))], axis=0)
    new_gripper = np.clip(pose_np[:, -1:] + action_np[:, -1:], 0.0, 1.0)
    new_pose = np.concatenate([new_xyz, new_rot.astype(np.float32), new_gripper.astype(np.float32)], axis=-1)
    return torch.from_numpy(new_pose).to(device=device, dtype=dtype)[:, None]


class OfflineEvaluator:
    def __init__(self, train_config: Path, checkpoint: Path, device: str):
        from app.vjepa_droid.utils import init_video_model, load_checkpoint

        cfg = yaml.safe_load(train_config.read_text())
        requested_device = device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.normalize_reps = bool(cfg["loss"].get("normalize_reps", True))
        self.history_len = int(max(cfg["data"]["dataset_fpcs"]))
        crop_size = int(cfg["data"]["crop_size"])
        patch_size = int(cfg["data"]["patch_size"])
        tubelet_size = int(cfg["data"]["tubelet_size"])
        model_cfg = cfg["model"]

        self.transform = InferenceTransform(crop_size=crop_size)
        self.tokens_per_frame = int((crop_size // patch_size) ** 2)
        self.encoder, self.predictor = init_video_model(
            device=self.device,
            patch_size=patch_size,
            max_num_frames=512,
            tubelet_size=tubelet_size,
            model_name=model_cfg["model_name"],
            crop_size=crop_size,
            pred_depth=model_cfg["pred_depth"],
            pred_num_heads=model_cfg.get("pred_num_heads"),
            pred_embed_dim=model_cfg["pred_embed_dim"],
            uniform_power=model_cfg.get("uniform_power", False),
            use_sdpa=cfg["meta"].get("use_sdpa", False),
            use_rope=model_cfg.get("use_rope", False),
            use_silu=model_cfg.get("use_silu", False),
            use_pred_silu=model_cfg.get("use_pred_silu", False),
            wide_silu=model_cfg.get("wide_silu", True),
            pred_is_frame_causal=model_cfg.get("pred_is_frame_causal", True),
            use_activation_checkpointing=False,
            action_embed_dim=7,
            use_extrinsics=model_cfg.get("use_extrinsics", False),
        )
        load_checkpoint(
            r_path=str(checkpoint),
            encoder=self.encoder,
            predictor=self.predictor,
            target_encoder=None,
            opt=None,
            scaler=None,
        )
        self.encoder.eval()
        self.predictor.eval()

    @torch.inference_mode()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        clip = self.transform(frames)[None, :]
        batch, channels, time_steps, _, _ = clip.shape
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)
        reps = self.encoder(clip)
        reps = reps.view(batch, time_steps, -1, reps.size(-1))
        if self.normalize_reps:
            reps = F.layer_norm(reps, (reps.size(-1),))
        return reps

    @torch.inference_mode()
    def rollout_gt_actions(
        self,
        history_frames: np.ndarray,
        future_frames: np.ndarray,
        history_states: np.ndarray,
        future_actions: np.ndarray,
        future_states: np.ndarray,
        goal_frame: np.ndarray,
        state_mode: str,
    ):
        context_rep = self.encode_frames(history_frames)
        gt_future_rep = self.encode_frames(future_frames)
        goal_rep = self.encode_frames(goal_frame)[0:1, -1:, :, :]
        pose_window = torch.as_tensor(history_states, dtype=torch.float32, device=self.device)[None, ...]
        frame_window = context_rep
        predicted_reps = []
        stats = []

        for step_idx in range(future_actions.shape[0]):
            history_actions = poses_to_diffs(pose_window[0].detach().cpu().numpy())
            history_actions = torch.as_tensor(history_actions, dtype=torch.float32, device=self.device)[None, ...]
            next_action = torch.as_tensor(future_actions[step_idx], dtype=torch.float32, device=self.device)[None, None, :]
            predictor_actions = torch.cat([history_actions, next_action], dim=1)
            flat_reps = frame_window.flatten(1, 2)
            next_rep = self.predictor(flat_reps, predictor_actions, pose_window)[:, -self.tokens_per_frame :]
            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(1, 1, self.tokens_per_frame, -1)

            if state_mode == "gt":
                next_pose = torch.as_tensor(future_states[step_idx], dtype=torch.float32, device=self.device)[None, None, :]
            else:
                next_pose = compute_new_pose(pose_window[:, -1:], next_action)

            pred_goal_loss = torch.mean(torch.abs(next_rep.flatten(1) - goal_rep.flatten(1))).item()
            gt_rep_step = gt_future_rep[:, step_idx : step_idx + 1]
            per_step_gt_image_loss = torch.mean(torch.abs(next_rep.flatten(1) - gt_rep_step.flatten(1))).item()
            stats.append(
                {
                    "step": step_idx,
                    "per_step_gt_image_loss": per_step_gt_image_loss,
                    "pred_goal_loss": pred_goal_loss,
                }
            )
            predicted_reps.append(next_rep.cpu())
            frame_window = torch.cat([frame_window[:, 1:], next_rep], dim=1)
            pose_window = torch.cat([pose_window[:, 1:], next_pose], dim=1)

        pred_rollout = torch.cat(predicted_reps, dim=1).to(self.device)
        offline_sloss_like = torch.mean(torch.abs(pred_rollout - gt_future_rep)).item()
        return stats, offline_sloss_like


def load_trajectory_segment(
    traj_path: Path,
    camera_view: str,
    fps: int,
    history_len: int,
    rollout: int,
    start_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = json.loads((traj_path / "metadata.json").read_text())
    mp4_rel = metadata[camera_view]
    video_path = traj_path / mp4_rel
    h5_path = traj_path / "trajectory.h5"

    with h5py.File(h5_path, "r") as f:
        states = np.concatenate(
            [
                np.array(f["observation/robot_state/cartesian_position"], dtype=np.float32),
                np.array(f["observation/robot_state/gripper_position"], dtype=np.float32)[:, None],
            ],
            axis=1,
        )

    vr = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
    vfps = vr.get_avg_fps()
    fstp = int(np.ceil(vfps / fps))
    needed = history_len + rollout
    indices = np.arange(start_step * fstp, (start_step + needed) * fstp, fstp).astype(np.int64)
    if indices[-1] >= len(vr):
        raise ValueError(
            f"Requested segment exceeds trajectory length: last_index={indices[-1]}, video_len={len(vr)}"
        )
    frames = vr.get_batch(indices).asnumpy()
    sampled_states = states[indices]
    history_frames = frames[:history_len]
    future_frames = frames[history_len:]
    history_states = sampled_states[:history_len]
    future_states = sampled_states[history_len:]
    return history_frames, future_frames, history_states, future_states


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.train_config.read_text())
    history_len = int(args.history_len or max(cfg["data"]["dataset_fpcs"]))
    fps = int(args.fps or cfg["data"]["fps"])

    traj_paths = load_traj_paths(args.paths_file, args.dataset_root)
    traj_path = traj_paths[args.traj_index]
    print(f"[offline_gt_eval] trajectory={traj_path}", flush=True)

    history_frames, future_frames, history_states, future_states = load_trajectory_segment(
        traj_path=traj_path,
        camera_view=args.camera_view,
        fps=fps,
        history_len=history_len,
        rollout=args.rollout,
        start_step=args.start_step,
    )
    future_actions = poses_to_diffs(np.concatenate([history_states[-1:], future_states], axis=0))

    evaluator = OfflineEvaluator(
        train_config=args.train_config,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    goal_frame = future_frames[-1]
    stats, offline_sloss_like = evaluator.rollout_gt_actions(
        history_frames=history_frames,
        future_frames=future_frames,
        history_states=history_states,
        future_actions=future_actions,
        future_states=future_states,
        goal_frame=goal_frame,
        state_mode=args.state_mode,
    )

    print(f"[offline_gt_eval] offline_sloss_like={offline_sloss_like:.6f}", flush=True)
    print("[offline_gt_eval] GT-action rollout losses", flush=True)
    for item in stats:
        print(
            f"step={item['step']} "
            f"per_step_gt_image_loss={item['per_step_gt_image_loss']:.6f} "
            f"pred_goal_loss={item['pred_goal_loss']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
