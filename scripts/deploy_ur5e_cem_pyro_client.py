#!/usr/bin/env python3

import argparse
import base64
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import cv2
import numpy as np
import Pyro5.api
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DummyWandbRun:
    def log(self, *args, **kwargs):
        if not args:
            return
        payload = args[0]
        if not isinstance(payload, dict):
            return
        loss_keys = [key for key in payload.keys() if key.endswith("/loss")]
        if not loss_keys:
            return
        step = payload.get("step")
        for key in loss_keys:
            value = payload[key]
            try:
                loss_value = float(value)
            except (TypeError, ValueError):
                loss_value = value
            step_str = "?" if step is None else str(step)
            print(f"[CEM] {key} step={step_str} loss={loss_value}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("/home/yang/vjepa2/configs/train/vitg16/ur5e-256px-8f.yaml"),
    )
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=Path("/home/yang/vjepa2/configs/deploy/vitg16/ur5e-256px-8f-cem.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--goal-image-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--rollout", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--cem-steps", type=int, default=None)
    parser.add_argument("--maxnorm", type=float, default=None)
    parser.add_argument("--single-frame-input", action="store_true", default=None)
    parser.add_argument("--execute-steps", type=int, default=None)
    parser.add_argument("--max-replans", type=int, default=None)
    parser.add_argument("--pyro-uri", type=str, default=None)
    parser.add_argument("--motion-command", type=str, choices=["moveL", "servoL"], default=None)
    parser.add_argument("--state-rotation-mode", type=str, choices=["euler", "rotvec"], default=None)
    parser.add_argument("--action-rotation-mode", type=str, choices=["euler", "rotvec"], default=None)
    parser.add_argument("--use-gripper", action="store_true", default=None)
    parser.add_argument("--gripper-state-scale", type=float, default=None)
    parser.add_argument("--gripper-action-mode", type=str, choices=["delta", "absolute"], default=None)
    parser.add_argument("--gripper-action-scale", type=float, default=None)
    parser.add_argument("--close-gripper-after", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--print-actions-only", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_nested(cfg: Dict, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_arg(args, name: str, cfg: Dict, path: str, default=None):
    value = getattr(args, name)
    if value is not None:
        return value
    cfg_value = _get_nested(cfg, path, None)
    if cfg_value is not None:
        return cfg_value
    return default


def resolve_runtime_settings(args):
    deploy_cfg = _load_yaml(args.deploy_config)

    checkpoint_value = _resolve_arg(args, "checkpoint", deploy_cfg, "runtime.checkpoint", None)
    goal_image_value = _resolve_arg(args, "goal_image_path", deploy_cfg, "runtime.goal_image_path", None)

    settings = {
        "device": _resolve_arg(args, "device", deploy_cfg, "runtime.device", "cuda:0"),
        "history_len": _resolve_arg(args, "history_len", deploy_cfg, "planner.history_len", None),
        "rollout": int(_resolve_arg(args, "rollout", deploy_cfg, "planner.rollout", 2)),
        "topk": int(_resolve_arg(args, "topk", deploy_cfg, "planner.topk", 16)),
        "num_samples": int(_resolve_arg(args, "num_samples", deploy_cfg, "planner.num_samples", 128)),
        "cem_steps": int(_resolve_arg(args, "cem_steps", deploy_cfg, "planner.cem_steps", 4)),
        "maxnorm": float(_resolve_arg(args, "maxnorm", deploy_cfg, "planner.maxnorm", 0.05)),
        "single_frame_input": bool(
            _resolve_arg(args, "single_frame_input", deploy_cfg, "planner.single_frame_input", False)
        ),
        "execute_steps": int(_resolve_arg(args, "execute_steps", deploy_cfg, "runtime.execute_steps", 1)),
        "max_replans": int(_resolve_arg(args, "max_replans", deploy_cfg, "runtime.max_replans", 200)),
        "pyro_uri": _resolve_arg(args, "pyro_uri", deploy_cfg, "robot.pyro_uri", "PYRONAME:vjepa2_deploy"),
        "motion_command": _resolve_arg(args, "motion_command", deploy_cfg, "robot.motion_command", "moveL"),
        "state_rotation_mode": _resolve_arg(
            args, "state_rotation_mode", deploy_cfg, "robot.state_rotation_mode", "rotvec"
        ),
        "action_rotation_mode": _resolve_arg(
            args, "action_rotation_mode", deploy_cfg, "robot.action_rotation_mode", "rotvec"
        ),
        "use_gripper": bool(_resolve_arg(args, "use_gripper", deploy_cfg, "robot.use_gripper", False)),
        "gripper_state_scale": float(
            _resolve_arg(args, "gripper_state_scale", deploy_cfg, "robot.gripper_state_scale", 255.0)
        ),
        "gripper_action_mode": _resolve_arg(
            args, "gripper_action_mode", deploy_cfg, "robot.gripper_action_mode", "delta"
        ),
        "gripper_action_scale": float(
            _resolve_arg(args, "gripper_action_scale", deploy_cfg, "robot.gripper_action_scale", 255.0)
        ),
        "close_gripper_after": int(
            _resolve_arg(args, "close_gripper_after", deploy_cfg, "planner.close_gripper_after", -1)
        ),
        "save_dir": Path(_resolve_arg(args, "save_dir", deploy_cfg, "runtime.save_dir", "deploy_outputs")),
        "checkpoint": Path(checkpoint_value) if checkpoint_value is not None else None,
        "goal_image_path": Path(goal_image_value) if goal_image_value is not None else None,
    }

    if settings["checkpoint"] is None:
        raise ValueError("Checkpoint is required. Set --checkpoint or runtime.checkpoint in the deploy config.")
    if settings["goal_image_path"] is None:
        raise ValueError("Goal image path is required. Set --goal-image-path or runtime.goal_image_path in the deploy config.")
    return settings


class PyroRobotBridge:
    def __init__(
        self,
        pyro_uri: str,
        motion_command: str,
        state_rotation_mode: str,
        action_rotation_mode: str,
        gripper_state_scale: float,
        gripper_action_mode: str,
        gripper_action_scale: float,
        use_gripper: bool,
    ):
        self.proxy = Pyro5.api.Proxy(pyro_uri)
        self.motion_command = motion_command
        self.state_rotation_mode = state_rotation_mode
        self.action_rotation_mode = action_rotation_mode
        self.gripper_state_scale = float(gripper_state_scale)
        self.gripper_action_mode = gripper_action_mode
        self.gripper_action_scale = float(gripper_action_scale)
        self.use_gripper = bool(use_gripper)

    def _image_from_payload(self, payload: Dict[str, object]) -> np.ndarray:
        shape = tuple(int(v) for v in payload["shape"])
        dtype = np.dtype(str(payload["dtype"]))
        raw_data = payload["data"]
        if isinstance(raw_data, dict):
            if raw_data.get("encoding") == "base64":
                raw_data = base64.b64decode(raw_data["data"])
            else:
                raise TypeError(f"Unsupported serialized image payload format: {raw_data.keys()}")
        elif isinstance(raw_data, memoryview):
            raw_data = raw_data.tobytes()
        elif isinstance(raw_data, bytearray):
            raw_data = bytes(raw_data)
        image = np.frombuffer(raw_data, dtype=dtype).reshape(shape)
        return image.copy()

    def _state_to_proprio(self, state: Dict[str, object]) -> np.ndarray:
        tcp_pose = np.asarray(state["tcp_pose"], dtype=np.float32).reshape(-1)
        pos = tcp_pose[:3]
        rotvec = tcp_pose[3:6]
        if self.state_rotation_mode == "euler":
            rot = R.from_rotvec(rotvec.astype(np.float64)).as_euler("xyz", degrees=False).astype(np.float32)
        elif self.state_rotation_mode == "rotvec":
            rot = rotvec.astype(np.float32)
        else:
            raise ValueError(f"Unsupported state rotation mode: {self.state_rotation_mode}")

        gripper_raw = float(state["gripper_state"]) if self.use_gripper else 0.0
        gripper_value = np.float32(gripper_raw / max(self.gripper_state_scale, 1e-6))
        return np.concatenate([pos, rot, np.array([gripper_value], dtype=np.float32)], axis=0)

    def get_observation(self) -> Dict[str, np.ndarray]:
        state = self.proxy.get_robot_state()
        image_payload = self.proxy.get_scene1_image()
        return {
            "visual": self._image_from_payload(image_payload),
            "proprio": self._state_to_proprio(state),
        }

    def execute_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        command = {
            "action": action.tolist(),
            "motion_command": self.motion_command,
            "action_rotation_mode": self.action_rotation_mode,
            "gripper_action_mode": self.gripper_action_mode,
            "gripper_action_scale": self.gripper_action_scale,
        }
        self.proxy.execute_action(command)

    def close(self) -> None:
        try:
            self.proxy._pyroRelease()
        except Exception:
            pass


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


def compute_new_pose(pose: torch.Tensor, action: torch.Tensor, rotation_mode: str) -> torch.Tensor:
    device, dtype = pose.device, pose.dtype
    pose_np = pose[:, 0].detach().cpu().numpy()
    action_np = action[:, 0].detach().cpu().numpy()

    new_xyz = pose_np[:, :3] + action_np[:, :3]
    if rotation_mode == "euler":
        matrices = [R.from_euler("xyz", theta, degrees=False).as_matrix() for theta in pose_np[:, 3:6]]
        delta_matrices = [R.from_euler("xyz", theta, degrees=False).as_matrix() for theta in action_np[:, 3:6]]
        new_rot = np.stack(
            [R.from_matrix(delta_matrices[i] @ matrices[i]).as_euler("xyz", degrees=False) for i in range(len(matrices))],
            axis=0,
        )
    elif rotation_mode == "rotvec":
        current_rot = [R.from_rotvec(theta) for theta in pose_np[:, 3:6]]
        delta_rot = [R.from_rotvec(theta) for theta in action_np[:, 3:6]]
        new_rot = np.stack([(current_rot[i] * delta_rot[i]).as_rotvec() for i in range(len(current_rot))], axis=0)
    else:
        raise ValueError(f"Unsupported rotation mode: {rotation_mode}")

    new_gripper = np.clip(pose_np[:, -1:] + action_np[:, -1:], 0.0, 1.0)
    new_pose = np.concatenate([new_xyz, new_rot.astype(np.float32), new_gripper.astype(np.float32)], axis=-1)
    return torch.from_numpy(new_pose).to(device=device, dtype=dtype)[:, None]


def poses_to_diffs(poses: np.ndarray, rotation_mode: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape[0] < 2:
        return np.zeros((0, 7), dtype=np.float32)
    xyz = poses[:, :3]
    rot = poses[:, 3:6]
    xyz_diff = xyz[1:] - xyz[:-1]
    if rotation_mode == "euler":
        matrices = [R.from_euler("xyz", theta, degrees=False).as_matrix() for theta in rot]
        rot_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        rot_diff = [R.from_matrix(mat).as_euler("xyz", degrees=False) for mat in rot_diff]
        rot_diff = np.stack(rot_diff, axis=0).astype(np.float32)
    elif rotation_mode == "rotvec":
        rotations = [R.from_rotvec(theta) for theta in rot]
        rot_diff = [(rotations[t].inv() * rotations[t + 1]).as_rotvec() for t in range(len(rotations) - 1)]
        rot_diff = np.stack(rot_diff, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported rotation mode: {rotation_mode}")
    gripper = poses[:, -1:]
    gripper_diff = gripper[1:] - gripper[:-1]
    return np.concatenate([xyz_diff, rot_diff, gripper_diff], axis=1)


def ensure_history(bridge: PyroRobotBridge, history_len: int) -> Tuple[Deque[np.ndarray], Deque[np.ndarray]]:
    visual_hist: Deque[np.ndarray] = deque(maxlen=history_len)
    proprio_hist: Deque[np.ndarray] = deque(maxlen=history_len)
    first_obs = bridge.get_observation()
    for _ in range(history_len):
        visual_hist.append(first_obs["visual"])
        proprio_hist.append(first_obs["proprio"])
    return visual_hist, proprio_hist


class VJEPA2ACPolicy:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: str, rotation_mode: str, planner_args: Dict):
        from app.vjepa_droid.utils import init_video_model, load_checkpoint

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        requested_device = device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.rotation_mode = rotation_mode
        self.planner_args = planner_args
        self.normalize_reps = bool(config["loss"].get("normalize_reps", True))
        self.wandb_run = DummyWandbRun()

        crop_size = int(config["data"]["crop_size"])
        patch_size = int(config["data"]["patch_size"])
        tubelet_size = int(config["data"]["tubelet_size"])
        num_frames = int(max(config["data"]["dataset_fpcs"]))
        # Match training-time model construction. The predictor's internal
        # causal attention mask is sized from max_num_frames, not the current
        # rollout/history length.
        model_max_num_frames = 512
        model_cfg = config["model"]
        self.train_history_len = int(num_frames)
        self.single_frame_input = bool(planner_args.get("single_frame_input", False))
        self.history_len = 1 if self.single_frame_input else int(planner_args["history_len"] or num_frames)
        if self.history_len not in (1, self.train_history_len):
            raise ValueError(
                f"History length must be either 1 (single-frame mode) or the training "
                f"frames-per-clip for alignment: history_len={self.history_len}, train_fpc={num_frames}"
            )

        self.transform = InferenceTransform(crop_size=crop_size)
        self.tokens_per_frame = int((crop_size // patch_size) ** 2)
        self.encoder, self.predictor = init_video_model(
            device=self.device,
            patch_size=patch_size,
            max_num_frames=model_max_num_frames,
            tubelet_size=tubelet_size,
            model_name=model_cfg["model_name"],
            crop_size=crop_size,
            pred_depth=model_cfg["pred_depth"],
            pred_num_heads=model_cfg.get("pred_num_heads"),
            pred_embed_dim=model_cfg["pred_embed_dim"],
            uniform_power=model_cfg.get("uniform_power", False),
            use_sdpa=config["meta"].get("use_sdpa", False),
            use_rope=model_cfg.get("use_rope", False),
            use_silu=model_cfg.get("use_silu", False),
            use_pred_silu=model_cfg.get("use_pred_silu", False),
            wide_silu=model_cfg.get("wide_silu", True),
            pred_is_frame_causal=model_cfg.get("pred_is_frame_causal", True),
            # Disable activation checkpointing for deployment; it is a
            # training-time memory optimization and slows inference/CEM rollout.
            use_activation_checkpointing=False,
            action_embed_dim=7,
            use_extrinsics=model_cfg.get("use_extrinsics", False),
        )
        load_checkpoint(
            r_path=str(checkpoint_path),
            encoder=self.encoder,
            predictor=self.predictor,
            target_encoder=None,
            opt=None,
            scaler=None,
        )
        self.encoder.eval()
        self.predictor.eval()

    @torch.inference_mode()
    def encode_history(self, images: np.ndarray) -> torch.Tensor:
        clip = self.transform(images)[None, :]
        batch, channels, time_steps, height, width = clip.shape
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)
        reps = self.encoder(clip)
        reps = reps.view(batch, time_steps, -1, reps.size(-1))
        if self.normalize_reps:
            reps = F.layer_norm(reps, (reps.size(-1),))
        return reps

    @torch.inference_mode()
    def encode_goal(self, image: np.ndarray) -> torch.Tensor:
        return self.encode_history(image)[0:1, -1:, :, :]

    @torch.inference_mode()
    def plan_action(self, image_history: np.ndarray, state_history: np.ndarray, goal_rep: torch.Tensor) -> np.ndarray:
        if self.single_frame_input:
            image_history = np.asarray(image_history)[-1:]
            state_history = np.asarray(state_history)[-1:]
        context_rep = self.encode_history(image_history)
        context_pose = torch.as_tensor(state_history, dtype=torch.float32, device=self.device).view(
            1, context_rep.shape[1], 7
        )
        planned_actions = self._cem(context_rep, context_pose, goal_rep)
        return planned_actions[0].detach().cpu().numpy()

    def _states_to_action_history(self, poses: torch.Tensor) -> torch.Tensor:
        pose_np = poses.detach().cpu().numpy()
        action_hist = [poses_to_diffs(pose_np[idx], self.rotation_mode) for idx in range(pose_np.shape[0])]
        return torch.from_numpy(np.stack(action_hist, axis=0)).to(device=poses.device, dtype=poses.dtype)

    def _step_predictor(
        self,
        reps: torch.Tensor,
        poses: torch.Tensor,
        next_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time_steps, num_tokens, dim = reps.size()
        flat_reps = reps.flatten(1, 2)
        history_actions = self._states_to_action_history(poses)
        predictor_actions = torch.cat([history_actions, next_action], dim=1)
        next_rep = self.predictor(flat_reps, predictor_actions, poses)[:, -self.tokens_per_frame :]
        if self.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        next_rep = next_rep.view(batch, 1, num_tokens, dim)
        next_pose = compute_new_pose(poses[:, -1:], next_action, self.rotation_mode)
        return next_rep, next_pose

    def _cem(self, context_frame: torch.Tensor, context_pose: torch.Tensor, goal_frame: torch.Tensor) -> torch.Tensor:
        rollout = self.planner_args["rollout"]
        samples = self.planner_args["num_samples"]
        topk = self.planner_args["topk"]
        cem_steps = self.planner_args["cem_steps"]
        maxnorm = self.planner_args["maxnorm"]
        close_gripper_after = self.planner_args["close_gripper_after"]

        momentum_mean = 0.15
        momentum_mean_gripper = 0.15
        momentum_std = 0.75
        momentum_std_gripper = 0.15

        context_frame = context_frame.repeat(samples, 1, 1, 1)
        goal_frame = goal_frame.repeat(samples, 1, 1, 1)
        context_pose = context_pose.repeat(samples, 1, 1)

        mean = torch.zeros((rollout, 7), device=self.device)
        std = torch.cat(
            [
                torch.ones((rollout, 3), device=self.device) * maxnorm,
                torch.ones((rollout, 3), device=self.device) * maxnorm,
                torch.ones((rollout, 1), device=self.device),
            ],
            dim=-1,
        )

        for cem_step in range(cem_steps):
            action_traj = None
            frame_window = context_frame
            pose_window = context_pose

            for horizon in range(rollout):
                action_samples = torch.randn(samples, mean.size(1), device=self.device) * std[horizon] + mean[horizon]
                action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
                action_samples[:, 3:6] = torch.clip(action_samples[:, 3:6], min=-maxnorm, max=maxnorm)
                action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
                action_samples = action_samples[:, None]
                if close_gripper_after >= 0 and horizon >= close_gripper_after:
                    action_samples[:, :, -1] = 1.0
                action_traj = action_samples if action_traj is None else torch.cat([action_traj, action_samples], dim=1)
                next_frame, next_pose = self._step_predictor(frame_window, pose_window, action_samples)
                frame_window = torch.cat([frame_window[:, 1:], next_frame], dim=1)
                pose_window = torch.cat([pose_window[:, 1:], next_pose], dim=1)

            losses = torch.mean(torch.abs(frame_window[:, -1].flatten(1) - goal_frame.flatten(1)), dim=-1)
            indices = losses.topk(topk, largest=False).indices
            selected_actions = action_traj[indices]
            selected_losses = losses[indices]
            mean_selected = selected_actions.mean(dim=0)
            std_selected = selected_actions.std(dim=0)
            self.wandb_run.log(
                {
                    "step": cem_step,
                    "cem/best/loss": losses.min().item(),
                    "cem/mean/loss": losses.mean().item(),
                    "cem/topk_mean/loss": selected_losses.mean().item(),
                }
            )
            mean = torch.cat(
                [
                    mean_selected[..., :6] * (1.0 - momentum_mean) + mean[..., :6] * momentum_mean,
                    mean_selected[..., -1:] * (1.0 - momentum_mean_gripper)
                    + mean[..., -1:] * momentum_mean_gripper,
                ],
                dim=-1,
            )
            std = torch.cat(
                [
                    std_selected[..., :6] * (1.0 - momentum_std) + std[..., :6] * momentum_std,
                    std_selected[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
                ],
                dim=-1,
            )

        planned = mean.clone()
        planned[:, -1:] = torch.where(
            torch.abs(planned[:, -1:]) < 0.25,
            torch.zeros_like(planned[:, -1:]),
            planned[:, -1:],
        )
        return planned[None, :]


def save_cycle_artifacts(save_dir: Path, cycle_idx: int, primitive_actions: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"cycle_{cycle_idx:03d}_primitive_actions.npy", primitive_actions)


def main():
    args = parse_args()
    print("[vjepa2_deploy] parsing configs...", flush=True)
    settings = resolve_runtime_settings(args)
    planner_args = {
        "rollout": settings["rollout"],
        "topk": settings["topk"],
        "num_samples": settings["num_samples"],
        "cem_steps": settings["cem_steps"],
        "maxnorm": settings["maxnorm"],
        "close_gripper_after": settings["close_gripper_after"],
        "history_len": settings["history_len"],
    }

    print(
        f"[vjepa2_deploy] loading policy "
        f"(train_config={args.train_config}, checkpoint={settings['checkpoint']}, device={settings['device']})",
        flush=True,
    )
    policy = VJEPA2ACPolicy(
        config_path=args.train_config,
        checkpoint_path=settings["checkpoint"],
        device=settings["device"],
        rotation_mode=settings["state_rotation_mode"],
        planner_args=planner_args,
    )
    print(
        f"[vjepa2_deploy] policy ready "
        f"(history_len={policy.history_len}, rollout={settings['rollout']}, num_samples={settings['num_samples']})",
        flush=True,
    )

    print(f"[vjepa2_deploy] loading goal image from {settings['goal_image_path']}...", flush=True)
    goal_bgr = cv2.imread(str(settings["goal_image_path"]), cv2.IMREAD_COLOR)
    if goal_bgr is None:
        raise FileNotFoundError(f"Failed to read goal image: {settings['goal_image_path']}")
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
    print("[vjepa2_deploy] encoding goal image...", flush=True)
    goal_rep = policy.encode_goal(goal_rgb)
    print(
        f"[vjepa2_deploy] goal encoded (shape={tuple(goal_rep.shape)})",
        flush=True,
    )

    print(f"[vjepa2_deploy] connecting to Pyro server at {settings['pyro_uri']}...", flush=True)
    bridge = PyroRobotBridge(
        pyro_uri=settings["pyro_uri"],
        motion_command=settings["motion_command"],
        state_rotation_mode=settings["state_rotation_mode"],
        action_rotation_mode=settings["action_rotation_mode"],
        gripper_state_scale=settings["gripper_state_scale"],
        gripper_action_mode=settings["gripper_action_mode"],
        gripper_action_scale=settings["gripper_action_scale"],
        use_gripper=settings["use_gripper"],
    )
    print("[vjepa2_deploy] Pyro bridge ready, fetching bootstrap history...", flush=True)
    history_len = int(policy.history_len)
    visual_hist, proprio_hist = ensure_history(bridge, history_len)
    print(
        f"[vjepa2_deploy] bootstrap history ready "
        f"(visual={len(visual_hist)}, proprio={len(proprio_hist)})",
        flush=True,
    )
    try:
        for cycle_idx in range(int(settings["max_replans"])):
            print(f"[vjepa2_deploy] cycle={cycle_idx} fetching observation...", flush=True)
            current_obs = bridge.get_observation()
            visual_hist.append(current_obs["visual"])
            proprio_hist.append(current_obs["proprio"])

            print(f"[vjepa2_deploy] cycle={cycle_idx} planning...", flush=True)
            primitive_actions = policy.plan_action(
                image_history=np.stack(list(visual_hist), axis=0),
                state_history=np.stack(list(proprio_hist), axis=0),
                goal_rep=goal_rep,
            )
            print(
                f"[vjepa2_deploy] cycle={cycle_idx} planning done "
                f"(planned_actions={primitive_actions.shape[0]})",
                flush=True,
            )
            save_cycle_artifacts(settings["save_dir"], cycle_idx, primitive_actions)

            exec_steps = min(int(settings["execute_steps"]), primitive_actions.shape[0])
            print(
                f"[vjepa2_deploy] cycle={cycle_idx} executing {exec_steps} action(s)",
                flush=True,
            )
            for action_idx in range(exec_steps):
                action = primitive_actions[action_idx]
                print(f"[vjepa2_deploy] action[{action_idx}]={action.tolist()}", flush=True)
                if args.print_actions_only:
                    continue
                bridge.execute_action(action)
                latest_obs = bridge.get_observation()
                visual_hist.append(latest_obs["visual"])
                proprio_hist.append(latest_obs["proprio"])
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
