import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

from app.vjepa_droid.droid import DROIDVideoDataset, get_json
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Offline CEM planning test for V-JEPA Droid.")
    parser.add_argument("--config", required=True, help="Training yaml config.")
    parser.add_argument("--planner-config", default=None, help="Optional yaml config for this script only.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <folder>/latest.pt.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode-path", default=None, help="Absolute trajectory path.")
    parser.add_argument("--episode-index", type=int, default=None, help="Episode index within the config CSV split.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Split used with --episode-index.")
    parser.add_argument("--start-frame", type=int, default=0, help="Raw video start frame for the clip.")
    parser.add_argument("--goal-step", type=int, default=None, help="Goal frame index within the 8-step clip.")
    parser.add_argument("--plan-horizon", type=int, default=2, help="Number of actions to plan.")
    parser.add_argument("--samples", type=int, default=256, help="CEM samples per iteration.")
    parser.add_argument("--topk", type=int, default=32, help="Top-k samples kept by CEM.")
    parser.add_argument("--cem-steps", type=int, default=8, help="Number of CEM refinement iterations.")
    parser.add_argument("--maxnorm", type=float, default=0.075, help="Notebook-style raw xyz action clamp.")
    parser.add_argument("--momentum-mean", type=float, default=0.25)
    parser.add_argument("--momentum-std", type=float, default=0.75)
    parser.add_argument("--momentum-mean-gripper", type=float, default=0.15)
    parser.add_argument("--momentum-std-gripper", type=float, default=0.15)
    parser.add_argument("--plot-energy-landscape", action="store_true", help="Save notebook-style xyz energy heatmap.")
    parser.add_argument("--energy-nsamples", type=int, default=5, help="Grid samples per xyz axis.")
    parser.add_argument("--energy-grid-size", type=float, default=0.075, help="Raw xyz grid range for energy plot.")
    parser.add_argument("--energy-output", default=None, help="PNG path for the energy landscape plot.")
    parser.add_argument("--cem-verbose", action="store_true", help="Print per-iteration CEM losses.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def apply_planner_overrides(args, planner_cfg):
    if planner_cfg is None:
        return args

    config = dict(planner_cfg.get("planner", planner_cfg))
    for key, value in config.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and value is not None:
            setattr(args, attr, value)
    return args


def get_dtype(which_dtype):
    which_dtype = which_dtype.lower()
    if which_dtype == "bfloat16":
        return torch.bfloat16, True
    if which_dtype == "float16":
        return torch.float16, True
    return torch.float32, False


def autocast_kwargs(device, dtype, mixed_precision):
    return {
        "device_type": device.type,
        "dtype": dtype,
        "enabled": mixed_precision and device.type == "cuda",
    }


def build_transform(cfg):
    data = cfg["data"]
    aug = cfg["data_aug"]
    return make_transforms(
        random_horizontal_flip=aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=aug.get("random_resize_scale", [0.3, 1.0]),
        reprob=aug.get("reprob", 0.0),
        auto_augment=aug.get("auto_augment", False),
        motion_shift=aug.get("motion_shift", False),
        crop_size=data.get("crop_size", 256),
    )


def build_dataset(cfg, transform, split):
    data = cfg["data"]
    meta = cfg["meta"]
    dataset_path = data["datasets"]
    camera_views = data.get("camera_views", ["left_mp4_path"])
    return DROIDVideoDataset(
        data_path=dataset_path,
        camera_views=camera_views,
        frameskip=1,
        frames_per_clip=max(data["dataset_fpcs"]),
        fps=data.get("fps", 5),
        transform=transform,
        camera_frame=data.get("camera_frame", False),
        action_from_state_rotation_mode=data.get("action_from_state_rotation_mode", "euler"),
        enumerate_clips=data.get("enumerate_clips", False),
        clip_stride_frames=data.get("clip_stride_frames", 1),
        split=split,
        holdout_trajectories=data.get("holdout_trajectories", 4),
        val_clip_ratio=data.get("val_clip_ratio"),
        split_seed=meta.get("seed", 0),
    )


def resolve_episode_path(args, dataset):
    if args.episode_path is not None:
        return args.episode_path, args.start_frame
    if args.episode_index is None:
        raise ValueError("Provide either --episode-path or --episode-index.")
    if dataset.clip_index is not None:
        if args.episode_index < 0 or args.episode_index >= len(dataset.clip_index):
            raise IndexError(
                f"episode_index={args.episode_index} is out of range for split clip count {len(dataset.clip_index)}"
            )
        return dataset.clip_index[args.episode_index]
    if args.episode_index < 0 or args.episode_index >= len(dataset.samples):
        raise IndexError(f"episode_index={args.episode_index} is out of range for split size {len(dataset.samples)}")
    return dataset.samples[args.episode_index], args.start_frame


def load_planning_example(dataset, episode_path, start_frame, plan_horizon, goal_step):
    metadata = get_json(episode_path)
    if metadata is None:
        raise ValueError(f"Missing metadata for episode: {episode_path}")

    camera_key = dataset.camera_views[0]
    mp4_name = metadata[camera_key].split("recordings/MP4/")[-1]
    camera_name = mp4_name.split(".")[0]
    vpath = os.path.join(episode_path, "recordings/MP4", mp4_name)
    tpath = os.path.join(episode_path, dataset.h5_name)

    vr = VideoReader(vpath, num_threads=1, ctx=cpu(0))
    vfps = vr.get_avg_fps()
    fps = dataset.fps if dataset.fps is not None else vfps
    fstp = int(np.ceil(vfps / fps))

    with h5py.File(tpath, "r") as trajectory:
        extrinsics = np.array(trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"])
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )

    sampled_len = max(plan_horizon, goal_step) + 1
    indices = start_frame + np.arange(sampled_len, dtype=np.int64) * fstp
    if start_frame < 0 or indices[-1] >= len(vr):
        raise ValueError(
            f"Requested sequence overruns video: start_frame={start_frame}, "
            f"last_index={int(indices[-1])}, video_len={len(vr)}, fstp={fstp}"
        )

    sampled_states = states[indices]
    sampled_extrinsics = extrinsics[indices]
    if dataset.camera_frame:
        sampled_states = dataset.transform_frame(sampled_states, sampled_extrinsics)
    if dataset.action_from_state_rotation_mode == "rotvec":
        sampled_actions = dataset.rotvec_poses_to_diffs(sampled_states)
    else:
        sampled_actions = dataset.poses_to_diffs(sampled_states)

    buffer = vr.get_batch(indices).asnumpy()
    if dataset.transform is not None:
        buffer = dataset.transform(buffer)

    return buffer, sampled_actions, sampled_states, sampled_extrinsics, indices, fstp


def apply_action_to_state(state, action, rotation_mode):
    device, dtype = state.device, state.dtype
    state_np = state[:, 0].detach().cpu().numpy()
    action_np = action[:, 0].detach().cpu().numpy()

    new_xyz = state_np[:, :3] + action_np[:, :3]

    if rotation_mode == "rotvec":
        state_rot = [Rotation.from_rotvec(rot).as_matrix() for rot in state_np[:, 3:6]]
        delta_rot = [Rotation.from_rotvec(rot).as_matrix() for rot in action_np[:, 3:6]]
        new_rot = [Rotation.from_matrix(s @ d).as_rotvec() for s, d in zip(state_rot, delta_rot)]
    else:
        state_rot = [Rotation.from_euler("xyz", ang, degrees=False).as_matrix() for ang in state_np[:, 3:6]]
        delta_rot = [Rotation.from_euler("xyz", ang, degrees=False).as_matrix() for ang in action_np[:, 3:6]]
        new_rot = [Rotation.from_matrix(d @ s).as_euler("xyz", degrees=False) for s, d in zip(state_rot, delta_rot)]

    new_rot = np.stack(new_rot, axis=0)
    new_gripper = np.clip(state_np[:, -1:] + action_np[:, -1:], 0.0, 1.0)
    new_state = np.concatenate([new_xyz, new_rot, new_gripper], axis=-1)
    return torch.from_numpy(new_state).to(device=device, dtype=dtype)[:, None]


def load_models(cfg, checkpoint_path, device):
    meta = cfg["meta"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    crop_size = data_cfg["crop_size"]
    patch_size = data_cfg["patch_size"]
    tubelet_size = data_cfg["tubelet_size"]
    max_num_frames = max(data_cfg["dataset_fpcs"])
    model_max_num_frames = max_num_frames * tubelet_size

    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=model_max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_cfg["model_name"],
        crop_size=crop_size,
        pred_depth=model_cfg["pred_depth"],
        pred_num_heads=model_cfg.get("pred_num_heads"),
        pred_embed_dim=model_cfg["pred_embed_dim"],
        uniform_power=model_cfg.get("uniform_power", False),
        use_sdpa=meta.get("use_sdpa", False),
        use_rope=model_cfg.get("use_rope", False),
        use_silu=model_cfg.get("use_silu", False),
        use_pred_silu=model_cfg.get("use_pred_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        pred_is_frame_causal=model_cfg.get("pred_is_frame_causal", True),
        use_activation_checkpointing=model_cfg.get("use_activation_checkpointing", False),
        action_embed_dim=7,
        use_extrinsics=model_cfg.get("use_extrinsics", False),
    )

    encoder, predictor, _, _, _, epoch = load_checkpoint(
        checkpoint_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=None,
        opt=None,
        scaler=None,
        replace_kw=["module.", "backbone."],
    )
    encoder.eval()
    predictor.eval()
    return encoder, predictor, epoch


def encode_clip(encoder, clip, frames_per_clip, normalize_reps, dtype, mixed_precision):
    with torch.autocast(**autocast_kwargs(clip.device, dtype, mixed_precision)):
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(clip.size(0), frames_per_clip, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h


def rollout_world_model(
    predictor,
    context_rep,
    context_state,
    context_extr,
    action_traj,
    tokens_per_frame,
    normalize_reps,
    rotation_mode,
    dtype,
    mixed_precision,
):
    reps = context_rep
    states = context_state
    extrinsics = context_extr

    with torch.autocast(**autocast_kwargs(reps.device, dtype, mixed_precision)):
        for step in range(action_traj.size(1)):
            cur_actions = action_traj[:, : step + 1]
            pred = predictor(reps, cur_actions, states, extrinsics)
            next_rep = pred[:, -tokens_per_frame:]
            if normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_state = apply_action_to_state(states[:, -1:], cur_actions[:, -1:], rotation_mode)
            next_extr = extrinsics[:, -1:].clone()
            reps = torch.cat([reps, next_rep], dim=1)
            states = torch.cat([states, next_state], dim=1)
            extrinsics = torch.cat([extrinsics, next_extr], dim=1)

    return reps, states


def cem_plan(
    predictor,
    context_rep,
    context_state,
    context_extr,
    goal_rep,
    tokens_per_frame,
    normalize_reps,
    rotation_mode,
    dtype,
    mixed_precision,
    plan_horizon,
    samples,
    topk,
    cem_steps,
    maxnorm,
    momentum_mean,
    momentum_std,
    momentum_mean_gripper,
    momentum_std_gripper,
    cem_verbose,
):
    device = context_rep.device
    context_rep = context_rep.repeat(samples, 1, 1)
    context_state = context_state.repeat(samples, 1, 1)
    context_extr = context_extr.repeat(samples, 1, 1)
    goal_rep = goal_rep.repeat(samples, 1, 1)
    topk = min(topk, samples)

    mean = torch.cat(
        [
            torch.zeros((plan_horizon, 3), device=device, dtype=torch.float32),
            torch.zeros((plan_horizon, 1), device=device, dtype=torch.float32),
        ],
        dim=-1,
    )
    std = torch.cat(
        [
            torch.ones((plan_horizon, 3), device=device, dtype=torch.float32) * maxnorm,
            torch.ones((plan_horizon, 1), device=device, dtype=torch.float32),
        ],
        dim=-1,
    )

    def sample_actions():
        action_samples = (
            torch.randn(samples, plan_horizon, 4, device=device, dtype=torch.float32) * std + mean
        )
        action_samples[..., :3] = torch.clip(action_samples[..., :3], min=-maxnorm, max=maxnorm)
        action_samples[..., -1:] = torch.clip(action_samples[..., -1:], min=-0.75, max=0.75)
        return torch.cat(
            [
                action_samples[..., :3],
                torch.zeros((samples, plan_horizon, 3), device=device, dtype=torch.float32),
                action_samples[..., -1:],
            ],
            dim=-1,
        )

    for step_idx in range(cem_steps):
        action_traj = sample_actions()
        pred_reps, pred_states = rollout_world_model(
            predictor,
            context_rep,
            context_state,
            context_extr,
            action_traj,
            tokens_per_frame,
            normalize_reps,
            rotation_mode,
            dtype,
            mixed_precision,
        )
        final_rep = pred_reps[:, -tokens_per_frame:]
        scores = torch.mean(torch.abs(final_rep.flatten(1) - goal_rep.flatten(1)), dim=1)
        top_idx = torch.topk(scores, k=topk, largest=False).indices
        selected = action_traj[top_idx]
        selected_scores = scores[top_idx]
        if cem_verbose:
            print(
                f"[CEM] iter={step_idx + 1}/{cem_steps} "
                f"best={scores.min().item():.6f} "
                f"topk_mean={selected_scores.mean().item():.6f} "
                f"topk_std={selected_scores.std().item():.6f}"
            )
        selected_mean = selected.mean(dim=0)
        selected_std = selected.std(dim=0)
        mean = torch.cat(
            [
                selected_mean[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                selected_mean[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                selected_std[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                selected_std[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

    gripper = mean[..., -1:].clone()
    gripper[torch.abs(gripper) < 0.25] = 0.0
    return torch.cat(
        [
            mean[..., :3],
            torch.zeros((plan_horizon, 3), device=device, dtype=torch.float32),
            gripper,
        ],
        dim=-1,
    ).unsqueeze(0)


def compute_energy_landscape(
    predictor,
    context_rep,
    context_state,
    context_extr,
    goal_rep,
    tokens_per_frame,
    normalize_reps,
    rotation_mode,
    dtype,
    mixed_precision,
    nsamples,
    grid_size,
):
    device = context_rep.device
    action_samples = []
    for dx in np.linspace(-grid_size, grid_size, nsamples):
        for dy in np.linspace(-grid_size, grid_size, nsamples):
            for dz in np.linspace(-grid_size, grid_size, nsamples):
                action_samples.append([dx, dy, dz, 0.0, 0.0, 0.0, 0.0])
    action_grid = torch.tensor(action_samples, device=device, dtype=torch.float32).unsqueeze(1)

    pred_reps, pred_states = rollout_world_model(
        predictor,
        context_rep.repeat(action_grid.size(0), 1, 1),
        context_state.repeat(action_grid.size(0), 1, 1),
        context_extr.repeat(action_grid.size(0), 1, 1),
        action_grid,
        tokens_per_frame,
        normalize_reps,
        rotation_mode,
        dtype,
        mixed_precision,
    )
    losses = torch.mean(
        torch.abs(pred_reps[:, -tokens_per_frame:].flatten(1) - goal_rep.repeat(action_grid.size(0), 1, 1).flatten(1)),
        dim=1,
    )
    best_idx = torch.argmin(losses)
    return action_grid[:, 0], losses, pred_states, best_idx


def save_energy_landscape_plot(actions, losses, gt_actions, output_path, nsamples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions_np = actions.detach().cpu().numpy()
    losses_np = losses.detach().cpu().numpy()
    gt_np = gt_actions.detach().cpu().numpy()

    heatmap, xedges, zedges = np.histogram2d(
        actions_np[:, 0],
        actions_np[:, 2],
        weights=losses_np,
        bins=nsamples,
    )
    counts, _, _ = np.histogram2d(actions_np[:, 0], actions_np[:, 2], bins=[xedges, zedges])
    heatmap = heatmap / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]],
        cmap="viridis",
        aspect="auto",
    )
    ax.scatter([gt_np[0, 0, 0]], [gt_np[0, 0, 2]], c="red", marker="x", s=80, label="GT")
    best_idx = int(np.argmin(losses_np))
    ax.scatter([actions_np[best_idx, 0]], [actions_np[best_idx, 2]], c="white", marker="o", s=45, label="best grid")
    ax.set_xlabel("Action Delta x")
    ax.set_ylabel("Action Delta z")
    ax.set_title("Energy Landscape")
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label="latent L1")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def tensor_to_list(tensor):
    return np.asarray(tensor.detach().cpu().tolist()).round(6).tolist()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    planner_cfg = load_config(args.planner_config) if args.planner_config is not None else None
    args = apply_planner_overrides(args, planner_cfg)
    data_cfg = cfg["data"]
    meta_cfg = cfg["meta"]
    loss_cfg = cfg["loss"]
    planner_checkpoint = None if planner_cfg is None else planner_cfg.get("planner", planner_cfg).get("checkpoint")
    checkpoint_path = args.checkpoint or planner_checkpoint or os.path.join(cfg["folder"], "latest.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[cem_plan] using checkpoint: {checkpoint_path}")

    dtype, mixed_precision = get_dtype(meta_cfg.get("dtype", "float32"))
    device = torch.device(args.device)
    if device.type != "cuda":
        mixed_precision = False
    torch.manual_seed(meta_cfg.get("seed", 0))
    np.random.seed(meta_cfg.get("seed", 0))

    transform = build_transform(cfg)
    dataset = build_dataset(cfg, transform=transform, split=args.split)
    episode_path, start_frame = resolve_episode_path(args, dataset)

    goal_step = args.goal_step if args.goal_step is not None else args.plan_horizon

    clip, actions, states, extrinsics, indices, fstp = load_planning_example(
        dataset,
        episode_path=episode_path,
        start_frame=start_frame,
        plan_horizon=args.plan_horizon,
        goal_step=goal_step,
    )
    if goal_step <= 0:
        raise ValueError(f"goal_step must be >= 1, got {goal_step}")
    if args.plan_horizon <= 0:
        raise ValueError(f"plan_horizon must be >= 1, got {args.plan_horizon}")
    if args.plan_horizon > goal_step:
        raise ValueError("plan_horizon must be <= goal_step")

    clip = clip.unsqueeze(0).to(device, non_blocking=True)
    actions = torch.from_numpy(actions).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    states = torch.from_numpy(states).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    encoder, predictor, epoch = load_models(cfg, checkpoint_path, device)
    frames_per_clip = clip.size(2)
    normalize_reps = loss_cfg.get("normalize_reps", True)
    rotation_mode = data_cfg.get("action_from_state_rotation_mode", "euler")

    with torch.inference_mode():
        encoded = encode_clip(
            encoder,
            clip,
            frames_per_clip=frames_per_clip,
            normalize_reps=normalize_reps,
            dtype=dtype,
            mixed_precision=mixed_precision,
        )
        tokens_per_frame = encoded.size(1) // frames_per_clip
        if encoded.size(1) % frames_per_clip != 0:
            raise ValueError(
                f"Encoded token count {encoded.size(1)} is not divisible by frames_per_clip={frames_per_clip}"
            )
        context_rep = encoded[:, :tokens_per_frame]
        context_state = states[:, :1]
        context_extr = extrinsics[:, :1]
        goal_start = goal_step * tokens_per_frame
        goal_end = (goal_step + 1) * tokens_per_frame
        goal_rep = encoded[:, goal_start:goal_end]

        planned_actions = cem_plan(
            predictor=predictor,
            context_rep=context_rep,
            context_state=context_state,
            context_extr=context_extr,
            goal_rep=goal_rep,
            tokens_per_frame=tokens_per_frame,
            normalize_reps=normalize_reps,
            rotation_mode=rotation_mode,
            dtype=dtype,
            mixed_precision=mixed_precision,
            plan_horizon=args.plan_horizon,
            samples=args.samples,
            topk=args.topk,
            cem_steps=args.cem_steps,
            maxnorm=args.maxnorm,
            momentum_mean=args.momentum_mean,
            momentum_std=args.momentum_std,
            momentum_mean_gripper=args.momentum_mean_gripper,
            momentum_std_gripper=args.momentum_std_gripper,
            cem_verbose=args.cem_verbose,
        )

        pred_reps, pred_states = rollout_world_model(
            predictor,
            context_rep,
            context_state,
            context_extr,
            planned_actions,
            tokens_per_frame,
            normalize_reps,
            rotation_mode,
            dtype,
            mixed_precision,
        )
        pred_goal_l1 = torch.mean(torch.abs(pred_reps[:, -tokens_per_frame:].flatten(1) - goal_rep.flatten(1)), dim=1)

        gt_actions = actions[:, : args.plan_horizon]
        gt_reps, gt_states = rollout_world_model(
            predictor,
            context_rep,
            context_state,
            context_extr,
            gt_actions,
            tokens_per_frame,
            normalize_reps,
            rotation_mode,
            dtype,
            mixed_precision,
        )
        gt_goal_l1 = torch.mean(torch.abs(gt_reps[:, -tokens_per_frame:].flatten(1) - goal_rep.flatten(1)), dim=1)

        energy_summary = None
        if args.plot_energy_landscape:
            energy_actions, energy_losses, energy_states, best_idx = compute_energy_landscape(
                predictor=predictor,
                context_rep=context_rep,
                context_state=context_state,
                context_extr=context_extr,
                goal_rep=goal_rep,
                tokens_per_frame=tokens_per_frame,
                normalize_reps=normalize_reps,
                rotation_mode=rotation_mode,
                dtype=dtype,
                mixed_precision=mixed_precision,
                nsamples=args.energy_nsamples,
                grid_size=args.energy_grid_size,
            )
            energy_output = args.energy_output
            if energy_output is None:
                energy_output = os.path.join(
                    os.getcwd(),
                    f"energy_{args.split}_episode{args.episode_index if args.episode_index is not None else 'path'}"
                    f"_start{start_frame}_goal{goal_step}.png",
                )
            save_energy_landscape_plot(
                actions=energy_actions,
                losses=energy_losses,
                gt_actions=gt_actions,
                output_path=energy_output,
                nsamples=args.energy_nsamples,
            )
            energy_summary = {
                "output_path": energy_output,
                "nsamples": int(args.energy_nsamples),
                "grid_size": float(args.energy_grid_size),
                "best_grid_action": tensor_to_list(energy_actions[best_idx]),
                "best_grid_loss": float(energy_losses[best_idx].item()),
                "gt_grid_action_loss": float(
                    torch.mean(torch.abs(gt_reps[:, -tokens_per_frame:].flatten(1) - goal_rep.flatten(1)), dim=1).item()
                ),
            }

    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(epoch),
        "episode_path": episode_path,
        "split": args.split,
        "raw_start_frame": int(start_frame),
        "raw_goal_frame": int(indices[goal_step]),
        "sampling_stride_frames": int(fstp),
        "clip_frame_indices": indices.tolist(),
        "plan_horizon": int(args.plan_horizon),
        "goal_step": int(goal_step),
        "rotation_mode": rotation_mode,
        "cem_action_space": "raw_notebook",
        "maxnorm": float(args.maxnorm),
        "planned_actions": tensor_to_list(planned_actions[0]),
        "ground_truth_actions": tensor_to_list(gt_actions[0]),
        "start_state": tensor_to_list(states[0, 0]),
        "goal_state": tensor_to_list(states[0, goal_step]),
        "predicted_final_state_from_plan": tensor_to_list(pred_states[0, -1]),
        "predicted_final_state_from_gt_actions": tensor_to_list(gt_states[0, -1]),
        "pred_goal_l1": float(pred_goal_l1.item()),
        "gt_goal_l1": float(gt_goal_l1.item()),
        "energy_landscape": energy_summary,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
