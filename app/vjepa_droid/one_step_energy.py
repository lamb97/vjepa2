import argparse
import json
import os

import numpy as np
import torch

from app.vjepa_droid.cem_plan import (
    build_dataset,
    build_transform,
    compute_energy_landscape,
    encode_clip,
    get_dtype,
    load_config,
    load_models,
    load_planning_example,
    rollout_world_model,
    save_energy_landscape_plot,
    tensor_to_list,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/val one-step action energy landscape diagnostic.")
    parser.add_argument("--config", required=True, help="Training yaml config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <folder>/latest.pt.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--goal-step", type=int, default=None, help="Goal frame index within the sampled clip.")
    parser.add_argument("--train-index", type=int, default=None, help="Optional fixed train clip/episode index.")
    parser.add_argument("--val-index", type=int, default=None, help="Optional fixed val clip/episode index.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for train/val examples.")
    parser.add_argument("--energy-nsamples", type=int, default=5, help="Grid samples per xyz axis.")
    parser.add_argument("--energy-grid-size", type=float, default=0.075, help="Raw xyz grid range.")
    parser.add_argument("--output-dir", default="one_step_energy_outputs", help="Directory for PNG and JSON outputs.")
    return parser.parse_args()


def choose_example(dataset, index, rng):
    if index is None:
        index = int(rng.integers(0, len(dataset)))
    if dataset.clip_index is not None:
        episode_path, start_frame = dataset.clip_index[index]
    else:
        episode_path, start_frame = dataset.samples[index], 0
    return int(index), episode_path, int(start_frame)


def one_step_action_from_states(dataset, states, goal_step):
    pair = states[[0, goal_step]]
    if dataset.action_from_state_rotation_mode == "rotvec":
        return dataset.rotvec_poses_to_diffs(pair)
    return dataset.poses_to_diffs(pair)


def run_split(
    split,
    cfg,
    transform,
    encoder,
    predictor,
    device,
    dtype,
    mixed_precision,
    normalize_reps,
    rotation_mode,
    goal_step,
    index,
    rng,
    args,
):
    dataset = build_dataset(cfg, transform=transform, split=split)
    example_index, episode_path, start_frame = choose_example(dataset, index=index, rng=rng)

    clip, _, states_np, extrinsics_np, indices, fstp = load_planning_example(
        dataset,
        episode_path=episode_path,
        start_frame=start_frame,
        plan_horizon=1,
        goal_step=goal_step,
    )
    gt_action_np = one_step_action_from_states(dataset, states_np, goal_step)

    clip = clip.unsqueeze(0).to(device, non_blocking=True)
    states = torch.from_numpy(states_np).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    extrinsics = torch.from_numpy(extrinsics_np).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    gt_action = torch.from_numpy(gt_action_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        encoded = encode_clip(
            encoder,
            clip,
            frames_per_clip=clip.size(2),
            normalize_reps=normalize_reps,
            dtype=dtype,
            mixed_precision=mixed_precision,
        )
        tokens_per_frame = encoded.size(1) // clip.size(2)
        context_rep = encoded[:, :tokens_per_frame]
        context_state = states[:, :1]
        context_extr = extrinsics[:, :1]
        goal_rep = encoded[:, goal_step * tokens_per_frame : (goal_step + 1) * tokens_per_frame]

        energy_actions, energy_losses, _, best_idx = compute_energy_landscape(
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
        gt_reps, gt_states = rollout_world_model(
            predictor,
            context_rep,
            context_state,
            context_extr,
            gt_action,
            tokens_per_frame,
            normalize_reps,
            rotation_mode,
            dtype,
            mixed_precision,
        )
        gt_loss = torch.mean(torch.abs(gt_reps[:, -tokens_per_frame:].flatten(1) - goal_rep.flatten(1)), dim=1)

    png_path = os.path.join(args.output_dir, f"{split}_idx{example_index}_start{start_frame}_goal{goal_step}.png")
    save_energy_landscape_plot(
        actions=energy_actions,
        losses=energy_losses,
        gt_actions=gt_action,
        output_path=png_path,
        nsamples=args.energy_nsamples,
    )

    return {
        "split": split,
        "example_index": example_index,
        "episode_path": episode_path,
        "raw_start_frame": int(start_frame),
        "raw_goal_frame": int(indices[goal_step]),
        "sampling_stride_frames": int(fstp),
        "clip_frame_indices": indices.tolist(),
        "goal_step": int(goal_step),
        "rotation_mode": rotation_mode,
        "output_path": png_path,
        "gt_one_step_action": tensor_to_list(gt_action[0, 0]),
        "gt_one_step_final_state": tensor_to_list(gt_states[0, -1]),
        "goal_state": tensor_to_list(states[0, goal_step]),
        "gt_loss": float(gt_loss.item()),
        "best_grid_action": tensor_to_list(energy_actions[best_idx]),
        "best_grid_loss": float(energy_losses[best_idx].item()),
        "energy_nsamples": int(args.energy_nsamples),
        "energy_grid_size": float(args.energy_grid_size),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or os.path.join(cfg["folder"], "latest.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    dtype, mixed_precision = get_dtype(cfg["meta"].get("dtype", "float32"))
    device = torch.device(args.device)
    if device.type != "cuda":
        mixed_precision = False

    rng = np.random.default_rng(args.seed)
    transform = build_transform(cfg)
    encoder, predictor, epoch = load_models(cfg, checkpoint_path, device)
    normalize_reps = cfg["loss"].get("normalize_reps", True)
    rotation_mode = cfg["data"].get("action_from_state_rotation_mode", "euler")
    goal_step = args.goal_step if args.goal_step is not None else max(cfg["data"]["dataset_fpcs"]) - 1
    if goal_step <= 0 or goal_step >= max(cfg["data"]["dataset_fpcs"]):
        raise ValueError(f"goal_step must be in [1, {max(cfg['data']['dataset_fpcs']) - 1}], got {goal_step}")

    results = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(epoch),
        "goal_step": int(goal_step),
        "splits": [
            run_split(
                "train",
                cfg,
                transform,
                encoder,
                predictor,
                device,
                dtype,
                mixed_precision,
                normalize_reps,
                rotation_mode,
                goal_step,
                args.train_index,
                rng,
                args,
            ),
            run_split(
                "val",
                cfg,
                transform,
                encoder,
                predictor,
                device,
                dtype,
                mixed_precision,
                normalize_reps,
                rotation_mode,
                goal_step,
                args.val_index,
                rng,
                args,
            ),
        ],
    }

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"[one_step_energy] wrote {json_path}")


if __name__ == "__main__":
    main()
