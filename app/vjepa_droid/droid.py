# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    action_from_state_rotation_mode="euler",
    enumerate_clips=False,
    clip_stride_frames=1,
    split="train",
    holdout_trajectories=4,
    val_clip_ratio=None,
    tubelet_size=2,
    split_seed=0,
):
    dataset = DROIDVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
        action_from_state_rotation_mode=action_from_state_rotation_mode,
        enumerate_clips=enumerate_clips,
        clip_stride_frames=clip_stride_frames,
        split=split,
        holdout_trajectories=holdout_trajectories,
        val_clip_ratio=val_clip_ratio,
        split_seed=split_seed,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        action_from_state_rotation_mode="euler",
        enumerate_clips=False,
        clip_stride_frames=1,
        split="train",
        holdout_trajectories=4,
        val_clip_ratio=None,
        split_seed=0,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.action_from_state_rotation_mode = action_from_state_rotation_mode
        self.enumerate_clips = enumerate_clips
        self.clip_stride_frames = max(int(clip_stride_frames), 1)
        self.split = split
        self.holdout_trajectories = max(int(holdout_trajectories), 0)
        self.val_clip_ratio = None if val_clip_ratio is None else float(val_clip_ratio)
        self.split_seed = int(split_seed)
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        data_paths = [data_path] if isinstance(data_path, str) else list(data_path)
        samples = []
        for path in data_paths:
            samples.extend(list(pd.read_csv(path, header=None, delimiter=" ").values[:, 0]))
        if self.val_clip_ratio is None and self.holdout_trajectories > 0:
            rng = np.random.default_rng(self.split_seed)
            permuted_indices = rng.permutation(len(samples))
            samples = [samples[idx] for idx in permuted_indices]
            if self.split == "train":
                samples = samples[self.holdout_trajectories :]
            elif self.split == "val":
                samples = samples[: self.holdout_trajectories]
            else:
                raise ValueError(f"Unsupported split={self.split}")
        self.samples = samples
        self.clip_index = None
        if self.enumerate_clips:
            self.clip_index = self._build_clip_index()
            if self.val_clip_ratio is not None:
                if not 0.0 < self.val_clip_ratio < 1.0:
                    raise ValueError(f"val_clip_ratio must be in (0, 1), got {self.val_clip_ratio}")
                rng = np.random.default_rng(self.split_seed)
                permuted_indices = rng.permutation(len(self.clip_index))
                val_count = max(1, int(round(len(self.clip_index) * self.val_clip_ratio)))
                if self.split == "train":
                    chosen_indices = permuted_indices[val_count:]
                elif self.split == "val":
                    chosen_indices = permuted_indices[:val_count]
                else:
                    raise ValueError(f"Unsupported split={self.split}")
                self.clip_index = [self.clip_index[idx] for idx in chosen_indices]
                logger.info(
                    "Clip-level split produced %d %s clips from %d total clips",
                    len(self.clip_index),
                    self.split,
                    len(permuted_indices),
                )

    def __getitem__(self, index):
        if self.clip_index is None:
            path = self.samples[index]
            start_frame = None
        else:
            path, start_frame = self.clip_index[index]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(path, start_frame=start_frame)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {path=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                if self.clip_index is None:
                    path = self.samples[index]
                    start_frame = None
                else:
                    path, start_frame = self.clip_index[index]

        return buffer, actions, states, extrinsics, indices

    def _build_clip_index(self):
        clip_index = []
        for path in self.samples:
            metadata = get_json(path)
            if metadata is None:
                logger.warning(f"Missing metadata while indexing clips for {path}")
                continue
            camera_key = self.camera_views[0]
            mp4_name = metadata[camera_key].split("recordings/MP4/")[-1]
            vpath = os.path.join(path, "recordings/MP4", mp4_name)
            vr = VideoReader(vpath, num_threads=1, ctx=cpu(0))
            vfps = vr.get_avg_fps()
            fps = self.fps if self.fps is not None else vfps
            fstp = ceil(vfps / fps)
            nframes = int(self.frames_per_clip * fstp)
            vlen = len(vr)
            if vlen < nframes:
                logger.warning(f"Skipping short trajectory during clip indexing: {vpath}, {vlen=} < {nframes=}")
                continue
            max_start = vlen - nframes
            starts = range(0, max_start + 1, self.clip_stride_frames)
            clip_index.extend((path, sf) for sf in starts)
        logger.info(f"Enumerated {len(clip_index)} clips from {len(self.samples)} trajectories")
        return clip_index

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def rotvec_poses_to_diffs(self, poses):
        xyz = poses[:, :3]
        rotvecs = poses[:, 3:6]
        matrices = [Rotation.from_rotvec(rotvec).as_matrix() for rotvec in rotvecs]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t].T @ matrices[t + 1] for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_rotvec() for mat in angle_diff]
        angle_diff = np.stack(angle_diff, axis=0)
        gripper = poses[:, -1:]
        gripper_term = gripper[1:] - gripper[:-1]
        return np.concatenate([xyz_diff, angle_diff, gripper_term], axis=1)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path, start_frame=None):
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath)

        # -- randomly sample a camera view
        camera_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"]
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        if start_frame is None:
            ef = np.random.randint(nframes, vlen)
            sf = ef - nframes
        else:
            sf = int(start_frame)
            if sf < 0 or sf + nframes > vlen:
                raise Exception(f"Invalid clip start {sf} for {vpath=}, {nframes=}, {vlen=}")
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
        # --
        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]
        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)
        if self.action_from_state_rotation_mode == "rotvec":
            actions = self.rotvec_poses_to_diffs(states)
        else:
            actions = self.poses_to_diffs(states)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples) if self.clip_index is None else len(self.clip_index)
