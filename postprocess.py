import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from show import plot_joints_anim
from propose.wrappers.poseout_process import ProPoseOutputPostProcess
from tofeature import tofeature
import sys
import os
print(os.getcwd())

name = sys.argv[1]
input_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

print(input_dir, output_dir)

def align(joints_position):
    def rotate_x(a):
        s, c = np.sin(a), np.cos(a)
        return np.array([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]]).astype(np.float32)
    time, joint, _ = joints_position.shape
    joints_position = joints_position.reshape(-1, 3)
    joints_position = (rotate_x(np.pi)[:3,:3] @ joints_position.T).T
    return joints_position.reshape(time, joint, 3)

poses = []
pkls = sorted(map(lambda p: p.as_posix(), input_dir.glob("*.pkl")))
print(len(pkls))
for p in pkls:
    with open(p, "rb") as f:
        pose_output = pickle.load(f)
        pose_output['transl'][..., 1] = 0
        poses.append(pose_output)


joints_position = np.concatenate(
    [
        ProPoseOutputPostProcess(pose).to_smpl_output().joints.cpu().numpy()
        for pose in poses
    ],
    axis=0,
)

joints_position = align(joints_position)
feature = tofeature(joints_position)

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir.joinpath(f"{name}_joints"), joints_position)
np.save(output_dir.joinpath(f"{name}_feat"), feature)