import json
import pdb
import numpy as np
from omegaconf import DictConfig
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
[sys.path.append(i) for i in ['.', '..', '../../animation_lib/BEATS']]

from anim import bvh, quat, txform
from utils_zeggs import write_bvh
import torch
from scipy.signal import savgol_filter


bone_names = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumb4",
    "RightHandThumb4_End",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndex4_End",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddle4_End",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRing4_End",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinky4_End",
    "RightForeArmEnd",
    "RightForeArmEnd_End",
    "RightArmEnd",
    "RightArmEnd_End",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumb4",
    "LeftHandThumb4_End",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndex4_End",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddle4_End",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRing4_End",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinky4_End",
    "LeftForeArmEnd",
    "LeftForeArmEnd_End",
    "LeftArmEnd",
    "LeftArmEnd_End",
    "Neck",
    "Neck1",
    "Head",
    "HeadEnd",
    "HeadEnd_End",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightToeBaseEnd",
    "RightToeBaseEnd_End",
    "RightLegEnd",
    "RightLegEnd_End",
    "RightUpLegEnd",
    "RightUpLegEnd_End",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeBaseEnd",
    "LeftToeBaseEnd_End",
    "LeftLegEnd",
    "LeftLegEnd_End",
    "LeftUpLegEnd",
    "LeftUpLegEnd_End"
    ]



def preprocess_animation(animation_file, fps=60):
    anim_data = bvh.load(animation_file)       #  'rotations' (8116, 75, 3), 'positions', 'offsets' (75, 3), 'parents', 'names' (75,), 'order' 'zyx', 'frametime' 0.016667
    nframes = len(anim_data["rotations"])

    if fps != 60 :
        rate = 60 // fps
        anim_data["rotations"] = anim_data["rotations"][0:nframes:rate]
        anim_data["positions"] = anim_data["positions"][0:nframes:rate]
        dt = 1 / fps
        nframes = anim_data["positions"].shape[0]
    else:
        dt = anim_data["frametime"]

    njoints = len(anim_data["parents"])

    lrot = quat.unroll(quat.from_euler(np.radians(anim_data["rotations"]), anim_data["order"]))
    lpos = anim_data["positions"]
    grot, gpos = quat.fk(lrot, lpos, anim_data["parents"])
    # Find root (Projected hips on the ground)
    root_pos = gpos[:, anim_data["names"].index("Spine2")] * np.array([1, 0, 1])
    # Root direction
    root_fwd = quat.mul_vec(grot[:, anim_data["names"].index("Hips")], np.array([[0, 0, 1]]))
    root_fwd[:, 1] = 0
    root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]
    # Root rotation
    root_rot = quat.normalize(
        quat.between(np.array([[0, 0, 1]]).repeat(len(root_fwd), axis=0), root_fwd)
    )

    # Find look at direction
    gaze_lookat = quat.mul_vec(grot[:, anim_data["names"].index("Head")], np.array([0, 0, 1]))
    gaze_lookat[:, 1] = 0
    gaze_lookat = gaze_lookat / np.sqrt(np.sum(np.square(gaze_lookat), axis=-1))[..., np.newaxis]
    # Find gaze position
    gaze_distance = 100  # Assume other actor is one meter away
    gaze_pos_all = root_pos + gaze_distance * gaze_lookat
    gaze_pos = np.median(gaze_pos_all, axis=0)
    gaze_pos = gaze_pos[np.newaxis].repeat(nframes, axis=0)

    # Visualize Gaze Pos
    visualize_gaze = False
    if visualize_gaze:
        import matplotlib.pyplot as plt

        plt.scatter(gaze_pos_all[:, 0], gaze_pos_all[:, 2], s=0.1, marker=".")
        plt.scatter(gaze_pos[0, 0], gaze_pos[0, 2])
        plt.scatter(root_pos[:, 0], root_pos[:, 2], s=0.1, marker=".")
        plt.quiver(root_pos[::60, 0], root_pos[::60, 2], root_fwd[::60, 0], root_fwd[::60, 2])
        plt.gca().set_aspect("equal")
        plt.savefig('1.jpg')
        plt.show()

    # Compute local gaze dir
    gaze_dir = gaze_pos - root_pos
    # gaze_dir = gaze_dir / np.sqrt(np.sum(np.square(gaze_dir), axis=-1))[..., np.newaxis]
    gaze_dir = quat.mul_vec(quat.inv(root_rot), gaze_dir)

    # Make relative to root
    lrot[:, 0] = quat.mul(quat.inv(root_rot), lrot[:, 0])
    lpos[:, 0] = quat.mul_vec(quat.inv(root_rot), lpos[:, 0] - root_pos)

    # Local velocities
    lvel = np.zeros_like(lpos)
    lvel[1:] = (lpos[1:] - lpos[:-1]) / dt
    lvel[0] = lvel[1] - (lvel[3] - lvel[2])

    lvrt = np.zeros_like(lpos)
    lvrt[1:] = quat.to_helical(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) / dt
    lvrt[0] = lvrt[1] - (lvrt[3] - lvrt[2])

    # Root velocities
    root_vrt = np.zeros_like(root_pos)
    root_vrt[1:] = quat.to_helical(quat.abs(quat.mul(root_rot[1:], quat.inv(root_rot[:-1])))) / dt
    root_vrt[0] = root_vrt[1] - (root_vrt[3] - root_vrt[2])
    root_vrt[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vrt[1:])
    root_vrt[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vrt[0])

    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel[0] = root_vel[1] - (root_vel[3] - root_vel[2])
    root_vel[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vel[1:])
    root_vel[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vel[0])

    # Compute character space
    crot, cpos, cvrt, cvel = quat.fk_vel(lrot, lpos, lvrt, lvel, anim_data["parents"])

    # Compute 2-axis transforms
    ltxy = np.zeros(dtype=np.float32, shape=[len(lrot), njoints, 2, 3])
    ltxy[..., 0, :] = quat.mul_vec(lrot, np.array([1.0, 0.0, 0.0]))
    ltxy[..., 1, :] = quat.mul_vec(lrot, np.array([0.0, 1.0, 0.0]))

    ctxy = np.zeros(dtype=np.float32, shape=[len(crot), njoints, 2, 3])
    ctxy[..., 0, :] = quat.mul_vec(crot, np.array([1.0, 0.0, 0.0]))
    ctxy[..., 1, :] = quat.mul_vec(crot, np.array([0.0, 1.0, 0.0]))


    lpos = lpos.reshape(nframes, -1)
    ltxy = ltxy.reshape(nframes, -1)
    lvel = lvel.reshape(nframes, -1)
    lvrt = lvrt.reshape(nframes, -1)

    all_poses = np.concatenate((root_pos, root_rot, root_vel, root_vrt, lpos, ltxy, lvel, lvrt, gaze_dir), axis=1)

    return all_poses, anim_data["parents"], dt, anim_data["order"], njoints


def pose2bvh(poses, outpath, length, smoothing=False, smooth_foot=False):
    parents = np.array([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,8,14,15,16,17,8,19,20,21,22,8,24,25,26,27,8,29,30,31,32,8,34,7,36,4,38,39,40,41,42,43,44,45,41,47,48,49,50,41,52,53,54,55,41,57,58,59,60,41,62,63,64,65,41,67,40,69,4,71,72,73,74,0,76,77,78,79,80,78,82,77,84,0,86,87,88,89,90,88,92,87,94], dtype=np.int32)



    order = 'zyx'
    dt = 0.05
    njoints = 96

    # smoothing
    if smoothing:
        n_poses = poses.shape[0]
        out_poses = np.zeros((n_poses, poses.shape[1]))
        for i in range(out_poses.shape[1]):
            # if (13 + (njoints - 14) * 9) <= i < (13 + njoints * 9): out_poses[:, i] = savgol_filter(poses[:, i], 41, 2)  # NOTE: smoothing on rotation matrices is not optimal
            # else:
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
    else:
        out_poses = poses

    # Extract predictions
    P_root_pos = out_poses[:, 0:3]
    P_root_rot = out_poses[:, 3:7]
    P_root_vel = out_poses[:, 7:10]
    P_root_vrt = out_poses[:, 10:13]
    P_lpos = out_poses[:, 13 + njoints * 0: 13 + njoints * 3].reshape([length, njoints, 3])
    P_ltxy = out_poses[:, 13 + njoints * 3: 13 + njoints * 9].reshape([length, njoints, 2, 3])
    P_lvel = out_poses[:, 13 + njoints * 9: 13 + njoints * 12].reshape([length, njoints, 3])
    P_lvrt = out_poses[:, 13 + njoints * 12: 13 + njoints * 15].reshape([length, njoints, 3])

    P_ltxy = torch.as_tensor(P_ltxy, dtype=torch.float32)
    P_lrot = quat.from_xform(txform.xform_orthogonalize_from_xy(P_ltxy).cpu().numpy())        #

    if smooth_foot:
        pdb.set_trace()
        next_poses_LeftToeBase = P_lrot[:, -7]      # (length, 4)       7/14, 5/12
        next_poses_RightToeBase = P_lrot[:, -14]
        next_poses_LeftToeBase = np.zeros_like(next_poses_LeftToeBase)
        next_poses_RightToeBase = np.zeros_like(next_poses_RightToeBase)
        P_lrot[:, -7] = next_poses_LeftToeBase
        P_lrot[:, -14] = next_poses_RightToeBase

    # 20fps -> 60fps
    dt = 1 / 20
    # P_root_pos = P_root_pos.repeat(3, axis=0)
    # P_root_rot = P_root_rot.repeat(3, axis=0)
    # P_lpos = P_lpos.repeat(3, axis=0)
    # P_lrot = P_lrot.repeat(3, axis=0)

    write_bvh(outpath,
              P_root_pos,
              P_root_rot,
              P_lpos,
              P_lrot,
              parents, bone_names, order, dt
              )

if __name__ == '__main__':
    '''
    cd mymdm/process
    python process_zeggs_bvh.py
    '''
    all_poses, parents, dt, order, njoints = preprocess_animation('/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/untitled.bvh', fps=20)       # 20
    pose2bvh(poses=all_poses, outpath='ori_zegg_1114.bvh', length=all_poses.shape[0], smoothing=True, smooth_foot=False)

