import json
import pdb
import numpy as np
from omegaconf import DictConfig
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
[sys.path.append(i) for i in ['.', '..', '../../animation_lib/BEATS']]

from anim import bvh, quat, txform
from utils_zeggs import write_bvh, write_bvh_upper
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


from scipy.spatial.transform import Rotation
from beat_data_proc.MyBVH import load_bvh_data

from torch.nn import functional as F
from process_beats_npz_2_smplx_20fps import pose2json2, quat2json





def batch_rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x):
    x = x.reshape(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d



def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def euler2mat(angles, euler_orders):
    assert angles.ndim == 3 and angles.shape[2] == 3, f"wrong shape: {angles.shape}"
    assert angles.shape[1] == len(euler_orders)

    nJoints = len(euler_orders)
    nFrames = len(angles)
    rot_mats = np.zeros((nFrames, nJoints, 3, 3), dtype=np.float32)

    for j in range(nJoints):
        # {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations
        R = Rotation.from_euler(euler_orders[j].upper(), angles[:, j, :], degrees=True)  # upper for intrinsic rotation
        rot_mats[:, j, :, :] = R.as_matrix()
    return rot_mats


bone_names = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",#6
    "Head",
    "HeadEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",#11
    "RightHand",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandRing",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandPinky",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandIndex",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumb4",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandRing",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandPinky",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandIndex",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumb4",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightForeFoot",
    "RightToeBase",
    "RightToeBaseEnd",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftForeFoot",
    "LeftToeBase",
    "LeftToeBaseEnd",
    ]


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def preprocess_animation(animation_file, fps=20):
    anim_data = bvh.load(animation_file)       #  'rotations' (8116, 75, 3), 'positions', 'offsets' (75, 3), 'parents', 'names' (75,), 'order' 'zyx', 'frametime' 0.016667
    info = load_bvh_data(animation_file)
    euler_orders=info["euler_orders"][: 75]
    nframes = len(anim_data["rotations"])
    if fps != 20 :
        rate = 20 // fps
        anim_data["rotations"] = anim_data["rotations"][0:nframes:rate]
        anim_data["positions"] = anim_data["positions"][0:nframes:rate]
        dt = 1 / fps
        nframes = anim_data["positions"].shape[0]
    else:
        dt = anim_data["frametime"]
    njoints = len(anim_data["parents"])

    transl=anim_data["positions"][:,0,:]

    #anim_data["rotations"][:,0,0]+=90

    beat_xyz_euler_cp=np.array(anim_data["rotations"])
    pred_pose_rotmat=euler2mat(beat_xyz_euler_cp, euler_orders).astype(np.float32)#34,47,3,3
    print("pred_pose_rotmat shape:",pred_pose_rotmat.shape)
    
    pred_pose_rotmat=pred_pose_rotmat.reshape(-1,75,3,3)
    pred_pose_rotmat=pred_pose_rotmat.reshape(-1,3,3)
    pred_pose_rotmat=torch.tensor(pred_pose_rotmat)
    
    if pred_pose_rotmat.shape[1:] == (3,3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = pred_pose_rotmat.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        pred_pose_rotmat = torch.cat([rot_mat, hom_mat], dim=-1)
    pred_pose_quat=rotation_matrix_to_quaternion(pred_pose_rotmat).reshape(-1,75,4).numpy()
    
    lrot=quat.unroll(pred_pose_quat.copy())
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

    return all_poses, anim_data["parents"], dt, anim_data["order"], njoints,transl


def pose2bvh(poses, outpath, length, smoothing=False, smooth_foot=False, upper=False):
    parents = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15,
    12, 17, 18, 19, 20, 17, 22, 23, 24, 25, 12, 27, 28, 29, 30, 27, 32,
    33, 34, 4, 36, 37, 38, 39, 40, 41, 42, 39, 44, 45, 46, 47, 44, 49,
    50, 51, 52, 39, 54, 55, 56, 57, 54, 59, 60, 61, 0, 63, 64, 65, 66,
    67, 0, 69, 70, 71, 72, 73], dtype=np.int32)
    order = 'zyx'
    dt = 0.05
    njoints = 75

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

    # 20fps
    dt = 1 / 20
    
    # #60fps
    # dt = 1 / 30
    # P_root_pos = P_root_pos.repeat(3, axis=0)
    # P_root_rot = P_root_rot.repeat(3, axis=0)
    # P_lpos = P_lpos.repeat(3, axis=0)
    # P_lrot = P_lrot.repeat(3, axis=0)

    # P_root_pos = P_root_pos[0:len(P_root_pos):2]
    # P_root_rot = P_root_pos[0:len(P_root_rot):2]
    # P_lpos = P_root_pos[0:len(P_lpos):2]
    # P_lrot = P_root_pos[0:len(P_lrot):2]



    if upper:
        write_bvh_upper(outpath,
                P_root_pos,
                P_root_rot,
                P_lpos,
                P_lrot,
                parents, bone_names, order, dt
                )
    else:
        write_bvh(outpath,
                P_root_pos,
                P_root_rot,
                P_lpos,
                P_lrot,
                parents, bone_names, order, dt
                )





def pose2json(bvh_file):
    animation_file = bvh_file
    item=animation_file
    all_poses, parents,dt,order,njoints,transl = preprocess_animation(item, fps=20)
    pose2bvh(poses=all_poses, outpath='1114.bvh', length=all_poses.shape[0], smoothing=True, smooth_foot=True)
    beats_quat=np.load("./beat_zyx_quat.npy")
    beats_aa=quaternion_to_angle_axis(torch.tensor(beats_quat).reshape(-1,4)).numpy().reshape(-1,75,3)
    corr_index =  np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54])
    smplx_index = np.array([0,69,63,1,70,64,2,71,65,3,72,66,4,36,9,5,37,10,38,11,39,12,55,56,57,40,41,42,50,51,52,45,46,47,59,60,61,28,29,30,13,14,15,23,24,25,18,19,20,32,33,34])
    
    smplx_aa = np.zeros((len(beats_aa),55,3))
    smplx_aa[:,corr_index,:]=beats_aa[:,smplx_index,:]
    np.save("./beats_smplx_aa.npy",smplx_aa)

    smplx_aa_body=smplx_aa[:,:22,:]
    smplx_aa_body[:,[7,8,10,11],:] = 0
    smplx_aa_hand=smplx_aa[:,25:,:]
    smplx_aa_jaw=np.zeros([smplx_aa_body.shape[0],3,3])
    smplx_aa=np.concatenate((smplx_aa_body,smplx_aa_jaw,smplx_aa_hand),axis=1)
    transl=np.load("./beat_zyx_transl.npy")
    np.savez("./audio_only_smplx.npz",poses=smplx_aa,trans=transl*0.01,betas=np.zeros(10),gender='male',mocap_framerate=30,)
    print("smplx_aa shape:",smplx_aa.shape)

 
    pose2json2("/apdcephfs/share_1290939/shaolihuang/ykcheng/SIGASIA_realtime_A2G/HoloGest/HoloGest_audio2gesture/main/mydiffusion_zeggs/beat2_our_rot.npz")
    #pose2json2("./audio_only_smplx.npz")







import os

import torch
import numpy as np
from torch.nn import functional as F

if __name__ == '__main__':
    pose2json('/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/demo/main/mydiffusion_zeggs/sample_dir/20240529_223051_smoothing_SG_minibatch_1293_[1, 0, 0, 0, 0, 0, 0, 0]_66233.bvh')


    # animation_file = '/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/trimmed_english_allbeats_emo/train/1_wayne_0_82_82.bvh'
    # item=animation_file
    # all_poses, parents,dt,order,njoints,transl = preprocess_animation(item, fps=30)
    # pose2bvh(poses=all_poses, outpath='1114.bvh', length=all_poses.shape[0], smoothing=True, smooth_foot=True)
    # beats_quat=np.load("./beat_zyx_quat.npy")
    # print("beats_quat shape:",beats_quat.shape)
    # beats_aa=quaternion_to_angle_axis(torch.tensor(beats_quat).reshape(-1,4)).numpy().reshape(-1,75,3)
    # print("beats_aa shape:",beats_aa.shape)
    # corr_index =  np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54])
    # smplx_index = np.array([0,69,63,1,70,64,2,71,65,3,72,66,4,36,9,5,37,10,38,11,39,12,55,56,57,40,41,42,50,51,52,45,46,47,59,60,61,28,29,30,13,14,15,23,24,25,18,19,20,32,33,34])
    
    # smplx_aa = np.zeros((len(beats_aa),55,3))
    # smplx_aa[:,corr_index,:]=beats_aa[:,smplx_index,:]
    # np.save("./beats_smplx_aa.npy",smplx_aa)

    # smplx_aa_body=smplx_aa[:,:22,:]
    # #smplx_aa_body[:,[7,8,10,11],:] = 0
    # smplx_aa_hand=smplx_aa[:,25:,:]
    # smplx_aa_jaw=np.zeros([smplx_aa_body.shape[0],3,3])
    # smplx_aa=np.concatenate((smplx_aa_body,smplx_aa_jaw,smplx_aa_hand),axis=1)
    # transl=np.load("./beat_zyx_transl.npy")
    # np.savez("./audio_only_smplx.npz",poses=smplx_aa,trans=transl*0.01,betas=np.zeros(10),gender='male',mocap_framerate=30,)

    # smplx_fitting_aa = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/remote_down_dir/t3_wavlm_60W_beats_1speaker_didiffges_seed2.npz")['poses']
    # transl = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/remote_down_dir/t3_wavlm_60W_beats_1speaker_didiffges_seed2.npz")['trans']
    # body_indexs = [0,1,2,4,5,7,8,10,11]
    # smplx_aa[:,body_indexs,:] = smplx_fitting_aa[:,body_indexs,:]
    # np.savez("./audio_only_smplx.npz",poses=smplx_aa,trans=transl,betas=np.zeros(10),gender='male',mocap_framerate=30,)

    # np.save('./joby_transl.npy',transl)
    


    # theta = torch.tensor(smplx_aa.reshape(-1,3))
    # l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    # angle = torch.unsqueeze(l1norm, -1)
    # normalized = torch.div(theta, angle)
    # angle = angle * 0.5
    # v_cos = torch.cos(angle)
    # v_sin = torch.sin(angle)
    # quat = torch.cat([v_cos, v_sin * normalized], dim = 1)

    # quat = quat.reshape(-1,55,4).detach().cpu().numpy()
    # np.save('./joby_quat.npy',quat)






    # smplx_aa = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/beats_sig/t2_wavlm_100W_beats_1speaker_didffges_seed2_imitation.npz")['poses']
    # transl = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/beats_sig/t2_wavlm_100W_beats_1speaker_didffges_seed2_imitation.npz")['trans']
    # #np.savez("./audio_only_smplx.npz",poses=smplx_aa,trans=transl,betas=np.zeros(10),gender='male',mocap_framerate=30,)

    # np.save('./joby_transl.npy',transl)
    # smplx_aa[:,7,:]*=0
    # smplx_aa[:,8,:]*=0
    


    # theta = torch.tensor(smplx_aa.reshape(-1,3))
    # l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    # angle = torch.unsqueeze(l1norm, -1)
    # normalized = torch.div(theta, angle)
    # angle = angle * 0.5
    # v_cos = torch.cos(angle)
    # v_sin = torch.sin(angle)
    # quat = torch.cat([v_cos, v_sin * normalized], dim = 1)

    # quat = quat.reshape(-1,55,4).detach().cpu().numpy()
    # np.save('./joby_quat.npy',quat)



    
    





