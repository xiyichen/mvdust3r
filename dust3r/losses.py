from copy import copy, deepcopy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from pytorch3d.ops import knn_points
from dust3r.utils.geometry import xy_grid

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud, normalize_pointclouds
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale, get_joint_pointcloud_depths, get_joint_pointcloud_center_scales

from torch.utils.data import default_collate
import pdb

import random
from pytorch3d.transforms import so3_relative_angle

def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2

def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix

def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    try:
        rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-1)
    except:
        R_diff = rot_gt @ rot_pred.transpose(-1, -2)
        trace_R_diff = R_diff[:,0,0] + R_diff[:,1,1] + R_diff[:,2,2]
        cos = (trace_R_diff - 1) / 2
        cos = torch.clamp(cos, -1 + 1e-3, 1 - 1e-3)
        rel_angle_cos = torch.acos(cos)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def camera_to_rel_deg(pred_se3, gt_se3, device, batch_size): # w2c
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # # Convert cameras to 4x4 SE3 transformation matrices
        # gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        # pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0] // batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        relative_pose_gt = torch.linalg.inv(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = torch.linalg.inv(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
        rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg

def estimate_focal_knowing_depth(pts3d, valid_mask, min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape # valid_mask: [1, H, W], bs = 1
    assert THREE == 3

    # centered pixel grid
    pp = torch.tensor([[W/2, H/2]], dtype=torch.float32, device=pts3d.device)
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)
    valid_mask = valid_mask.flatten(1, 2)  # (B, HW, 1)
    pixels = pixels[valid_mask].unsqueeze(0)  # (1, N, 2)
    pts3d = pts3d[valid_mask].unsqueeze(0)  # (1, N, 3)

    # init focal with l2 closed form
    # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
    xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

    dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
    dot_xy_xy = xy_over_z.square().sum(dim=-1)

    focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

    # iterative re-weighted least-squares
    for iter in range(10):
        # re-weighting by inverse of distance
        dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
        # print(dis.nanmean(-1))
        w = dis.clip(min=1e-8).reciprocal()
        # update the scaling with the new weights
        focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal

def recursive_concat_collate(batch):
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(batch, dim=0)
    
    elif isinstance(batch[0], dict):
        return {key: recursive_concat_collate([d[key] for d in batch]) for key in batch[0]}
    
    elif isinstance(batch[0], list):
        return [recursive_concat_collate([d[i] for d in batch]) for i in range(len(batch[0]))]
    
    else:
        return batch

def recursive_repeat_interleave_collate(data, dim = 0, rp = 1):

    if torch.is_tensor(data):
        return data.repeat_interleave(rp, dim=dim)
    elif isinstance(data, dict):
        return {key: recursive_repeat_interleave_collate(value, dim, rp) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursive_repeat_interleave_collate(element, dim, rp) for element in data]
    elif isinstance(data, tuple):
        return tuple(recursive_repeat_interleave_collate(element, dim, rp) for element in data)
    else:
        return data

def combine_dict(dicts, make_list = False):
    if make_list:
        dict_all = {k:[] for k in dicts[0].keys()}
        for dict_i in dicts:
            for k in dict_i.keys():
                dict_all[k].append(dict_i[k])
        return dict_all
    else:
        dict_all = deepcopy(dicts[0])
        for dict_i in dicts[1:]:
            for k in dict_i.keys():
                dict_all[k] = dict_all[k] + dict_i[k]
        for k in dict_all.keys():
            dict_all[k] = dict_all[k] / len(dicts)
        return dict_all

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            # if loss2 is list
            if isinstance(loss2, list): # for MV use
                for loss2_i in loss2:
                    loss = loss + loss2_i
            else:
                loss = loss + loss2
        return loss

def extend_gts(gts, n_ref, bs):
        gts = recursive_repeat_interleave_collate(gts, 0, n_ref)
        for data_id in range(bs):
            for ref_id in range(1, n_ref):
                for k in gts[0].keys():
                    recursive_swap(gts[0][k], gts[ref_id][k], data_id * n_ref + ref_id)
        return gts

def swap(a, b):
    if type(a) == torch.Tensor:
        return b.clone(), a.clone()
    else:
        raise NotImplementedError

def swap_ref(a, b):
    if type(a) == torch.Tensor:
        temp = a.clone()
        a[:] = b.clone()
        b[:] = temp
    else:
        raise NotImplementedError

def recursive_swap(a, b, pos):

    if torch.is_tensor(a):
        a[pos], b[pos] = swap(a[pos], b[pos])
    elif isinstance(a, dict):
        for key in a.keys():
            recursive_swap(a[key], b[key], pos)
    elif isinstance(a, list):
        for i in range(len(a)):
            recursive_swap(a[i], b[i], pos)
    elif isinstance(a, tuple):
        for i in range(len(a)):
            recursive_swap(a[i], b[i], pos)
    else:
        return 

def calculate_RRA_RTA(c2w_pred1, c2w_pred2, c2w_gt1, c2w_gt2, eps = 1e-15): # [bs, 3, 3], [bs, 3, 3], [bs, 3, 1], [bs, 3, 1]
    """
    Return:
        RRA: [bs,]
        RTA: [bs,]
    """
    # R1, R2, R1_gt, R2_gt, t1, t2, t1_gt, t2_gt
    R1 = c2w_pred1[:, :3, :3]
    R2 = c2w_pred2[:, :3, :3]
    R1_gt = c2w_gt1[:, :3, :3]
    R2_gt = c2w_gt2[:, :3, :3]
    t1 = c2w_pred1[:, :3, 3:]
    t2 = c2w_pred2[:, :3, 3:]
    t1_gt = c2w_gt1[:, :3, 3:]
    t2_gt = c2w_gt2[:, :3, 3:]

    bs = R1.shape[0]
    R_pred = R1 @ R2.transpose(-1, -2)
    R_gt = R1_gt @ R2_gt.transpose(-1, -2)
    R_diff = R_pred @ R_gt.transpose(-1, -2)

    P_pred_diff = c2w_pred1 @ torch.linalg.inv(c2w_pred2)
    P_gt_diff = c2w_gt1 @ torch.linalg.inv(c2w_gt2)
    # print(R_diff)
    trace_R_diff = R_diff[:,0,0] + R_diff[:,1,1] + R_diff[:,2,2]
    # print(trace_R_diff.shape, 'trace')
    theta = torch.acos((trace_R_diff - 1) / 2)
    theta[theta > torch.pi] = theta[theta > torch.pi] - 2 * torch.pi
    theta[theta < -torch.pi] = theta[theta < -torch.pi] + 2 * torch.pi
    theta = theta * 180 / torch.pi

    # t_pred = t1 - t2
    # t_gt = t1_gt - t2_gt
    # t_pred = P_pred_diff[:, :3, 3:]
    # t_gt = P_gt_diff[:, :3, 3:]
    t_pred = P_pred_diff[:, 3, :3]
    t_gt = P_gt_diff[:, 3, :3]

    # cos_t = (t_pred * t_gt).sum((1,2)) / (torch.norm(t_pred, dim = (-1, -2)) + 1e-8) / (torch.norm(t_gt, dim = (-1, -2)) + 1e-8)
    # theta_t = torch.acos(cos_t)

    default_err = 1e6
    t_norm = torch.norm(t_pred, dim=1, keepdim=True)
    t = t_pred / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    theta_t = torch.acos(torch.sqrt(1 - loss_t))

    theta_t[torch.isnan(theta_t) | torch.isinf(theta_t)] = default_err



    theta_t[theta_t > torch.pi] = theta_t[theta_t > torch.pi] - 2 * torch.pi
    theta_t[theta_t < -torch.pi] = theta_t[theta_t < -torch.pi] + 2 * torch.pi
    theta_t = theta_t * 180 / torch.pi

    return theta.abs(), theta_t.abs()

def calibrate_camera_pnpransac(pointclouds, img_points, masks, intrinsics):
    """
    Input:
        pointclouds: (bs, N, 3) 
        img_points: (bs, N, 2) 
    Return:
        rotations: (bs, 3, 3) 
        translations: (bs, 3, 1) 
        c2ws: (bs, 4, 4) 
    """
    bs = pointclouds.shape[0]
    
    camera_matrix = intrinsics.cpu().numpy()  # (bs, 3, 3)
    
    dist_coeffs = np.zeros((5, 1))

    rotations = []
    translations = []
    
    for i in range(bs):
        obj_points = pointclouds[i][masks[i]].cpu().numpy()
        img_pts = img_points[i][[masks[i]]].cpu().numpy()

        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_pts, camera_matrix[i], dist_coeffs)

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotations.append(torch.tensor(rotation_matrix, dtype=torch.float32))
            translations.append(torch.tensor(tvec, dtype=torch.float32))
        else:
            rotations.append(torch.eye(3))
            translations.append(torch.ones(3, 1))

    rotations = torch.stack(rotations).to(pointclouds.device)
    translations = torch.stack(translations).to(pointclouds.device)
    w2cs = torch.eye(4).repeat(bs, 1, 1).to(pointclouds.device)
    w2cs[:, :3, :3] = rotations
    w2cs[:, :3, 3:] = translations
    return torch.linalg.inv(w2cs)

def umeyama_alignment(P1, P2, mask_): # [bs, N, 3], [bs, N, 3], [bs, N] all are torch.Tensor

    """
    Return:
    R: (bs, 3, 3)
    sigma: (bs, )
    t: (bs, 3)
    """
    from pytorch3d import ops
    ya = ops.corresponding_points_alignment
    R, T, s = ya(P1, P2, weights = mask_.float(), estimate_scale = True)
    return R, s, T

    bs, _ = P1.shape[0:2]
    ns = mask_.sum(1) # [bs, ]
    mask = mask_[:,:,None] # [bs, N, 1]

    mu1 = (P1 * mask).sum(1) / ns[:, None] # [bs, 3]
    mu2 = (P2 * mask).sum(1) / ns[:, None]

    X1 = P1 - mu1[:, None] # [bs, N, 3]
    X2 = P2 - mu2[:, None]
    
    X1_zero = X1 * mask
    X2_zero = X2 * mask
    # Calculate the Cov
    S = (X2_zero.transpose(-1, -2) @ X1_zero) / (ns[:, None, None] + 1e-8) # (bs, 3, 3)

    # SVD decomposition
    U, D, Vt = torch.linalg.svd(S) # [bs, 3, 3], [bs, 3], [bs, 3, 3]

    # diag mat D_mat
    d = torch.ones((bs, 3)).to(P1.device)
    det = torch.linalg.det(U @ Vt) # [bs, ]
    d[:, -1][det < 0] = -1

    D_mat = torch.eye(3).to(P1.device).repeat(bs, 1, 1) # [bs, 3, 3]
    D_mat[:, -1, -1] = d[:, -1]

    # rotation R
    R = U @ D_mat @ Vt

    D_diag = torch.diag_embed(D) # [bs, 3, 3]

    # scale sigma
    var1 = torch.square(X1_zero).sum(dim = (1, 2)) / (ns + 1e-8) # [bs, ]
    sigma = (D_mat @ D_diag).diagonal(dim1 = -2, dim2 = -1).sum(-1) / (var1 + 1e-8) # [bs, ]
    # sigma = np.trace(D_mat @ D_diag) / var1

    # translation t
    # t = mu2 - sigma * R @ mu1 
    t = mu2 - sigma[:, None] * (R @ mu1[:, :, None])[:, :, 0]
    
    return R, sigma, t # [bs, 3, 3], [bs, ], [bs, 3]

def chamfer_distance(pts1, pts2, mask): # [bs, N, 3], [bs, N, 3], [bs, N]
    bs = pts1.shape[0]
    cd = []
    for i in range(bs):
        disAB = knn_points(pts1[i:i+1][mask[i:i+1]][None], pts2[i:i+1][mask[i:i+1]][None])[0].mean()
        disBA = knn_points(pts2[i:i+1][mask[i:i+1]][None], pts1[i:i+1][mask[i:i+1]][None])[0].mean()
        cd.append(disAB + disBA)
    cd = torch.stack(cd, 0)
    return cd

def rotationInvMSE(pts3d_normalized, gts3d_normalized, mask_all):

    R, sigma, t = umeyama_alignment(pts3d_normalized, gts3d_normalized, mask_all)
    pts3d_normalized_rot = (sigma[:,None,None] * (R @ pts3d_normalized.transpose(-1, -2)).transpose(-1, -2)) + t[:, None] # [bs, h*w, 3]
    local_loss = (pts3d_normalized_rot - gts3d_normalized).norm(dim = -1)[mask_all].mean()

class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b, mask = None, reduction=None):
        if mask is not None:
            dist = self.distance(a, b)
            assert reduction == "mean_bs"
            bs = dist.shape[0]
            dist_mean = []
            for i in range(bs):
                mask_dist_i = dist[i][mask[i]]
                if mask_dist_i.numel() > 0:
                    dist_mean.append(mask_dist_i.mean())
                else:
                    dist_mean.append(dist.new_zeros(()))
            dist_mean = torch.stack(dist_mean, 0)
            return dist_mean

        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        reduction_effective = self.reduction
        if reduction is not None:
            reduction_effective = reduction
        if reduction_effective == 'none':
            return dist
        if reduction_effective == 'sum':
            return dist.sum()
        if reduction_effective == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        if reduction_effective == 'mean_bs':
            bs = dist.shape[0]
            return (dist.mean([i for i in range(1, dist.ndim)]) if dist.ndim >= 2 else dist) if dist.numel() > 0 else dist.new_zeros((bs,))
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'+bb()
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res

    def rearrange_for_mref(self, gt1, gt2s, pred1, pred2s):
        
        if gt1['img'].shape[0] == pred1['pts3d'].shape[0]:
            return gt1, gt2s, pred1, pred2s, 1
        bs = gt1['img'].shape[0]
        bs_pred = pred1['pts3d'].shape[0]
        n_ref = bs_pred // bs
        gts = [gt1] + gt2s
        preds = [pred1] + pred2s

        gts = extend_gts(gts, n_ref, bs)

        return gts[0], gts[1:], preds[0], preds[1:], n_ref


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        # print('before compute loss', args, kwargs)
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, mv = False, rot_invariant = False, dummy = False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.mv = mv
        self.rot_invariant = rot_invariant
        self.dummy = dummy
        if mv:
            self.compute_loss = self.compute_loss_mv

    def get_all_pts3ds(self, gt1, gt2s, pred1, pred2s, dist_clip=None, **kw):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose']) # c2w -> w2c
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2s = [geotrf(in_camera1, gt2['pts3d']) for gt2 in gt2s]  # list of B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2s = [gt2['valid_mask'].clone() for gt2 in gt2s]  # list of B,H,W
        
        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2s = [gt_pts2.norm(dim=-1) for gt_pts2 in gt_pts2s]  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2s = [valid2 & (dis2 <= dist_clip) for (valid2, dis2) in zip(valid2s, dis2s)]  # (B, H, W)
        
        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2s = [get_pred_pts3d(gt2, pred2, use_pose=True) for (gt2, pred2) in zip(gt2s, pred2s)]  # (B, H, W, 3)
        
        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2s = normalize_pointclouds(pr_pts1, pr_pts2s, self.norm_mode, valid1, valid2s)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2s = normalize_pointclouds(gt_pts1, gt_pts2s, self.norm_mode, valid1, valid2s)
        
        return gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, valid1, valid2s, {}

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose']) # c2w -> w2c
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # loss on gt2 side
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])
        self_name = type(self).__name__
        details = {self_name+'_pts3d_1': float(l1.mean()), self_name+'_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)

    def compute_loss_mv(self, gt1, gt2s_all, pred1, pred2s, log, **kw):
        C_avg = gt1['C_avg'].mean()
        gt1, gt2s_all, pred1, pred2s, n_ref = self.rearrange_for_mref(gt1, gt2s_all, pred1, pred2s)
        num_render_views = gt2s_all[0].get("num_render_views", torch.zeros([0]).long())[0].item()
        gt2s = gt2s_all[:-num_render_views] if num_render_views else gt2s_all
        
        gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring = \
            self.get_all_pts3ds(gt1, gt2s, pred1, pred2s, **kw)

        nv = len(gt_pts2s) + 1
        bs = gt_pts1.shape[0]
        h = gt_pts1.shape[1]
        w = gt_pts1.shape[2]
        # gt_pts1 (bs, h, w, 3), mask1 (bs, h, w)

        if log:
            pts_thres = 10.
            gt_pcds_original = [gt1['pts3d']] + [gt2['pts3d'] for gt2 in gt2s]
            gt_pcds_original = torch.stack(gt_pcds_original, 1) # [bs, nv, h, w, 3]
            gt_pcds_original = gt_pcds_original.flatten(1, 3) # [bs, nv*h*w, 3]
            gt_pcds_original_0_1 = torch.quantile(gt_pcds_original, 0.1, dim = 1) # [bs, 3]
            gt_pcds_original_0_9 = torch.quantile(gt_pcds_original, 0.9, dim = 1) # [bs, 3]
            gt_pcds_original_diff = gt_pcds_original_0_9 - gt_pcds_original_0_1
            l1 = self.criterion(pred_pts1, gt_pts1, mask1, reduction='mean_bs')
            l2s = [self.criterion(pred_pts2, gt_pts2, mask2, reduction='mean_bs') for (gt_pts2, pred_pts2, mask2) in zip(gt_pts2s, pred_pts2s, mask2s)]
            
            self_name = type(self).__name__
            details = {self_name+'_pts3d_1': float(l1.mean()), self_name+'_pts3d_2': np.mean([float(l2.mean()) for l2 in l2s]).item()}
            ls = torch.stack([l1.reshape(-1)] + [l2.reshape(-1) for l2 in l2s], -1) # [bs, nv]
            ls[ls > pts_thres] = pts_thres
            ls_mref = ls.reshape(bs // n_ref, n_ref, nv)
            ls_only_first = ls_mref[:, 0] # [bs // n_ref, nv]
            ls_best = torch.min(ls_mref, dim = 1)[0]
            
            details[self_name+'_area0_list'] = (gt_pcds_original_diff[:,1] * gt_pcds_original_diff[:,2]).detach().cpu().tolist()
            details[self_name+'_area1_list'] = (gt_pcds_original_diff[:,0] * gt_pcds_original_diff[:,2]).detach().cpu().tolist()
            details[self_name+'_area2_list'] = (gt_pcds_original_diff[:,0] * gt_pcds_original_diff[:,1]).detach().cpu().tolist()
            details[self_name+'_volume_list'] = (gt_pcds_original_diff[:,2] * gt_pcds_original_diff[:,0] * gt_pcds_original_diff[:,1]).detach().cpu().tolist()
            details[self_name+'_pts3d_list'] =  ls.mean(-1).detach().cpu().tolist() # [bs, nv] -> [bs]
            details[self_name+'_pts3d_1_first'] = ls_only_first[:, 0].mean().item()
            details[self_name+'_pts3d_2_first'] = ls_only_first[:, 1:].mean().item()
            details[self_name+'_pts3d_first'] = ls_only_first.mean().item()
            details[self_name+'_pts3d_best'] = ls_best.mean().item()
            details[self_name+'_pts3d_0.5_accu_list'] = (ls.mean(-1) < 0.5).float().detach().cpu().tolist()
            details[self_name+'_pts3d_0.3_accu_list'] = (ls.mean(-1) < 0.3).float().detach().cpu().tolist()
            details[self_name+'_pts3d_0.2_accu_list'] = (ls.mean(-1) < 0.2).float().detach().cpu().tolist()
            details[self_name+'_pts3d_0.1_accu_list'] = (ls.mean(-1) < 0.1).float().detach().cpu().tolist()
            details['n_ref'] = n_ref
            details[self_name+"_C_avg"] = C_avg.item()

            pred_pts = torch.stack([pred_pts1, *pred_pts2s], 1).flatten(1, 3) # [bs, nv, h, w, 3] -> [bs, nv*h*w, 3]
            gt_pts = torch.stack([gt_pts1, *gt_pts2s], 1).flatten(1, 3) # [bs, nv, h, w, 3] -> [bs, nv*h*w, 3]
            masks = torch.stack([mask1, *mask2s], 1).flatten(1, 3) # [bs, nv, h, w] -> [bs, nv*h*w]

            R, sigma, t = umeyama_alignment(pred_pts, gt_pts, masks)
            pts3d_normalized_rot = (sigma[:,None,None] * (R @ pred_pts.transpose(-1, -2)).transpose(-1, -2)) + t[:, None] # [bs, nv*h*w, 3]
            ls_rotInv = (pts3d_normalized_rot - gt_pts).norm(dim = -1).reshape(bs, -1) # [bs, nv*h*w] # TODO: need debug, seems it's wrong
            ls_rotInv[masks][:] = 0.
            ls_rotInv = ls_rotInv.sum(-1) / (masks.sum(-1) + 1e-8) # [bs,]
            details[self_name+'_pts3d_rotInv_list'] =  ls_rotInv.detach().cpu().tolist()
            
            if self.dummy: # CD's calculation cost much time, if we do not need it, just set to zero.
                details[self_name+'_cd_list'] =  [0. for i in range(bs)] # [bs, nv] -> [bs]
                details[self_name+'_cd'] = 0.
                details[self_name+'_cd_first'] = 0.
                details[self_name+'_cd_best'] = 0.
            else:
                cd = chamfer_distance(pred_pts, gt_pts, masks) # [bs]
                cd[cd > pts_thres] = pts_thres
                cd_mref = cd.reshape(bs // n_ref, n_ref)
                details[self_name+'_cd_list'] =  cd.detach().cpu().tolist() # [bs, nv] -> [bs]
                details[self_name+'_cd'] = cd.mean().item()
                details[self_name+'_cd_first'] = cd_mref[:, 0].mean().item()
                details[self_name+'_cd_best'] = torch.min(cd_mref, dim = 1)[0].mean().item()
            
            l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
            l2s = [self.criterion(pred_pts2[mask2], gt_pts2[mask2]) for (gt_pts2, pred_pts2, mask2) in zip(gt_pts2s, pred_pts2s, mask2s)]
            
        else:
            if self.rot_invariant:
                pred_pts = torch.stack([pred_pts1, *pred_pts2s], 1).flatten(1,3) # [bs, nv, h, w, 3] -> [bs, nv*h*w, 3]
                gt_pts = torch.stack([gt_pts1, *gt_pts2s], 1).flatten(1,3)
                mask = torch.stack([mask1, *mask2s], 1).flatten(1,3) # [bs, nv, h, w] -> [bs, nv*h*w]

                R, sigma, t = umeyama_alignment(pred_pts, gt_pts, mask)
                pts3d_normalized_rot = (sigma[:,None,None] * (R @ pred_pts.transpose(-1, -2)).transpose(-1, -2)) + t[:, None] # [bs, nv*h*w, 3]
                ls = (pts3d_normalized_rot - gt_pts).norm(dim = -1).reshape(bs, nv, -1) # [bs, nv, h*w]
                mask = mask.reshape(bs, nv, -1) # [bs, nv, h*w]
                
                mask1 = mask[:, 0].reshape(bs, h, w)
                l1 = ls[:, 0][mask[:, 0]]

                mask2s = [mask[:, i].reshape(bs, h, w) for i in range(1, nv)]
                l2s = [ls[:, i][mask[:, i]] for i in range(1, nv)]

            else:
                l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
                l2s = [self.criterion(pred_pts2[mask2], gt_pts2[mask2]) for (gt_pts2, pred_pts2, mask2) in zip(gt_pts2s, pred_pts2s, mask2s)]
                
            details = {}
        
        return Sum((l1, mask1), (l2s, mask2s)), (details | monitoring)


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha >= 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')
        self.mv = self.pixel_loss.mv
        if self.mv:
            self.compute_loss = self.compute_loss_mv

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss_mv(self, gt1, gt2s, pred1, pred2s, **kw):
        ((loss1, msk1), (loss2s, msk2s)), details = self.pixel_loss(gt1, gt2s, pred1, pred2s, **kw)

        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0  # averaged across pixels accross samples in one batch
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)

        conf_loss2s = []
        for (loss2, msk2, pred2) in zip(loss2s, msk2s, pred2s):
            if loss2.numel() == 0:
                print('NO VALID POINTS in img2', force=True)

            # weight by confidence
            conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
            # print('conf loss debug', log_conf2.mean(), conf2.min(), conf2.max())
            conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

            # average + nan protection (in case of no valid pixels at all)
            conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
            conf_loss2s.append(conf_loss2)
        conf_loss2 = sum(conf_loss2s)

        return conf_loss1 + conf_loss2, dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), **details)


    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), **details)


class Regr3D_ShiftInv (Regr3D): # first this
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):

        if self.mv:
            return self.get_all_pts3ds(gt1, gt2, pred1, pred2)
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring

    def get_all_pts3ds(self, gt1, gt2s, pred1, pred2s, **kw):
        # compute unnormalized points
        gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring = \
            super().get_all_pts3ds(gt1, gt2s, pred1, pred2s)

        # compute median depth
        gt_z1, gt_z2s = gt_pts1[..., 2], [gt_pts2[..., 2] for gt_pts2 in gt_pts2s]
        pred_z1, pred_z2s = pred_pts1[..., 2], [pred_pts2[..., 2] for pred_pts2 in pred_pts2s]
        gt_shift_z = get_joint_pointcloud_depths(gt_z1, gt_z2s, mask1, mask2s)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depths(pred_z1, pred_z2s, mask1, mask2s)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        for gt_z2 in gt_z2s:
            gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        for pred_z2 in pred_z2s:
            pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring

class Regr3D_ShiftAllInv (Regr3D): # first this
    """ Same than Regr3D but invariant to shift of xyz (center to original)
    """

    def get_all_pts3ds(self, gt1, gt2s, pred1, pred2s, **kw):
        # compute unnormalized points
        gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring = \
            super().get_all_pts3ds(gt1, gt2s, pred1, pred2s)
        
        for coor_id in range(3):
            # compute median depth
            gt_z1, gt_z2s = gt_pts1[..., coor_id], [gt_pts2[..., coor_id] for gt_pts2 in gt_pts2s]
            pred_z1, pred_z2s = pred_pts1[..., coor_id], [pred_pts2[..., coor_id] for pred_pts2 in pred_pts2s]
            gt_shift_z = get_joint_pointcloud_depths(gt_z1, gt_z2s, mask1, mask2s)[:, None, None]
            pred_shift_z = get_joint_pointcloud_depths(pred_z1, pred_z2s, mask1, mask2s)[:, None, None]

            # subtract the median depth
            gt_z1 -= gt_shift_z
            for gt_z2 in gt_z2s:
                gt_z2 -= gt_shift_z
            pred_z1 -= pred_shift_z
            for pred_z2 in pred_z2s:
                pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring
    

class Regr3D_ScaleInv (Regr3D): # then this
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring

    def get_all_pts3ds(self, gt1, gt2s, pred1, pred2s, **kw):
        # compute depth-normalized points
        gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring = super().get_all_pts3ds(gt1, gt2s, pred1, pred2s)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scales(gt_pts1, gt_pts2s, mask1, mask2s)
        _, pred_scale = get_joint_pointcloud_center_scales(pred_pts1, pred_pts2s, mask1, mask2s)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            for pred_pts2 in pred_pts2s:
                pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            for gt_pts2 in gt_pts2s:
                gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            for pred_pts2 in pred_pts2s:
                pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):

    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass

class Regr3D_ScaleShiftAllInv (Regr3D_ScaleInv, Regr3D_ShiftAllInv):

    # calls Regr3D_ShiftAllInv first, then Regr3D_ScaleInv
    pass

class CalcMetrics(): # note that this is not differentiable

    def __init__(self, random_crop_size = None, resize = None):
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        from torchvision import transforms
        self.random_crop = transforms.RandomCrop(random_crop_size) if random_crop_size is not None else None
        self.resize = transforms.Resize(resize) if resize is not None else None

        self.laplacian_kernel = torch.tensor([[1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def calc_metrics(self, img_gt, img_, log = True): # img and img_gt should be in -1~1
        
        self.psnr = self.psnr.to(img_.device)
        self.lpips = self.lpips.to(img_.device)
        img = torch.clip(img_, -1, 1)

        results = {}
        img_01 = (img + 1) / 2
        img_gt_01 = (img_gt + 1) / 2
        if log:
            results['psnr'] = float(self.psnr(img_gt_01, img_01).item())
            results['ssim'] = float(self.ssim(img_gt_01[None], img_01[None]).item())
            results['lpips'] = float(self.lpips(img_gt[None], img[None]).item())
        else:
            results['psnr'] = 0.
            results['ssim'] = 0.
            results['lpips'] = 0.
        return results

    def calc_lpips(self, img_gt, img): # [bs, 3, h, w]

        self.lpips = self.lpips.to(img.device)
        if self.random_crop is not None:
            all_img = torch.cat([img_gt, img], dim=0) # [2bs, 3, h, w]
            all_img = self.random_crop(all_img)
            img_gt, img = all_img[:img_gt.shape[0]], all_img[img_gt.shape[0]:] # [bs, 3, h, w]
        if self.resize is not None:
            img_gt = self.resize(img_gt)
            img = self.resize(img)
        img = torch.clip(img, -1, 1)
        img_gt = torch.clip(img_gt, -1, 1)
        return self.lpips(img_gt, img) # a float (averaged)

    def laplace(self, img): # [bs, h, w, 1]

        # [out_channels, in_channels, kernel_height, kernel_width]
        laplacian_kernel = self.laplacian_kernel.to(img.device)
        return F.conv2d(img.permute(0,3,1,2), laplacian_kernel, padding=1).permute(0,2,3,1) # [bs, h, w, 1]

# calc_metrics = CalcMetrics()
# calc_metrics = CalcMetrics(random_crop_size = (224, 224))
calc_metrics = CalcMetrics(resize = (384, 512))

class GSRenderLoss (Criterion, MultiLoss):

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, mv = False, render_included = False, scale_scaled = True, use_gt_pcd = False, lpips_coeff = 0., rgb_coeff = 1.0, copy_rgb_coeff = 10.0, use_img_rgb = False, cam_relocation = False, local_loss_coeff = 0., lap_loss_coeff = 0.): # criterion = L21, the others are as default
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.mv = mv
        self.compute_loss = None
        if mv:
            self.compute_loss = self.compute_loss_mv
        self.render_included = render_included
        self.scale_scaled = scale_scaled
        self.use_gt_pcd = use_gt_pcd
        self.lpips_coeff = lpips_coeff
        self.rgb_coeff = rgb_coeff
        self.copy_rgb_coeff = copy_rgb_coeff
        self.use_img_rgb = use_img_rgb
        self.cam_relocation = cam_relocation
        self.local_loss_coeff = local_loss_coeff
        self.lap_loss_coeff = lap_loss_coeff

        from dust3r.gs import GaussianRenderer
        self.gs_renderer = GaussianRenderer()


    def get_all_pts3ds_to_canonical_cam(self, gt1, gt2s, pred1, pred2s, log = False, dist_clip=None, **kw):
        # everything is normalized w.r.t. camera of view1
        # W1 -> W2     W1->W2 c2w1
        in_camera1 = inv(gt1['camera_pose']) # c2w -> w2c # B,3,3
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2s = [geotrf(in_camera1, gt2['pts3d']) for gt2 in gt2s]  # list of B,H,W,3

        c2ws = [gt1['camera_pose']] + [gt2['camera_pose'] for gt2 in gt2s] # [n_v, B, 3, 3]
        c2ws = [in_camera1 @ c2w for c2w in c2ws]

        valid1 = gt1['valid_mask'].clone()
        valid2s = [gt2['valid_mask'].clone() for gt2 in gt2s]  # list of B,H,W

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2s = [gt_pts2.norm(dim=-1) for gt_pts2 in gt_pts2s]  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2s = [valid2 & (dis2 <= dist_clip) for (valid2, dis2) in zip(valid2s, dis2s)]  # (B, H, W)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2s = [get_pred_pts3d(gt2, pred2, use_pose=True) for (gt2, pred2) in zip(gt2s, pred2s)]  # (B, H, W, 3)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2s, pr_norm_factor = normalize_pointclouds(pr_pts1, pr_pts2s, self.norm_mode, valid1, valid2s, return_norm_factor = True)
            pr_norm_factor = pr_norm_factor.detach()
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2s, gt_norm_factor = normalize_pointclouds(gt_pts1, gt_pts2s, self.norm_mode, valid1, valid2s, return_norm_factor = True)
        while gt_norm_factor.ndim < c2ws[0].ndim: # -> (B, 1, 1)
            gt_norm_factor.unsqueeze_(-1)
            pr_norm_factor.unsqueeze_(-1)
        for c2w in c2ws:
            # print('c2w shape', c2w.shape, gt_norm_factor.shape) # [4,4,4]
            c2w[:,:3,3:] = c2w[:,:3,3:] / gt_norm_factor
        if 'scale' in pred1.keys():
            while gt_norm_factor.ndim < pred1['scale'].ndim: # -> (B, 1, 1)
                gt_norm_factor.unsqueeze_(-1)
                pr_norm_factor.unsqueeze_(-1)
            for pred in [pred1] + pred2s:
                if self.scale_scaled:
                    pred['scale'][:] = pred['scale'][:] / pr_norm_factor
        extra_info = {'monitering': {}}
        if log:
            with torch.no_grad():
                nv = len(pred2s) + 1
                bs, h, w = gt_pts1.shape[0:3]

                gts = [gt1] + gt2s
                preds = [pred1] + pred2s
                pr_pts = [pr_pts1] + pr_pts2s
                valids = [valid1] + valid2s
                # calc the 2D pixel os a map [H, W] -> [H, W, 2]
                y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                pixel_coords = torch.stack([x_coords, y_coords], dim=-1) # [H, W, 2]
                pixel_coords = pixel_coords.to(gt_pts1.device).repeat(gt_pts1.shape[0], 1, 1, 1).float() # [B, h, w, 3]
                
                conf = preds[0]['conf'].reshape(bs, -1) # [bs, H * W]
                conf_sorted = conf.sort()[0] # [bs, h * w]
                conf_thres = conf_sorted[:, int(conf.shape[1] * 0.03)]
                valid1 = (conf >= conf_thres[:, None]) # & valids[0].reshape(bs, -1)
                valid1 = valid1.reshape(bs, h, w)
                intrinsics = []
                pts3d = preds[0]['pts3d'] # [bs, H, W, 3]
                    
                for i in range(bs):
                    focal_i = estimate_focal_knowing_depth(pts3d[i:i+1], valid1[i:i+1])
                    intrinsics_i = torch.eye(3, device=pts3d.device)
                    intrinsics_i[0, 0] = focal_i
                    intrinsics_i[1, 1] = focal_i
                    intrinsics_i[0, 2] = w / 2
                    intrinsics_i[1, 2] = h / 2
                    intrinsics.append(intrinsics_i)
                intrinsics = torch.stack(intrinsics, dim=0) # [bs, 3, 3]
                # if "intrinsics_pred" in preds[0].keys():
                #     intrinsics_list = preds[0]["intrinsics_pred"]
                for (gt, pr_pt, pred, valid) in zip(gts, pr_pts, preds, valids):
                    
                    gt_intrinsics = gt['camera_intrinsics'][:,:3,:3] # [B, 3, 3]
                    # intrinsics = intrinsics_list
                    # print(intrinsics[0], gt_intrinsics)
                    # fbvscode.set_trace()
                    if 'c2ws_pred' not in pred.keys():
                        c2ws_pred = calibrate_camera_pnpransac(pr_pt.flatten(1,2), pixel_coords.flatten(1,2), valid.flatten(1,2), intrinsics)
                        pred['c2ws_pred'] = c2ws_pred
                    else:
                        intrinsics = preds[0]['intrinsics_pred']
                        c2ws_pred = calibrate_camera_pnpransac(pr_pt.flatten(1,2), pixel_coords.flatten(1,2), valid.flatten(1,2), intrinsics)
                        pred['c2ws_pred'] = c2ws_pred
                        # print('c2ws already predicted in GO!')


                # for (gt, pr_pt, pred, valid) in zip(gts, pr_pts, preds, valids):
                                    
                #     gt_intrinsics = gt['camera_intrinsics'][:,:3,:3] # [B, 3, 3]
                #     # intrinsics = intrinsics_list
                #     # print(intrinsics[0], gt_intrinsics)
                #     # fbvscode.set_trace()
                #     c2ws_pred = calibrate_camera_pnpransac(pr_pt.flatten(1,2), pixel_coords.flatten(1,2), valid.flatten(1,2), gt_intrinsics)
                #     pred['c2ws_pred_'] = c2ws_pred
    
                # theta_Rs, theta_ts = [], []
                # for i in range(nv):
                #     for j in range(i + 1, nv):
                #         c2w_gt1, c2w_gt2 = c2ws[i], c2ws[j]
                #         c2w_pred1, c2w_pred2 = preds[i]['c2ws_pred'], preds[j]['c2ws_pred']
                #         theta_R, theta_t = calculate_RRA_RTA(c2w_pred1, c2w_pred2, c2w_gt1, c2w_gt2)
                #         theta_Rs.append(theta_R)
                #         theta_ts.append(theta_t)
                # theta_Rs = torch.stack(theta_Rs, dim=1) # [bs, N*(N-1)/2]
                # theta_ts = torch.stack(theta_ts, dim=1) # [bs, N*(N-1)/2]
                c2ws_all = torch.stack(c2ws, dim=1)[:,:nv].flatten(0,1) # [bs, N, 4, 4]
                c2ws_pred = torch.stack([pred['c2ws_pred'] for pred in preds], dim=1)[:,:nv].flatten(0, 1)
                r, t = camera_to_rel_deg(torch.linalg.inv(c2ws_pred), torch.linalg.inv(c2ws_all), c2ws_all.device, bs)
                theta_Rs = r.reshape(bs, -1)
                theta_ts = t.reshape(bs, -1)
                
                RRA_thres = 15
                RTA_thres = 15
                mAA_thres = 30
                RRA = (theta_Rs < RRA_thres).float().mean(-1)
                RTA = (theta_ts < RTA_thres).float().mean(-1)
                mAA = torch.zeros((bs,)).to(RRA.device)
                for thres in range(1, mAA_thres + 1):
                    mAA += ((theta_Rs < thres) * (theta_ts < thres)).float().mean(-1)
                mAA /= mAA_thres
                extra_info['RRA'] = RRA
                extra_info['RTA'] = RTA
                extra_info['mAA'] = mAA
                        
        return gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, valid1, valid2s, c2ws, pr_norm_factor, extra_info

    def get_all_pts3ds(self, gt1, gt2s, pred1, pred2s, dist_clip=None, **kw):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose']) # c2w -> w2c
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2s = [geotrf(in_camera1, gt2['pts3d']) for gt2 in gt2s]  # list of B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2s = [gt2['valid_mask'].clone() for gt2 in gt2s]  # list of B,H,W
        
        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2s = [gt_pts2.norm(dim=-1) for gt_pts2 in gt_pts2s]  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2s = [valid2 & (dis2 <= dist_clip) for (valid2, dis2) in zip(valid2s, dis2s)]  # (B, H, W)
        
        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2s = [get_pred_pts3d(gt2, pred2, use_pose=True) for (gt2, pred2) in zip(gt2s, pred2s)]  # (B, H, W, 3)
        
        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2s = normalize_pointclouds(pr_pts1, pr_pts2s, self.norm_mode, valid1, valid2s)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2s = normalize_pointclouds(gt_pts1, gt_pts2s, self.norm_mode, valid1, valid2s)
        
        return gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, valid1, valid2s, {}

    # def compute_loss_mv(self, gt1, gt2s, pred1, pred2s, log, scale_range = [0.0001, 0.02], **kw):
    # def compute_loss_mv(self, gt1, gt2s, pred1, pred2s, log, scale_range = [0.00001, 0.001], **kw):
    def local_lap_loss(self, pts3d, gts3d, c2ws_all, mask_all): # [bs, h, w, 3], [bs, h, w, 3], [bs, 4, 4], [bs, h, w]
        
        cam_centers = c2ws_all[:, :3, 3] # [bs, 3]
        pts_dis = (pts3d - cam_centers[:,None,None]).norm(dim = -1) # [bs, h, w]
        gts_dis = (gts3d - cam_centers[:,None,None]).norm(dim = -1) # [bs, h, w]
        pts_dis_lap = calc_metrics.laplace(pts_dis.unsqueeze(-1)) # [bs, h, w, 1]
        gts_dis_lap = calc_metrics.laplace(gts_dis.unsqueeze(-1)) # [bs, h, w, 1]
        lap_loss = (pts_dis_lap - gts_dis_lap).abs().squeeze(-1)
        lap_loss[~mask_all] = 0
        lap_loss = lap_loss.mean()
        return lap_loss
    
    def local_loss(self, pts3d_, gts3d_, c2ws_all_, mask_all_, conf_all_, real_bs): # [bs, h, w, 3], [bs, h, w, 3], [bs, 4, 4], [bs, h, w] (bs = real_bs * n_inference)
        
        loss_type = "dis"
        loss_type = "only_T"
        # loss_type = "RT"
        
        bs = pts3d_.shape[0]

        pts3d = pts3d_.clone()
        gts3d = gts3d_.clone()
        c2ws_all = c2ws_all_.clone()
        mask_all = mask_all_.clone()
        conf_all = conf_all_.clone()
        # mask_all = mask_all_.clone()
        
        pts3d[~mask_all] = 0.
        gts3d[~mask_all] = 0.

        if "dis" in loss_type:
            mask_all_flatten = mask_all.reshape(real_bs, -1) # [real_bs, n_inference * h * w]
            pts3d = pts3d.reshape(real_bs, -1, 3) # [real_bs, n_inference * h * w]
            gts3d = gts3d.reshape(real_bs, -1, 3) # [real_bs, n_inference * h * w]
            id_list_1 = torch.randint(0, pts3d.shape[1], (real_bs, pts3d.shape[1],), device = pts3d.device) # [real_bs, M], M = n_inference * h * w
            valid_1 = mask_all_flatten.gather(1, id_list_1)
            id_list_2 = torch.randint(0, pts3d.shape[1], (real_bs, pts3d.shape[1],), device = pts3d.device)
            valid_2 = mask_all_flatten.gather(1, id_list_2)
            valid = valid_1.bool() & valid_2.bool() # [real_bs, M]
            pts3d_1 = torch.gather(pts3d, 1, id_list_1[:, :, None].repeat(1, 1, 3))
            pts3d_2 = torch.gather(pts3d, 1, id_list_2[:, :, None].repeat(1, 1, 3))
            gts3d_1 = torch.gather(gts3d, 1, id_list_1[:, :, None].repeat(1, 1, 3))
            gts3d_2 = torch.gather(gts3d, 1, id_list_2[:, :, None].repeat(1, 1, 3))
            pts3d_dis = (pts3d_1 - pts3d_2).norm(dim = -1) # [real_bs, M]
            gts3d_dis = (gts3d_1 - gts3d_2).norm(dim = -1)
            pts3d_dis = pts3d_dis * valid
            gts3d_dis = gts3d_dis * valid
            pts3d_dis_normalized = pts3d_dis / (pts3d_dis.mean(-1, keepdim = True) + 1e-8)
            gts3d_dis_normalized = gts3d_dis / (gts3d_dis.mean(-1, keepdim = True) + 1e-8)
            local_loss = (pts3d_dis_normalized - gts3d_dis_normalized).abs().mean()
            
            return local_loss

        pts3d = pts3d.reshape(bs, -1, 3) # [bs, h*w, 3]
        gts3d = gts3d.reshape(bs, -1, 3) # [bs, h
        mask_all = mask_all.reshape(bs, -1) # [bs, h*w]
        n_p = mask_all.sum(-1) # [bs, ]
        pts3d_mean = pts3d.sum(1) / (n_p[:, None] + 1e-8) # [bs, 3]
        gts3d_mean = gts3d.sum(1) / (n_p[:, None] + 1e-8) # [bs, 3]
        pts3d_center = pts3d - pts3d_mean[:, None] # [bs, h*w, 3]
        gts3d_center = gts3d - gts3d_mean[:, None] # [bs, h*w, 3]
        pts3d_norm = pts3d_center.norm(dim = -1) # [bs, h*w]
        pts3d_norm_mean = pts3d_norm.sum(-1) / (n_p + 1e-8) # [bs, ]
        gts3d_norm = gts3d_center.norm(dim = -1) # [bs, h*w]
        gts3d_norm_mean = gts3d_norm.sum(-1) / (n_p + 1e-8) # [bs, ]
        pts3d_normalized = pts3d_center / (pts3d_norm_mean[:, None, None] + 1e-8) # [bs, h*w, 3]
        gts3d_normalized = gts3d_center / (gts3d_norm_mean[:, None, None] + 1e-8) # [bs, h*w, 3]
        
        if "only_T" in loss_type:
            local_loss = (pts3d_normalized - gts3d_normalized).norm(dim = -1).mean()
        # return local_loss

        # conf_sorted = conf_all.sort()[0] # [n_inference, h * w]
        # conf_thres = conf_sorted[:, int(conf_all.shape[1] * 0.03)]
        # mask_all = mask_all & (conf_all >= conf_thres[:, None])
        #     def umeyama_alignment(self, P1, P2, mask_): # [bs, N, 3], [bs, N, 3], [bs, N] all are torch.Tensor
        """
        Return:
        R: (bs, 3, 3)
        sigma: (bs, )
        t: (bs, 3)
        """
        if "RT" in loss_type:
            R, sigma, t = umeyama_alignment(pts3d_normalized, gts3d_normalized, mask_all)
            pts3d_normalized_rot = (sigma[:,None,None] * (R @ pts3d_normalized.transpose(-1, -2)).transpose(-1, -2)) + t[:, None] # [bs, h*w, 3]
            local_loss = (pts3d_normalized_rot - gts3d_normalized).norm(dim = -1)[mask_all].mean()
            # print('local_loss min', (local_loss - local_loss_rot).min())
        return local_loss

    def compute_loss_mv(self, gt1, gt2s, pred1, pred2s, log, scale_range = [0.0001, 0.004], **kw):
        gt1, gt2s, pred1, pred2s, n_ref = self.rearrange_for_mref(gt1, gt2s, pred1, pred2s)
        gt_pts1, gt_pts2s, pred_pts1, pred_pts2s, mask1, mask2s, c2ws, pr_norm_factor, extra_info = \
            self.get_all_pts3ds_to_canonical_cam(gt1, gt2s, pred1, pred2s, log, **kw)
        monitoring = extra_info['monitering']
        # gt1: ['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'only_render', 'num_render_views', 'idx', 'true_shape', 'pts3d', 'valid_mask', 'rng']
        # pred2s[0]: ['conf', 'rgb', 'opacity', 'scale', 'rotation', 'pts3d_in_other_view']
        # pred1 : ['conf', 'rgb', 'opacity', 'scale', 'rotation', 'pts3d'] # ('pts3d', torch.Size([8, 224, 224, 3]), 3.4764482975006104, -1.5572370290756226), ('conf', torch.Size([8, 224, 224]), 41.92277908325195, 1.0040476322174072), ('rgb', torch.Size([8, 224, 224, 3]), 0.8159868121147156, -0.8702595829963684), ('opacity', torch.Size([8, 224, 224, 1]), 0.999699592590332, 7.182779518188909e-05), ('scale', torch.Size([8, 224, 224, 3]), 0.03545345366001129, -0.04244176670908928), ('rotation', torch.Size([8, 224, 224, 4]), 0.9999783039093018, -0.9999967813491821)
        num_render_views = gt2s[0].get("num_render_views", torch.zeros([0]).long())[0].item()
        gt_pts2s_inference = gt_pts2s[:-num_render_views] if num_render_views else gt_pts2s

        preds = [pred1] + pred2s
        gts = [gt1] + gt2s
        
        bs, h, w = gt_pts1.shape[:3]
        nv = len(gt2s) + 1
        n_inference = len(pred_pts2s) + 1
        mask_all = torch.stack([mask1] + mask2s, 1) # (bs, nv, h, w)
        conf_all = torch.stack([pred1['conf']] + [pred2['conf'] for pred2 in pred2s], 1) # (bs, n_inference, h, w)
        self.gs_renderer.set_view_info(height = h, width = w)
        
        loss_rgb = 0.
        loss_scale = 0.
        loss_lpips = 0.
        loss_copy_rgb = 0.
        loss_local = 0.
        loss_lap = 0.
        loss_opacity = 0.
        
        log_scale = 0.
        
        render_all = []
        render_relocated_all = []
        photometrics_all = []
        photometrics_inference_all = []
        photometrics_render_all = []
        pts3d_all = torch.stack([pred_pts1] + [pred_pts2 for pred_pts2 in pred_pts2s], 1) # [bs, n_inference, 224, 224, 3]
        gts3d_all = torch.stack([gt_pts1] + [gt_pts2 for gt_pts2 in gt_pts2s_inference], 1) # [bs, n_inference, 224, 224, 3]
        c2ws_gs_all = torch.stack([c2w for c2w in c2ws], 1) # [bs, nv, 4, 4]

        if self.local_loss_coeff:
            loss_local = self.local_loss(pts3d_all.flatten(0, 1), gts3d_all.flatten(0, 1), c2ws_gs_all[:,:n_inference].flatten(0, 1), mask_all[:,:n_inference].flatten(0, 1), conf_all.flatten(0, 1), bs)
        if self.lap_loss_coeff:
            loss_lap = self.local_lap_loss(pts3d_all.flatten(0, 1), gts3d_all.flatten(0, 1), c2ws_gs_all[:,:n_inference].flatten(0, 1), mask_all[:,:n_inference].flatten(0, 1))
        for dp_id in range(bs):
            # gt1, gt2s, pred1, pred2s = torch.load('/home/zgtang/gt_pred.pt')
            # gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, c2ws = torch.load('/home/zgtang/others.pt')
            # c2ws_gs = torch.stack([c2w[dp_id] for c2w in c2ws], 0) # [nv, 4, 4]
            c2ws_gs = c2ws_gs_all[dp_id] # [nv, 4, 4]
            gt_imgs = torch.stack([gt1['img'][dp_id]] + [gt2['img'][dp_id] for gt2 in gt2s], 0).permute(0,2,3,1) # [nv, 224, 224, 3] -1~1
            gt_masks = torch.stack([gt1['foreground_mask'][dp_id]] + [gt2['foreground_mask'][dp_id] for gt2 in gt2s], 0) # [nv, 224, 224] [0,1]
            
            intrinsics = torch.stack([gt1['camera_intrinsics'][dp_id][:3,:3]] + [gt2['camera_intrinsics'][dp_id][:3,:3] for gt2 in gt2s]).cuda() # [nv, 3, 3]
            # pts3d = torch.stack([pred_pts1[dp_id]] + [pred_pts2[dp_id] for pred_pts2 in pred_pts2s], 0) # [n_inference, 224, 224, 3]
            # gts3d = torch.stack([gt_pts1[dp_id]] + [gt_pts2[dp_id] for gt_pts2 in gt_pts2s_inference], 0) # [n_inference, 224, 224, 3]
            pts3d = pts3d_all[dp_id]
            gts3d = gts3d_all[dp_id]
            if self.use_gt_pcd:
                pts3d = pts3d * 0 + gts3d.detach()

            pts3d_gs = pts3d.reshape(-1, 3)
            # rgb_gs = torch.stack([pred1['rgb'][dp_id]] + [pred2['rgb'][dp_id] for pred2 in pred2s], 0).flatten(0, -2)
            rot_gs = torch.stack([pred1['rotation'][dp_id]] + [pred2['rotation'][dp_id] for pred2 in pred2s], 0).reshape(-1, 4)
            scale_gs = torch.stack([pred1['scale'][dp_id]] + [pred2['scale'][dp_id] for pred2 in pred2s], 0).reshape(-1, 3)

            scale_clip_above = torch.clip(scale_gs - scale_range[1], min = 0)
            scale_clip_below = torch.clip(scale_range[0] - scale_gs, min = 0)
            scale_clip_loss = torch.square(scale_clip_above + scale_clip_below).mean()            
            scale_gs = torch.clip(scale_gs, scale_range[0], scale_range[1])

            opacity_gs = torch.stack([pred1['opacity'][dp_id]] + [pred2['opacity'][dp_id] for pred2 in pred2s], 0).reshape(-1)
            # print('debug middle', c2ws_gs.shape, rgb_gs.shape, pts3d_gs.shape, opacity_gs.shape, scale_gs.shape, rot_gs.shape, intrinsics.shape) # debug middle torch.Size([4, 4, 4]) torch.Size([200704, 12]) torch.Size([200704, 3]) torch.Size([200704]) torch.Size([200704, 3]) torch.Size([200704, 4]) torch.Size([4, 3, 3])
            
            if self.rgb_coeff or dp_id == 0:
                # sh_base = rgb_gs.shape[-1] // 3
                sh_base = 1
                SH = False if (self.use_img_rgb and sh_base == 1) else True
                if self.use_img_rgb:
                    if sh_base == 1:
                        # rgb_gs = (rgb_gs[:] * 0).mean() + gt_imgs[:-num_render_views] if num_render_views else gt_imgs
                        rgb_gs = gt_imgs[:-num_render_views] if num_render_views else gt_imgs
                        rgb_gs = rgb_gs.reshape(-1, 3)
                    else:
                        sh_degree = int(np.sqrt(sh_base)) - 1
                        pts3d_gs_copy = pts3d.reshape(n_inference, -1, 3)
                        rgb_gs_copy = rgb_gs.reshape(n_inference, -1, sh_base, 3)
                        rgb_copy = self.gs_renderer.calc_color_from_sh(pts3d_gs_copy, c2ws_gs[:n_inference], rgb_gs_copy, sh_degree)
                        # def calc_color_from_sh(self, pcds, c2ws, sh, sh_degree): # pcds: [nv, N, 3], c2ws: [nv, 4, 4], sh: [nv, N, K, 3] -> colors: [nv, N, 3] -1~1
                        l_rgb_copy = self.criterion(rgb_copy.reshape(-1, 3), gt_imgs[:n_inference].reshape(-1, 3))
                        loss_copy_rgb = loss_copy_rgb + l_rgb_copy
                if self.cam_relocation:
                    valid_mask = mask_all[dp_id][:n_inference].reshape(n_inference, -1) # [n_inference, h * w]
                    conf = conf_all[dp_id].reshape(n_inference, -1) # [n_inference, h * w]
                    conf_sorted = conf.sort()[0] # [n_inference, h * w]
                    conf_thres = conf_sorted[:, int(conf.shape[1] * 0.03)]
                    valid_mask = valid_mask & (conf >= conf_thres[:, None])
                    R, sigma, t = umeyama_alignment(pts3d.reshape(n_inference, -1, 3), gts3d.reshape(n_inference, -1, 3), valid_mask)
                    Rt = torch.eye(4).to(R.device).repeat(n_inference, 1, 1)
                    Rt[:, :3, :3] = R
                    Rt[:, :3, 3] = t
                    # c2ws_gs_relocated = c2ws_gs[:n_inference] @ Rt
                    # rgb_render_relocated = [
                    #     self.gs_renderer(torch.linalg.inv(c2ws_gs_relocated[i:i+1]), intrinsics[i:i+1], sigma[i] * pts3d_gs, rgb_gs, opacity_gs, sigma[i] * scale_gs, rot_gs, eps2d=0.1, SH = SH)['rgb'] # [1, h, w, 3]
                    #     for i in range(n_inference)
                    # ]
                    np_per_view = pts3d_gs.shape[0] // n_inference
                    rgb_render_relocated = [
                        self.gs_renderer(torch.linalg.inv(c2ws_gs[i:i+1]), intrinsics[i:i+1], pts3d_gs[i * np_per_view: (i + 1) * np_per_view], rgb_gs[i * np_per_view: (i + 1) * np_per_view], opacity_gs[i * np_per_view: (i + 1) * np_per_view], scale_gs[i * np_per_view: (i + 1) * np_per_view], rot_gs[i * np_per_view: (i + 1) * np_per_view], eps2d=0.1, SH = SH)['rgb'] # [1, h, w, 3]
                        for i in range(n_inference)
                    ]
                    # print('scale and Rt', sigma, '\n', Rt)
                    rgb_render_relocated = torch.cat(rgb_render_relocated, 0) # [n_inference, h, w, 3]
                    render_relocated_all.append(rgb_render_relocated.detach().cpu())
                # print('gs debug', c2ws_gs.shape, pts3d_gs.shape, rgb_gs.shape, opacity_gs.shape, scale_gs.shape, rot_gs.shape, intrinsics.shape)
                res = self.gs_renderer(torch.linalg.inv(c2ws_gs), intrinsics, pts3d_gs, rgb_gs, opacity_gs, scale_gs, rot_gs, eps2d=0.1, SH = SH)
                # res = self.gs_renderer(torch.linalg.inv(c2ws_gs), intrinsics, pts3d_gs, rgb_gs, torch.ones_like(opacity_gs), torch.ones_like(scale_gs) * 1e-3, rot_gs, eps2d=0.1, SH = SH)
                rgb_render = res['rgb'] # [nv, 224, 224, 3]
                opacity_render = res['mask'][...,0] # [nv, 224, 224]
                
                if '0' in str(rgb_render.device) and dp_id == 0:
                    for vi in range(len(rgb_render)):
                        pred = (rgb_render[vi]+1)/2
                        gt = (gt_imgs[vi]+1)/2
                        mask = opacity_render[vi]
                        cv2.imwrite(f'./debug/{vi}_gt.png', gt.detach().cpu().numpy()[:,:,::-1]*255)
                        cv2.imwrite(f'./debug/{vi}_pred.png', pred.detach().cpu().numpy()[:,:,::-1]*255)
                        cv2.imwrite(f'./debug/{vi}_mask.png', mask.detach().cpu().numpy()*255)
                # pdb.set_trace()
            else:
                rgb_render = torch.zeros_like(gt_imgs)
            photometric_results = [calc_metrics.calc_metrics(gt_imgs[i].permute(2,0,1), rgb_render[i].permute(2,0,1), log) for i in range(nv)]
            photometrics_all.append(combine_dict(photometric_results))
            photometrics_inference_all.append(combine_dict(photometric_results[:-num_render_views] if num_render_views else photometric_results))
            photometrics_render_all.append(combine_dict(photometric_results[-num_render_views:]))
            
            log_scale = log_scale + scale_gs.mean()
            
            ls = self.criterion(rgb_render[mask_all[dp_id]], gt_imgs[mask_all[dp_id]]) # in criterion (L21), mean is calculated
            mask_loss = self.criterion(opacity_render.reshape(-1,1), gt_masks.reshape(-1,1).to(opacity_render.device))
            if self.render_included:
                render_all.append(rgb_render.detach().cpu())
            loss_rgb = loss_rgb + ls
            loss_scale = loss_scale + scale_clip_loss
            if self.lpips_coeff > 0:
                loss_lpips = loss_lpips + calc_metrics.calc_lpips(gt_imgs.permute(0,3,1,2), rgb_render.permute(0,3,1,2))
                        
        loss_rgb = loss_rgb / bs
        loss_opacity = loss_opacity + mask_loss
        loss_scale = loss_scale / bs
        loss_lpips = loss_lpips / bs
        loss_copy_rgb = loss_copy_rgb / bs

        log_scale = log_scale / bs

        loss = loss_opacity + self.rgb_coeff * loss_rgb + 1.0 * loss_scale + self.lpips_coeff * loss_lpips + self.copy_rgb_coeff * loss_copy_rgb + self.local_loss_coeff * loss_local + self.lap_loss_coeff * loss_lap

        self_name = type(self).__name__
        details = {}
        if log:
            photometrics_all_ = combine_dict(photometrics_all)
            photometrics_inference_all_ = combine_dict(photometrics_inference_all)
            photometrics_render_all_ = combine_dict(photometrics_render_all)
            photometrics_all_list = combine_dict(photometrics_all, make_list=True)
            photometrics_inference_all_list = combine_dict(photometrics_inference_all, make_list=True)
            photometrics_render_all_list = combine_dict(photometrics_render_all, make_list=True)
            details[self_name+'_gs_rgb'] = float(loss_rgb)
            details[self_name+'_gs_mask'] = float(loss_opacity)
            details[self_name+'_gs_copy_rgb'] = float(loss_copy_rgb)
            details[self_name+'_gs_scale_clip'] = float(loss_scale)
            details[self_name+'_gs_scale'] = float(log_scale)
            details[self_name+'_gs_loss_all'] = float(loss)
            details[self_name+'_local_loss'] = float(loss_local)
            details[self_name+'_lap_loss'] = float(loss_lap)
            details[self_name+'_RRA'] = extra_info['RRA'].mean().item()
            details[self_name+'_RTA'] = extra_info['RTA'].mean().item()
            details[self_name+'_mAA'] = extra_info['mAA'].mean().item()
            details[self_name+'_RRA_list'] = extra_info['RRA'].detach().cpu().tolist()
            details[self_name+'_RTA_list'] = extra_info['RTA'].detach().cpu().tolist()
            details[self_name+'_mAA_list'] = extra_info['mAA'].detach().cpu().tolist()
            
            for k in photometrics_all_.keys():
                details[self_name+f'_gs_all_{k}'] = float(photometrics_all_[k])
                details[self_name+f'_gs_inference_{k}'] = float(photometrics_inference_all_[k])
                details[self_name+f'_gs_render_{k}'] = float(photometrics_render_all_[k])
                details[self_name+f'_gs_all_{k}_list'] = photometrics_all_list[k]
                details[self_name+f'_gs_inference_{k}_list'] = photometrics_inference_all_list[k]
                details[self_name+f'_gs_render_{k}_list'] = photometrics_render_all_list[k]
        if self.render_included:
            details['render_all'] = render_all
            if self.cam_relocation:
                details['render_relocated_all'] = render_relocated_all
        return loss, (details | monitoring)
