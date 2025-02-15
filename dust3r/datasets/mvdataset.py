import sys, os
import os.path as osp
import json
import itertools
from collections import deque
import imageio
from copy import deepcopy

import cv2
import numpy as np
import random
import h5py

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *
    
import os
from scipy.ndimage import label
import PIL.Image

def read_transparent_png(filename, white_bg=False, keep_largest_component=True):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    if white_bg:
        background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
    else:
        background_image = np.zeros_like(rgb_channels, dtype=np.uint8)

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)
    alpha_factor[alpha_factor<0.5] = 0
    alpha_factor[alpha_factor>=0.5] = 1
    
    if keep_largest_component:
        mask = alpha_factor[...,0]
        labeled_mask, _ = label(mask)
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0
        largest_component_label = component_sizes.argmax()
        mask = (labeled_mask == largest_component_label)
        mask = mask.astype(np.float32)
        alpha_factor = mask[...,None]

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    bg = background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + bg
    # final_image = np.ascontiguousarray((final_image)[:,:,::-1]).astype(np.float32)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return final_image, alpha_factor[...,0]


def recenter(image, mask, bg_color=0, size=None, border_ratio = 0.2):
    """ recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """
    
    H, W, C = image.shape
    if size is None:
        size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)*255*bg_color
    else:
        result = np.zeros((size, size, C), dtype=np.float32)
        result[:,:,:3] = bg_color
            
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    
    return result, x_min, y_min, h2/(x_max-x_min), w2/(y_max-y_min), x2_min, y2_min
    
class MVDataset(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, from_tar = False, random_order = False, random_render_order = False, debug = False, *args, ROOT, n_test=1000, num_render_views = 0, n_vis_test = 8, n_vis_train = 8, split_thres = 0.9, render_start = None, tb_name = None, ref_all = False, n_ref = 1, random_nv_nr = None, dps_name = 'dps.h5', n_all = None, single_id = None, reverse = False, **kwargs):
        self.ROOT = ROOT
        self.num_render_views = num_render_views
        self.from_tar = from_tar
        self.random_order = random_order
        self.random_render_order = random_render_order
        super().__init__(*args, **kwargs) # self.num_views, split set inside
        if "test" in dps_name:
            self.test = True
        else:
            self.test = False
        self.num_inference_views = self.num_views - self.num_render_views
        self.num_vis_test = n_vis_test
        self.num_vis_train = n_vis_train
        if render_start is None:
            render_start = self.num_inference_views
        self.render_start = render_start
        self.tb_name = tb_name if tb_name is not None else self.split
        self.ref_all = ref_all
        self.n_ref = n_ref
        if random_nv_nr is None:
            random_nv_nr = [[self.num_views, self.num_render_views]]
        self.random_nv_nr = random_nv_nr
        self.dps_name = dps_name
        self.single_id = single_id

        # load all scenes
        self.data_name = osp.basename(self.ROOT)
        # self.h5f_path = g_pathmgr.get_local_path(osp.join(self.ROOT, self.dps_name))
        # with h5py.File(self.h5f_path, 'r') as h5f:
        #     self.dps = [i for i in range(len(h5f['json_strs']))]
        
        # if self.split != "all":
        #     split_ind = int(len(self.dps) * split_thres)
        #     test_id_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_test)]
        #     train_id_list = np.setdiff1d(np.arange(split_ind), test_id_list)
        #     train_test_id_list = [x + 1 for x in range(0, split_ind, split_ind // n_test)]
        #     vis_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_vis_test)][-n_vis_test:] + [train_test_id_list[x] for x in range(0, len(train_test_id_list), len(train_test_id_list) // n_vis_train)][-n_vis_train:]
        #     # vis_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_vis_test)][-n_vis_test:] + [train_test_id_list[x] for x in range(0, len(train_test_id_list), len(train_test_id_list) // n_vis_train)][-n_vis_train:]
        #     print('vis list', len(vis_list), vis_list)
        
        # if self.split == "all":
        #     if n_all is not None:
        #         self.dps = [self.dps[int(id / n_all * len(self.dps))] for id in range(n_all)]
        #     pass
        # elif self.split == "train":
        #     self.dps = [self.dps[x] for x in train_id_list]
        # elif self.split == "train_test":
        #     self.dps = [self.dps[x] for x in train_test_id_list]
        # elif self.split == "vis":
        #     self.dps = [self.dps[x] for x in vis_list]
        # elif self.split == "test":
        #     self.dps = [self.dps[x] for x in test_id_list]
        # if self.single_id is not None:
        #     self.dps = [self.dps[self.single_id]]
        # if reverse:
        #     self.dps = list(reversed(self.dps))
        # self.dps = ['0000']
        test_subjects = {'0003', '0132', '0019', '0516', '0126', '0001', '0114', '0000', '0054', '0432', '0522', '0078', '0066', '0192', '0330', '0498', '0264', '0306', '0108', '0006', '0354', '0302', '0036', '0072', '0240', '0324', '0156', '0138', '0042', '0258', '0402', '0450', '0084', '0360', '0312', '0096', '0456', '0486', '0300', '0012', '0510', '0523', '0216', '0480', '0474', '0366', '0228', '0150', '0390', '0009', '0246', '0060', '0414', '0426', '0348', '0068', '0002', '0276', '0018', '0090', '0048', '0270', '0518', '0168', '0384', '0144', '0444', '0222', '0492', '0186', '0396', '0438', '0180', '0294', '0204', '0252', '0462', '0342', '0024', '0210', '0102', '0378', '0504', '0372', '0174', '0468', '0030', '0234', '0318', '0023', '0198', '0162', '0282', '0420', '0408', '0021', '0336', '0288', '0120', '0025'}

        self.dps = []
        for i in range(2445):
            subject_id = str(i).zfill(4)
            if self.split == 'train':
                if subject_id not in test_subjects:
                    self.dps.append(subject_id)
                # self.dps = ['0000' for _ in range(400)]
            else:
                if subject_id in test_subjects:
                    self.dps.append(subject_id)
                self.dps = self.dps[:5]
        
    
    def __len__(self):
        
        print('len in dataset', len(self.dps), self.split)
        # if self.ref_all:
        #     return len(self.dps) * self.num_inference_views
        return len(self.dps)

    def _get_views(self, idx, resolution, rng, from_tar = False, ref_view_id = None):
        subject = self.dps[idx]
        random_nv_nr = random.choice(self.random_nv_nr)
        # self.num_views = random_nv_nr[0]
        # self.num_render_views = random_nv_nr[1]
        # self.num_inference_views = self.num_views - self.num_render_views
        if ref_view_id is None:
            # by default, use view 0 as ref, otherwise, use specified ref view
            ref_view_id = 0
        C_avg = np.array(0.).astype(np.float32)
        # if self.ref_all:
        #     ref_view_id = idx % self.num_inference_views
        #     idx = idx // self.num_inference_views
        # with h5py.File(self.h5f_path, 'r') as h5f:
        #     json_str = h5f['json_strs'][self.dps[idx]]
        #     data_dict = json.loads(json_str)
        #     C = data_dict['C'] if 'C' in data_dict.keys() else None
        #     if C is None:
        #         C_avg = np.array(0.).astype(np.float32)
        #     else:
        #         C = np.array(C)
        #         C = C[:self.num_inference_views,:self.num_inference_views]
        #         C_avg = C.mean()
        #         if C[0][0] > 0.9:
        #             C_avg -= 1 / self.num_inference_views
        #         C_avg = np.array(C_avg).astype(np.float32)
            
        #     rgb_list = data_dict['rgb_list']
        #     depth_list = data_dict['depth_list']
        #     pose_raw_list, pose_list = [], []
        #     if "pose_raw_list" in data_dict.keys():
        #         pose_raw_list = data_dict['pose_raw_list']
        #     else:
        #         pose_list = data_dict['pose_list']
        #     intrinsic_raw, intrinsic_list = None, []
        #     if "intrinsic_raw" in data_dict.keys():
        #         intrinsic_raw = data_dict['intrinsic_raw']
        #     else:
        #         intrinsic_list = data_dict['intrinsic_list']

        #     if len(rgb_list) < self.num_views and 'nv' not in data_dict.keys(): # not implemented
        #         n_repeat = (self.num_views - 1) // len(rgb_list) + 1
        #         rgb_list = (rgb_list * n_repeat)[:self.num_views]
        #         depth_list = (depth_list * n_repeat)[:self.num_views]
        #         pose_raw_list = (pose_raw_list * n_repeat)[:self.num_views]
        #         pose_list = (pose_list * n_repeat)[:self.num_views]
        #         intrinsic_list = (intrinsic_list * n_repeat)[:self.num_views]
        
        # num_tuple = len(rgb_list)
        # if 'nv' in data_dict.keys():
        #     num_tuple = data_dict['nv']
        #     rgb_all = imageio.imread(g_pathmgr.get_local_path(rgb_list[0])).astype(np.uint8)
        #     depth_all = imageio.imread(g_pathmgr.get_local_path(depth_list[0])).astype(np.float32) / 1000
        #     rgb_all = np.split(rgb_all, num_tuple, axis=1)
        #     depth_all = np.split(depth_all, num_tuple, axis=1)
        #     if len(rgb_all) < self.num_views:
        #         n_repeat = (self.num_views - 1) // len(rgb_all) + 1
        #         pose_raw_list = (pose_raw_list * n_repeat)[:self.num_views]
        #         pose_list = (pose_list * n_repeat)[:self.num_views]
        #         intrinsic_list = (intrinsic_list * n_repeat)[:self.num_views]
        #         rgb_all = (rgb_all * n_repeat)[:self.num_views]
        #         depth_all = (depth_all * n_repeat)[:self.num_views]
        # if self.random_render_order:
        #     # randomly choose num_render_views from num_views w/o replacement:
        #     render_set = np.random.choice(range(self.num_views), size = self.num_render_views, replace = False)
        # else:
        render_set = [self.render_start + i for i in range(self.num_render_views)] # test: 20 views [4,23] # train: 6 views
        
        inference_set = [] # 4 views [0,3]
        for i in range(self.num_views):
            if i not in render_set:
                inference_set.append(i)
            if len(inference_set) == self.num_inference_views:
                break

        # rgb_list, depth_list, pose_list, intrinsic_list = change_to_sr([rgb_list, depth_list, pose_list, intrinsic_list])
        views = []
        
        if self.split == 'train':
            input_views = np.random.permutation(np.arange(0,24))[:1].tolist() + np.random.permutation(np.arange(24,48))[:1].tolist() + np.random.permutation(np.arange(48,72))[:1].tolist() + np.random.permutation(np.arange(72,96))[:1].tolist()
            input_views = np.random.permutation(input_views).tolist()
            other_views = [v for v in np.random.permutation(96).tolist() if v not in input_views][:2]
            all_views = input_views + other_views
        else:
            input_views = [0,24,48,72]
            all_views = input_views + [i for i in np.arange(0, 97, 10).tolist() if i not in input_views][:10]
        # print(len(all_views), len(inference_set), len(render_set))
        # assert len(all_views) == len(inference_set) + len(render_set), f'len(all_views) {all_views}, len(inference_set) {inference_set}, len(render_set), {render_set}'
        with open(f"/fs/gamma-datasets/MannequinChallenge/training_examples_cam_aug_depths/{subject}/depth_max.json", "r") as json_file:
            depth_max_dict = json.load(json_file)
        input_metadata = np.load(os.path.join(f'/fs/gamma-datasets/MannequinChallenge/training_examples_cam_aug/{subject}/meta.pkl'), allow_pickle=True)
        for i in inference_set + render_set:
            
            # if 'nv' in data_dict.keys():
            #     rgb = rgb_all[i]
            #     depth = depth_all[i]
            # else:
            #     rgb = imageio.imread(g_pathmgr.get_local_path(rgb_list[i])).astype(np.uint8) # / 256
            #     if depth_list[i].endswith('.npy'):
            #         depth = np.load(g_pathmgr.get_local_path(depth_list[i]))
            #     else:
            #         depth = imageio.imread(g_pathmgr.get_local_path(depth_list[i])).astype(np.float32) / 1000
            
            # if intrinsic_raw is not None:
            #     intrinsic_ = np.array(intrinsic_raw).astype(np.float32)
            #     intrinsic = np.zeros((4,4)).astype(np.float32)
            #     intrinsic[:3,:3] = intrinsic_[:3,:3]
            # else:
            #     intrinsic = np.loadtxt(g_pathmgr.get_local_path(intrinsic_list[i])).astype(np.float32)
            # if len(pose_raw_list):
            #     camera_pose = np.array(pose_raw_list[i]).astype(np.float32)
            # else:
            #     camera_pose = np.loadtxt(g_pathmgr.get_local_path(pose_list[i])).astype(np.float32)
            # rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

            # rgb, depth, intrinsic = self._crop_resize_if_necessary(
            #     rgb, depth, intrinsic, resolution, rng=rng, info="[Empty]")
            # if self.save_results:
            #     if i < len(rgb_list):
            #         if 'gibson' in rgb_list[i]:
            #             scene_name = rgb_list[i].split('/')[-2]
            #         elif 'hm3d' in rgb_list[i]:
            #             scene_name = rgb_list[i].split('/')[-3]
            #         elif 'mp3d' in rgb_list[i]:
            #             scene_name = rgb_list[i].split('/')[-2]
            #         else:
            #             scene_name = rgb_list[i].split('/')
            #             for x in scene_name:
            #                 if "scene" in x:
            #                     scene_name = x
            #                     break
            #     label=f"dataName_{self.tb_name}_id_{str(idx).zfill(9)}_sceneName_{scene_name}_refId_{str(ref_view_id).zfill(3)}"
            # else:
            #     label=f"{str(idx).zfill(9)}"
            im_idx = all_views[i]
            camera_pose = np.eye(4)
            camera_pose[:3,:4] = input_metadata[4][im_idx]
            camera_pose = np.linalg.inv(camera_pose)
            intrinsics = input_metadata[0].copy()
            
            image, mask = read_transparent_png(f'/fs/gamma-datasets/MannequinChallenge/training_examples_cam_aug/{subject}/{str(im_idx).zfill(3)}.png', white_bg=False, keep_largest_component=True)
            depthmap = imread_cv2(f'/fs/gamma-datasets/MannequinChallenge/training_examples_cam_aug_depths/{subject}/{im_idx}.png', cv2.IMREAD_UNCHANGED)
            depthmap = ((depthmap.astype(np.float32) / 65535) * depth_max_dict[str(im_idx)]).astype(np.float32)
            rgba = np.zeros((image.shape[0], image.shape[1], 4))
            rgba[:,:,:3] = image
            rgba[:,:,3] = mask
            
            # if 224 in resolution:
            if False:
                rgba, x1, y1, s1, s2, x2, y2 = recenter(rgba, mask, np.zeros(3).astype(np.float32)*255, size=224, border_ratio=0.05)
                depth_a = np.zeros((image.shape[0], image.shape[1], 4))
                depth_a[:,:,0] = depthmap
                depth_a[:,:,1] = depthmap
                depth_a[:,:,2] = depthmap
                depth_a[:,:,3] = mask
                depth_a, _, _, _, _, _, _ = recenter(depth_a, mask, np.zeros(3).astype(np.float32)*255, size=224, border_ratio=0.05)
                intrinsics[0][2] -= y1
                intrinsics[1][2] -= x1
                intrinsics[0] *= s2
                intrinsics[1] *= s1
                intrinsics[0][2] += y2
                intrinsics[1][2] += x2
                intrinsics[-1,-1] = 1
                
                rgb_image = rgba[:,:,:3]
                maskmap = rgba[:,:,3]
                depthmap = depth_a[:,:,0]
            else:
                rgba, x1, y1, s1, s2, x2, y2 = recenter(rgba, mask, np.zeros(3).astype(np.float32)*255, size=384, border_ratio=0.1)
                depth_a = np.zeros((image.shape[0], image.shape[1], 4))
                depth_a[:,:,0] = depthmap
                depth_a[:,:,1] = depthmap
                depth_a[:,:,2] = depthmap
                depth_a[:,:,3] = mask
                depth_a, _, _, _, _, _, _ = recenter(depth_a, mask, np.zeros(3).astype(np.float32)*255, size=384, border_ratio=0.1)
                intrinsics[0][2] -= y1
                intrinsics[1][2] -= x1
                intrinsics[0] *= s2
                intrinsics[1] *= s1
                intrinsics[0][2] += y2
                intrinsics[1][2] += x2
                intrinsics[-1,-1] = 1
                
                rgb_image = rgba[:,:,:3]
                maskmap = rgba[:,:,3]
                depthmap = depth_a[:,:,0]
                rgb_image_padded = np.zeros((384, 512, 3))
                rgb_image_padded[:,64:64+384,:] = rgb_image
                maskmap_padded = np.zeros((384, 512))
                maskmap_padded[:,64:64+384] = maskmap
                depthmap_padded = np.zeros((384, 512))
                depthmap_padded[:,64:64+384] = depthmap
                depthmap = depthmap_padded
                rgb_image = rgb_image_padded
                maskmap = maskmap_padded
                intrinsics[0][2] += 64
            
            maskmap_roundup = maskmap.copy()
            maskmap_roundup = (maskmap_roundup > 0.99)
            depthmap *= maskmap_roundup
            rgb_image *= maskmap_roundup[...,None]
            rgb_image = rgb_image.astype(np.uint8)
            rgb_image = PIL.Image.fromarray(rgb_image)
                
            views.append(dict(
                random_nv_nr=np.array(random_nv_nr),
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                foreground_mask=maskmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='thuman',
                label=str(subject),
                instance=str(int(im_idx)),
                only_render = i in render_set,
                num_render_views = random_nv_nr[1],
                n_ref = self.n_ref,
                C_avg = C_avg,
            ))
        
        if ref_view_id != 0:
            views[0], views[ref_view_id] = deepcopy(views[ref_view_id]), deepcopy(views[0])

        if self.random_order:
            random.shuffle(views)
        else:
            views[0], views[1] = deepcopy(views[1]), deepcopy(views[0])
        
        if self.num_inference_views < 12:
            if len(views) > 3:
                views[1], views[3] = deepcopy(views[3]), deepcopy(views[1])
            
            if len(views) > 6:
                views[2], views[6] = deepcopy(views[6]), deepcopy(views[2])
        else:
            
            change_id = self.num_inference_views // 4 + 1
            views[1], views[change_id] = deepcopy(views[change_id]), deepcopy(views[1])
            change_id = (self.num_inference_views * 2) // 4 + 1
            views[2], views[change_id] = deepcopy(views[change_id]), deepcopy(views[2])
            change_id = (self.num_inference_views * 3) // 4 + 1
            views[3], views[change_id] = deepcopy(views[change_id]), deepcopy(views[3])

        views_inference, views_render = [], []
        for view in views:
            if view['only_render']:
                views_render.append(view)
            else:
                views_inference.append(view)
        views = views_inference + views_render
        assert len(views) == self.num_views
        
        return views

if __name__ == "__main__":
    pass
