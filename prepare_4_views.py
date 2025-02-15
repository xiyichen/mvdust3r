import cv2
import numpy as np
from scipy.ndimage import label
import json

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

subject_id = '0000'
cam = np.load(f'/fs/gamma-datasets/MannequinChallenge/training_examples_cam_aug/{subject_id}/meta.pkl', allow_pickle=True)
d = {}
for cam_id in [0, 24, 48, 72]:
# for cam_id in range(97):
    w2c = np.eye(4)
    w2c[:3,:4] = cam[4][cam_id]
    image, mask = read_transparent_png(f'/fs/gamma-datasets/MannequinChallenge/training_examples_cam_aug/{subject_id}/{str(cam_id).zfill(3)}.png', white_bg=False, keep_largest_component=True)
    rgba = np.zeros((image.shape[0], image.shape[1], 4))
    rgba[...,:3] = image
    rgba[...,3] = mask
    rgba, x1, y1, s1, s2, x2, y2 = recenter(rgba, mask, np.zeros(3).astype(np.float32)*255, size=384, border_ratio=0.1)
    K = cam[0].copy()
    K[0][2] -= y1
    K[1][2] -= x1
    K[0] *= s2
    K[1] *= s1
    K[0][2] += y2
    K[1][2] += x2
    rgba_padded = np.zeros((384, 512, 4))
    rgba_padded[:,64:64+384,:] = rgba
    rgba_padded[...,:3] = rgba_padded[...,:3][:,:,::-1]
    rgba_padded[...,3] *= 255
    K[0][2] += 64
    d[cam_id] = {}
    d[cam_id]['K'] = K.tolist()
    d[cam_id]['w2c'] = w2c.tolist()
    cv2.imwrite(f'./test_inputs/{subject_id}_{cam_id}.png', rgba_padded)
    with open(f'./test_inputs/{subject_id}.json', 'w') as json_file:
        json.dump(d, json_file, indent=4)
    