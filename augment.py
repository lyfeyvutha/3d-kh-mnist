import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import rotate, zoom, affine_transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration (Update FONT_PATH if needed) ---
FONT_PATH = 'KantumruyPro-Regular.ttf'
VOXEL_SIZE = 64
IMG_SIZE = 64
EXTRUSION_DEPTH = 12
CHARACTER = '១'

# --- Helper Functions ---
def rasterize_char(char, font, size):
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    try:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        x_pos = (size - char_width) // 2 - bbox[0]
        y_pos = (size - char_height) // 2 - bbox[1]
    except TypeError:
        char_width, char_height = font.getsize(char)
        x_pos = (size - char_width) / 2
        y_pos = (size - char_height) / 2
    draw.text((x_pos, y_pos), char, font=font, fill=255)
    return np.array(image) / 255.0

def extrude_to_3d(img_2d, depth, voxel_size):
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    start_depth = (voxel_size - depth) // 2
    for i in range(depth):
        z_slice = start_depth + i
        if 0 <= z_slice < voxel_size:
            grid[:, :, z_slice] = img_2d
    return grid

# --- Individual Augmentation Functions ---

def apply_rotation(voxel_grid):
    angle_y = 25
    print(f"Applying Rotation: Tilting {angle_y} degrees.")
    return rotate(voxel_grid, angle_y, axes=(0, 2), reshape=False, mode='constant', cval=0, order=0)

def apply_scaling(voxel_grid):
    scale_factor = 1.2
    print(f"Applying Scaling: Zooming by a factor of {scale_factor}.")
    zoomed = zoom(voxel_grid, scale_factor, mode='constant', cval=0, order=0)
    (zx, zy, zz) = zoomed.shape
    (vx, vy, vz) = voxel_grid.shape
    x_start = max(0, (zx - vx) // 2)
    y_start = max(0, (zy - vy) // 2)
    z_start = max(0, (zz - vz) // 2)
    return zoomed[x_start:x_start+vx, y_start:y_start+vy, z_start:z_start+vz]

def apply_shearing(voxel_grid):
    shear_val = 0.4
    print(f"Applying Shearing: Slanting with a value of {shear_val}.")
    shear_matrix = np.array([[1, shear_val, 0], [0, 1, 0], [0, 0, 1]])
    return affine_transform(voxel_grid, shear_matrix, mode='constant', cval=0, order=0)

#def apply_noise(voxel_grid):
    #noise_probability = 0.015
    #noise = np.random.rand(*voxel_grid.shape) < noise_probability
    #return np.logical_xor(voxel_grid > 0.5, noise).astype(np.float32)

def apply_all_augmentations(voxel_grid):
    angle_y = np.random.uniform(-15, 15)
    voxel_grid = rotate(voxel_grid, angle_y, axes=(0, 2), reshape=False, mode='constant', cval=0, order=0)
    shear_val = np.random.uniform(-0.2, 0.2)
    shear_matrix = np.array([[1, shear_val, 0], [0, 1, 0], [0, 0, 1]])
    voxel_grid = affine_transform(voxel_grid, shear_matrix, mode='constant', cval=0, order=0)
    #noise = np.random.rand(*voxel_grid.shape) < 0.005
    #return np.logical_xor(voxel_grid > 0.5, noise).astype(np.float32)
    return voxel_grid

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(FONT_PATH):
        print(f"Font file not found at '{FONT_PATH}'. Please update the path.")
        exit()

    font = ImageFont.truetype(FONT_PATH, int(IMG_SIZE * 1.1))
    base_2d = rasterize_char(CHARACTER, font, IMG_SIZE)
    base_3d = extrude_to_3d(base_2d, EXTRUSION_DEPTH, VOXEL_SIZE)

    rotated_grid = apply_rotation(base_3d.copy())
    scaled_grid = apply_scaling(base_3d.copy())
    sheared_grid = apply_shearing(base_3d.copy())
    #noisy_grid = apply_noise(base_3d.copy())
    all_augmentations_grid = apply_all_augmentations(base_3d.copy())

    fig = plt.figure(figsize=(15, 10))
    grids = [base_3d, rotated_grid, scaled_grid, sheared_grid, all_augmentations_grid] #noisy_grid, all_augmentations_grid]
    titles = ['Original', '1. Rotation (Final)', '2. Scaling (Final)', '3. Shearing (Final)', '4. All Augmentations'] #'4. Noise Only', '5. All Augmentations']
    colors = ['cyan', 'cyan', 'cyan', 'cyan', 'magenta']
    positions = [231, 232, 233, 234, 235]

    for i in range(len(grids)):
        ax = fig.add_subplot(positions[i], projection='3d')
        ax.voxels(grids[i], facecolors=colors[i], edgecolor='k')
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()
