import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import rotate, zoom, affine_transform, gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.measure as measure

# --- CONFIGURATION ---
FONT_PATH = 'KantumruyPro-Regular.ttf'
VOXEL_SIZE = 64  
IMG_SIZE = 64    
EXTRUSION_DEPTH = 12  
CHARACTER = '១'

# --- HELPER FUNCTIONS  ---
def rasterize_char_smooth(char, font, size):
    """High-quality font with anti-aliasing (same size)"""
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
    
    # Convert + ANTI-ALIASING
    img_array = np.array(image).astype(np.float32) / 255.0
    return gaussian_filter(img_array, sigma=0.5)  # Light smoothing

def extrude_to_3d_smooth(img_2d, depth, voxel_size):
    """Smooth extrusion with gradient falloff"""
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    start_depth = (voxel_size - depth) // 2
    
    for i in range(depth):
        z_slice = start_depth + i
        if 0 <= z_slice < voxel_size:
            # GRADIENT EXTRUSION: smoother edges
            fade_factor = 1.0 - (abs(i - depth/2) / (depth/2)) * 0.3
            grid[:, :, z_slice] = img_2d * fade_factor
    
    # 3D GAUSSIAN SMOOTHING 
    return gaussian_filter(grid, sigma=1.0)

# --- SMOOTH AUGMENTATIONS (ORDER=3 INTERPOLATION) ---
def apply_rotation(voxel_grid):
    angle_y = 25
    print(f"Applying Smooth Rotation: {angle_y}°")
    return rotate(voxel_grid, angle_y, axes=(0, 2), reshape=False, 
                  mode='constant', cval=0, order=3)  # CUBIC INTERPOLATION

def apply_scaling(voxel_grid):
    scale_factor = 1.2
    print(f"Applying Smooth Scaling: {scale_factor}x")
    zoomed = zoom(voxel_grid, scale_factor, mode='constant', cval=0, order=3)
    
    (zx, zy, zz) = zoomed.shape
    (vx, vy, vz) = voxel_grid.shape
    x_start = max(0, (zx - vx) // 2)
    y_start = max(0, (zy - vy) // 2)
    z_start = max(0, (zz - vz) // 2)
    return zoomed[x_start:x_start+vx, y_start:y_start+vy, z_start:z_start+vz]

def apply_shearing(voxel_grid):
    shear_val = 0.4
    print(f"Applying Smooth Shearing: {shear_val}")
    shear_matrix = np.array([[1, shear_val, 0], [0, 1, 0], [0, 0, 1]])
    return affine_transform(voxel_grid, shear_matrix, mode='constant', 
                           cval=0, order=3)

def apply_all_augmentations(voxel_grid):
    angle_y = np.random.uniform(-15, 15)
    voxel_grid = rotate(voxel_grid, angle_y, axes=(0, 2), reshape=False, 
                       mode='constant', cval=0, order=3)
    
    shear_val = np.random.uniform(-0.2, 0.2)
    shear_matrix = np.array([[1, shear_val, 0], [0, 1, 0], [0, 0, 1]])
    voxel_grid = affine_transform(voxel_grid, shear_matrix, mode='constant', 
                                 cval=0, order=3)
    return voxel_grid

def plot_smooth_voxels(ax, grid, color='cyan', title=''):
    """Render smooth voxels using density thresholding"""
    # Threshold for visibility
    visible = grid > 0.1
    
    # Use alpha blending for smoother appearance
    alphas = np.clip(grid[visible] * 2, 0, 1)  # Density → transparency
    
    ax.voxels(visible, facecolors=color, edgecolor='none', 
              alpha=alphas.mean() if alphas.size > 0 else 0.7)
    ax.set_title(title)

def plot_smooth_surface(ax, grid, color='cyan', title=''):
    """MARCHING CUBES = TRUE SMOOTH SURFACES"""
    # Extract isosurface (level=0.1)
    verts, faces, _, _ = measure.marching_cubes(grid, level=0.1)
    
    # Plot smooth mesh
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                    color=color, alpha=0.8, shade=True)
    ax.set_title(title)

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(FONT_PATH):
        print(f"Font file not found at '{FONT_PATH}'. Please update the path.")
        exit()

    # SMOOTH FONT RENDERING
    font = ImageFont.truetype(FONT_PATH, int(IMG_SIZE * 1.1))
    base_2d = rasterize_char_smooth(CHARACTER, font, IMG_SIZE)
    base_3d = extrude_to_3d_smooth(base_2d, EXTRUSION_DEPTH, VOXEL_SIZE)

    # Apply smooth augmentations
    rotated_grid = apply_rotation(base_3d.copy())
    scaled_grid = apply_scaling(base_3d.copy())
    sheared_grid = apply_shearing(base_3d.copy())
    all_augmentations_grid = apply_all_augmentations(base_3d.copy())

    # Plot ALL SMOOTH
    fig = plt.figure(figsize=(15, 10))
    grids = [base_3d, rotated_grid, scaled_grid, sheared_grid, all_augmentations_grid]
    titles = ['Original (Smooth)', '1. Rotation', '2. Scaling', '3. Shearing', '4. All Augmentations']
    colors = ['cyan', 'cyan', 'cyan', 'cyan', 'magenta']
    positions = [231, 232, 233, 234, 235]

    for i in range(len(grids)):
        ax = fig.add_subplot(positions[i], projection='3d')
        plot_smooth_surface(ax, grids[i], colors[i], titles[i])

    plt.tight_layout()
    plt.show()
