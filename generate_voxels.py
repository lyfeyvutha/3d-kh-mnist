import os
import shutil
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import rotate, zoom, affine_transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
FONT_PATH = 'KantumruyPro-Regular.ttf' 
DATASET_DIR = "khmer_numeral_dataset"
NUMERALS = "០១២៣៤៥៦៧៨៩"
SAMPLES_PER_NUMERAL = 200  # Create 200 variations for each numeral
IMG_SIZE = 64             # The 2D image resolution
VOXEL_SIZE = 32            # The final 3D voxel grid resolution (32x32x32)
EXTRUSION_DEPTH = 12       # How "thick" to make the numeral in 3D

def rasterize_char(char, font, size):
    """Renders a single character to a 2D numpy array."""
    # Create a grayscale image ('L') with a black background
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    # Get bounding box of the character and center it
    try:
        bbox = font.getbbox(char) # (left, top, right, bottom)
        char_width = bbox[2] - bbox[0]  # right - left
        char_height = bbox[3] - bbox[1] # bottom - top
        x_pos = (size - char_width) // 2 - bbox[0]
        y_pos = (size - char_height) // 2 - bbox[1]
    except TypeError: # A fallback for older Pillow versions
        char_width, char_height = font.getsize(char)
        x_pos = (size - char_width) / 2
        y_pos = (size - char_height) / 2
        
    # Draw the character in white
    draw.text((x_pos, y_pos), char, font=font, fill=255)
    
    # Convert to numpy array and normalize to [0, 1]
    return np.array(image) / 255.0

def extrude_to_3d(img_2d, depth, voxel_size):
    """Extrudes a 2D image into a 3D voxel grid."""
    if img_2d.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    
    # Create an empty box (3D cube)
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    # Center the extrusion
    start_depth = (voxel_size - depth) // 2 # voxel_size - depth (empty space along Z axis)
    
    # Stack the slices 'depth' times
    for i in range(depth):
        z_slice = start_depth + i
        # Safety check to ensure we don't go outside the grid
        if 0 <= z_slice < voxel_size:
            # Copy the 2D image onto one slice of the 3D grid
            grid[:, :, z_slice] = img_2d 
            
    return grid

def apply_augmentations(voxel_grid):
    """Applies random transformations to a voxel grid."""

    # Rotation
    angle_x, angle_y, angle_z = np.random.uniform(-15, 15, 3)
    # Applies rotation
    voxel_grid = rotate(voxel_grid, angle_x, axes=(1, 2), reshape=False, mode='constant', cval=0, order=0)
    voxel_grid = rotate(voxel_grid, angle_y, axes=(0, 2), reshape=False, mode='constant', cval=0, order=0)
    voxel_grid = rotate(voxel_grid, angle_z, axes=(0, 1), reshape=False, mode='constant', cval=0, order=0)

    # Scaling
    scale_factor = np.random.uniform(0.9, 1.1) # Between 90% and 110% of the original size

    # To maintain size, we zoom and then crop/pad back to original size
    zoomed = zoom(voxel_grid, scale_factor, mode='constant', cval=0, order=0)
    (zx, zy, zz) = zoomed.shape
    (vx, vy, vz) = voxel_grid.shape
    x_start = max(0, (zx - vx) // 2)
    y_start = max(0, (zy - vy) // 2)
    z_start = max(0, (zz - vz) // 2)
    
    cropped_zoomed = zoomed[
        x_start : x_start + vx,
        y_start : y_start + vy,
        z_start : z_start + vz,
    ]
    
    # Pad if necessary
    (cx, cy, cz) = cropped_zoomed.shape
    pad_x = vx - cx
    pad_y = vy - cy
    pad_z = vz - cz
    voxel_grid = np.pad(cropped_zoomed, (
        (pad_x//2, pad_x - pad_x//2), 
        (pad_y//2, pad_y - pad_y//2), 
        (pad_z//2, pad_z - pad_z//2)
    ), 'constant')
    
    # Shearing
    shear_val = np.random.uniform(-0.2, 0.2)
    shear_matrix = np.array([[1, shear_val, 0], [0, 1, 0], [0, 0, 1]])
    voxel_grid = affine_transform(voxel_grid, shear_matrix, mode='constant', cval=0, order=0)

    # Noise
    #noise = np.random.rand(*voxel_grid.shape) < 0.005 # 0.5% chance to flip a voxel
    #voxel_grid = np.logical_xor(voxel_grid > 0.5, noise).astype(np.float32)
    
    return voxel_grid

def generate_dataset():
    """Main function to generate the entire dataset."""
    if not os.path.exists(FONT_PATH):
        print(f"Font file not found at '{FONT_PATH}'. Please download it and update the path.")
        return False

    if os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' already exists. Recreating it.")
        shutil.rmtree(DATASET_DIR)
        
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Created dataset directory: '{DATASET_DIR}'")

    font = ImageFont.truetype(FONT_PATH, int(IMG_SIZE * 0.8)) # Font size is 80% of image height
    
    total_files = len(NUMERALS) * SAMPLES_PER_NUMERAL
    count = 0
    
    for i, char in enumerate(NUMERALS):
        print(f"\nGenerating data for numeral '{char}' ({i+1}/{len(NUMERALS)})...")
        
        # 1. Create the base 2D and 3D shapes
        base_2d = rasterize_char(char, font, IMG_SIZE)
        base_3d = extrude_to_3d(base_2d, EXTRUSION_DEPTH, VOXEL_SIZE)
        
        for j in range(SAMPLES_PER_NUMERAL):
            # 2. Apply augmentations
            augmented_voxel = apply_augmentations(base_3d.copy())
            
            # 3. Save the file
            filename = f"numeral_{i}_{j:04d}.npy"
            filepath = os.path.join(DATASET_DIR, filename)
            np.save(filepath, augmented_voxel)
            
            count += 1
            print(f"\r Saved {count}/{total_files}", end="")
    
    print(f"\n\n Dataset generation complete. {total_files} files created.")
    return True

def visualize_sample(dataset_dir):
    """Loads and plots one random sample from the generated dataset."""
    print("\n Visualizing a random sample...")
    try:
        files = os.listdir(dataset_dir)
        if not files:
            print("No files found in dataset directory.")
            return
        
        sample_file = np.random.choice(files)
        sample_path = os.path.join(dataset_dir, sample_file)
        
        voxel_grid = np.load(sample_path)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(voxel_grid, edgecolor='k', facecolors='cyan')
        ax.set_title(f"Sample: {sample_file}")
        plt.show()

    except Exception as e:
        print(f"Could not visualize sample. Error: {e}")

# --- Main execution ---
if __name__ == "__main__":
    if generate_dataset():
        visualize_sample(DATASET_DIR)