import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
VOXEL_SIZE = 32      # The size of our 3D box (32x32x32)
EXTRUSION_DEPTH = 12 # How "thick" our final 3D shape should be
SHAPE_SIZE = 16      # The size of our initial 2D shape (16x16)

# --- The Function to Demonstrate ---
def extrude_to_3d(img_2d, depth, voxel_size):
    """Extrudes a 2D image into a 3D voxel grid."""
    if img_2d.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    
    # Create an empty 3D cube filled with zeros
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    # Calculate where to start stacking to center the object
    start_depth = (voxel_size - depth) // 2
    
    # This loop stacks the 2D image 'depth' times
    for i in range(depth):
        z_slice = start_depth + i
        # Safety check to ensure we don't go outside the grid
        if 0 <= z_slice < voxel_size:
            # Copy the 2D image onto one slice of the 3D grid
            grid[:, :, z_slice] = img_2d
            
    return grid

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Create a simple 2D shape to extrude
    # This creates a 16x16 array of zeros (a black square)
    print(f"Step 1: Creating a simple 2D shape of size {SHAPE_SIZE}x{SHAPE_SIZE}.")
    two_d_shape = np.zeros((SHAPE_SIZE, SHAPE_SIZE))

    # Make the center of the shape a smaller white square
    # This creates a white 8x8 square inside the black 16x16 square
    margin = SHAPE_SIZE // 4
    two_d_shape[margin:-margin, margin:-margin] = 1.0
    
    # We need to pad the 2D image to match the voxel grid size
    pad_amount = (VOXEL_SIZE - SHAPE_SIZE) // 2
    padded_shape = np.pad(two_d_shape, pad_width=pad_amount, mode='constant', constant_values=0)
    print(f"Shape of 2D input after padding: {padded_shape.shape}")

    # 2. Call the function to perform the extrusion
    print(f"\nStep 2: Calling extrude_to_3d with a depth of {EXTRUSION_DEPTH}.")
    three_d_object = extrude_to_3d(padded_shape, EXTRUSION_DEPTH, VOXEL_SIZE)
    print(f"Shape of the final 3D object: {three_d_object.shape}")

    # 3. Visualize the result
    print("\nStep 3: Visualizing the 2D input and 3D output...")
    
    fig = plt.figure(figsize=(12, 6))

    # Plot the 2D input shape
    ax1 = fig.add_subplot(121)
    ax1.imshow(padded_shape, cmap='gray', interpolation='nearest')
    ax1.set_title("Input: 2D Shape")

    # Plot the 3D output object
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.voxels(three_d_object, facecolors='cyan', edgecolor='k')
    ax2.set_title("Output: 3D Extruded Object")
    
    plt.show()