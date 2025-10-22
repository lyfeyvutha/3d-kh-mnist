import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Set this to the name of the folder containing .npy files.
DATASET_DIR = "khmer_numeral_dataset"

def visualize_npy_file(file_path):
    """Loads a single .npy file and displays it as a 3D voxel plot."""
    try:
        # Load the data from the specified file
        voxel_grid = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    print(f"Successfully loaded '{os.path.basename(file_path)}'. Shape: {voxel_grid.shape}")

    # Prepare a 3D canvas for plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the voxels. This command draws a cube for every non-zero value in the grid.
    ax.voxels(voxel_grid, facecolors='cyan', edgecolor='k')

    ax.set_title(f"Visualization of: {os.path.basename(file_path)}")
    plt.show()

if __name__ == "__main__":
    # --- 1. Check if the dataset directory exists ---
    if not os.path.isdir(DATASET_DIR):
        print(f"Error: The directory '{DATASET_DIR}' was not found.")
        print("Please make sure this script is in the same parent folder as your dataset.")
        exit()

    # --- 2. Find all .npy files in the directory ---
    npy_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.npy')]

    if not npy_files:
        print(f"Error: No .npy files were found in the '{DATASET_DIR}' directory.")
        exit()
    
    # --- 3. Create a user selection menu ---
    while True:
        print("\n--- Select a file to visualize ---")
        for i, filename in enumerate(npy_files):
            # The format [1] filename.npy makes it easy to read
            print(f"  [{i+1}] {filename}")
        print("------------------------------------")
        
        # --- 4. Get and validate input ---
        try:
            choice = input(f"Enter a number (1-{len(npy_files)}) or 'q' to quit: ")

            if choice.lower() == 'q':
                print("Exiting.")
                break

            choice_index = int(choice) - 1 # Convert input's 1-based number to 0-based index

            if 0 <= choice_index < len(npy_files):
                selected_file = npy_files[choice_index]
                full_path = os.path.join(DATASET_DIR, selected_file)
                visualize_npy_file(full_path)
            else:
                print("Invalid number. Please enter a number from the list.")

        except ValueError:
            print("That's not a valid number. Please try again.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break