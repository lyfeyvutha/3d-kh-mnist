import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

# --- Configuration ---
# Make sure this font file is in the same folder as the script
FONT_PATH = 'KantumruyPro-Regular.ttf'
IMG_SIZE = 32 # The 2D image resolution
CHARACTER_TO_RASTERIZE = '1' # The Khmer numeral for "1"

# --- The Function to Illustrate ---
def rasterize_char(char, font, size):
    """Renders a single character to a 2D numpy array."""
    # Create a grayscale image ('L') with a black background
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    # Get bounding box of the character and center it
    try:
        bbox = font.getbbox(char) 
        char_width = bbox[2] - bbox[0] # right - left
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

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load the font file
    try:
        # Font size is set to 80% of the image height for good padding
        font = ImageFont.truetype(FONT_PATH, int(IMG_SIZE * 0.8))
    except FileNotFoundError:
        print(f"❌ Error: Font file not found at '{FONT_PATH}'.")
        print("Please download a Khmer font and place it in the correct path.")
        exit()

    # 2. Call the function to get the pixel grid
    print(f"Rasterizing the character '{CHARACTER_TO_RASTERIZE}'...")
    pixel_array = rasterize_char(CHARACTER_TO_RASTERIZE, font, IMG_SIZE)
    print("Done. Shape of the resulting array:", pixel_array.shape)

    # 3. Use Matplotlib to visualize the array
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pixel_array, cmap='gray', interpolation='nearest')
    
    # Add gridlines to show individual pixels
    ax.set_xticks(np.arange(-.5, IMG_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-.5, IMG_SIZE, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    ax.set_title(f"Rasterized Character: '{CHARACTER_TO_RASTERIZE}'\n(Resolution: {IMG_SIZE}x{IMG_SIZE})")
    plt.show()