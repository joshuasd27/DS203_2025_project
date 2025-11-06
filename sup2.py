import os
#from tensorflow as tf  import is_tensor()
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import cv2

# Global grid configuration
GRID_SIZE_H = 16  # Grid height (rows)
GRID_SIZE_W = 32  # Grid width (columns)

def load_images_and_labels(folder, exclude_file, annotations_file,
                           skip_offset=0):
    """
    Loads images (no resizing) and labels into memory, excluding specified indices,
    and splits into train/test by skip pattern (every 5th image → test).
    
    Returns lists of TensorFlow image tensors (variable size)
    and label tensors (fixed 64D vectors).
    """

    # --- Parse exclusion list ---
    with open(exclude_file, 'r') as f:
        exclude_lines = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]
    exclude_set = set(exclude_lines)

    # --- Parse annotation file ---
    label_dict = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = parts[0].lstrip('0')  # normalize numeric keys
            label_vals = np.array(
                [float(x) for x in parts[1].split(',')],
                dtype=np.float32
            )
            label_dict[idx] = label_vals

    # --- Collect image paths ---
    image_files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    # --- Filter excluded images ---
    valid_files = []
    for f in image_files:
        base = os.path.splitext(os.path.basename(f))[0].lstrip('0')
        if base not in exclude_set:
            valid_files.append(f)

    # --- Load images (native size, no resize) ---
    images, labels = [], []
    for p in valid_files:
        base = os.path.splitext(os.path.basename(p))[0].lstrip('0')
        # Replace TensorFlow with OpenCV
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img.astype(np.uint8)  # Ensure uint8 type for consistency
        images.append(img)
        labels.append(label_dict[base])

    # --- Split using stride pattern ---
    indices = list(range(len(images)))
    test_indices = [i for i in indices if (i % 5) == skip_offset]
    train_indices = [i for i in indices if (i % 5) != skip_offset]

    train_imgs = [images[i] for i in train_indices]
    test_imgs  = [images[i] for i in test_indices]
    train_lbls = np.stack([labels[i] for i in train_indices])
    test_lbls  = np.stack([labels[i] for i in test_indices])
    print(f"Loaded {len(train_imgs)} of {type(train_imgs[0])} training images and {len(test_imgs)} of {type(test_imgs[0])} testing images.")
    print(f"Training labels shape: {train_lbls.shape}, Testing labels shape: {test_lbls.shape}")
    print(f"Example training image shape: {train_imgs[0].shape}, Example training label: {train_lbls[0]}")
    print(f"Example testing image shape: {test_imgs[0].shape}, Example testing label: {test_lbls[0]}")

    # `images` is a list of NumPy arrays
    total_bytes = sum(img.nbytes for img in train_imgs)
    total_mb = total_bytes / (1024 * 1024)

    print("Images loaded: {} | Total size: {:.2f} MB".format(len(images), total_mb))
    return train_imgs, train_lbls, test_imgs, test_lbls


# def display_image_with_grids(image, labels, title="Image with Grid Overlays", extra_comments=None):
#     """
#     Display an image with superimposed grids colored according to labels.
    
#     Args:
#         image: TensorFlow tensor or numpy array of shape (H, W, 3) with values in [0, 1]
#         labels: 1D numpy array of length 256 with class labels
#         title: Title for the plot
#     """
#     # Convert TensorFlow tensor to numpy if needed
#     if hasattr(image, 'numpy'):
#         image = image.numpy()
    
#     # Ensure labels is numpy array
#     labels = np.array(labels)
    
#     # Reshape vector to grid (row-wise)
#     labels_grid = labels.reshape(GRID_SIZE_H, GRID_SIZE_W)
    
#     # Get unique labels and create colormap
#     unique_labels = np.unique(labels)

#     # Define colors for different labels
#     label_colors = ['red', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
#     color_map = {label: label_colors[i % len(label_colors)] for i, label in enumerate(unique_labels)}
    
#     # Create figure with 1x2 subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
#     # Left subplot: Original image
#     ax1.imshow(image)
#     ax1.set_title("Original Image")
#     ax1.set_xticks([])
#     ax1.set_yticks([])
    
#     # Right subplot: Image with grid overlays
#     ax2.imshow(image)
#     ax2.set_title(title)
    
#     # Get image dimensions
#     height, width = image.shape[:2]
    
#     # Set axis limits to image boundaries
#     ax2.set_xlim(0, width)
#     ax2.set_ylim(height, 0)  # Note: y-axis is flipped for images
    
#     # Draw grid with GRID_SIZE_H x GRID_SIZE_W cells
#     for i in range(GRID_SIZE_H + 1):  # GRID_SIZE_H+1 lines for GRID_SIZE_H cells
#         # Horizontal lines
#         y = i * height / GRID_SIZE_H
#         if y < height:  # Only draw lines within image bounds
#             ax2.axhline(y, color='blue', linewidth=1, alpha=0.7)
    
#     for j in range(GRID_SIZE_W + 1):  # GRID_SIZE_W+1 lines for GRID_SIZE_W cells
#         # Vertical lines
#         x = j * width / GRID_SIZE_W
#         if x < width:  # Only draw lines within image bounds
#             ax2.axvline(x, color='blue', linewidth=1, alpha=0.7)
       
#     # Draw 8x8 green grid (thicker lines)
#     for i in range(9):  # 9 lines for 8 cells
#         # Vertical lines
#         x = i * width / 8
#         if x < width:  # Only draw lines within image bounds
#             ax2.axvline(x, color='green', linewidth=2, alpha=0.8)
#         # Horizontal lines
#         y = i * height / 8
#         if y < height:  # Only draw lines within image bounds
#             ax2.axhline(y, color='green', linewidth=2, alpha=0.8)
    
#     # Color the grid cells according to labels
#     for i in range(GRID_SIZE_H):
#         for j in range(GRID_SIZE_W):
#             label = labels_grid[i, j]
#             color = color_map[label]
            
#             # Calculate cell boundaries
#             x_start = j * width / GRID_SIZE_W
#             x_end = (j + 1) * width / GRID_SIZE_W
#             y_start = i * height / GRID_SIZE_H
#             y_end = (i + 1) * height / GRID_SIZE_H
#                # Create colored rectangle with transparency
#             rect = patches.Rectangle(
#                 (x_start, y_start), 
#                 x_end - x_start, 
#                 y_end - y_start,
#                 linewidth=0, 
#                 facecolor=color, 
#                 alpha=0.3
#             )
#             ax2.add_patch(rect)
    
#     # Create legend
#     legend_elements = []
#     for label in sorted(unique_labels):
#         legend_elements.append(
#             patches.Patch(color=color_map[label], label=f'Class {int(label)}')
#         )
    
#     ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
#     # Remove axis ticks for cleaner look
#     ax2.set_xticks([])
#     ax2.set_yticks([])
    
#     plt.tight_layout()
#     if extra_comments is not None:
#         plt.suptitle(extra_comments, fontsize=16, fontweight='bold')
#     plt.show()

def display_image_with_grids(image, labels_list, title_list=None, extra_comments=None):
    """
    Display an image with multiple grid overlays for different clustering results.
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1] or [0, 255]
        labels_list: List of 1D numpy arrays, each of length GRID_SIZE_H*GRID_SIZE_W with class labels
        title_list: List of titles for each grid overlay (optional)
        extra_comments: Overall title for the entire figure
    """
    # Convert TensorFlow tensor to numpy if needed
    if hasattr(image, 'numpy'):
        image = image.numpy()
    
    # Ensure labels_list is a list
    if not isinstance(labels_list, list):
        labels_list = [labels_list]
    
    num_grids = len(labels_list)
    
    # Create default titles if not provided
    if title_list is None:
        title_list = [f"Grid Overlay {i+1}" for i in range(num_grids)]
    elif not isinstance(title_list, list):
        title_list = [title_list]
    
    # Ensure title_list has the same length as labels_list
    while len(title_list) < num_grids:
        title_list.append(f"Grid Overlay {len(title_list)+1}")
    
    # Calculate optimal subplot layout
    if num_grids == 1:
        rows, cols = 1, 2  # Original + 1 grid
    elif num_grids == 2:
        rows, cols = 1, 3  # Original + 2 grids
    elif num_grids <= 4:
        rows, cols = 2, 3  # Original + up to 4 grids (2x3 layout)
    elif num_grids <= 6:
        rows, cols = 2, 4  # Original + up to 6 grids (2x4 layout)
    elif num_grids <= 9:
        rows, cols = 3, 4  # Original + up to 9 grids (3x4 layout)
    else:
        # For more than 9 grids, calculate square-ish layout
        cols = int(np.ceil(np.sqrt(num_grids + 1)))
        rows = int(np.ceil((num_grids + 1) / cols))
    
    # Calculate figure size based on number of subplots
    fig_width = cols * 6
    fig_height = rows * 5
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle case where we have only one row or column
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # First subplot: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create color palette for consistent coloring across all grids
    base_colors = ['red', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Process each grid overlay
    for idx, (labels, title) in enumerate(zip(labels_list, title_list)):
        ax = axes[idx + 1]
        
        # Ensure labels is numpy array
        labels = np.array(labels)
        
        # Reshape vector to grid (row-wise)
        labels_grid = labels.reshape(GRID_SIZE_H, GRID_SIZE_W)
        
        # Get unique labels and create colormap
        unique_labels = np.unique(labels)
        color_map = {label: base_colors[int(label) % len(base_colors)] for label in unique_labels}
        
        # Display image with grid overlay
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        
        # Set axis limits to image boundaries
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Note: y-axis is flipped for images
        
        # Draw main grid with GRID_SIZE_H x GRID_SIZE_W cells
        for i in range(GRID_SIZE_H + 1):  # GRID_SIZE_H+1 lines for GRID_SIZE_H cells
            y = i * height / GRID_SIZE_H
            if y <= height:  # Only draw lines within image bounds
                ax.axhline(y, color='blue', linewidth=0.8, alpha=0.6)
        
        for j in range(GRID_SIZE_W + 1):  # GRID_SIZE_W+1 lines for GRID_SIZE_W cells
            x = j * width / GRID_SIZE_W
            if x <= width:  # Only draw lines within image bounds
                ax.axvline(x, color='blue', linewidth=0.8, alpha=0.6)
        
        # Draw 8x8 green reference grid (optional - can be removed if too cluttered)
        if GRID_SIZE_H <= 16 and GRID_SIZE_W <= 16:  # Only show for smaller grids
            for i in range(9):  # 9 lines for 8 cells
                x = i * width / 8
                y = i * height / 8
                if x <= width:
                    ax.axvline(x, color='green', linewidth=1.5, alpha=0.5)
                if y <= height:
                    ax.axhline(y, color='green', linewidth=1.5, alpha=0.5)
        
        # Color the grid cells according to labels
        for i in range(GRID_SIZE_H):
            for j in range(GRID_SIZE_W):
                label = labels_grid[i, j]
                color = color_map[label]
                
                # Calculate cell boundaries
                x_start = j * width / GRID_SIZE_W
                x_end = (j + 1) * width / GRID_SIZE_W
                y_start = i * height / GRID_SIZE_H
                y_end = (i + 1) * height / GRID_SIZE_H
                
                # Create colored rectangle with transparency
                rect = patches.Rectangle(
                    (x_start, y_start), 
                    x_end - x_start, 
                    y_end - y_start,
                    linewidth=0, 
                    facecolor=color, 
                    alpha=0.4
                )
                ax.add_patch(rect)
        
        # Create legend for this subplot
        legend_elements = []
        for label in sorted(unique_labels):
            legend_elements.append(
                patches.Patch(color=color_map[label], label=f'Class {int(label)}')
            )
        
        # Position legend outside the plot area
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for idx in range(num_grids + 1, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    if extra_comments is not None:
        fig.suptitle(extra_comments, fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    if extra_comments is not None:
        plt.subplots_adjust(top=0.90)  # Make room for the main title
    
    plt.show()


def display_clustering_comparison(image, grids_dict, method_name="Clustering Comparison"):
    """
    Convenience function to display clustering results from analyze_image_clusters.
    
    Args:
        image: Input image
        grids_dict: Dictionary from analyze_image_clusters {k: label_grid}
        method_name: Overall method name for the title
    """
    labels_list = []
    title_list = []
    
    for k, grid in grids_dict.items():
        labels_list.append(grid.flatten())
        title_list.append(f"K={k} Clustering")
    
    display_image_with_grids(image, labels_list, title_list, extra_comments=method_name)

train_imgs, train_lbls, test_imgs, test_lbls = load_images_and_labels("./resized_images", "./bad_image_indexes.txt", "./annotations.txt", skip_offset=0)
import sys

# def srgb_to_linear(component):
#     """
#     Convert sRGB component to linear RGB using gamma correction.
    
#     Args:
#         component: sRGB component value in range [0, 1]
    
#     Returns:
#         Linear RGB component value
#     """
#     return np.where(component <= 0.03928, 
#                     component / 12.92, 
#                     np.power((component + 0.055) / 1.055, 2.4))
# def calculate_luminance(image):
#     """
#     Calculate luminance using proper sRGB to linear RGB conversion.
    
#     Args:
#         image: RGB image with values in range [0, 1] or [0, 255]
    
#     Returns:
#         Luminance values
#     """
#     # Normalize R, G, B values: Divide each component by 255 if needed
#     if image.max() > 1.0:
#         R_sRGB = image[..., 0] / 255.0
#         G_sRGB = image[..., 1] / 255.0
#         B_sRGB = image[..., 2] / 255.0
#     else:
#         R_sRGB = image[..., 0]
#         G_sRGB = image[..., 1]
#         B_sRGB = image[..., 2]
    
#     # Linearize the values: Apply gamma correction to each component
#     R = srgb_to_linear(R_sRGB)
#     G = srgb_to_linear(G_sRGB)
#     B = srgb_to_linear(B_sRGB)
    
#     # Calculate the luminance using ITU-R BT.709 coefficients
#     L = (0.2126 * R) + (0.7152 * G) + (0.0722 * B)
    
#     return L    

# def visualize_luminance_analysis(image, title="Luminance Analysis"):
#     """
#     Comprehensive luminance visualization: pixel-level, statistics, and patch-wise analysis.
    
#     Args:
#         image: numpy array of shape (H, W, 3) with values in [0, 1]
#         title: Title for the plots
#     """
#     # Calculate proper luminance using sRGB conversion
#     L = calculate_luminance(image)
    
#     # Create figure with 2x2 subplots
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
#     # 1. Original image
#     ax1.imshow(image)
#     ax1.set_title("Original Image")
#     ax1.set_xticks([])
#     ax1.set_yticks([])
    
#     # 2. Pixel-level luminance heatmap
#     im2 = ax2.imshow(L, cmap='viridis')
#     ax2.set_title("Pixel-level Luminance")
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#     cbar2.set_label('Luminance', rotation=270, labelpad=20)
    
#     # 3. Luminance histogram
#     ax3.hist(L.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
#     ax3.set_title("Luminance Distribution")
#     ax3.set_xlabel("Luminance Value")
#     ax3.set_ylabel("Frequency")
#     ax3.grid(True, alpha=0.3)
    
#     # Add statistics text
#     stats_text = f"""Statistics:
# Min: {L.min():.4f}
# Max: {L.max():.4f}
# Mean: {L.mean():.4f}
# Std: {L.std():.4f}
# Median: {np.median(L):.4f}
# Q1: {np.percentile(L, 25):.4f}
# Q3: {np.percentile(L, 75):.4f}"""
#     ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
#     # 4. Patch-wise average luminance
#     H, W = L.shape
#     grid_h, grid_w = GRID_SIZE, GRID_SIZE
#     ph, pw = H // grid_h, W // grid_w
    
#     # Calculate patch-wise average luminance
#     patch_luminance = np.zeros((grid_h, grid_w), dtype=np.float32)
#     for i in range(grid_h):
#         for j in range(grid_w):
#             patch_y = L[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
#             patch_luminance[i, j] = patch_y.mean()
    
#     im4 = ax4.imshow(patch_luminance, cmap='viridis', interpolation='nearest')
#     ax4.set_title(f"Patch-wise Average Luminance ({GRID_SIZE}x{GRID_SIZE})")
#     ax4.set_xlabel("Patch Column")
#     ax4.set_ylabel("Patch Row")
    
#     # Add grid lines for patches
#     for i in range(GRID_SIZE + 1):
#         ax4.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
#         ax4.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
#     cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
#     cbar4.set_label('Average Luminance', rotation=270, labelpad=20)
    
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()
    
#     # Print detailed statistics
#     print(f"\n{'='*60}")
#     print(f"LUMINANCE ANALYSIS REPORT")
#     print(f"{'='*60}")
#     print(f"Image Shape: {image.shape}")
#     print(f"Luminance Shape: {L.shape}")
#     print(f"\nPIXEL-LEVEL STATISTICS:")
#     print(f"  Minimum:     {L.min():.6f}")
#     print(f"  Maximum:     {L.max():.6f}")
#     print(f"  Mean:        {L.mean():.6f}")
#     print(f"  Std Dev:     {L.std():.6f}")
#     print(f"  Median:      {np.median(L):.6f}")
#     print(f"  25th Percentile: {np.percentile(L, 25):.6f}")
#     print(f"  75th Percentile: {np.percentile(L, 75):.6f}")
#     print(f"  Range:       {L.max() - L.min():.6f}")
#     print(f"  Variance:    {L.var():.6f}")
    
#     print(f"\nPATCH-LEVEL STATISTICS ({GRID_SIZE}x{GRID_SIZE} patches):")
#     print(f"  Patch size:  {ph}x{pw} pixels")
#     print(f"  Min patch avg:   {patch_luminance.min():.6f}")
#     print(f"  Max patch avg:   {patch_luminance.max():.6f}")
#     print(f"  Mean patch avg:  {patch_luminance.mean():.6f}")
#     print(f"  Std patch avg:   {patch_luminance.std():.6f}")
#     print(f"  Patch range:     {patch_luminance.max() - patch_luminance.min():.6f}")
    
#     return L, patch_luminance
# def visualize_luminance_overlay(image, title="Luminance Overlay", alpha=0.6):
#     """
#     Show original image with luminance overlay and patch grid with values.
    
#     Args:
#         image: numpy array of shape (H, W, 3) with values in [0, 1]
#         title: Title for the plot
#         alpha: Transparency of overlay
#     """
#     L = calculate_luminance(image)
    
#     # Create figure with 1x3 subplots
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
#     # 1. Original image
#     ax1.imshow(image)
#     ax1.set_title("Original Image")
#     ax1.set_xticks([])
#     ax1.set_yticks([])
    
#     # 2. Original with luminance overlay
#     ax2.imshow(image)
#     im2 = ax2.imshow(L, cmap='plasma', alpha=alpha)
#     ax2.set_title(f"Luminance Overlay (α={alpha})")
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#     cbar2.set_label('Luminance', rotation=270, labelpad=20)
    
#     # 3. Patch grid with luminance values
#     ax3.imshow(image)
    
#     H, W = L.shape
#     grid_h, grid_w = GRID_SIZE, GRID_SIZE
#     ph, pw = H // grid_h, W // grid_w
    
#     # Calculate and overlay patch luminance
#     for i in range(grid_h):
#         for j in range(grid_w):
#             patch_y = L[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
#             patch_avg = patch_y.mean()
            
#             # Get patch boundaries in image coordinates
#             x_start = j * W / grid_w
#             x_end = (j + 1) * W / grid_w
#             y_start = i * H / grid_h
#             y_end = (i + 1) * H / grid_h
            
#             # Normalize for colormap
#             norm_val = (patch_avg - L.min()) / (L.max() - L.min()) if L.max() > L.min() else 0
#             color = plt.cm.plasma(norm_val)
            
#             # Draw colored rectangle
#             rect = patches.Rectangle(
#                 (x_start, y_start), x_end - x_start, y_end - y_start,
#                 linewidth=1, edgecolor='white', facecolor=color, alpha=0.7
#             )
#             ax3.add_patch(rect)
            
#             # Add text with luminance value
#             text_x = x_start + (x_end - x_start) / 2
#             text_y = y_start + (y_end - y_start) / 2
#             ax3.text(text_x, text_y, f'{patch_avg:.3f}', 
#                     ha='center', va='center', fontsize=8, 
#                     color='white', weight='bold')
    
#     # Draw grid lines
#     for i in range(grid_h + 1):
#         y = i * H / grid_h
#         ax3.axhline(y, color='white', linewidth=1, alpha=0.8)
#     for j in range(grid_w + 1):
#         x = j * W / grid_w
#         ax3.axvline(x, color='white', linewidth=1, alpha=0.8)
    
#     ax3.set_title(f"Patch Luminance Grid ({GRID_SIZE}x{GRID_SIZE})")
#     ax3.set_xlim(0, W)
#     ax3.set_ylim(H, 0)
#     ax3.set_xticks([])
#     ax3.set_yticks([])
    
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()
# def compare_luminance_methods(image):
#     """
#     Compare simple RGB average vs proper sRGB luminance calculation.
#     """
#     # Simple RGB average (old method)
#     Y_simple = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    
#     # Proper sRGB luminance
#     L_proper = calculate_luminance(image)
    
#     # Create comparison plot
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
#     # Simple method
#     im1 = ax1.imshow(Y_simple, cmap='viridis')
#     ax1.set_title("Simple RGB Average")
#     plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
#     # Proper sRGB method
#     im2 = ax2.imshow(L_proper, cmap='viridis')
#     ax2.set_title("Proper sRGB Luminance")
#     plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
#     # Difference
#     diff = L_proper - Y_simple
#     im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
#     ax3.set_title("Difference (sRGB - Simple)")
#     plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
#     # Histogram comparison
#     ax4.hist(Y_simple.flatten(), bins=50, alpha=0.7, label='Simple RGB', color='blue')
#     ax4.hist(L_proper.flatten(), bins=50, alpha=0.7, label='sRGB Luminance', color='red')
#     ax4.set_title("Luminance Distribution Comparison")
#     ax4.set_xlabel("Luminance Value")
#     ax4.set_ylabel("Frequency")
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     print(f"Comparison Statistics:")
#     print(f"Simple RGB - Mean: {Y_simple.mean():.6f}, Std: {Y_simple.std():.6f}")
#     print(f"sRGB Luminance - Mean: {L_proper.mean():.6f}, Std: {L_proper.std():.6f}")
#     print(f"Mean difference: {diff.mean():.6f}")
#     print(f"Max absolute difference: {np.abs(diff).max():.6f}")

# # Example usage:
# img = train_imgs[6]

# # Comprehensive luminance analysis
# L, patch_lum = visualize_luminance_analysis(img, "Training Image Luminance Analysis")

# # Luminance overlay visualization
# visualize_luminance_overlay(img, "Luminance Overlay Visualization", alpha=0.6)

# # Compare different luminance calculation methods
# compare_luminance_methods(img)

def extract_patch_features(image, P=8, R=1, lbp_method='ror', lbp_bins=36, hsv_bins=10, lbp_weight=0.5, hsv_weight=0.5):
    """
    Split image tensor into GRID_SIZE x GRID_SIZE patches and extract [LBP + Color] features.
    Returns (feature_matrix, grid_shape).
    """
    (lbp_weight,hsv_weight) = (lbp_weight/(lbp_weight+hsv_weight),hsv_weight/(lbp_weight+hsv_weight))

    H, W, C = image.shape
    grid_h, grid_w = GRID_SIZE_H, GRID_SIZE_W
    ph, pw = H // grid_h, W // grid_w  # patch height/width
    # print(f"Image shape: {image.shape}, Patch size: {ph}x{pw}")

    features = []

    coordinates = []
    for i in range(grid_h):
        for j in range(grid_w):
            #coordinates of patch
            (x,y) = (j*pw,i*ph)
            coordinates.append([x,y])
    coordinates_norm = np.array(coordinates)
    coordinates_norm = norm(coordinates_norm, axis=0)

            
    for i in range(grid_h):
        for j in range(grid_w):
            patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]

            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            # --- LBP ---
            lbp = local_binary_pattern(gray, P=P, R=R, method=lbp_method)
            # lbp_bins = lbp.shape[0]
            lbp_hist, _ = np.histogram(lbp, bins=lbp_bins, range=(0, lbp_bins), density=False)
            lbp_hist = lbp_hist.astype(float) / np.sum(lbp_hist)
            lbp_hist = lbp_hist*np.sqrt(lbp_bins)
            print(lbp_hist.shape)
            print(lbp_hist)

            # --- Hue Histograms ---

            # Ensure patch is uint8 for OpenCV
            if patch.dtype != np.uint8:
                patch = (255 * np.clip(patch, 0, 1)).astype(np.uint8)

            # Convert RGB to HSV using OpenCV
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            
            # Extract hue channel (first channel)
            hue = hsv[..., 0]
            
            # Create histogram of hue values
            # Hue ranges from 0 to 179 in OpenCV (180 degrees)
            hue_hist, _ = np.histogram(hue, bins=hsv_bins, range=(0, 180), density=False)
            hue_hist = hue_hist.astype(float) / np.sum(hue_hist)
            hue_hist = hue_hist*np.sqrt(hsv_bins)
            print(hue_hist.shape)
            print(hue_hist)

            lbp_hist = lbp_hist / norm(lbp_hist)
            hue_hist = hue_hist / norm(hue_hist)

            #coordinates of patch
            (x,y) = (j*pw,i*ph)


            # --- Combine ---
            feature = np.concatenate([lbp_weight * lbp_hist, hsv_weight * hue_hist])

            #scaling = 1/10000000
            #feature = np.append(feature, [i*ph*scaling, j*pw*scaling])  # Append grid cell coordinates
            features.append(feature)
            print(f"Patch ({i}, {j}) feature shape: {feature.shape}")
            print(feature)
            print("-----")
            # sys.exit()
    
    features = np.array(features)  # shape (GRID_SIZE^2, lbp_bins + 3*color_bins)
    #features = StandardScaler().fit_transform(features)
    return features, (grid_h, grid_w)

def cluster_patches(features, grid_shape, k_values=[2,3.4]):
    """
    Run K-means for multiple k, assign class labels sorted by cluster size.
    Returns a dict of {k: GRID_SIZE x GRID_SIZE label grid}.
    """
    H, W = grid_shape
    results = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, max_iter=1000,random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)

        # sort clusters by frequency → largest = class 0, next = class 1, ...
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)  # descending order
        remap = {old: new for new, old in enumerate(unique[order])}
        labels = np.array([remap[l] for l in labels])

        results[k] = labels.reshape(H, W)
    return results



def analyze_image_clusters(image_tensor, k_values=[3,5,7], P=8, R=1, lbp_method='ror', lbp_bins=19, hsv_bins=28, lbp_weight=0.7, hsv_weight=0.2):
    """
    Complete pipeline: extract features, run clustering, return grids.
    """
    features, grid_shape = extract_patch_features(image_tensor, P=P, R=R, lbp_method=lbp_method, lbp_bins=lbp_bins, hsv_bins=hsv_bins, lbp_weight=lbp_weight, hsv_weight=hsv_weight)
    grids = cluster_patches(features, grid_shape, k_values)
    return grids

img_idxs = [324,23,38,453]
num_classes =3
labels_list = []
title_list=[]
for img_idx in img_idxs:
    img = train_imgs[img_idx]
    for lbp_weight in np.linspace(0,1,num=11):
        grids = analyze_image_clusters(img, k_values=[num_classes], lbp_weight=lbp_weight, hsv_weight=1-lbp_weight)
        labels_list.append(grids[num_classes])
        title_list.append(f"Lbp weight = {lbp_weight}")
    display_image_with_grids(img, labels_list=labels_list, title_list=title_list)
# display_image_with_grids(img, grids[3], title="Training Image with Grid Overlays")
# display_image_with_grids(img, grids[4], title="Training Image with Grid Overlays")
# display_image_with_grids(img, grids[5], title="Training Image with Grid Overlays")

print(64*18)
#HOG things to decide -> patch size in percentage of image, number of bins, 
# use unsigned gradients as need only orientation not light to dark or dark to light

