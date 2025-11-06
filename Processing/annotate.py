import cv2
import numpy as np
import matplotlib.pyplot as plt

def annotate_image_with_grid(img_path, vec):
    # Load image (BGR)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or invalid path")

    # Convert to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure vector length is 64 (8x8)
    if len(vec) != 64:
        raise ValueError("Vector must be of length 64")

    vec = np.array(vec).reshape(8, 8)
    h, w, _ = img.shape
    cell_h, cell_w = h // 8, w // 8

    annotated = img_rgb.copy()

    # Draw grid and annotate
    for i in range(8):
        for j in range(8):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            # Draw grid
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # Add text (0 or 1)
            label = str(int(vec[i, j]))
            color = (255, 0, 0) if vec[i, j] == 1 else (0, 255, 0)  # red for 1, green for 0
            cv2.putText(
                annotated,
                label,
                (x1 + cell_w // 3, y1 + int(cell_h * 0.6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

    # Display side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(annotated)
    plt.title("Annotated Image (8×8 Grid)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def grayscale_masked_grid(img_path, vec, img_in=None):
    if img_in is None:
        img = cv2.imread(img_path)
    else:
        img = img_in

    if img is None:
        raise ValueError("Image not found or invalid path")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(vec) != 64:
        raise ValueError("Vector must be of length 64")

    vec = np.array(vec).reshape(8, 8)
    h, w, _ = img.shape
    cell_h, cell_w = h // 8, w // 8

    annotated = img_rgb.copy()

    for i in range(8):
        for j in range(8):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            if vec[i, j] == 1:
                region = annotated[y1:y2, x1:x2].astype(np.float32)

                # Convert to grayscale and then tint slightly blue-gray
                gray = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

                # Blend gray with original color to desaturate (keep 40% color)
                desaturated = cv2.addWeighted(region, 0.4, gray_rgb, 0.6, 0)

                # Add subtle blue tint overlay (for visibility)
                tint = np.zeros_like(desaturated)
                tint[:, :, 2] = 30  # blue channel boost
                desaturated = cv2.addWeighted(desaturated, 0.9, tint, 0.1, 0)

                annotated[y1:y2, x1:x2] = desaturated.clip(0, 255).astype(np.uint8)

            # Draw borders
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(annotated)
    plt.title("Desaturated on Vector=1 Cells (with Borders)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return annotated


def color_overlay_grid(img_path, vec, img_in = None, alpha=0.5):
    """
    Overlay color-coded 8x8 grid based on vec values (0/1)
    on top of the image with visible gridlines.
    """
    if img_in is None:
        img = cv2.imread(img_path)
    else:
        img = img_in
    if img is None:
        raise ValueError("Image not found or invalid path")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    if len(vec) != 64:
        raise ValueError("Vector must be of length 64")

    vec = np.array(vec).reshape(8, 8)
    cell_h, cell_w = h // 8, w // 8

    overlay = img_rgb.copy()

    for i in range(8):
        for j in range(8):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            if vec[i, j] == 1:
                color = (255, 0, 0)   # red block for 1
            else:
                color = (0, 255, 0)   # green block for 0

            # Fill rectangle with color (semi-transparent)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlays
    blended = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

    # Draw grid borders
    for i in range(9):
        y = i * cell_h
        cv2.line(blended, (0, y), (w, y), (255, 255, 255), 2)
    for j in range(9):
        x = j * cell_w
        cv2.line(blended, (x, 0), (x, h), (255, 255, 255), 2)

    # Display side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(blended)
    plt.title("Color Overlay Grid (Red=1, Green=0)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return blended

# Example usage:
if __name__ == "__main__":
    vec = np.random.randint(0, 2, 64)  # random 0/1 vector
    print(vec.reshape(8,8))
    grayscale_masked_grid(r"E:\Sem_3\DS 203\E7\resized_images\image_1.jpg", vec)

    # Example usage:
    #vec = np.random.randint(0, 2, 64)  # random 0/1 vector
    #annotate_image_with_grid(r"E:\Sem_3\DS 203\E7\resized_images\image_1.jpg", vec)

