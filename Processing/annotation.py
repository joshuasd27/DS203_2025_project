import os
import cv2
import numpy as np

# --- Configuration ---
IMAGE_FOLDER = '/home/joshua/Downloads/resized_images'
ANNOTATION_FILE = os.path.join(IMAGE_FOLDER, 'annotations.txt')
GRID_SIZE = 8

# --- Global variables to manage state ---
clicked_cells = set()
current_image = None
current_image_display = None
cell_width = 0
cell_height = 0
rectangle_mode = False # Are we in rectangle selection mode?
rect_start_cell = None # Stores the (row, col) of the first corner


def get_annotated_images(annotation_file):
    """Reads the annotation file and returns a set of already annotated image numbers."""
    if not os.path.exists(annotation_file):
        return set()
    
    annotated_ids = set()
    with open(annotation_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    image_id = int(line.strip().split()[0])
                    annotated_ids.add(image_id)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse line in annotation file: {line.strip()}")
    return annotated_ids
def draw_grid_and_text(image, image_name):
    """Draws the grid, cell numbers, and highlights selected cells."""
    global cell_width, cell_height
    
    # Create a copy to draw on
    display_img = image.copy()
    h, w, _ = display_img.shape
    cell_width = w // GRID_SIZE
    cell_height = h // GRID_SIZE

    # Create a transparent overlay for selections
    overlay = display_img.copy()
    
    # Draw grid lines and cell numbers
    for i in range(1, GRID_SIZE):
        # Vertical lines
        cv2.line(display_img, (i * cell_width, 0), (i * cell_width, h), (255, 255, 255), 1)
        # Horizontal lines
        cv2.line(display_img, (0, i * cell_height), (w, i * cell_height), (255, 255, 255), 1)

    # Highlight clicked cells and draw numbers
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_index = row * GRID_SIZE + col
            
            # Highlight if selected
            if cell_index in clicked_cells:
                cv2.rectangle(overlay, (col * cell_width, row * cell_height),
                              ((col + 1) * cell_width, (row + 1) * cell_height),
                              (0, 255, 0), -1) # Green fill

    # Apply the transparent overlay
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

    # Display image name/number at the top-left
    info_text = f"Image: {image_name}"
    if rectangle_mode:
        if rect_start_cell is None:
            info_text += " | Rect Mode: Click 1st corner"
        else:
            info_text += " | Rect Mode: Click 2nd corner"

    cv2.putText(display_img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA) # Black outline
    cv2.putText(display_img, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # White text

    return display_img

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select/deselect grid cells."""
    global current_image_display, rectangle_mode, rect_start_cell

    if event == cv2.EVENT_LBUTTONDOWN:
        if cell_width == 0 or cell_height == 0: return

        col = x // cell_width
        row = y // cell_height
        cell_index = row * GRID_SIZE + col

        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return

        image_name = param

        if rectangle_mode:
            if rect_start_cell is None:
                # This is the first corner click
                rect_start_cell = (row, col)
                print("First corner selected. Click the second corner.")
            else:
                # This is the second corner click
                r1, c1 = rect_start_cell
                r2, c2 = row, col

                start_row, end_row = min(r1, r2), max(r1, r2)
                start_col, end_col = min(c1, c2), max(c1, c2)

                for r in range(start_row, end_row + 1):
                    for c in range(start_col, end_col + 1):
                        idx = r * GRID_SIZE + c
                        clicked_cells.add(idx)
                
                print("Rectangle selected. Returning to single-click mode.")
                rectangle_mode = False
                rect_start_cell = None
        else:
            # Normal single-cell toggle mode
            if cell_index in clicked_cells:
                clicked_cells.remove(cell_index)
            else:
                clicked_cells.add(cell_index)
        
        current_image_display = draw_grid_and_text(current_image, image_name)
        cv2.imshow('Image Annotator', current_image_display)

def main():
    """Main function to run the annotation tool."""
    global clicked_cells, current_image, current_image_display, rectangle_mode, rect_start_cell

    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Error: Image directory not found at '{IMAGE_FOLDER}'")
        return

    annotated_ids = get_annotated_images(ANNOTATION_FILE)
    print(f"Found {len(annotated_ids)} previously annotated images.")

    # Find all jpg images and sort them numerically
    try:
        image_files = sorted(
            [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
    except ValueError:
        print("Error: Could not sort image files numerically. Ensure they are named like '1.jpg', '2.jpg', etc.")
        return

    cv2.namedWindow('Image Annotator')

    for image_file in image_files:
        try:
            image_id = int(os.path.splitext(image_file)[0])
        except ValueError:
            continue # Skip files not named with a number

        if image_id in annotated_ids:
            continue # Skip already annotated images

        image_path = os.path.join(IMAGE_FOLDER, image_file)
        current_image = cv2.imread(image_path)

        if current_image is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        print(f"\nAnnotating image: {image_file}")
        print("Click on cells with animals. Press ENTER to save and go to the next image.")
        print("Press 'r' to toggle rectangle selection mode.")
        print("Press ESC to quit without saving the current image.")

        clicked_cells.clear()
        rectangle_mode = False
        rect_start_cell = None
        cv2.setMouseCallback('Image Annotator', mouse_callback, param=str(image_id))
        
        current_image_display = draw_grid_and_text(current_image, str(image_id))
        cv2.imshow('Image Annotator', current_image_display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            # On ENTER, save the annotation
            if key == 13: # 13 is the Enter key
                annotation_vector = [0] * (GRID_SIZE * GRID_SIZE)
                for cell_idx in clicked_cells:
                    annotation_vector[cell_idx] = 1
                
                annotation_str = ",".join(map(str, annotation_vector))
                
                with open(ANNOTATION_FILE, 'a') as f:
                    f.write(f"{image_id} {annotation_str}\n")
                
                print(f"Saved annotation for image {image_id}.")
                break # Move to the next image

            # On 'r' key, toggle rectangle mode
            if key == ord('r'):
                rectangle_mode = not rectangle_mode
                rect_start_cell = None # Reset corner on mode toggle
                if rectangle_mode:
                    print("Rectangle mode enabled. Click the first corner.")
                else:
                    print("Rectangle mode disabled.")
                # Redraw to update the info text
                current_image_display = draw_grid_and_text(current_image, str(image_id))
                cv2.imshow('Image Annotator', current_image_display)

            # On ESC, quit the program
            if key == 27: # 27 is the Esc key
                print("Exiting annotator.")
                cv2.destroyAllWindows()
                return

    print("\nAll images have been annotated!")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
