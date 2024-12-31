import os
import cv2

def read_labels(label_path):
    """Reads the contents of a YOLO format label file."""
    with open(label_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_label(label_path, data):
    """Writes data to a YOLO format label file."""
    with open(label_path, 'w') as file:
        for line in data:
            file.write(line)

def apply_horizontal_mirror(image_path, label_path):
    """Applies a horizontal mirror effect to an image and updates the 
    corresponding YOLO format label."""
    # Read the image
    img = cv2.imread(image_path)

    # Apply horizontal mirror effect
    img_mirror = cv2.flip(img, 1)

    # Get the dimensions of the image
    height, width, _ = img.shape

    # Read the labels
    lines_label = read_labels(label_path)

    # Apply mirror to bounding box coordinates
    new_lines_label = []
    for line in lines_label:
        class_, x_center, y_center, box_width, box_height = map(float, line.split())
        x_center_mirror = 1.0 - x_center
        new_lines_label.append(f"{int(class_)} {x_center_mirror} {y_center} {box_width} {box_height}\n")

    # Save the mirrored image
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    mirrored_image_path = os.path.join(os.path.dirname(image_path), f"{file_name}_horizontal_mirror{file_extension}")
    cv2.imwrite(mirrored_image_path, img_mirror)

    # Save the new label
    mirrored_label_path = os.path.join(os.path.dirname(label_path), f"{file_name}_horizontal_mirror.txt")
    write_label(mirrored_label_path, new_lines_label)

    return mirrored_image_path, mirrored_label_path

def apply_vertical_mirror(image_path, label_path):
    """Applies a vertical mirror effect to an image and 
    updates the corresponding YOLO format label."""
    # Read the image
    img = cv2.imread(image_path)

    # Apply vertical mirror effect
    img_mirror = cv2.flip(img, 0)

    # Get the dimensions of the image
    height, width, _ = img.shape

    # Read the labels
    lines_label = read_labels(label_path)

    # Apply mirror to bounding box coordinates
    new_lines_label = []
    for line in lines_label:
        class_, x_center, y_center, box_width, box_height = map(float, line.split())
        y_center_mirror = 1.0 - y_center
        new_lines_label.append(f"{int(class_)} {x_center} {y_center_mirror} {box_width} {box_height}\n")

    # Save the mirrored image
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    mirrored_image_path = os.path.join(os.path.dirname(image_path), f"{file_name}_vertical_mirror{file_extension}")
    cv2.imwrite(mirrored_image_path, img_mirror)

    # Save the new label
    mirrored_label_path = os.path.join(os.path.dirname(label_path), f"{file_name}_vertical_mirror.txt")
    write_label(mirrored_label_path, new_lines_label)

    return mirrored_image_path, mirrored_label_path

def mirror_images_in_directory(image_directory, label_directory):
    """Mirrors all images in a directory and updates corresponding YOLO format labels."""
    # List all files in the image directory
    image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.jpg'))]

    # Apply mirror effect to each image
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        label_path = os.path.join(label_directory, f"{os.path.splitext(image_file)[0]}.txt")
        mirrored_image_path, mirrored_label_path = apply_horizontal_mirror(image_path, label_path)
        print(f"Mirrored image saved: {mirrored_image_path}")
        print(f"New label file saved: {mirrored_label_path}")
        mirrored_image_path, mirrored_label_path = apply_vertical_mirror(image_path, label_path)
        print(f"Mirrored image saved: {mirrored_image_path}")
        print(f"New label file saved: {mirrored_label_path}")

def main():
    """
    Get images from directory 
    Apply mirror effect
    Change YoLo file
    Store new images and labels on database

    7 - 500: train
    500 - 590: val
    0 - 7 && 580 - 647: inference
    """
    for i in range(0, 1):
        # Specify the directories
        dataset_images_val_directory = f'datasets/dataset_{i}/images/val'
        dataset_labels_val_directory = f'datasets/dataset_{i}/labels/val'

        # Call the function to mirror images and generate new YOLO labels in the specified directories
        mirror_images_in_directory(dataset_images_val_directory, dataset_labels_val_directory)
        

if __name__ == "__main__":
    main()
