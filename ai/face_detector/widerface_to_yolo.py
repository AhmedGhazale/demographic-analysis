import os
from PIL import Image
import shutil


def convert_widerface_to_yolo(wider_dir, output_dir):
    """
    Converts the WIDERFace dataset to YOLO format.

    Args:
        wider_dir (str): Path to the root directory of the WIDERFace dataset.
        output_dir (str): Path to the output directory where YOLO-formatted data will be saved.
    """
    splits = ['train', 'val']  # WIDERFace splits to process

    for split in splits:

        print(f"started processing {split} split, (this can take some time)")
        # Path to the annotation file
        ann_file = os.path.join(wider_dir, 'wider_face_split', f'wider_face_{split}_bbx_gt.txt')

        # Create output directories
        images_output_dir = os.path.join(output_dir,'images', split )
        labels_output_dir = os.path.join(output_dir,'labels', split )
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        # Read annotation file
        with open(ann_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        i = 0
        while i < len(lines):
            # Read image filename
            filename = lines[i]
            i += 1

            # Read number of bounding boxes
            num_boxes = int(lines[i])
            i += 1
            if(num_boxes == 0):
                i+=1

            # Read bounding box annotations
            bbox_lines = []
            for _ in range(num_boxes):
                bbox_lines.append(lines[i])
                i += 1

            # Process image and annotations
            src_img_path = os.path.join(wider_dir, f'WIDER_{split}', 'images', filename)
            if not os.path.exists(src_img_path):
                print(f"Warning: Image not found - {src_img_path}")
                continue

            # Get image dimensions
            try:
                with Image.open(src_img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening {src_img_path}: {e}")
                continue

            # Process annotations
            yolo_annotations = []
            for line in bbox_lines:
                values = list(map(int, line.split()))
                if len(values) < 4:
                    continue  # Skip invalid lines

                x1, y1, w, h = values[:4]

                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue

                # Convert to YOLO format
                x_center = (x1 + w / 2) / img_width
                y_center = (y1 + h / 2) / img_height
                yolo_w = w / img_width
                yolo_h = h / img_height

                # Clamp values between 0 and 1
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                yolo_w = max(0, min(1, yolo_w))
                yolo_h = max(0, min(1, yolo_h))

                yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}")

            # Create destination filenames
            dest_filename = filename.replace('/', '_')  # Flatten directory structure
            dest_img_path = os.path.join(images_output_dir, dest_filename)
            dest_label_path = os.path.join(labels_output_dir, dest_filename.replace('.jpg', '.txt'))

            # Copy image
            shutil.copy2(src_img_path, dest_img_path)

            # Write label file
            with open(dest_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

        print(f"Finished processing {split} split")
    # Create dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    yaml_content = f"""# Auto-generated YOLO dataset configuration
path: {os.path.abspath(output_dir)}  # dataset root directory
train: images/train  # relative to 'path'
val: images/val      # relative to 'path'
test:                # test images (optional)

# Classes
names:
  0: face

# Class counts
nc: 1
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated dataset configuration file at: {yaml_path}")

if __name__ == '__main__':
    # Configure your paths here
    widerface_root = 'widerface'  # Root directory of WIDERFace dataset
    output_directory = 'widerface-yolo'  # Output directory

    convert_widerface_to_yolo(widerface_root, output_directory)
    print("Conversion complete!")