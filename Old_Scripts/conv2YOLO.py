import os
import json
from PIL import Image

def convert_bbox_to_yolo_format(bbox, img_size):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (bbox[0] + bbox[2]) / 2.0 - bbox[0]
    y = (bbox[1] + bbox[3]) / 2.0 - bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def prepare_dataset(image_dir, annotation_dir):
    class_names = sorted(os.listdir(image_dir))  # Assuming each class has its own folder
    class_to_index = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(image_dir, class_name)
        class_annotation_folder = os.path.join(annotation_dir, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = Image.open(img_path)
            img_w, img_h = img.size

            json_file = img_file.replace('.jpg', '.json')
            annotation_json_path = os.path.join(class_annotation_folder, json_file)

            if not os.path.exists(annotation_json_path):
                print(f"Warning: JSON file not found for {img_file}")
                continue

            with open(annotation_json_path, 'r') as f:
                annotations = json.load(f)

            # Check if annotations is a list of items or a single dictionary
            if isinstance(annotations, dict):  # If it's a single dictionary, make it a list
                annotations = [annotations]

            annotation_file = img_file.replace('.jpg', '.txt')
            annotation_path = os.path.join(class_annotation_folder, annotation_file)
            with open(annotation_path, 'w') as file:
                for ann in annotations:
                    bbox = ann['bbox']
                    yolo_bbox = convert_bbox_to_yolo_format(bbox, (img_w, img_h))
                    class_id = class_to_index[ann['class']]
                    file.write(f"{class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")


image_directory = r"C:\Users\janva\PycharmProjects\boat\venv\boat-types-recognition"
annotation_directory = r"C:\Users\janva\PycharmProjects\boat\venv\annotations"
prepare_dataset(image_directory, annotation_directory)
