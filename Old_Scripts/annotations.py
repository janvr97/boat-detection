import json
from PIL import Image
import os
import shutil
import random

def create_centered_bbox(image_path, scale=0.6):
    with Image.open(image_path) as img:
        width, height = img.size
        bbox_width = width * scale
        bbox_height = height * scale
        left = (width - bbox_width) / 2
        top = (height - bbox_height) / 2
        right = left + bbox_width
        bottom = top + bbox_height
        return [int(left), int(top), int(right), int(bottom)]

def save_annotations(image_folder, annotation_folder):
    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        class_annotation_folder = os.path.join(annotation_folder, class_folder)
        os.makedirs(class_annotation_folder, exist_ok=True)  # Ensure subfolder exists
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                bbox = create_centered_bbox(image_path)
                annotation = {
                    "class": class_folder,
                    "bbox": bbox
                }
                # Save annotation to JSON file in the corresponding subfolder
                annotation_filename = f"{os.path.splitext(image_file)[0]}.json"
                annotation_path = os.path.join(class_annotation_folder, annotation_filename)
                with open(annotation_path, 'w') as f:
                    json.dump(annotation, f, indent=4)


""" Convert to YOLO format """

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


# def split_dataset(image_folder, annotation_folder, train_dir, val_dir, train_ratio=0.8):
#     images = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
#     random.shuffle(images)
#
#     split_index = int(len(images) * train_ratio)
#     train_images = images[:split_index]
#     val_images = images[split_index:]
#
#     # Ensure directories exist
#     os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
#     os.makedirs(os.path.join(train_dir, 'annotations'), exist_ok=True)
#     os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
#     os.makedirs(os.path.join(val_dir, 'annotations'), exist_ok=True)
#
#     # Move images and annotations
#     for img in train_images:
#         shutil.move(os.path.join(image_folder, img), os.path.join(train_dir, 'images', img))
#         shutil.move(os.path.join(annotation_folder, img.replace('.jpg', '.txt')), os.path.join(train_dir, 'annotations', img.replace('.jpg', '.txt')))
#     for img in val_images:
#         shutil.move(os.path.join(image_folder, img), os.path.join(val_dir, 'images', img))
#         shutil.move(os.path.join(annotation_folder, img.replace('.jpg', '.txt')), os.path.join(val_dir, 'annotations', img.replace('.jpg', '.txt')))


# def split_dataset(image_folder, annotation_folder, train_dir, val_dir, train_ratio=0.8):
#     # Function to find all jpg files recursively
#     def find_images(folder):
#         for root, dirs, files in os.walk(folder):
#             for file in files:
#                 if file.lower().endswith('.jpg'):
#                     yield os.path.join(root, file)  # Yield the full path of the file
#
#     images = list(find_images(image_folder))
#     print(f"Total images found: {len(images)}")  # Debugging output
#
#     if len(images) == 0:
#         print("No images found, check the directory path and file extensions.")
#         return  # Exit if no images are found to avoid further errors
#
#     random.shuffle(images)
#     split_index = int(len(images) * train_ratio)
#     train_images = images[:split_index]
#     val_images = images[split_index:]
#
#     print(f"Training images count: {len(train_images)}")  # Debugging output
#     print(f"Validation images count: {len(val_images)}")  # Debugging output
#
#     # Ensure directories exist
#     os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
#     os.makedirs(os.path.join(train_dir, 'annotations'), exist_ok=True)
#     os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
#     os.makedirs(os.path.join(val_dir, 'annotations'), exist_ok=True)
#
#     # Move images and annotations
#     for img_path in train_images:
#         img_filename = os.path.basename(img_path)
#         annotation_filename = img_filename.replace('.jpg', '.txt')
#         shutil.move(img_path, os.path.join(train_dir, 'images', img_filename))
#         shutil.move(os.path.join(annotation_folder, annotation_filename), os.path.join(train_dir, 'annotations', annotation_filename))
#
#     for img_path in val_images:
#         img_filename = os.path.basename(img_path)
#         annotation_filename = img_filename.replace('.jpg', '.txt')
#         shutil.move(img_path, os.path.join(val_dir, 'images', img_filename))
#         shutil.move(os.path.join(annotation_folder, annotation_filename), os.path.join(val_dir, 'annotations', annotation_filename))


def split_dataset(image_folder, annotation_folder, train_dir, val_dir, train_ratio=0.8):
    def find_images(folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.jpg'):
                    yield os.path.join(root, file)  # Yield the full path of the file

    images = list(find_images(image_folder))
    print(f"Total images found: {len(images)}")  # Debugging output

    random.shuffle(images)
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"Training images count: {len(train_images)}")  # Debugging output
    print(f"Validation images count: {len(val_images)}")  # Debugging output

    for img_path in train_images:
        img_filename = os.path.basename(img_path)
        annotation_filename = img_filename.replace('.jpg', '.txt')
        src_annotation_path = os.path.join(annotation_folder, annotation_filename)
        dst_annotation_path = os.path.join(train_dir, 'annotations', annotation_filename)

        print(f"Moving image {img_path} to {os.path.join(train_dir, 'images', img_filename)}")  # Debugging output
        shutil.move(img_path, os.path.join(train_dir, 'images', img_filename))

        print(f"Trying to move annotation {src_annotation_path} to {dst_annotation_path}")  # Debugging output
        shutil.move(src_annotation_path, dst_annotation_path)

    for img_path in val_images:
        img_filename = os.path.basename(img_path)
        annotation_filename = img_filename.replace('.jpg', '.txt')
        src_annotation_path = os.path.join(annotation_folder, annotation_filename)
        dst_annotation_path = os.path.join(val_dir, 'annotations', annotation_filename)

        print(f"Moving image {img_path} to {os.path.join(val_dir, 'images', img_filename)}")  # Debugging output
        shutil.move(img_path, os.path.join(val_dir, 'images', img_filename))

        print(f"Trying to move annotation {src_annotation_path} to {dst_annotation_path}")  # Debugging output
        shutil.move(src_annotation_path, dst_annotation_path)


# Ensure your paths are correctly defined here
base_dir = r"C:\Users\janva\PycharmProjects\boat\venv"
image_folder = os.path.join(base_dir, 'boat-types-recognition')  # Images are directly under this directory
annotation_folder = os.path.join(base_dir, 'annotations')  # Annotations are expected to be here

train_dir = os.path.join(base_dir, 'train')  # Directory for training split
val_dir = os.path.join(base_dir, 'val')  # Directory for validation split

if __name__ == '__main__':
    split_dataset(image_folder, annotation_folder, train_dir=train_dir, val_dir=val_dir)

# # Example usage
# base_dir = r"C:\Users\janva\PycharmProjects\boat\venv"
# image_folder = os.path.join(base_dir, 'boat-types-recognition')  # Assumes images are directly under this directory
# annotation_folder = os.path.join(base_dir, 'annotations')  # Assumes annotations to be stored here
#
# train_dir = os.path.join(base_dir, 'train')  # Directory for training split
# val_dir = os.path.join(base_dir, 'val')  # Directory for validation split
#
# print("Listing contents of image_folder:", image_folder)
# print(os.listdir(image_folder))
#
# if __name__ == '__main__':
#     # Create annotations in JSON format
#     save_annotations(image_folder, annotation_folder)
#
#     # Convert JSON annotations to YOLO format
#     prepare_dataset(image_folder, annotation_folder)
#
#     # Split into training/validation set
#     split_dataset(image_folder, annotation_folder, train_dir=train_dir, val_dir=val_dir)






