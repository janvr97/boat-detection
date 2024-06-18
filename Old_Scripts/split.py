import os
import shutil
import random

# to train, enter in terminal:
#python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --cache

def split_dataset(image_folder, annotation_folder, train_dir, val_dir, train_ratio=0.8):
    images = os.listdir(image_folder)
    random.shuffle(images)  # Randomize the order to split data randomly

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Ensure the directories are clean before splitting
    clear_directory(train_dir)
    clear_directory(val_dir)

    # Move images and annotations
    move_files(train_images, image_folder, train_dir, annotation_folder)
    move_files(val_images, image_folder, val_dir, annotation_folder)

def move_files(files, src_image_folder, dest_folder, src_annotation_folder):
    os.makedirs(os.path.join(dest_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'annotations'), exist_ok=True)
    for file in files:
        image_src = os.path.join(src_image_folder, file)
        image_dest = os.path.join(dest_folder, 'images', file)
        annotation_src = os.path.join(src_annotation_folder, file.replace('.jpg', '.txt'))
        annotation_dest = os.path.join(dest_folder, 'annotations', file.replace('.jpg', '.txt'))
        shutil.copy(image_src, image_dest)
        shutil.copy(annotation_src, annotation_dest)

def clear_directory(directory):
    for subdir in ['images', 'annotations']:
        folder = os.path.join(directory, subdir)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            os.unlink(file_path)

base_dir = r"C:\Users\janva\PycharmProjects\boat\venv"
image_folder = os.path.join(base_dir, 'images')
annotation_folder = os.path.join(base_dir, 'annotations')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

if __name__ == '__main__':
    split_dataset(image_folder, annotation_folder, train_dir, val_dir)
