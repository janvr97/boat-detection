import os
import random
import shutil
import subprocess


# 1. Download images from Kaggle
def download_images():
    dataset = 'kunalgupta2616/boat-types-recognition'
    images_folder = 'C:/Users/janva/PycharmProjects/boat/images'
    os.makedirs(images_folder, exist_ok=True)
    os.system(f'kaggle datasets download -d {dataset} -p {images_folder} --unzip')


# 2. Split images and annotations into train and val folders
def split_dataset(base_dir='C:/Users/janva/PycharmProjects/boat'):
    image_folder = os.path.join(base_dir, 'images')
    annotation_folder = os.path.join(base_dir, 'annotations')
    train_folder = os.path.join(base_dir, 'train')
    val_folder = os.path.join(base_dir, 'val')

    # Create train/val folders
    for folder in [train_folder, val_folder]:
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

    images = os.listdir(image_folder)
    random.shuffle(images)
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    def move_files(file_list, dest_folder):
        for file in file_list:
            shutil.copy(os.path.join(image_folder, file), os.path.join(dest_folder, 'images', file))
            annotation_file = file.replace('.jpg', '.txt')
            shutil.copy(os.path.join(annotation_folder, annotation_file),
                        os.path.join(dest_folder, 'labels', annotation_file))

    move_files(train_images, train_folder)
    move_files(val_images, val_folder)


# 3. Download YOLOv5 pre-trained model
def download_yolov5():
    if not os.path.exists('yolov5'):
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    os.chdir('yolov5')
    os.system('pip install -r requirements.txt')


# 4. Train and validate the model
def train_yolov5():
    os.chdir('C:/Users/janva/PycharmProjects/boat/yolov5')
    train_command = "python train.py --img 640 --batch 16 --epochs 50 --data ../data.yaml --weights yolov5s.pt --device 0"
    subprocess.run(train_command, shell=True)
    val_command = "python val.py --data ../data.yaml --weights runs/train/exp/weights/best.pt --device 0"
    subprocess.run(val_command, shell=True)


# 5. Use the trained model on a video file
def detect_video():
    video_path = 'C:/Users/janva/PycharmProjects/boat/videos/test_video_0.mp4'
    detect_command = f"python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source {video_path} --view-img --nosave --device 0"
    subprocess.run(detect_command, shell=True)

if __name__ == '__main__':
    download_images()
    split_dataset()
    download_yolov5()
    train_yolov5()
    detect_video()
