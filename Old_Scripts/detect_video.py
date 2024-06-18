import sys
import os
import torch
import cv2
from pathlib import Path

# Dynamically add the yolov5 directory to sys.path
project_dir = r"C:\Users\janva\PycharmProjects\boat\yolov5"
yolov5_dir = os.path.join(project_dir, '..yolov5')
sys.path.append(yolov5_dir)

sys.path.append(r'C:\Users\janva\PycharmProjects\boat\yolov5')

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, [0, 2]].clamp_(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]].clamp_(0, img0_shape[0])  # y1, y2
    return coords

def check_for_gpu():
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used for training.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Check if MPS (Metal Performance Shaders) is available for macOS
        print("MPS is available. GPU will be used for training.")
        return torch.device("mps")
    else:
        print("CUDA is not available. CPU will be used for training.")
        return torch.device("cpu")

def run_real_time_detection(weights='runs/train/exp2/weights/best.pt', source=0, imgsz=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
    device = check_for_gpu()
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model.warmup(imgsz=(1 if pt else 1, 3, imgsz, imgsz))  # warmup

    # Check if the video source is valid
    if not os.path.isfile(source):
        print(f"Error: The source file {source} does not exist.")
        return

    # Load video source
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, imgsz, imgsz))  # warmup

    # Initialize video stream
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        pred = model(img, augment=False, visualize=False)
        t2 = time_sync()

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()

            annotator = Annotator(im0, line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            # Stream results
            cv2.imshow(str(path), im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

# Example usage for real-time detection
video_path = r'../videos/test_video_0.mp4'
run_real_time_detection(weights='runs/train/exp2/weights/best.pt', source=video_path)  # source=0 for webcam
