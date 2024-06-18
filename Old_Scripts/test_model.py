import subprocess

def test_model(video_path, weights_path, output_dir):
    command = [
        'python', 'detect.py',
        '--weights', weights_path,
        '--source', video_path,
        '--conf', '0.25',
        '--name', output_dir
    ]
    subprocess.run(command, cwd=r'C:\Users\janva\PycharmProjects\boat\yolov5')

video_path = r'C:\Users\janva\PycharmProjects\boat\test_video.mp4'
weights_path = r'runs/train/exp2/weights/last.pt'
output_dir = 'test_output'

test_model(video_path, weights_path, output_dir)


""" Command Terminal Examples"""
#python detect.py --weights runs/train/exp2/weights/best.pt --conf 0.25 --img-size 420 --source C:\Users\janva\PycharmProjects\boat\videos\test_video_0.mp4
#python detect.py --weights runs/train/exp2/weights/best.pt --conf 0.25 --source C:\Users\janva\PycharmProjects\boat\videos\test_video_3.mp4 --view-img --nosave --vid-stride 2 --device 0


