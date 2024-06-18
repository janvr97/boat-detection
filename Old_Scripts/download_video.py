import requests
import os

def download_video(url, output_path, video_name="test_video"):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        video_count = len(os.listdir(output_path))
        file_path = os.path.join(output_path, f"{video_name}_{video_count}.mp4")
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Video downloaded and saved as {file_path}")
    else:
        print("Failed to download video")

# Example usage:
video_url = 'https://videos.pexels.com/video-files/5667410/5667410-hd_1366_720_30fps.mp4'  # Replace with your video URL
output_path = r'C:\Users\janva\PycharmProjects\boat\videos'
os.makedirs(output_path, exist_ok=True)
download_video(video_url, output_path)
