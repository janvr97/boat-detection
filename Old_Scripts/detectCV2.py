import torch
import cv2

# Load the trained model
weights_path = r'C:\Users\janva\PycharmProjects\boat\runs\train\exp2\weights\best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Path to the test video
video_path = r'C:\Users\janva\PycharmProjects\boat\videos\test_video_0.mp4'
cap = cv2.VideoCapture(video_path)


while True:
    ret, img = cap.read()
    if not ret:
        break

    # Perform detection on image
    results = model(img)

    # Convert detected results to pandas DataFrame
    data_frame = results.pandas().xyxy[0]

    # Draw bounding boxes and labels
    for index, row in data_frame.iterrows():
        # Find the coordinates of the bounding box
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Find label name and confidence score
        label = row['name']
        conf = row['confidence']
        text = f'{label} {conf:.2f}'

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
