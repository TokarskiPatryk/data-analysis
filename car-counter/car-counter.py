# %%
print(12)

# %%
powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\ProgramData\anaconda3' "

# %% [markdown]
# detect from image

# %%
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['image3.png'], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

# %% [markdown]
# yolo predict model=yolov8n.pt source=''http://192.168.191.50:8080/browserfs.html' imgsz=32

# %%
import cv2

cap = cv2.VideoCapture('rtsp://192.168.191.50:8080/h264.sdp')
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img= cap.read()
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# %% [markdown]
# webcam

# %%
# source from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture('rtsp://192.168.191.50:8080/h264.sdp')


frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

cap.set(3, 400)
cap.set(4, 300)

# Define the vertical line position
success, img = cap.read()
line_position = int(img.shape[1] / 2)

# Initialize variables for counting cars
cars_crossed_l_to_r = 0
cars_crossed_r_to_l = 0
car_ids = []
car_ids_crossed = []
previous_frame=[]

# model
model = YOLO("yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness) #classNames[cls]+str(box.id)

            # Count cars crossing the line
            if x1 < line_position and x2 > line_position:
                if box.id not in car_ids:
                    car_ids.append(box.id)
                    if y2 < img.shape[0]/2: #changed
                        cars_crossed_r_to_l += 1                   
                    else:
                        cars_crossed_l_to_r += 1
                        car_ids_crossed.append(box.id)




    # Display the number of cars crossed     
    cv2.putText(img, "Cars Crossed l to r: " + str(cars_crossed_l_to_r), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Cars Crossed r to l: " + str(cars_crossed_r_to_l), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Draw the vertical line
    cv2.line(img, (line_position, 0), (line_position, img.shape[0]), (0, 255, 0), 2)


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







