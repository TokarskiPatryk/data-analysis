# Car Counter

Detect and count vehicles in a live camera stream using object detection models.

## Approach

Two notebooks explore different libraries:

| File | Library | Model |
|---|---|---|
| `main.ipynb` | ImageAI | YOLOv3 |
| `main2.ipynb` | ultralytics | YOLOv8 |
| `car-counter.py` | ultralytics | YOLOv8n (pretrained) |

Video input is an RTSP stream (`cv2.VideoCapture`). Detected objects are overlaid on frames in real time.

## Tech stack

Python · ultralytics · ImageAI · OpenCV (cv2)

> Note: `main.ipynb` requires a working ImageAI + YOLOv3 setup. The notebook currently shows a `ModuleNotFoundError` — the `models` module from an older YOLOv3 implementation is missing.
