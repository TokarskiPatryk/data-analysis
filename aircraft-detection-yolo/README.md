# Aircraft Detection with YOLOv8

Fine-tune YOLOv8 to detect 20 types of aircraft in satellite imagery.

## Dataset

3,821 satellite images with 22,341 labelled objects across 20 aircraft classes (A1–A20). Labels were originally in PascalVOC XML format and converted to YOLO TXT format.

Split: 72% train (2,778) / 14% val (544) / 14% test (521).

Kaggle dataset: https://www.kaggle.com/datasets/tokarooo/aircraft-detection-with-yolov8

## Models trained

Four variants were compared, all trained for 100 epochs at 640×640 on Kaggle (P100 GPU):

| ID | Base model | Frozen layers |
|---|---|---|
| model1 | YOLOv8n | No |
| model2 | YOLOv8n | Yes |
| model3 | YOLOv8m | No |
| model4 | YOLOv8m | Yes |

Trained weights are stored in `results/modelX/best.pt`. Base weights used for fine-tuning are in `yolo-default-models/`.

## Evaluation metrics

mAP50 and mAP50-95 (reported by `ultralytics` during training).

## Tech stack

Python · ultralytics (YOLOv8) · Quarto (report)

## Report

https://tokarskipatryk.github.io/data-analysis/aircraft-detection-yolo/project.html
