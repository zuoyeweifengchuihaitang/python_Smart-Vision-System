# -*- coding: utf-8 -*-
import torch
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. MediaPipe 人脸检测
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.15)

# 2. FaceNet 特征提取
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 3. YOLO 模型（只是辅助作用，帮助我们找到mp没找到的人脸）

yolo_person = YOLO('yolov8n.pt')      # 行人检测
yolo_face = YOLO('yolov8n-face.pt')   # 辅助人脸检测