# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from core.models import device, face_detection, resnet, yolo_face

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def extract_embeddings(image):
    """级联检测 + 特征提取"""
    if isinstance(image, str):
        img_array = np.fromfile(image, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return []
    else:
        img = image.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # MediaPipe 检测
    mp_results = face_detection.process(img_rgb)
    mp_bboxes = []
    if mp_results.detections:
        for detection in mp_results.detections:
            if detection.score[0] < 0.15: continue
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            mp_bboxes.append((x1, y1, x2, y2))

    # YOLO 检测
    yolo_results = yolo_face(img, conf=0.3, verbose=False)[0]
    yolo_bboxes = [tuple(box.astype(int)) for box in yolo_results.boxes.xyxy.cpu().numpy()]

    # 融合去重
    final_bboxes = list(mp_bboxes)
    for y_box in yolo_bboxes:
        is_duplicate = any(compute_iou(y_box, m_box) > 0.4 for m_box in mp_bboxes)
        if not is_duplicate:
            final_bboxes.append(y_box)

    if not final_bboxes:
        return []

    # 提取特征
    embeddings = []
    for bbox in final_bboxes:
        x1, y1, x2, y2 = bbox
        expand_w = int((x2 - x1) * 0.1)
        expand_h = int((y2 - y1) * 0.1)
        cx1 = max(0, x1 - expand_w)
        cy1 = max(0, y1 - expand_h)
        cx2 = min(w, x2 + expand_w)
        cy2 = min(h, y2 + expand_h)

        face_crop = img[cy1:cy2, cx1:cx2]
        if face_crop.size == 0: continue

        face_crop = cv2.resize(face_crop, (160, 160))
        face_tensor = torch.from_numpy(face_crop.transpose(2, 0, 1)).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy().flatten()

        embeddings.append((emb, (cx1, cy1, cx2, cy2), 1.0))

    return embeddings