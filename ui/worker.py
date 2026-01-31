# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import winsound
from PyQt5.QtCore import QThread, pyqtSignal

# 核心模型导入
from core.models import yolo_person, yolo_face
from core.recognition import extract_embeddings
from core.tracking import CentroidTracker, PedestrianFlowManager
from database.logger import log_unified

# 尝试导入语音库，如果失败则禁用，防止报错
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class VisionEngine(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    flow_ready = pyqtSignal(dict)
    log_signal = pyqtSignal(str, str, str)
    count_ready = pyqtSignal(int)

    def __init__(self, source=0, face_db=None, bl=None, wl=None, mode='face', flow_config=None, density_config=None):
        super().__init__()
        self._active = True
        self.source = source
        self.mode = mode
        self.face_db, self.bl, self.wl = face_db, bl, wl
        self.tracker = CentroidTracker()
        self.flow_mgr = PedestrianFlowManager()
        self.log_cd = {}
        self.density_config = density_config or {}
        self.density_threshold = self.density_config.get('threshold', 10)
        self.alert_interval = self.density_config.get('alert_interval', 5)
        self.interval_start = time.time()
        self.max_count = 0

        # 设置源名称
        if mode == 'flow': self.src = u"流量统计"
        elif mode == 'density': self.src = u"人群密度统计"
        elif isinstance(source, int): self.src = u"实时监控"
        elif isinstance(source, str): self.src = u"视频分析"
        else: self.src = u"图片识别"

        # 流量线设置
        if flow_config:
            self.flow_mgr.set_line(flow_config['p1'], flow_config['p2'])
            self.flow_mgr.in_side_sign = flow_config['sign']
        
        # 初始化语音引擎 (懒加载 + 异常保护)
        self.ts = None
        if TTS_AVAILABLE:
            try:
                # 某些系统初始化 init 会导致 Crash，这里加保护
                self.ts = pyttsx3.init()
                self.ts.setProperty('rate', 150)
            except Exception as e:
                print(f"⚠️ 语音模块初始化失败 (已自动禁用): {e}")
                self.ts = None

    def _get_identity(self, emb):
        """统一处理身份比对逻辑，返回 (name, color, status)"""
        name, color, status = "Stranger", (0, 165, 255), "Stranger"
        if self.face_db:
            sims = {p: np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e) + 1e-8)
                    for p, e in self.face_db.items()}
            mid = max(sims, key=sims.get, default=None)
            if mid is not None:
                score = sims[mid]
                if score > 0.75:
                    name = mid
                    color = (0, 255, 0) if mid in self.wl else (0, 0, 255)
                    status = u"白名单" if mid in self.wl else u"黑名单"
                    current_time = time.time()
                    if mid not in self.log_cd or (current_time - self.log_cd[mid] > 10):
                        self.log_signal.emit(self.src, mid, status)
                        log_unified(self.src, mid, status, f"Sim:{score:.2f}")
                        
                        # 触发警报
                        if status == u"黑名单":
                            winsound.Beep(1000, 500)
                            # 尝试语音播报 (如果引擎可用)
                            if self.ts:
                                try:
                                    self.ts.say(f"发现黑名单 {name}")
                                    self.ts.runAndWait()
                                except:
                                    pass # 语音失败不影响主程序

                        self.log_cd[mid] = current_time
        return name, color, status

    def run(self):
        if isinstance(self.source, np.ndarray):
            frame = self.source.copy()
            self.process_frame(frame)
            self.frame_ready.emit(frame)
            return

        cap = cv2.VideoCapture(self.source)
        while self._active:
            ret, frame = cap.read()
            if not ret: break
            
            self.process_frame(frame)
            self.frame_ready.emit(frame)
            
            if isinstance(self.source, str):
                time.sleep(0.03)  # 视频文件播放控制速度
        cap.release()

    def process_frame(self, frame):

        if self.mode == 'flow' or self.mode == 'density':
            res = yolo_person(frame, classes=[0], verbose=False, conf=0.3)[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            
            if self.mode == 'flow':
                rects = [box.astype(int) for box in boxes]
                for (x1,y1,x2,y2) in rects: cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
                objs = self.tracker.update(rects)
                for tid, cent in objs.items():
                    cross = self.flow_mgr.check_crossing(tid, cent)
                    if cross: 
                        msg = u"越线进入" if cross=="IN" else u"越线离开"
                        self.log_signal.emit(self.src, f"ID:{tid}", msg)
                        log_unified(self.src, f"ID:{tid}", "Flow", msg)
                    cv2.circle(frame, tuple(cent), 5, (0,255,255), -1)
                st = self.flow_mgr.get_status()
                self.flow_ready.emit(st)
                p1, p2 = self.flow_mgr.line_pts
                cv2.line(frame, p1, p2, (0,0,255), 3)
                cv2.arrowedLine(frame, p1, p2, (0,255,0), 3)
                cv2.putText(frame, f"IN:{st['in']} OUT:{st['out']}", (20,60), 0, 1.2, (0,255,0), 3)
            
            elif self.mode == 'density':
                count = 0
                roi = self.density_config.get('roi')
                if roi:
                    rx1, ry1, rx2, ry2 = roi
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                            count += 1
                else:
                    count = len(boxes)
                self.count_ready.emit(count)

                now = time.time()
                if count > self.max_count:
                    self.max_count = count
                if now - self.interval_start >= self.alert_interval:
                    msg = f"间隔内最大人数: {self.max_count} (阈值: {self.density_threshold})"
                    self.log_signal.emit(self.src, "密度统计", msg)
                    if self.max_count > self.density_threshold:
                        alert = f"【密度告警】{msg} - 超标！"
                        self.log_signal.emit(self.src, "密度告警", alert)
                        winsound.Beep(2500, 1200)
                    self.interval_start = now
                    self.max_count = 0
        else:
            # 人脸识别模式
            faces_mp = extract_embeddings(frame)
            detected_centers = set()

            for emb, bbox, score in faces_mp:
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                detected_centers.add(center)
                name, color, status = self._get_identity(emb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), 0, 0.8, color, 2)

            if not faces_mp:
                res_face = yolo_face(frame, verbose=False, conf=0.3)[0]
                boxes_face = res_face.boxes.xyxy.cpu().numpy().astype(int)

                for box in boxes_face:
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if center in detected_centers: continue

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0: continue
                    emb_list = extract_embeddings(face_crop)
                    if not emb_list: continue
                    
                    emb, bbox_yolo, conf = emb_list[0]
                    name, color, status = self._get_identity(emb)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), 0, 0.8, color, 2)

    def stop(self): 
        self._active = False 
        self.wait()