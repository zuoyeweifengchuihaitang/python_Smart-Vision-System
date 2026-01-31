# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from database.operations import delete_face, startup_self_check
from config import FACES_DIR

class CaptureWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(u"人像采集")
        self.captured_frame = None
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.init_ui()

    def init_ui(self):
        v = QVBoxLayout(self)
        self.v_l = QLabel()
        self.v_l.setFixedSize(640, 480)
        v.addWidget(self.v_l)
        
        btn = QPushButton(u"📸 立即拍照") 
        btn.clicked.connect(self.snap)
        v.addWidget(btn)
        
        self.timer.timeout.connect(self.upd)
        self.timer.start(30)

    def upd(self):
        ret, f = self.cap.read()
        if ret:
            self.cur_f = f
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            self.v_l.setPixmap(QPixmap.fromImage(QImage(rgb.data, 640, 480, 640*3, QImage.Format_RGB888)))

    def snap(self): 
        self.captured_frame = self.cur_f.copy() 
        self.cap.release() 
        self.accept()

    def closeEvent(self, event): 
        self.cap.release()
        event.accept()

class ManageDialog(QDialog):
    def __init__(self, f_db, bl, wl, parent=None):
        super().__init__(parent)
        self.db, self.bl, self.wl = f_db, bl, wl
        self.selected_id = None
        self.setWindowTitle(u"人员身份管理中心")
        self.resize(900, 550)
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel(u"人员列表 (红=黑名单 / 绿=白名单)"))
        
        self.lw = QListWidget()
        self.refresh_list()
        self.lw.itemClicked.connect(self.show_p)
        left_layout.addWidget(self.lw)
        layout.addLayout(left_layout, 1)
        
        right_layout = QVBoxLayout()
        self.status_label = QLabel(u"请选择人员")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f8c8d;")
        right_layout.addWidget(self.status_label)

        self.img = QLabel(u"照片预览区域")
        self.img.setFixedSize(400, 350)
        self.img.setStyleSheet("background:#eee;border:2px dashed #999;")
        self.img.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.img)
        
        btn = QPushButton(u"🗑️ 彻底删除该人员")
        btn.clicked.connect(self.confirm)
        btn.setStyleSheet("background-color: #c0392b; color: white; padding: 10px; font-weight: bold;")
        right_layout.addWidget(btn)
        layout.addLayout(right_layout, 2)
    
    def refresh_list(self):
        self.lw.clear()
        for name in self.db.keys():
            is_black = name in self.bl
            tag = "[黑名单]" if is_black else "[白名单]"
            color = QColor("#c0392b") if is_black else QColor("#27ae60")
            item = QListWidgetItem(f"{tag} {name}")
            item.setForeground(color)
            item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            item.setData(Qt.UserRole, name) 
            self.lw.addItem(item)

    def show_p(self, item):
        real_id = item.data(Qt.UserRole)
        self.selected_id = real_id
        if real_id in self.bl:
            self.status_label.setText(f"当前身份：黑名单 (警报)")
            self.status_label.setStyleSheet("color: #c0392b; font-size: 20px; font-weight: bold;")
            sub = 'black'
        else:
            self.status_label.setText(f" 当前身份：白名单 (通行)")
            self.status_label.setStyleSheet("color: #27ae60; font-size: 20px; font-weight: bold;")
            sub = 'white'
            
        found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            p = os.path.join(FACES_DIR, sub, f"{real_id}{ext}")
            if os.path.exists(p): 
                img_data = np.fromfile(p, dtype=np.uint8)
                qimg = QImage.fromData(img_data)
                self.img.setPixmap(QPixmap.fromImage(qimg).scaled(400, 350, Qt.KeepAspectRatio))
                found = True
                break
        
        if not found: 
            self.img.setText(u"⚠️ 数据库有记录但图片已丢失")
            self.img.setPixmap(QPixmap())

    def confirm(self): 
        if not self.selected_id:
            return
        
        role = u"黑名单" if self.selected_id in self.bl else u"白名单"
        msg = f"确定要彻底删除【{role}】人员：\n\n{self.selected_id}\n\n吗？"
        
        if QMessageBox.question(self, u"删除警告", msg) == QMessageBox.Yes: 
            delete_face(self.selected_id)
            self.db, self.bl, self.wl = startup_self_check()
            self.refresh_list()
            self.img.clear()
            self.img.setText(u"已删除")
            self.status_label.setText(u"已删除")
            self.selected_id = None
            QMessageBox.information(self, u"完成", u"该人员档案已彻底移除")