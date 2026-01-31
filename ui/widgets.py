# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, Qt

class ClickLabel(QLabel):
    clicked_pos = pyqtSignal(int, int)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: 
            self.clicked_pos.emit(event.x(), event.y())