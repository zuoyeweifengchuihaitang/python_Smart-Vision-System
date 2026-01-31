# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import SmartVisionApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SmartVisionApp()
    win.show()
    sys.exit(app.exec_())