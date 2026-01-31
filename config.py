# -*- coding: utf-8 -*-
import os

# 字体配置
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # 黑体备用
if not os.path.exists(FONT_PATH):
    FONT_PATH = None

# 路径配置
DB_PATH = 'data/db/face_db.pkl'
FACES_DIR = 'data/faces'
LOG_PATH = 'data/access_log.csv'