# 👁️ Smart Vision System 

> 一个基于 PyQt5 和深度学习的智能化安防监控与数据分析平台。集成了多模态人脸识别、黑名单预警、行人流量统计及人群密度热力分析功能。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![MediaPipe](https://img.shields.io/badge/AI-MediaPipe-teal)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-orange)


---

## 📖 项目简介

**Smart Vision System** 是为应用型课程设计开发的桌面端视觉管理系统。旨在解决传统监控“只录不管”的痛点，通过计算机视觉技术实现主动式的安全管控。

系统采用 **MVC 架构** 设计，结合了 **MediaPipe** 的高实时性与 **YOLOv8** 的高鲁棒性，能够在普通 PC 上流畅运行，适用于校园门禁、办公楼宇、商场客流分析等场景。

## ✨ 核心功能

### 1. 👮 多模态人脸识别 & 门禁管理
- **三维识别**：支持实时摄像头、本地图片、视频文件三种输入源。
- **级联检测策略**：优先使用 MediaPipe 进行毫秒级初筛，YOLOv8-face 进行二次补检，有效解决侧脸、遮挡漏检问题。
- **黑名单报警**：识别到黑名单人员时，自动触发 **UI 红框闪烁 + 蜂鸣器报警 + TTS 语音播报**。

### 2. 🚶 交互式视频流量统计
- **自定义警戒线**：用户可在视频画面上鼠标点击绘制虚拟警戒线。
- **双向计数**：基于质心追踪算法 (CentroidTracker)，自动统计 IN (进入) 和 OUT (离开) 的人数。

### 3. 👥 人群密度热力分析
- **ROI 区域监控**：支持鼠标框选感兴趣区域 (ROI)。
- **密度告警**：区域内人数超过设定阈值（如 10人）时触发警报。
- **动态热力图**：实时生成 Crowd Heatmap，直观展示拥挤程度。

### 4. 📊 数据可视化看板
- **实时日志**：界面侧边栏实时滚动显示识别记录。
- **图表分析**：内置 Matplotlib 看板，展示黑白名单比例、各渠道数据统计。
- **报表导出**：支持一键导出 Excel 考勤表，异常记录自动标红。

---

## 🛠️ 技术栈

- **语言**: Python 3.12
- **界面**: PyQt5 (Qt Designer)
- **视觉算法**:
  - OpenCV (视频流处理)
  - MediaPipe (人脸检测)
  - YOLOv8n / YOLOv8n-face (行人与人脸检测)
  - FaceNet (InceptionResnetV1 特征提取)
- **数据处理**: Pandas, NumPy
- **其他**: Pyttsx3 (语音合成), Winsound (系统音效)

---
## 📂 项目结构 (Project Structure)

本项目采用模块化 (MVC) 架构设计，逻辑清晰，易于维护。

```text
SmartVisionSystem/
│
├── core/                      # [核心算法层] 存放 AI 模型与算法逻辑
│   ├── __init__.py
│   ├── models.py              # 模型初始化 (加载 YOLOv8, MediaPipe, FaceNet)
│   ├── recognition.py         # 人脸识别核心逻辑 (特征提取、比对)
│   └── tracking.py            # 物体追踪 (CentroidTracker) 与 流量统计逻辑
│
├── database/                  # [数据层] 负责数据持久化
│   ├── __init__.py
│   ├── operations.py          # 数据库核心操作 (增删改查、自检、物理同步)
│   └── logger.py              # 访问日志 (.csv) 的读写与统计分析
│
├── ui/                        # [视图层] PyQt5 界面与交互
│   ├── __init__.py
│   ├── main_window.py         # 主程序窗口逻辑
│   ├── worker.py              # [核心控制器] 多线程视觉处理引擎 (QThread)
│   ├── dialogs.py             # 弹窗组件 (注册窗口、管理窗口)
│   └── widgets.py             # 自定义 UI 控件 (如点击反馈 Label)
│
├── data/                      # [资源目录] (自动生成，无需手动创建)
│   ├── faces/                 # 存放录入的人脸物理图片 (.jpg)
│   │   ├── black/             # 黑名单照片
│   │   └── white/             # 白名单照片
│   ├── db/                    # 存放特征数据库文件
│   │   └── face_db.pkl        # 人脸特征向量索引文件
│   └── access_log.csv         # 访问与报警日志
│
├── config.py                  # 全局配置文件 (字体路径、阈值设置等)
├── main.py                    # [程序入口] 启动文件
├── requirements.txt           # 项目依赖库列表
└── README.md                  # 项目说明文档


```
---

## 🚀 快速开始

### 1. 环境要求
- Windows 10/11 (推荐，以支持语音和系统蜂鸣)
- Python 3.8+
- 建议使用 GPU (CUDA) 以获得最佳性能，但也完全支持 CPU 运行。

### 2. 安装依赖
克隆项目到本地并安装所需库：

```bash
git clone [https://github.com/你的用户名/Smart-Vision-System.git](https://github.com/你的用户名/Smart-Vision-System.git)
cd Smart-Vision-System

# 推荐使用清华源加速下载
pip install -r requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
