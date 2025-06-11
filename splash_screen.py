from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        # 创建一个空白的QPixmap作为背景
        pixmap = QPixmap(400, 200)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)
        
        # 创建一个widget来容纳进度条和标签
        self.content_widget = QWidget(self)
        layout = QVBoxLayout(self.content_widget)
        
        # 创建标签显示加载状态
        self.status_label = QLabel("正在初始化...", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 创建进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 设置widget的位置
        self.content_widget.setGeometry(10, 60, 380, 100)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QSplashScreen {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 10px;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
                font-weight: bold;
            }
        """)
    
    def set_progress(self, value, status_text):
        """更新进度条和状态文本"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status_text)
        self.repaint()  # 强制重绘，确保显示更新