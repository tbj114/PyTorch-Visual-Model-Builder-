#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Visual Model Builder
A modern, visual interface for building PyTorch models through drag-and-drop
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from model_builder import ModelBuilderWindow
import traceback

def log_uncaught_exceptions(ex_type, ex_value, ex_traceback):
    log_file = "error.log"
    with open(log_file, "a") as f:
        f.write(f"Timestamp: {QDateTime.currentDateTime().toString(Qt.ISODate)}\n")
        f.write(f"Exception Type: {ex_type.__name__}\n")
        f.write(f"Exception Value: {ex_value}\n")
        f.write("Traceback:\n")
        traceback.print_exception(ex_type, ex_value, ex_traceback, file=f)
        f.write("\n" + "-"*80 + "\n\n")
    sys.__excepthook__(ex_type, ex_value, ex_traceback)

def main():
    sys.excepthook = log_uncaught_exceptions
    app = QApplication(sys.argv)
    app.setApplicationName("PyTorch Visual Model Builder")
    app.setApplicationVersion("1.0.0")
    
    # 创建并显示启动画面
    from splash_screen import SplashScreen
    splash = SplashScreen()
    splash.show()
    
    # 确保启动画面显示
    app.processEvents()
    
    # 模拟加载过程
    splash.set_progress(10, "正在初始化应用程序...")
    app.processEvents()
    
    # Set application style
    app.setStyle('Fusion')
    splash.set_progress(30, "正在设置应用程序样式...")
    app.processEvents()
    
    # Apply dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # Create and show main window
    splash.set_progress(70, "正在创建主窗口...")
    app.processEvents()
    
    window = ModelBuilderWindow()
    
    splash.set_progress(90, "正在加载界面组件...")
    app.processEvents()
    
    window.show()
    
    splash.set_progress(100, "加载完成")
    app.processEvents()
    
    # 延迟关闭启动画面
    QTimer.singleShot(500, splash.close)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()