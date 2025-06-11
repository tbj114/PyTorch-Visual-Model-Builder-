#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Background - Provides a grid background for the canvas
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class GridBackground(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.grid_size = 20
        self.grid_color = QColor(60, 60, 60)
        self.major_grid_color = QColor(80, 80, 80)
        self.major_grid_interval = 5  # Every 5th line is major
        
        # Set Z-value to be behind everything
        self.setZValue(-1000)
        
    def boundingRect(self):
        """Return a very large bounding rectangle"""
        return QRectF(-10000, -10000, 20000, 20000)
        
    def paint(self, painter, option, widget):
        """Paint the grid"""
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        # Get the visible rect
        rect = option.exposedRect
        
        # Calculate grid bounds
        left = int(rect.left() // self.grid_size) * self.grid_size
        top = int(rect.top() // self.grid_size) * self.grid_size
        right = int(rect.right() // self.grid_size + 1) * self.grid_size
        bottom = int(rect.bottom() // self.grid_size + 1) * self.grid_size
        
        # Draw vertical lines
        x = left
        line_count = 0
        while x <= right:
            if line_count % self.major_grid_interval == 0:
                painter.setPen(QPen(self.major_grid_color, 1))
            else:
                painter.setPen(QPen(self.grid_color, 1))
                
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += self.grid_size
            line_count += 1
            
        # Draw horizontal lines
        y = top
        line_count = 0
        while y <= bottom:
            if line_count % self.major_grid_interval == 0:
                painter.setPen(QPen(self.major_grid_color, 1))
            else:
                painter.setPen(QPen(self.grid_color, 1))
                
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += self.grid_size
            line_count += 1
            
    def set_grid_size(self, size):
        """Set the grid size"""
        self.grid_size = size
        self.update()
        
    def set_grid_color(self, color):
        """Set the grid color"""
        self.grid_color = color
        self.update()
        
    def set_major_grid_color(self, color):
        """Set the major grid color"""
        self.major_grid_color = color
        self.update()