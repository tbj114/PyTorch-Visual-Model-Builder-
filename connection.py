#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connection - Visual connections between model nodes
"""

import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Connection(QGraphicsItem):
    def __init__(self, output_port, input_port, temporary=False):
        super().__init__()
        self.output_port = output_port
        self.input_port = input_port
        self.temporary = temporary
        self.end_point = None  # For temporary connections
        
        # Visual properties
        self.line_width = 3
        self.selected_width = 4
        self.arrow_size = 8
        
        # Colors
        self.normal_color = QColor(150, 150, 150)
        self.selected_color = QColor(100, 150, 255)
        self.active_color = QColor(100, 255, 150)
        self.temporary_color = QColor(255, 255, 100)
        
        # State
        self.is_active = True
        
        # Set flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, not temporary)
        self.setZValue(-1)  # Draw behind nodes
        
        # Add connection to ports
        if not temporary and output_port and input_port:
            output_port.add_connection(self)
            input_port.add_connection(self)
            
    def boundingRect(self):
        """Return the bounding rectangle"""
        if self.temporary and self.end_point:
            start = self.output_port.get_scene_position()
            end = self.end_point
        elif self.output_port and self.input_port and not self.temporary:
            start = self.output_port.get_scene_position()
            end = self.input_port.get_scene_position()
        else:
            return QRectF()
            
        # Convert to local coordinates
        start = self.mapFromScene(start)
        end = self.mapFromScene(end)
        
        # Calculate bounding rect with some padding
        left = min(start.x(), end.x()) - 10
        top = min(start.y(), end.y()) - 10
        width = abs(end.x() - start.x()) + 20
        height = abs(end.y() - start.y()) + 20
        
        return QRectF(left, top, width, height)
        
    def paint(self, painter, option, widget):
        """Paint the connection"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get start and end points
        if self.temporary and self.end_point:
            start = self.mapFromScene(self.output_port.get_scene_position())
            end = self.mapFromScene(self.end_point)
        elif self.output_port and self.input_port and not self.temporary:
            start = self.mapFromScene(self.output_port.get_scene_position())
            end = self.mapFromScene(self.input_port.get_scene_position())
        else:
            return
            
        # Choose color and width
        if self.temporary:
            color = self.temporary_color
            width = self.line_width
        elif self.isSelected():
            color = self.selected_color
            width = self.selected_width
        elif self.is_active:
            color = self.active_color
            width = self.line_width
        else:
            color = self.normal_color
            width = self.line_width
            
        # Create bezier curve path
        path = self.create_bezier_path(start, end)
        
        # Draw connection line
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw arrow at the end (for non-temporary connections)
        if not self.temporary:
            self.draw_arrow(painter, start, end, color)
            
        # Draw data flow animation (optional)
        if self.is_active and not self.temporary:
            self.draw_flow_animation(painter, path)
            
    def create_bezier_path(self, start, end):
        """Create a smooth bezier curve path"""
        path = QPainterPath()
        path.moveTo(start)
        
        # Calculate control points for smooth curve
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        
        # Control point offset (creates the curve)
        ctrl_offset = max(abs(dx) * 0.5, 50)
        
        ctrl1 = QPointF(start.x() + ctrl_offset, start.y())
        ctrl2 = QPointF(end.x() - ctrl_offset, end.y())
        
        path.cubicTo(ctrl1, ctrl2, end)
        return path
        
    def draw_arrow(self, painter, start, end, color):
        """Draw an arrow at the end of the connection"""
        # Calculate arrow direction
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
            
        # Normalize direction
        dx /= length
        dy /= length
        
        # Calculate arrow points
        arrow_length = self.arrow_size
        arrow_width = self.arrow_size * 0.6
        
        # Arrow tip is slightly before the end point
        tip = QPointF(end.x() - dx * 8, end.y() - dy * 8)
        
        # Arrow base points
        base1 = QPointF(
            tip.x() - dx * arrow_length + dy * arrow_width,
            tip.y() - dy * arrow_length - dx * arrow_width
        )
        base2 = QPointF(
            tip.x() - dx * arrow_length - dy * arrow_width,
            tip.y() - dy * arrow_length + dx * arrow_width
        )
        
        # Draw arrow
        arrow_path = QPainterPath()
        arrow_path.moveTo(tip)
        arrow_path.lineTo(base1)
        arrow_path.lineTo(base2)
        arrow_path.closeSubpath()
        
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPath(arrow_path)
        
    def draw_flow_animation(self, painter, path):
        """Draw animated data flow indicators"""
        # This could be enhanced with actual animation
        # For now, just draw some dots along the path
        
        # Get current time for animation
        import time
        t = time.time() * 2  # Animation speed
        
        # Draw moving dots
        for i in range(3):
            progress = ((t + i * 0.3) % 1.0)
            point = path.pointAtPercent(progress)
            
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(point, 2, 2)
            
    def update_end_point(self, point):
        """Update end point for temporary connections"""
        self.end_point = point
        self.update()
        
    def update_path(self):
        """Update the connection path"""
        self.prepareGeometryChange()
        self.update()
        
    def set_active(self, active):
        """Set the active state of the connection"""
        self.is_active = active
        self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.RightButton:
            # Show context menu
            self.show_context_menu(event.screenPos())
        else:
            super().mousePressEvent(event)
            
    def show_context_menu(self, position):
        """Show context menu for connection"""
        menu = QMenu()
        
        delete_action = menu.addAction("Delete Connection")
        delete_action.triggered.connect(self.delete_connection)
        
        # Show tensor shape info if available
        if (self.output_port and self.output_port.tensor_shape and
            self.input_port and self.input_port.tensor_shape):
            menu.addSeparator()
            info_action = menu.addAction(
                f"Shape: {self.output_port.tensor_shape} → {self.input_port.tensor_shape}"
            )
            info_action.setEnabled(False)
            
        menu.exec_(position)
        
    def delete_connection(self):
        """Delete this connection"""
        # Remove from ports
        if self.output_port:
            self.output_port.remove_connection(self)
        if self.input_port:
            self.input_port.remove_connection(self)
            
        # Remove from scene
        if self.scene():
            self.scene().removeItem(self)
            
        # Notify canvas
        if self.scene() and hasattr(self.scene(), 'views'):
            views = self.scene().views()
            if views and hasattr(views[0], 'model_modified'):
                views[0].model_modified.emit()
                
    def serialize(self):
        """Serialize connection to dictionary"""
        if self.temporary or not self.output_port or not self.input_port:
            return None
            
        # Find port indices in their parent nodes
        output_node = self.output_port.parent_node
        input_node = self.input_port.parent_node
        
        output_port_index = output_node.output_ports.index(self.output_port)
        input_port_index = input_node.input_ports.index(self.input_port)
        
        return {
            'type': 'Connection',
            'output_node_id': output_node.node_id,
            'output_port_index': output_port_index,
            'input_node_id': input_node.node_id,
            'input_port_index': input_port_index,
            'is_active': self.is_active
        }
        
    @staticmethod
    def deserialize(data, nodes):
        """Deserialize connection from dictionary"""
        # Find nodes by ID
        output_node = None
        input_node = None
        
        for node in nodes:
            if node.node_id == data['output_node_id']:
                output_node = node
            elif node.node_id == data['input_node_id']:
                input_node = node
                
        if not output_node or not input_node:
            return None
            
        # Get ports
        try:
            output_port = output_node.output_ports[data['output_port_index']]
            input_port = input_node.input_ports[data['input_port_index']]
        except IndexError:
            return None
            
        # Create connection
        connection = Connection(output_port, input_port)
        connection.is_active = data.get('is_active', True)
        
        return connection