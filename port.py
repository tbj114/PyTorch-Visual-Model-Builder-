#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Port - Connection points for model nodes
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Port(QGraphicsItem):
    def __init__(self, port_type, parent_node):
        super().__init__()
        self.port_type = port_type  # 'input' or 'output'
        self.parent_node = parent_node
        self.is_port = True  # Identifier for canvas
        self.is_connected = False
        self.connections = []
        self.port_radius = 6
        self.snap_radius = 30  # 增加吸附半径，提升连接体验
        self.hover_radius = 25  # 悬停检测半径
        self.tensor_shape = None
        
        # Set flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)
        
        # Visual state
        self.is_hovered = False
    
    def boundingRect(self):
        """Return the bounding rectangle"""
        return QRectF(-self.snap_radius, -self.snap_radius,
                     self.snap_radius * 2, self.snap_radius * 2)
                     
    def paint(self, painter, option, widget):
        """Paint the port"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get colors based on port type and state
        if self.port_type == 'input':
            if self.is_connected:
                fill_color = QColor(50, 150, 255)  # Connected blue
            else:
                fill_color = QColor(100, 100, 255)  # Unconnected blue
        else:  # output
            if self.is_connected:
                fill_color = QColor(50, 255, 150)  # Connected green
            else:
                fill_color = QColor(100, 255, 100)  # Unconnected green
                
        # Adjust for hover state
        if self.is_hovered:
            fill_color = fill_color.lighter(130)
            
        # Draw port circle
        painter.setBrush(QBrush(fill_color))
        
        if self.is_hovered:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        else:
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            
        painter.drawEllipse(-self.port_radius, -self.port_radius,
                          self.port_radius * 2, self.port_radius * 2)
                          
        # Draw inner circle for visual depth
        inner_radius = self.port_radius - 2
        if inner_radius > 0:
            painter.setBrush(QBrush(fill_color.lighter(120)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(-inner_radius, -inner_radius,
                              inner_radius * 2, inner_radius * 2)
                              
    def hoverEnterEvent(self, event):
        """Handle hover enter"""
        self.is_hovered = True
        self.update()
        
        # Show tooltip with port information
        tooltip = f"{self.port_type.capitalize()} Port"
        if self.tensor_shape:
            tooltip += f"\nShape: {self.tensor_shape}"
        if self.is_connected:
            tooltip += f"\nConnections: {len(self.connections)}"
        self.setToolTip(tooltip)
        
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """Handle hover leave"""
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            # Get the canvas from the scene
            canvas = None
            if self.scene() and hasattr(self.scene(), 'views'):
                views = self.scene().views()
                if views:
                    canvas = views[0]
                    
            if canvas and self.port_type == 'output':
                # Start connection from output port
                canvas.start_connection(self)
                event.accept()
                return
                
        super().mousePressEvent(event)
        
    def set_connected(self, connected):
        """Set the connection state"""
        self.is_connected = connected
        self.update()
        
    def add_connection(self, connection):
        """Add a connection to this port"""
        if connection not in self.connections:
            self.connections.append(connection)
            self.set_connected(True)
            
    def remove_connection(self, connection):
        """Remove a connection from this port"""
        if connection in self.connections:
            self.connections.remove(connection)
            if not self.connections:
                self.set_connected(False)
                
    def update_connections(self):
        """Update all connections attached to this port"""
        for connection in self.connections:
            connection.update_path()
            
    def get_scene_position(self):
        """Get the position of this port in scene coordinates"""
        return self.mapToScene(QPointF(0, 0))
        
    def can_connect_to(self, other_port):
        """Check if this port can connect to another port"""
        # Can't connect to same type
        if self.port_type == other_port.port_type:
            return False
            
        # Can't connect to same node
        if self.parent_node == other_port.parent_node:
            return False
            
        # Input ports can only have one connection
        if other_port.port_type == 'input' and other_port.is_connected:
            return False
            
        # Check tensor shape compatibility if available
        if (self.tensor_shape and other_port.tensor_shape and 
            not self.are_shapes_compatible(self.tensor_shape, other_port.tensor_shape)):
            return False
            
        return True
        
    def are_shapes_compatible(self, shape1, shape2):
        """Check if two tensor shapes are compatible"""
        # Basic compatibility check
        # This can be enhanced with more sophisticated logic
        if len(shape1) != len(shape2):
            return False
            
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 != -1 and dim2 != -1 and dim1 != dim2:
                return False
                
        return True
        
    def set_tensor_shape(self, shape):
        """Set the tensor shape for this port"""
        self.tensor_shape = shape
        
    def get_tensor_shape(self):
        """Get the tensor shape for this port"""
        return self.tensor_shape
        
    def serialize(self):
        """Serialize port to dictionary"""
        return {
            'type': 'Port',
            'port_type': self.port_type,
            'is_connected': self.is_connected,
            'tensor_shape': self.tensor_shape,
            'position': [self.pos().x(), self.pos().y()]
        }
        
    @staticmethod
    def deserialize(data, parent_node):
        """Deserialize port from dictionary"""
        port = Port(data['port_type'], parent_node)
        port.is_connected = data.get('is_connected', False)
        port.tensor_shape = data.get('tensor_shape')
        if 'position' in data:
            port.setPos(QPointF(data['position'][0], data['position'][1]))
        return port