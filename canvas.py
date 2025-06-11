#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Canvas - The main drawing area for building models
"""

import math
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from model_node import ModelNode
from connection import Connection
from grid_background import GridBackground

class ModelCanvas(QGraphicsView):
    selection_changed = pyqtSignal(list)
    model_modified = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.nodes = []
        self.connections = []
        self.selected_items = []
        self.current_module_type = None
        self.is_connecting = False
        self.connection_start = None
        self.temp_connection = None
        self.zoom_factor = 1.0
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
        self.grid_background = GridBackground()
        self.scene.addItem(self.grid_background)
        self.init_canvas()
        # 支持拖拽
        self.setAcceptDrops(True)

    def init_canvas(self):
        """Initialize canvas settings"""
        # Set scene rect
        self.scene.setSceneRect(-5000, -5000, 10000, 10000)
        
        # Configure view
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Set background
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        
        # Apply styling
        self.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #555555;
                background-color: #1e1e1e;
            }
        """)
        
    def set_current_module(self, module_type):
        """Set the current module type for placement"""
        self.current_module_type = module_type
        self.setCursor(Qt.CrossCursor)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        scene_pos = self.mapToScene(event.pos())
        
        if event.button() == Qt.LeftButton:
            if self.current_module_type:
                # Place new module
                self.place_module(scene_pos)
                self.current_module_type = None
                self.setCursor(Qt.ArrowCursor)
                return
                
            # Check if clicking on a port
            item = self.scene.itemAt(scene_pos, self.transform())
            if hasattr(item, 'is_port') and item.is_port:
                if item.port_type == 'output':
                    self.start_connection(item)
                    return
                    
            # Check if clicking on a node
            if isinstance(item, ModelNode) or (hasattr(item, 'parentItem') and isinstance(item.parentItem(), ModelNode)):
                node = item if isinstance(item, ModelNode) else item.parentItem()
                if not node.isSelected():
                    if not (event.modifiers() & Qt.ControlModifier):
                        self.clear_selection()
                    self.select_node(node)
                    
        elif event.button() == Qt.RightButton:
            # Show context menu
            self.show_context_menu(event.pos())
            return
            
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        scene_pos = self.mapToScene(event.pos())
        
        if self.is_connecting and self.temp_connection:
            # Update temporary connection
            self.temp_connection.update_end_point(scene_pos)
            
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton and self.is_connecting:
            scene_pos = self.mapToScene(event.pos())
            
            # 查找最近的输入端口进行吸附
            target_port = self.find_nearest_input_port(scene_pos)
            
            if target_port and self.is_valid_connection(self.connection_start, target_port):
                # Complete connection
                self.complete_connection(target_port)
            else:
                # Cancel connection - 没有找到有效的端口或连接无效
                self.cancel_connection()
                # 显示提示信息
                if target_port:
                    self.show_connection_error("无法连接：连接无效")
                else:
                    self.show_connection_error("未找到可连接的端口")
                
        super().mouseReleaseEvent(event)
        
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if event.modifiers() & Qt.ControlModifier:
            # Zoom
            zoom_in = event.angleDelta().y() > 0
            zoom_factor = 1.15 if zoom_in else 1 / 1.15
            
            self.scale(zoom_factor, zoom_factor)
            self.zoom_factor *= zoom_factor
            
            # Clamp zoom
            if self.zoom_factor < 0.1:
                self.scale(0.1 / self.zoom_factor, 0.1 / self.zoom_factor)
                self.zoom_factor = 0.1
            elif self.zoom_factor > 5.0:
                self.scale(5.0 / self.zoom_factor, 5.0 / self.zoom_factor)
                self.zoom_factor = 5.0
        else:
            super().wheelEvent(event)
            
    def place_module(self, position):
        """Place a new module at the given position"""
        if not self.current_module_type:
            return
            
        # Create new node
        node = ModelNode(self.current_module_type, position)
        self.scene.addItem(node)
        self.nodes.append(node)
        
        # Connect signals
        # 移除信号连接，避免属性不存在报错
        
        # Save state for undo
        self.save_state()
        self.model_modified.emit()
        
    def start_connection(self, output_port):
        """Start creating a connection from an output port"""
        self.is_connecting = True
        self.connection_start = output_port
        
        # Create temporary connection for visual feedback
        self.temp_connection = Connection(output_port, None, temporary=True)
        self.scene.addItem(self.temp_connection)
        
    def complete_connection(self, input_port):
        """Complete a connection to an input port"""
        if not self.connection_start or not input_port:
            self.cancel_connection()
            return
            
        # Check if connection is valid
        if not self.is_valid_connection(self.connection_start, input_port):
            self.cancel_connection()
            return
            
        # Remove temporary connection
        if self.temp_connection:
            self.scene.removeItem(self.temp_connection)
            self.temp_connection = None
            
        # Create actual connection
        connection = Connection(self.connection_start, input_port)
        self.scene.addItem(connection)
        self.connections.append(connection)
        
        # Update port states
        self.connection_start.set_connected(True)
        input_port.set_connected(True)
        
        # Reset connection state
        self.is_connecting = False
        self.connection_start = None
        
        # Update model activation
        self.update_model_activation()
        
        # Save state for undo
        self.save_state()
        self.model_modified.emit()
        
    def cancel_connection(self):
        """Cancel the current connection"""
        if self.temp_connection:
            self.scene.removeItem(self.temp_connection)
            self.temp_connection = None
            
        self.is_connecting = False
        self.connection_start = None
        
    def find_nearest_input_port(self, scene_pos):
        """查找最近的输入端口进行吸附"""
        nearest_port = None
        min_distance = float('inf')
        
        for node in self.nodes:
            for port in node.input_ports:
                port_pos = port.get_scene_position()
                distance = ((scene_pos.x() - port_pos.x()) ** 2 + 
                           (scene_pos.y() - port_pos.y()) ** 2) ** 0.5
                
                # 检查是否在吸附范围内
                if distance <= port.snap_radius and distance < min_distance:
                    min_distance = distance
                    nearest_port = port
                    
        return nearest_port
    
    def show_connection_error(self, message):
        """显示连接错误提示"""
        # 创建临时提示标签
        if hasattr(self, 'error_label') and self.error_label is not None and self.error_label.parent() is not None:
            self.error_label.deleteLater()
            self.error_label = None
            
        self.error_label = QLabel(message, self)
        self.error_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 100, 100, 200);
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.error_label.adjustSize()
        self.error_label.move(10, 10)
        self.error_label.show()
        
        # 2秒后自动隐藏
        QTimer.singleShot(2000, lambda: self.error_label.deleteLater() if hasattr(self, 'error_label') else None)
        
    def is_valid_connection(self, output_port, input_port):
        """Check if a connection between two ports is valid"""
        # Can't connect to same node
        if output_port.parentItem() == input_port.parentItem():
            return False
            
        # Input port can only have one connection
        if input_port.is_connected:
            return False
            
        # Check for cycles
        if self.would_create_cycle(output_port.parentItem(), input_port.parentItem()):
            return False
            
        # Check tensor shape compatibility (if available)
        if hasattr(output_port, 'tensor_shape') and hasattr(input_port, 'tensor_shape'):
            if not self.are_shapes_compatible(output_port.tensor_shape, input_port.tensor_shape):
                return False
                
        return True
    


        
    def would_create_cycle(self, from_node, to_node):
        """Check if connecting from_node to to_node would create a cycle"""
        visited = set()
        
        def dfs(node):
            if node == from_node:
                return True
            if node in visited:
                return False
            visited.add(node)
            
            # Check all nodes that this node connects to
            for connection in self.connections:
                if connection.output_port.parentItem() == node:
                    if dfs(connection.input_port.parentItem()):
                        return True
            return False
            
        return dfs(to_node)
        
    def are_shapes_compatible(self, output_shape, input_shape):
        """Check if tensor shapes are compatible"""
        # Implement shape compatibility logic
        # For now, return True (can be enhanced later)
        return True
    
    def calculate_model_stats(self):
        """计算模型统计信息"""
        stats = {
            'total_nodes': len(self.nodes),
            'total_connections': len(self.connections),
            'input_nodes': 0,
            'output_nodes': 0,
            'hidden_nodes': 0,
            'total_parameters': 0,
            'estimated_memory_mb': 0
        }
        
        # 计算各类节点数量
        for node in self.nodes:
            if node.module_type == 'Input':
                stats['input_nodes'] += 1
            elif not any(conn.output_port.parentItem() == node for conn in self.connections):
                stats['output_nodes'] += 1
            else:
                stats['hidden_nodes'] += 1
                
        # 计算参数数量和内存估算
        for node in self.nodes:
            params = self.estimate_node_parameters(node)
            stats['total_parameters'] += params
            stats['estimated_memory_mb'] += self.estimate_memory_usage(node, params)
            
        return stats

    def estimate_node_parameters(self, node):
        """Estimate the number of parameters for a given node."""
        # Placeholder implementation
        return 0

    def estimate_memory_usage(self, node, params):
        """Estimate the memory usage for a given node."""
        # Placeholder implementation
        return 0
        
    def update_model_activation(self):
        """Update which nodes are activated (connected to input)"""
        # Find input nodes (nodes with no input connections)
        input_nodes = []
        for node in self.nodes:
            # Ensure node.input_ports is iterable and contains valid ports
            has_input_connection = False
            for conn in self.connections:
                # Add None checks for connection.input_port and its parentItem
                if conn.input_port and conn.input_port.parentItem() == node:
                    has_input_connection = True
                    break
            if node.module_type == 'Input' or not has_input_connection:
                input_nodes.append(node)
                
        # Mark all nodes as inactive first
        for node in self.nodes:
            node.set_active(False);
            
        # Activate nodes reachable from input nodes
        visited = set()
        
        def activate_recursive(node):
            if node in visited:
                return
            visited.add(node)
            node.set_active(True)
            
            # Activate connected nodes
            for connection in self.connections:
                # Add None checks for connection.output_port and connection.input_port and their parentItems
                if (connection.output_port and connection.output_port.parentItem() == node and
                    connection.input_port and connection.input_port.parentItem()): # Ensure input_port and its parent are valid
                    activate_recursive(connection.input_port.parentItem())
                    
        # Start activation from input nodes
        for input_node in input_nodes:
            activate_recursive(input_node)
            
    def get_model_info(self):
        """Get model information including node count, connection count, and total parameters"""
        node_count = len(self.nodes)
        connection_count = len(self.connections)
        total_params = 0
        
        # Calculate total parameters
        for node in self.nodes:
            if hasattr(node, 'calculate_parameters'):
                total_params += node.calculate_parameters()
                
        return node_count, connection_count, total_params
                    
        for input_node in input_nodes:
            activate_recursive(input_node)
            
    def select_node(self, node):
        """Select a node"""
        if node not in self.selected_items:
            self.selected_items.append(node)
            node.setSelected(True)
            self.selection_changed.emit(self.selected_items)
            
    def clear_selection(self):
        """Clear all selections"""
        for item in self.selected_items:
            item.setSelected(False)
        self.selected_items.clear()
        self.selection_changed.emit(self.selected_items)
        
    def delete_selection(self):
        """Delete selected items"""
        if not self.selected_items:
            return
            
        # Save state for undo
        self.save_state()
        
        # Remove connections first
        connections_to_remove = []
        for connection in self.connections:
            if (connection.output_port.parentItem() in self.selected_items or
                connection.input_port.parentItem() in self.selected_items):
                connections_to_remove.append(connection)
                
        for connection in connections_to_remove:
            self.remove_connection(connection)
            
        # Remove nodes
        for item in self.selected_items:
            if isinstance(item, ModelNode):
                self.scene.removeItem(item)
                self.nodes.remove(item)
                
        self.selected_items.clear()
        self.update_model_activation()
        self.model_modified.emit()
        
    def remove_connection(self, connection):
        """Remove a connection"""
        if connection in self.connections:
            # Update port states
            connection.output_port.set_connected(False)
            connection.input_port.set_connected(False)
            
            # Remove from scene and list
            self.scene.removeItem(connection)
            self.connections.remove(connection)
            
    def copy_selection(self):
        """Copy selected items to clipboard"""
        if not self.selected_items:
            return
            
        # Serialize selected nodes
        copied_data = {
            'nodes': [],
            'connections': []
        }
        
        # Copy selected nodes
        selected_nodes = [item for item in self.selected_items if isinstance(item, ModelNode)]
        node_id_map = {}
        
        for node in selected_nodes:
            node_data = node.serialize()
            copied_data['nodes'].append(node_data)
            node_id_map[node.node_id] = node
            
        # Copy connections between selected nodes
        for connection in self.connections:
            output_node = connection.output_port.parentItem()
            input_node = connection.input_port.parentItem()
            
            if output_node in selected_nodes and input_node in selected_nodes:
                copied_data['connections'].append(connection.serialize())
                
        # Store in clipboard
        import json
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(json.dumps(copied_data))
        
    def cut_selection(self):
        """Cut selected items to clipboard"""
        self.copy_selection()
        self.delete_selection()
        
    def paste_selection(self):
        """Paste items from clipboard"""
        import json
        from PyQt5.QtWidgets import QApplication
        
        clipboard = QApplication.clipboard()
        clipboard_text = clipboard.text()
        
        if not clipboard_text:
            return
            
        try:
            copied_data = json.loads(clipboard_text)
            if 'nodes' not in copied_data:
                return
                
            # Clear current selection
            self.clear_selection()
            
            # Create new nodes with offset
            offset = QPointF(50, 50)
            node_id_map = {}
            new_nodes = []
            
            for node_data in copied_data['nodes']:
                # Create new node with new ID
                from model_node import ModelNode
                node = ModelNode.deserialize(node_data)
                old_id = node.node_id
                node.node_id = f"node_{len(self.nodes) + len(new_nodes) + 1}"
                node_id_map[old_id] = node.node_id
                
                # Apply offset
                node.setPos(node.pos() + offset)
                
                # Add to scene
                self.scene.addItem(node)
                self.nodes.append(node)
                new_nodes.append(node)
                
                # Select the new node
                self.select_node(node)
                
            # Recreate connections with new node IDs
            if 'connections' in copied_data:
                for conn_data in copied_data['connections']:
                    # Update node IDs in connection data
                    if conn_data['output_node_id'] in node_id_map and conn_data['input_node_id'] in node_id_map:
                        conn_data['output_node_id'] = node_id_map[conn_data['output_node_id']]
                        conn_data['input_node_id'] = node_id_map[conn_data['input_node_id']]
                        
                        # Create connection
                        from connection import Connection
                        connection = Connection.deserialize(conn_data, self.nodes)
                        if connection:
                            self.scene.addItem(connection)
                            self.connections.append(connection)
                            
            self.update_model_activation()
            self.model_modified.emit()
            
        except (json.JSONDecodeError, KeyError):
            # Invalid clipboard data, ignore
            pass
        
    def select_all(self):
        """Select all items"""
        self.clear_selection()
        for node in self.nodes:
            self.select_node(node)
            
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.resetTransform()
        self.zoom_factor = 1.0
        
    def zoom_in(self):
        """Zoom in"""
        self.scale(1.15, 1.15)
        self.zoom_factor *= 1.15
        
    def zoom_out(self):
        """Zoom out"""
        self.scale(1/1.15, 1/1.15)
        self.zoom_factor /= 1.15
        
    def validate_model(self):
        """Validate the current model and return list of errors"""
        errors = []
        
        # Check for input nodes
        input_nodes = [node for node in self.nodes if node.module_type == 'Input']
        if not input_nodes:
            errors.append("Model must have at least one input node")
            
        # Check for output nodes
        output_nodes = []
        for node in self.nodes:
            has_output_connection = any(
                conn.output_port.parentItem() == node for conn in self.connections
            )
            if not has_output_connection:
                output_nodes.append(node)
                
        if not output_nodes:
            errors.append("Model must have at least one output node")
            
        # Check for disconnected nodes
        inactive_nodes = [node for node in self.nodes if not node.is_active]
        if inactive_nodes:
            errors.append(f"{len(inactive_nodes)} nodes are not connected to the model")
            
        return errors
        
    def save_state(self):
        """Save current state for undo"""
        state = {
            'nodes': [node.serialize() for node in self.nodes],
            'connections': [conn.serialize() for conn in self.connections]
        }
        
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
            
        # Clear redo stack
        self.redo_stack.clear()

    def serialize(self):
        """Serialize the entire canvas state"""
        return {
            'nodes': [node.serialize() for node in self.nodes],
            'connections': [conn.serialize() for conn in self.connections],
            'zoom_factor': self.zoom_factor,
            'stats': self.calculate_model_stats()
        }

    def deserialize(self, model_data):
        """Deserialize and load the canvas state"""
        self.clear()

        # Restore nodes
        node_map = {}
        for node_data in model_data.get('nodes', []):
            node = ModelNode.deserialize(node_data)
            self.scene.addItem(node)
            self.nodes.append(node)
            node_map[node.node_id] = node

        # Restore connections
        for conn_data in model_data.get('connections', []):
            connection = Connection.deserialize(conn_data, list(node_map.values()))
            if connection:
                self.scene.addItem(connection)
                self.connections.append(connection)

        self.update_model_activation()
        self.model_modified.emit()

    def clear(self):
        """Clear all items from the canvas"""
        for item in self.scene.items():
            self.scene.removeItem(item)
        self.nodes.clear()
        self.connections.clear()
        self.selected_items.clear()
        self.grid_background = GridBackground()
        self.scene.addItem(self.grid_background)

    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            return
            
        # Save current state to redo stack
        current_state = {
            'nodes': [node.serialize() for node in self.nodes],
            'connections': [conn.serialize() for conn in self.connections]
        }
        self.redo_stack.append(current_state)
        
        # Restore previous state
        state = self.undo_stack.pop()
        self.restore_state(state)
        
    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            return
            
        # Save current state to undo stack
        current_state = {
            'nodes': [node.serialize() for node in self.nodes],
            'connections': [conn.serialize() for conn in self.connections]
        }
        self.undo_stack.append(current_state)
        
        # Restore redo state
        state = self.redo_stack.pop()
        self.restore_state(state)
        
    def restore_state(self, state):
        """Restore canvas to a saved state"""
        # Clear current items
        self.clear()
        
        # Restore nodes
        node_map = {}
        for node_data in state['nodes']:
            node = ModelNode.deserialize(node_data)
            self.scene.addItem(node)
            self.nodes.append(node)
            node_map[node.node_id] = node
            
        # Restore connections
        for conn_data in state['connections']:
            connection = Connection.deserialize(conn_data, list(node_map.values()))
            if connection:
                self.scene.addItem(connection)
                self.connections.append(connection)
                
        self.update_model_activation()
        

        
    def show_context_menu(self, pos):
        """Show context menu"""
        menu = QMenu(self)
        scene_pos = self.mapToScene(pos)
        item = self.scene.itemAt(scene_pos, self.transform())
        
        if isinstance(item, ModelNode):
            menu.addAction("Delete Node", lambda: self.delete_node(item))
            menu.addAction("Duplicate Node", lambda: self.duplicate_node(item))
            menu.addSeparator()
        elif isinstance(item, Connection):
            delete_action = menu.addAction("删除连接")
            delete_action.triggered.connect(lambda: self.delete_connection(item))
            menu.addSeparator()
        
        menu.addAction("Add Input Node", lambda: self.add_node_at_cursor('Input', scene_pos))
        menu.addAction("Add Linear Layer", lambda: self.add_node_at_cursor('Linear', scene_pos))
        menu.addAction("Add Conv2d Layer", lambda: self.add_node_at_cursor('Conv2d', scene_pos))
        menu.addAction("Add ReLU", lambda: self.add_node_at_cursor('ReLU', scene_pos))
        
        menu.exec_(self.mapToGlobal(pos))
    
    def add_node_at_cursor(self, module_type, position):
        """Add a node at the cursor position"""
        self.current_module_type = module_type
        self.place_module(position)
    
    def delete_node(self, node):
        """Delete a specific node"""
        self.clear_selection()
        self.select_node(node)
        self.delete_selection()
    
    def duplicate_node(self, node):
        """Duplicate a specific node"""
        new_position = node.pos() + QPointF(50, 50)
        self.current_module_type = node.module_type
        self.place_module(new_position)
    
    def delete_connection(self, connection):
        """Delete a connection"""
        if connection in self.connections:
            self.connections.remove(connection)
            connection.output_port.set_connected(False)
            connection.input_port.set_connected(False)
            self.scene.removeItem(connection)
            self.model_modified.emit()
    
    def cancel_connection(self):
        """Cancel the current connection attempt"""
        self.is_connecting = False
        if self.temp_connection:
            self.scene.removeItem(self.temp_connection)
            self.temp_connection = None
        self.connection_start = None
        
    def on_node_moved(self):
        """Handle node movement"""
        self.model_modified.emit()
        
    def on_selection_changed(self):
        """Handle selection changes"""
        self.selection_changed.emit(self.selected_items)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            module_type = event.mimeData().text()
            pos = self.mapToScene(event.pos())
            self.current_module_type = module_type
            self.place_module(pos)
            self.current_module_type = None
            event.acceptProposedAction()
        else:
            event.ignore()