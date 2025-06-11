#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property Panel - Panel for editing module properties
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from language_manager import language_manager

class PropertyPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.current_nodes = []
        self.property_widgets = {}
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        self.title_label = QLabel(language_manager.get_text('properties'))
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Create scroll area for properties
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Create widget to hold properties
        self.properties_widget = QWidget()
        self.properties_layout = QVBoxLayout(self.properties_widget)
        self.properties_layout.setContentsMargins(0, 0, 0, 0)
        self.properties_layout.setSpacing(5)
        
        scroll_area.setWidget(self.properties_widget)
        layout.addWidget(scroll_area)
        
        # Initially show no selection message
        self.show_no_selection()
        
        # Apply styling
        self.apply_styling()

    def update_language(self):
        """Update UI elements based on current language"""
        self.title_label.setText(language_manager.get_text('properties'))
        # Re-apply current selection to update all text within properties
        self.update_properties(self.current_nodes)

    def set_nodes(self, nodes):
        """Set the nodes for the property panel to display."""
        self.update_properties(nodes)
        
    def update_properties(self, selected_nodes):
        """Update properties panel with selected nodes"""
        self.current_nodes = selected_nodes
        self.clear_properties()
        
        if not selected_nodes:
            self.show_no_selection()
        elif len(selected_nodes) == 1:
            self.show_single_node_properties(selected_nodes[0])
        else:
            self.show_multiple_nodes_properties(selected_nodes)
            
    def clear_properties(self):
        """Clear all property widgets"""
        for widget in self.property_widgets.values():
            widget.setParent(None)
        self.property_widgets.clear()
        
        # Clear layout
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
                
    def show_no_selection(self):
        """Show message when no nodes are selected"""
        self.title_label.setText("Properties")
        
        message = QLabel("No module selected")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("color: #888888; font-style: italic;")
        self.properties_layout.addWidget(message)
        
        self.properties_layout.addStretch()
        
    def show_single_node_properties(self, node):
        """Show properties for a single node"""
        self.title_label.setText(f"Properties - {node.module_type}")
        
        # Module type (read-only)
        self.add_property_group("Module Information")
        self.add_read_only_property("Type", node.module_type)
        self.add_read_only_property("Active", "Yes" if node.is_active else "No")
        
        # Module parameters
        if node.config:
            self.add_property_group("Parameters")
            
            for key, value in node.config.items():
                self.add_editable_property(node, key, value)
                
        # Shape information (if available)
        input_shape = node.get_input_shape()
        output_shape = node.get_output_shape()
        
        if input_shape or output_shape:
            self.add_property_group("Shape Information")
            
            if input_shape:
                self.add_read_only_property("Input Shape", str(input_shape))
            if output_shape:
                self.add_read_only_property("Output Shape", str(output_shape))
                
        # Memory usage (placeholder)
        self.add_property_group("Memory Usage")
        self.add_read_only_property("Parameters", "Calculating...")
        self.add_read_only_property("Memory", "Calculating...")
        
        self.properties_layout.addStretch()
        
    def show_multiple_nodes_properties(self, nodes):
        """Show properties for multiple nodes"""
        self.title_label.setText(f"Properties - {len(nodes)} modules")
        
        # Summary information
        self.add_property_group("Selection Summary")
        self.add_read_only_property("Selected Modules", str(len(nodes)))
        
        # Count by type
        type_counts = {}
        for node in nodes:
            type_counts[node.module_type] = type_counts.get(node.module_type, 0) + 1
            
        for module_type, count in type_counts.items():
            self.add_read_only_property(module_type, str(count))
            
        # Common properties (if any)
        common_properties = self.find_common_properties(nodes)
        if common_properties:
            self.add_property_group("Common Properties")
            
            for key, value in common_properties.items():
                # Create a multi-node editor
                self.add_multi_node_property(nodes, key, value)
                
        self.properties_layout.addStretch()
        
    def add_property_group(self, title):
        """Add a property group header"""
        group_label = QLabel(title)
        group_label.setFont(QFont("Arial", 10, QFont.Bold))
        group_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #404040;
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
            }
        """)
        self.properties_layout.addWidget(group_label)
        
    def add_read_only_property(self, name, value):
        """Add a read-only property"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        
        name_label = QLabel(f"{name}:")
        name_label.setMinimumWidth(80)
        name_label.setStyleSheet("color: #cccccc;")
        
        value_label = QLabel(str(value))
        value_label.setStyleSheet("color: #ffffff;")
        value_label.setWordWrap(True)
        
        layout.addWidget(name_label)
        layout.addWidget(value_label)
        layout.addStretch()
        
        self.properties_layout.addWidget(container)
        
    def add_editable_property(self, node, key, value):
        """Add an editable property"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        
        name_label = QLabel(f"{key}:")
        name_label.setMinimumWidth(80)
        name_label.setStyleSheet("color: #cccccc;")
        
        # Create appropriate editor based on value type
        editor = self.create_property_editor(node, key, value)
        
        layout.addWidget(name_label)
        layout.addWidget(editor)
        
        self.properties_layout.addWidget(container)
        self.property_widgets[key] = editor
        
    def create_property_editor(self, node, key, value):
        """Create appropriate editor widget for property"""
        if isinstance(value, bool):
            editor = QCheckBox()
            editor.setChecked(value)
            editor.toggled.connect(lambda checked: self.update_node_property(node, key, checked))
            return editor
            
        elif isinstance(value, int):
            editor = QSpinBox()
            editor.setRange(-999999, 999999)
            editor.setValue(value)
            editor.valueChanged.connect(lambda val: self.update_node_property(node, key, val))
            return editor
            
        elif isinstance(value, float):
            editor = QDoubleSpinBox()
            editor.setRange(-999999.0, 999999.0)
            editor.setDecimals(6)
            editor.setValue(value)
            editor.valueChanged.connect(lambda val: self.update_node_property(node, key, val))
            return editor
            
        elif isinstance(value, list):
            editor = QLineEdit()
            editor.setText(str(value))
            editor.editingFinished.connect(lambda: self.update_list_property(node, key, editor.text()))
            return editor
            
        elif isinstance(value, str):
            editor = QLineEdit()
            editor.setText(value)
            editor.textChanged.connect(lambda text: self.update_node_property(node, key, text))
            return editor
            
        else:
            # Default to string editor
            editor = QLineEdit()
            editor.setText(str(value))
            editor.textChanged.connect(lambda text: self.update_node_property(node, key, text))
            return editor
            
    def add_multi_node_property(self, nodes, key, value):
        """Add property editor for multiple nodes"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        
        name_label = QLabel(f"{key}:")
        name_label.setMinimumWidth(80)
        name_label.setStyleSheet("color: #cccccc;")
        
        # Create editor that updates all selected nodes
        editor = self.create_multi_node_editor(nodes, key, value)
        
        layout.addWidget(name_label)
        layout.addWidget(editor)
        
        self.properties_layout.addWidget(container)
        
    def create_multi_node_editor(self, nodes, key, value):
        """Create editor for multiple nodes"""
        if isinstance(value, bool):
            editor = QCheckBox()
            editor.setChecked(value)
            editor.toggled.connect(lambda checked: self.update_multiple_nodes_property(nodes, key, checked))
            return editor
            
        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                editor = QSpinBox()
                editor.setRange(-999999, 999999)
            else:
                editor = QDoubleSpinBox()
                editor.setRange(-999999.0, 999999.0)
                editor.setDecimals(6)
                
            editor.setValue(value)
            editor.valueChanged.connect(lambda val: self.update_multiple_nodes_property(nodes, key, val))
            return editor
            
        else:
            editor = QLineEdit()
            editor.setText(str(value))
            editor.textChanged.connect(lambda text: self.update_multiple_nodes_property(nodes, key, text))
            return editor
            
    def update_node_property(self, node, key, value):
        """Update a single node's property"""
        node.set_config_value(key, value)
        
        # Emit signal to indicate model was modified
        if hasattr(node, 'scene') and node.scene():
            views = node.scene().views()
            if views and hasattr(views[0], 'model_modified'):
                views[0].model_modified.emit()
                
    def update_list_property(self, node, key, text):
        """Update a list property from text"""
        try:
            # Try to evaluate as Python literal
            value = eval(text)
            if isinstance(value, list):
                self.update_node_property(node, key, value)
        except:
            # If evaluation fails, keep as string
            self.update_node_property(node, key, text)
            
    def update_multiple_nodes_property(self, nodes, key, value):
        """Update property for multiple nodes"""
        for node in nodes:
            if key in node.config:
                node.set_config_value(key, value)
                
        # Emit signal to indicate model was modified
        if nodes and hasattr(nodes[0], 'scene') and nodes[0].scene():
            views = nodes[0].scene().views()
            if views and hasattr(views[0], 'model_modified'):
                views[0].model_modified.emit()
                
    def find_common_properties(self, nodes):
        """Find properties that are common to all selected nodes"""
        if not nodes:
            return {}
            
        # Start with first node's properties
        common_props = dict(nodes[0].config)
        
        # Remove properties that don't exist in all nodes or have different values
        for node in nodes[1:]:
            keys_to_remove = []
            for key, value in common_props.items():
                if key not in node.config or node.config[key] != value:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del common_props[key]
                
        return common_props
        
    def apply_styling(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            PropertyPanel {
                background-color: #2b2b2b;
                border-left: 1px solid #555555;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                color: #ffffff;
            }
            
            QLineEdit:focus {
                border-color: #0078d4;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                color: #ffffff;
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0078d4;
            }
            
            QCheckBox {
                color: #ffffff;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                background-color: #3c3c3c;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
        """)
        
    def update_language(self):
        """Update all property panel texts when language changes"""
        # Update title based on current state
        if not self.current_nodes:
            self.title_label.setText(language_manager.get_text('properties'))
        elif len(self.current_nodes) == 1:
            node = self.current_nodes[0]
            self.title_label.setText(f"{language_manager.get_text('properties')} - {node.module_type}")
        else:
            self.title_label.setText(f"{language_manager.get_text('properties')} - {len(self.current_nodes)} {language_manager.get_text('modules')}")
            
        # Refresh the properties display to update all text
        self.set_nodes(self.current_nodes)

    def set_nodes(self, nodes):
        """Sets the nodes for the property panel and updates the display."""
        self.update_properties(nodes)