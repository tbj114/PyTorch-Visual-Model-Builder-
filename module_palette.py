#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Palette - Panel for selecting PyTorch modules
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from language_manager import language_manager

class ModulePalette(QWidget):
    module_selected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        # Create widget to hold module categories
        self.modules_widget = QWidget()
        self.modules_layout = QVBoxLayout(self.modules_widget)
        self.modules_layout.setContentsMargins(0, 0, 0, 0)
        self.modules_layout.setSpacing(2)
        self.create_module_categories()

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        self.title_label = QLabel(language_manager.get_text('module_palette'))
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(language_manager.get_text('search_modules'))
        self.search_box.textChanged.connect(self.filter_modules)
        self.main_layout = layout # Store layout as an instance variable
        layout.addWidget(self.search_box)

        # Add scroll area for modules
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.modules_widget)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.main_layout.addWidget(self.scroll_area)

    def update_language(self):
        """Update UI elements based on current language"""
        self.title_label.setText(language_manager.get_text('module_palette'))
        self.search_box.setPlaceholderText(language_manager.get_text('search_modules'))
        # Clear existing categories and recreate them to ensure all text is updated
        while self.modules_layout.count():
            child = self.modules_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.create_module_categories()
        # The scroll area is already part of main_layout, no need to re-add it.
        
    def create_module_categories(self):
        """Create categorized module buttons"""
        categories = {
            "Input/Output": [
                ("Input", "Input layer for the model", "🔵"),
                ("Output", "Output layer for the model", "🔴")
            ],
            "Linear Layers": [
                ("Linear", "Fully connected layer", "📊"),
                ("Embedding", "Embedding layer", "📝")
            ],
            "Convolutional Layers": [
                ("Conv2d", "2D convolution layer", "🔲"),
                ("Conv1d", "1D convolution layer", "📏"),
                ("ConvTranspose2d", "2D transposed convolution", "🔳")
            ],
            "Normalization": [
                ("BatchNorm2d", "2D batch normalization", "⚖️"),
                ("BatchNorm1d", "1D batch normalization", "📐"),
                ("LayerNorm", "Layer normalization", "📊"),
                ("GroupNorm", "Group normalization", "👥")
            ],
            "Activation Functions": [
                ("ReLU", "Rectified Linear Unit", "⚡"),
                ("LeakyReLU", "Leaky ReLU activation", "🔋"),
                ("Sigmoid", "Sigmoid activation", "📈"),
                ("Tanh", "Hyperbolic tangent", "📉"),
                ("Softmax", "Softmax activation", "🎯"),
                ("GELU", "Gaussian Error Linear Unit", "🌊"),
                ("Swish", "Swish activation", "🌀")
            ],
            "Pooling Layers": [
                ("MaxPool2d", "2D max pooling", "⬇️"),
                ("AvgPool2d", "2D average pooling", "📊"),
                ("AdaptiveAvgPool2d", "Adaptive average pooling", "🎯"),
                ("AdaptiveMaxPool2d", "Adaptive max pooling", "🔽")
            ],
            "Recurrent Layers": [
                ("LSTM", "Long Short-Term Memory", "🔄"),
                ("GRU", "Gated Recurrent Unit", "🔁"),
                ("RNN", "Basic RNN layer", "↩️")
            ],
            "Attention": [
                ("MultiheadAttention", "Multi-head attention", "👁️"),
                ("TransformerEncoderLayer", "Transformer encoder", "🔀"),
                ("TransformerDecoderLayer", "Transformer decoder", "🔃")
            ],
            "Regularization": [
                ("Dropout", "Dropout regularization", "❌"),
                ("Dropout2d", "2D dropout", "🚫"),
                ("AlphaDropout", "Alpha dropout", "🔸")
            ],
            "Utility": [
                ("Flatten", "Flatten tensor", "📏"),
                ("Reshape", "Reshape tensor", "🔄"),
                ("Permute", "Permute dimensions", "🔀"),
                ("Squeeze", "Remove dimensions", "🗜️"),
                ("Unsqueeze", "Add dimensions", "📈")
            ]
        }
        
        self.category_widgets = {}
        self.module_buttons = []
        
        for category_name, modules in categories.items():
            # Create collapsible category
            category_widget = CollapsibleCategory(category_name)
            self.category_widgets[category_name] = category_widget
            self.modules_layout.addWidget(category_widget)
            
            # Add modules to category
            for module_name, description, icon in modules:
                button = ModuleButton(module_name, description, icon)
                button.clicked.connect(lambda checked, name=module_name: self.select_module(name))
                category_widget.add_module(button)
                self.module_buttons.append(button)
                
    def select_module(self, module_name):
        """Select a module for placement"""
        self.module_selected.emit(module_name)
        
        # Visual feedback
        for button in self.module_buttons:
            button.set_selected(button.module_name == module_name)
            
    def filter_modules(self, text):
        """Filter modules based on search text"""
        text = text.lower()
        
        for category_name, category_widget in self.category_widgets.items():
            has_visible_modules = False
            
            for button in category_widget.module_buttons:
                visible = (text in button.module_name.lower() or 
                          text in button.description.lower())
                button.setVisible(visible)
                if visible:
                    has_visible_modules = True
                    
            # Show/hide category based on whether it has visible modules
            category_widget.setVisible(has_visible_modules)
            
            # Expand category if it has matches
            if has_visible_modules and text:
                category_widget.set_expanded(True)
                
    def apply_styling(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            ModulePalette {
                background-color: #2b2b2b;
                border-right: 1px solid #555555;
            }
            
            QLabel {
                color: #ffffff;
                padding: 5px;
            }
            
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            
            QLineEdit:focus {
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

class CollapsibleCategory(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.is_expanded = True
        self.module_buttons = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create header button
        self.header_button = QPushButton(f"▼ {self.title}")
        self.header_button.setCheckable(False)
        self.header_button.clicked.connect(self.toggle_expanded)
        layout.addWidget(self.header_button)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 0, 0, 0)
        self.content_layout.setSpacing(2)
        layout.addWidget(self.content_widget)
        
        # Apply styling
        self.header_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                padding: 8px;
                text-align: left;
                color: #ffffff;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        
    def add_module(self, module_button):
        """Add a module button to this category"""
        self.content_layout.addWidget(module_button)
        self.module_buttons.append(module_button)
        
    def toggle_expanded(self):
        """Toggle the expanded state"""
        self.set_expanded(not self.is_expanded)
        
    def set_expanded(self, expanded):
        """Set the expanded state"""
        self.is_expanded = expanded
        self.content_widget.setVisible(expanded)
        
        # Update arrow
        arrow = "▼" if expanded else "▶"
        self.header_button.setText(f"{arrow} {self.title}")
        
    def update_language(self):
        """Update category title when language changes"""
        # Get translated category name
        translated_title = language_manager.get_text(f'category_{self.title.lower().replace(" ", "_")}')
        if translated_title == f'category_{self.title.lower().replace(" ", "_")}':
            # If no translation found, keep original title
            translated_title = self.title
        
        # Update button text with current arrow state
        arrow = "▼" if self.is_expanded else "▶"
        self.header_button.setText(f"{arrow} {translated_title}")
        
        # Update stored title for future use
        self.title = translated_title

class ModuleButton(QPushButton):
    def __init__(self, module_name, description, icon):
        super().__init__()
        self.module_name = module_name
        self.description = description
        self.icon = icon
        self.is_selected = False
        self.drag_start_pos = None
        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        self.setText(f"{self.icon} {self.module_name}")
        self.setToolTip(self.description)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(35)
        self.update_style()

    def set_selected(self, selected):
        """Set the selected state"""
        self.is_selected = selected
        self.update_style()

    def update_style(self):
        """Update the button style"""
        if self.is_selected:
            style = """
                QPushButton {
                    background-color: #0078d4;
                    border: 1px solid #106ebe;
                    border-radius: 4px;
                    padding: 8px;
                    text-align: left;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
            """
        else:
            style = """
                QPushButton {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                    border-radius: 4px;
                    padding: 8px;
                    text-align: left;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    border-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #0078d4;
                }
            """
        self.setStyleSheet(style)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.drag_start_pos is not None and (event.pos() - self.drag_start_pos).manhattanLength() > QApplication.startDragDistance():
                drag = QDrag(self)
                mime_data = QMimeData()
                mime_data.setText(self.module_name)
                drag.setMimeData(mime_data)
                drag.exec_(Qt.CopyAction)
                return
        super().mouseMoveEvent(event)
        
    def update_language(self):
        """Update the button's text and tooltip based on current language"""
        translated_module_name = language_manager.get_text(self.module_name.lower().replace(' ', '_'))
        translated_description = language_manager.get_text(self.description.lower().replace(' ', '_'))
        
        self.setText(f"{self.icon} {translated_module_name}")
        self.setToolTip(translated_description)
    def mouseReleaseEvent(self, event):
        self.drag_start_pos = None
        super().mouseReleaseEvent(event)