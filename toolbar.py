#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Toolbar - Provides quick access to common operations
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from language_manager import language_manager

class MainToolBar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        
        # Store actions for language updates
        self.actions = {}
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        self.init_toolbar()
        
    def init_toolbar(self):
        """Initialize the toolbar"""
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(24, 24))
        
        # File operations
        self.actions['new'] = self.add_action(language_manager.get_text('new'), language_manager.get_text('new_tooltip'), "", self.parent_window.new_model, "📄")
        self.actions['open'] = self.add_action(language_manager.get_text('open'), language_manager.get_text('open_tooltip'), "", self.parent_window.open_model, "📁")
        self.actions['save'] = self.add_action(language_manager.get_text('save'), language_manager.get_text('save_tooltip'), "", self.parent_window.save_model, "💾")
        
        self.addSeparator()
        
        # Edit operations
        self.actions['undo'] = self.add_action(language_manager.get_text('undo'), language_manager.get_text('undo_tooltip'), "Ctrl+Z", self.parent_window.undo, "↶")
        self.actions['redo'] = self.add_action(language_manager.get_text('redo'), language_manager.get_text('redo_tooltip'), "Ctrl+Y", self.parent_window.redo, "↷")
        
        self.addSeparator()
        
        self.actions['copy'] = self.add_action(language_manager.get_text('copy'), language_manager.get_text('copy_tooltip'), "", self.parent_window.copy_selection, "📋")
        self.actions['cut'] = self.add_action(language_manager.get_text('cut'), language_manager.get_text('cut_tooltip'), "", self.parent_window.cut_selection, "✂️")
        self.actions['paste'] = self.add_action(language_manager.get_text('paste'), language_manager.get_text('paste_tooltip'), "", self.parent_window.paste_selection, "📄")
        self.actions['delete'] = self.add_action(language_manager.get_text('delete'), language_manager.get_text('delete_tooltip'), "Del", self.parent_window.delete_selection, "🗑️")
        
        self.addSeparator()
        
        # View operations
        self.actions['zoom_in'] = self.add_action(language_manager.get_text('zoom_in'), language_manager.get_text('zoom_in_tooltip'), "", self.parent_window.zoom_in, "🔍")
        self.actions['zoom_out'] = self.add_action(language_manager.get_text('zoom_out'), language_manager.get_text('zoom_out_tooltip'), "", self.parent_window.zoom_out, "🔎")
        self.actions['reset_zoom'] = self.add_action(language_manager.get_text('reset_zoom'), language_manager.get_text('reset_zoom_tooltip'), "", self.parent_window.reset_zoom, "🎯")
        
        self.addSeparator()
        
        # Model operations
        self.add_action("Validate", "Validate model", "", self.parent_window.validate_model, "✅")
        self.add_action("Export", "Export to Python", "", self.parent_window.export_to_python, "🐍")
        
        self.addSeparator()
        
        # Add zoom level display
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                padding: 5px;
                background-color: #404040;
                border-radius: 3px;
                min-width: 40px;
            }
        """)
        self.addWidget(self.zoom_label)
        
        # Add spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
        
        # Add model statistics
        self.stats_label = QLabel("No model")
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                padding: 5px;
                font-size: 11px;
            }
        """)
        self.addWidget(self.stats_label)
        
        # Settings button
        self.actions['settings'] = self.add_action(language_manager.get_text('settings'), language_manager.get_text('settings_tooltip'), "Ctrl+,", self.parent_window.open_settings, "⚙️")
        
        # Connect to canvas signals to update stats
        if hasattr(self.parent_window, 'canvas'):
            self.parent_window.canvas.model_modified.connect(self.update_stats)
            self.parent_window.canvas.selection_changed.connect(self.update_selection_stats)
            
    def add_action(self, text, tooltip, shortcut, callback, icon_text):
        """Add an action to the toolbar"""
        action = QAction(text, self)
        action.setToolTip(f"{tooltip} ({shortcut})")
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        
        # Create icon from text (emoji or symbol)
        icon = self.create_text_icon(icon_text)
        action.setIcon(icon)
        
        self.addAction(action)
        return action
        
    def create_text_icon(self, text):
        """Create an icon from text"""
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set font
        font = QFont("Arial", 16)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        
        # Draw text centered
        rect = QRect(0, 0, 24, 24)
        painter.drawText(rect, Qt.AlignCenter, text)
        
        painter.end()
        
        return QIcon(pixmap)
        
    def update_zoom_display(self, zoom_factor):
        """Update the zoom level display"""
        percentage = int(zoom_factor * 100)
        self.zoom_label.setText(f"{percentage}%")
        
    def update_stats(self):
        """Update model statistics"""
        if not hasattr(self.parent_window, 'canvas'):
            return
            
        canvas = self.parent_window.canvas
        node_count = len(canvas.nodes)
        connection_count = len(canvas.connections)
        
        # Count active nodes
        active_nodes = sum(1 for node in canvas.nodes if node.is_active)
        
        # Calculate approximate parameter count
        param_count = self.estimate_parameter_count(canvas.nodes)
        
        if param_count > 1000000:
            param_str = f"{param_count/1000000:.1f}M"
        elif param_count > 1000:
            param_str = f"{param_count/1000:.1f}K"
        else:
            param_str = str(param_count)
            
        stats_text = f"Nodes: {active_nodes}/{node_count} | Connections: {connection_count} | Params: {param_str}"
        self.stats_label.setText(stats_text)
        
    def update_selection_stats(self, selected_items):
        """Update selection statistics"""
        if selected_items:
            selection_text = f" | Selected: {len(selected_items)}"
            current_text = self.stats_label.text()
            if " | Selected:" in current_text:
                current_text = current_text.split(" | Selected:")[0]
            self.stats_label.setText(current_text + selection_text)
        else:
            current_text = self.stats_label.text()
            if " | Selected:" in current_text:
                self.stats_label.setText(current_text.split(" | Selected:")[0])
                
    def estimate_parameter_count(self, nodes):
        """Estimate the total number of parameters in the model"""
        total_params = 0
        
        for node in nodes:
            if not node.is_active:
                continue
                
            module_type = node.module_type
            config = node.config
            
            if module_type == 'Linear':
                in_features = config.get('in_features', 0)
                out_features = config.get('out_features', 0)
                bias = config.get('bias', True)
                
                params = in_features * out_features
                if bias:
                    params += out_features
                total_params += params
                
            elif module_type == 'Conv2d':
                in_channels = config.get('in_channels', 0)
                out_channels = config.get('out_channels', 0)
                kernel_size = config.get('kernel_size', 1)
                bias = config.get('bias', True)
                
                if isinstance(kernel_size, (list, tuple)):
                    kernel_size = kernel_size[0] * kernel_size[1]
                else:
                    kernel_size = kernel_size * kernel_size
                    
                params = in_channels * out_channels * kernel_size
                if bias:
                    params += out_channels
                total_params += params
                
            elif module_type == 'BatchNorm2d':
                num_features = config.get('num_features', 0)
                total_params += num_features * 2  # weight and bias
                
            elif module_type == 'Embedding':
                num_embeddings = config.get('num_embeddings', 0)
                embedding_dim = config.get('embedding_dim', 0)
                total_params += num_embeddings * embedding_dim
                
            elif module_type == 'LSTM':
                input_size = config.get('input_size', 0)
                hidden_size = config.get('hidden_size', 0)
                num_layers = config.get('num_layers', 1)
                bias = config.get('bias', True)
                bidirectional = config.get('bidirectional', False)
                
                # LSTM has 4 gates, each with input and hidden weights
                params_per_layer = 4 * (input_size * hidden_size + hidden_size * hidden_size)
                if bias:
                    params_per_layer += 4 * hidden_size * 2  # input and hidden bias
                    
                total_params += params_per_layer * num_layers
                if bidirectional:
                    total_params *= 2
                    
        return total_params
        
    def update_language(self):
        """Update all toolbar texts when language changes"""
        # Update action texts and tooltips
        if 'new' in self.actions:
            self.actions['new'].setText(language_manager.get_text('new'))
            self.actions['new'].setToolTip(f"{language_manager.get_text('new_tooltip')} (Ctrl+N)")
            
        if 'open' in self.actions:
            self.actions['open'].setText(language_manager.get_text('open'))
            self.actions['open'].setToolTip(f"{language_manager.get_text('open_tooltip')} (Ctrl+O)")
            
        if 'save' in self.actions:
            self.actions['save'].setText(language_manager.get_text('save'))
            self.actions['save'].setToolTip(f"{language_manager.get_text('save_tooltip')} (Ctrl+S)")
            
        if 'undo' in self.actions:
            self.actions['undo'].setText(language_manager.get_text('undo'))
            self.actions['undo'].setToolTip(f"{language_manager.get_text('undo_tooltip')} (Ctrl+Z)")
            
        if 'redo' in self.actions:
            self.actions['redo'].setText(language_manager.get_text('redo'))
            self.actions['redo'].setToolTip(f"{language_manager.get_text('redo_tooltip')} (Ctrl+Y)")
            
        if 'copy' in self.actions:
            self.actions['copy'].setText(language_manager.get_text('copy'))
            self.actions['copy'].setToolTip(f"{language_manager.get_text('copy_tooltip')} (Ctrl+C)")
            
        if 'cut' in self.actions:
            self.actions['cut'].setText(language_manager.get_text('cut'))
            self.actions['cut'].setToolTip(f"{language_manager.get_text('cut_tooltip')} (Ctrl+X)")
            
        if 'paste' in self.actions:
            self.actions['paste'].setText(language_manager.get_text('paste'))
            self.actions['paste'].setToolTip(f"{language_manager.get_text('paste_tooltip')} (Ctrl+V)")
            
        if 'delete' in self.actions:
            self.actions['delete'].setText(language_manager.get_text('delete'))
            self.actions['delete'].setToolTip(f"{language_manager.get_text('delete_tooltip')} (Del)")
            
        if 'zoom_in' in self.actions:
            self.actions['zoom_in'].setText(language_manager.get_text('zoom_in'))
            self.actions['zoom_in'].setToolTip(f"{language_manager.get_text('zoom_in_tooltip')} (Ctrl++)")
            
        if 'zoom_out' in self.actions:
            self.actions['zoom_out'].setText(language_manager.get_text('zoom_out'))
            self.actions['zoom_out'].setToolTip(f"{language_manager.get_text('zoom_out_tooltip')} (Ctrl+-)")
            
        if 'reset_zoom' in self.actions:
            self.actions['reset_zoom'].setText(language_manager.get_text('reset_zoom'))
            self.actions['reset_zoom'].setToolTip(f"{language_manager.get_text('reset_zoom_tooltip')} (Ctrl+0)")
            
        if 'settings' in self.actions:
            self.actions['settings'].setText(language_manager.get_text('settings'))
            self.actions['settings'].setToolTip(f"{language_manager.get_text('settings_tooltip')} (Ctrl+,)")
            
        # Update stats label if no model is loaded
        if hasattr(self, 'stats_label') and self.stats_label.text() == "No model":
            self.stats_label.setText(language_manager.get_text('no_model'))