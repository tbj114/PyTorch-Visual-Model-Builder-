#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status Bar - Displays application status information
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import psutil
import time
from language_manager import language_manager

class StatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        """Initialize status bar UI"""
        # Main status label
        self.status_label = QLabel(language_manager.get_text('ready'))
        self.addWidget(self.status_label)
        
        # Add stretch to push right-side widgets to the right
        self.addPermanentWidget(QWidget(), 1)
        
        # Model info
        self.model_info_label = QLabel(f"{language_manager.get_text('nodes')}: 0 | {language_manager.get_text('connections')}: 0")
        self.model_info_label.setMinimumWidth(150)
        self.addPermanentWidget(self.model_info_label)
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        self.addPermanentWidget(separator1)
        
        # Memory usage
        self.memory_label = QLabel(f"{language_manager.get_text('memory')}: 0 MB")
        self.memory_label.setMinimumWidth(100)
        self.addPermanentWidget(self.memory_label)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        self.addPermanentWidget(separator2)
        
        # CPU usage
        self.cpu_label = QLabel(f"{language_manager.get_text('cpu')}: 0%")
        self.cpu_label.setMinimumWidth(80)
        self.addPermanentWidget(self.cpu_label)
        
        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setFrameShadow(QFrame.Sunken)
        self.addPermanentWidget(separator3)
        
        # Zoom level
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.addPermanentWidget(self.zoom_label)
        
        # Separator
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.VLine)
        separator4.setFrameShadow(QFrame.Sunken)
        self.addPermanentWidget(separator4)
        
        # Language indicator
        self.language_label = QLabel("EN")
        self.language_label.setMinimumWidth(30)
        self.language_label.setAlignment(Qt.AlignCenter)
        self.addPermanentWidget(self.language_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.addPermanentWidget(self.progress_bar)
        
        # Apply styling
        self.apply_styling()
        
    def apply_styling(self):
        """Apply modern styling to status bar"""
        self.setStyleSheet("""
            QStatusBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-top: 1px solid #404040;
                font-size: 11px;
            }
            
            QStatusBar QLabel {
                color: #cccccc;
                padding: 2px 8px;
                background-color: transparent;
            }
            
            QStatusBar QFrame {
                color: #555555;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #404040;
                text-align: center;
                color: #ffffff;
                font-size: 10px;
            }
            
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """)
        
    def setup_timer(self):
        """Setup timer for updating system information"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_info)
        self.update_timer.start(2000)  # Update every 2 seconds
        
    def update_system_info(self):
        """Update system information display"""
        try:
            # Update memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"{language_manager.get_text('memory')}: {memory_mb:.1f} MB")
            
            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_label.setText(f"{language_manager.get_text('cpu')}: {cpu_percent:.1f}%")
            
        except Exception as e:
            # Fallback if psutil fails
            self.memory_label.setText(f"{language_manager.get_text('memory')}: N/A")
            self.cpu_label.setText(f"{language_manager.get_text('cpu')}: N/A")
            
    def set_status(self, message, timeout=0):
        """Set status message"""
        self.status_label.setText(message)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("就绪"))
            
    def update_model_info(self, node_count, connection_count, parameter_count=None):
        """Update model information"""
        if parameter_count is not None:
            if parameter_count >= 1000000:
                param_str = f"{parameter_count/1000000:.1f}M"
            elif parameter_count >= 1000:
                param_str = f"{parameter_count/1000:.1f}K"
            else:
                param_str = str(parameter_count)
            self.model_info_label.setText(f"{language_manager.get_text('nodes')}: {node_count} | {language_manager.get_text('connections')}: {connection_count} | {language_manager.get_text('parameters')}: {param_str}")
        else:
            self.model_info_label.setText(f"{language_manager.get_text('nodes')}: {node_count} | {language_manager.get_text('connections')}: {connection_count}")
            
    def update_zoom_level(self, zoom_percent):
        """Update zoom level display"""
        self.zoom_label.setText(f"{zoom_percent:.0f}%")
        
    def set_language(self, language_code):
        """Set language indicator"""
        language_map = {
            'en': 'EN',
            'zh': '中',
            'ja': '日'
        }
        self.language_label.setText(language_map.get(language_code, 'EN'))
        
    def update_language(self):
        """Update all status bar texts when language changes"""
        # Update main status
        self.status_label.setText(language_manager.get_text('ready'))
        
        # Update system info labels
        try:
            # Get current values and update with new language
            memory = psutil.virtual_memory()
            memory_used = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            cpu_percent = psutil.cpu_percent(interval=None)
            
            self.memory_label.setText(f"{language_manager.get_text('memory')}: {memory_used:.0f} MB ({memory_percent:.1f}%)")
            self.cpu_label.setText(f"{language_manager.get_text('cpu')}: {cpu_percent:.1f}%")
        except:
            self.memory_label.setText(f"{language_manager.get_text('memory')}: N/A")
            self.cpu_label.setText(f"{language_manager.get_text('cpu')}: N/A")
            
        # Update language indicator
        current_lang = language_manager.get_current_language()
        self.set_language(current_lang)
        
        # Update model info if canvas is available
        if hasattr(self.parent_window, 'canvas') and self.parent_window.canvas:
            try:
                model_info = self.parent_window.canvas.get_model_info()
                self.update_model_info_display(
                    model_info['nodes'], 
                    model_info['connections'], 
                    model_info['parameters']
                )
            except:
                self.model_info_label.setText(f"{language_manager.get_text('nodes')}: 0 | {language_manager.get_text('connections')}: 0")
        
    def show_progress(self, title="Processing...", maximum=100):
        """Show progress bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.set_status(title)
        
    def update_progress(self, value):
        """Update progress bar value"""
        if self.progress_bar.isVisible():
            self.progress_bar.setValue(value)
            
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
        self.set_status("就绪")
        
    def show_temporary_message(self, message, duration=3000):
        """Show a temporary message"""
        original_text = self.status_label.text()
        self.status_label.setText(message)
        
        # Apply temporary styling for important messages
        if "Error" in message or "Failed" in message:
            self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        elif "Success" in message or "Completed" in message:
            self.status_label.setStyleSheet("color: #51cf66; font-weight: bold;")
        elif "Warning" in message:
            self.status_label.setStyleSheet("color: #ffd43b; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #74c0fc; font-weight: bold;")
            
        # Reset after duration
        QTimer.singleShot(duration, lambda: self.reset_status_styling(original_text))
        
    def reset_status_styling(self, original_text):
        """Reset status label styling"""
        self.status_label.setStyleSheet("color: #cccccc; background-color: transparent;")
        self.status_label.setText(original_text)
        
    def set_model_validation_status(self, is_valid, error_count=0):
        """Set model validation status"""
        if is_valid:
            self.show_temporary_message("✓ Model is valid", 2000)
        else:
            self.show_temporary_message(f"✗ Model has {error_count} error(s)", 5000)
            
    def set_file_operation_status(self, operation, filename, success=True):
        """Set file operation status"""
        if success:
            self.show_temporary_message(f"✓ {operation}: {filename}", 2000)
        else:
            self.show_temporary_message(f"✗ Failed to {operation.lower()}: {filename}", 5000)
            
    def mousePressEvent(self, event):
        """Handle mouse press events on status bar"""
        if event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())
        super().mousePressEvent(event)
        
    def show_context_menu(self, position):
        """Show context menu for status bar"""
        menu = QMenu(self)
        
        # Toggle system info
        toggle_system_action = QAction("Toggle System Info", self)
        toggle_system_action.triggered.connect(self.toggle_system_info)
        menu.addAction(toggle_system_action)
        
        # Reset status
        reset_action = QAction("Reset Status", self)
        reset_action.triggered.connect(lambda: self.set_status("Ready"))
        menu.addAction(reset_action)
        
        menu.exec_(self.mapToGlobal(position))
        
    def toggle_system_info(self):
        """Toggle system information visibility"""
        visible = self.memory_label.isVisible()
        self.memory_label.setVisible(not visible)
        self.cpu_label.setVisible(not visible)
        
        # Also toggle separators
        for i in range(self.count()):
            widget = self.widget(i)
            if isinstance(widget, QFrame):
                widget.setVisible(not visible)
                
    def update_language(self):
        """Update status bar texts when language changes"""
        # Update status label if it shows "Ready"
        if self.status_label.text() == "Ready" or self.status_label.text() == language_manager.get_text('ready'):
            self.status_label.setText(language_manager.get_text('ready'))
        
        # Update model info label
        # Extract current numbers from the label
        current_text = self.model_info_label.text()
        if ":" in current_text:
            parts = current_text.split("|")
            if len(parts) >= 2:
                nodes_part = parts[0].strip()
                connections_part = parts[1].strip()
                
                # Extract numbers
                nodes_num = nodes_part.split(":")[-1].strip() if ":" in nodes_part else "0"
                connections_num = connections_part.split(":")[-1].strip() if ":" in connections_part else "0"
                
                # Update with new language
                self.model_info_label.setText(f"{language_manager.get_text('nodes')}: {nodes_num} | {language_manager.get_text('connections')}: {connections_num}")
        
        # Update memory label
        current_memory = self.memory_label.text()
        if ":" in current_memory:
            memory_value = current_memory.split(":")[-1].strip()
            self.memory_label.setText(f"{language_manager.get_text('memory')}: {memory_value}")
        
        # Update CPU label
        current_cpu = self.cpu_label.text()
        if ":" in current_cpu:
            cpu_value = current_cpu.split(":")[-1].strip()
            self.cpu_label.setText(f"{language_manager.get_text('cpu')}: {cpu_value}")