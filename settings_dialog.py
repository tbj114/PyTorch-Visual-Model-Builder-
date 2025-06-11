#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings Dialog - Application configuration interface
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import os
from language_manager import language_manager

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.settings_file = "settings.json"
        self.settings = self.load_settings()
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        self.init_ui()
        self.load_current_settings()
        
    def init_ui(self):
        """Initialize the settings dialog UI"""
        self.setWindowTitle(language_manager.get_text('settings'))
        self.setFixedSize(600, 500)
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_general_tab()
        self.create_appearance_tab()
        self.create_editor_tab()
        self.create_performance_tab()
        self.create_language_tab()
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        layout.addWidget(button_box)
        
        # Apply styling
        self.apply_styling()
        
    def create_general_tab(self):
        """Create general settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Auto-save group
        auto_save_group = QGroupBox(language_manager.get_text('auto_save'))
        auto_save_layout = QVBoxLayout(auto_save_group)
        
        self.auto_save_enabled = QCheckBox(language_manager.get_text('enable_auto_save'))
        auto_save_layout.addWidget(self.auto_save_enabled)
        
        auto_save_interval_layout = QHBoxLayout()
        auto_save_interval_layout.addWidget(QLabel(language_manager.get_text('auto_save_interval')))
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setValue(5)
        auto_save_interval_layout.addWidget(self.auto_save_interval)
        auto_save_interval_layout.addStretch()
        auto_save_layout.addLayout(auto_save_interval_layout)
        
        layout.addWidget(auto_save_group)
        
        # Recent files group
        recent_files_group = QGroupBox("Recent Files")
        recent_files_layout = QVBoxLayout(recent_files_group)
        
        recent_files_count_layout = QHBoxLayout()
        recent_files_count_layout.addWidget(QLabel("Maximum recent files:"))
        self.recent_files_count = QSpinBox()
        self.recent_files_count.setRange(0, 20)
        self.recent_files_count.setValue(10)
        recent_files_count_layout.addWidget(self.recent_files_count)
        recent_files_count_layout.addStretch()
        recent_files_layout.addLayout(recent_files_count_layout)
        
        clear_recent_btn = QPushButton("Clear Recent Files")
        clear_recent_btn.clicked.connect(self.clear_recent_files)
        recent_files_layout.addWidget(clear_recent_btn)
        
        layout.addWidget(recent_files_group)
        
        # Startup group
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)
        
        self.restore_session = QCheckBox("Restore last session on startup")
        startup_layout.addWidget(self.restore_session)
        
        self.show_start_page = QCheckBox("Show start page")
        startup_layout.addWidget(self.show_start_page)
        
        layout.addWidget(startup_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "General")
        
    def create_appearance_tab(self):
        """Create appearance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Theme group
        theme_group = QGroupBox("Theme")
        theme_layout = QVBoxLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Auto"])
        theme_layout.addWidget(QLabel("Theme:"))
        theme_layout.addWidget(self.theme_combo)
        
        layout.addWidget(theme_group)
        
        # Colors group
        colors_group = QGroupBox("Colors")
        colors_layout = QFormLayout(colors_group)
        
        self.grid_color_btn = QPushButton()
        self.grid_color_btn.setFixedSize(50, 30)
        self.grid_color_btn.clicked.connect(lambda: self.choose_color('grid_color'))
        colors_layout.addRow("Grid color:", self.grid_color_btn)
        
        self.background_color_btn = QPushButton()
        self.background_color_btn.setFixedSize(50, 30)
        self.background_color_btn.clicked.connect(lambda: self.choose_color('background_color'))
        colors_layout.addRow("Background color:", self.background_color_btn)
        
        self.selection_color_btn = QPushButton()
        self.selection_color_btn.setFixedSize(50, 30)
        self.selection_color_btn.clicked.connect(lambda: self.choose_color('selection_color'))
        colors_layout.addRow("Selection color:", self.selection_color_btn)
        
        layout.addWidget(colors_group)
        
        # Font group
        font_group = QGroupBox("Font")
        font_layout = QVBoxLayout(font_group)
        
        font_selection_layout = QHBoxLayout()
        self.font_btn = QPushButton("Choose Font...")
        self.font_btn.clicked.connect(self.choose_font)
        font_selection_layout.addWidget(self.font_btn)
        
        self.font_label = QLabel("Current: System Default")
        font_selection_layout.addWidget(self.font_label)
        font_selection_layout.addStretch()
        
        font_layout.addLayout(font_selection_layout)
        
        layout.addWidget(font_group)
        
        # Animation group
        animation_group = QGroupBox("Animation")
        animation_layout = QVBoxLayout(animation_group)
        
        self.enable_animations = QCheckBox("Enable animations")
        animation_layout.addWidget(self.enable_animations)
        
        self.animation_speed = QSlider(Qt.Horizontal)
        self.animation_speed.setRange(1, 10)
        self.animation_speed.setValue(5)
        animation_layout.addWidget(QLabel("Animation speed:"))
        animation_layout.addWidget(self.animation_speed)
        
        layout.addWidget(animation_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Appearance")
        
    def create_editor_tab(self):
        """Create editor settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Grid group
        grid_group = QGroupBox("Grid")
        grid_layout = QVBoxLayout(grid_group)
        
        self.show_grid = QCheckBox("Show grid")
        grid_layout.addWidget(self.show_grid)
        
        self.snap_to_grid = QCheckBox("Snap to grid")
        grid_layout.addWidget(self.snap_to_grid)
        
        grid_size_layout = QHBoxLayout()
        grid_size_layout.addWidget(QLabel("Grid size:"))
        self.grid_size = QSpinBox()
        self.grid_size.setRange(10, 100)
        self.grid_size.setValue(20)
        grid_size_layout.addWidget(self.grid_size)
        grid_size_layout.addStretch()
        grid_layout.addLayout(grid_size_layout)
        
        layout.addWidget(grid_group)
        
        # Connection group
        connection_group = QGroupBox("Connections")
        connection_layout = QVBoxLayout(connection_group)
        
        self.show_connection_animation = QCheckBox("Show connection animation")
        connection_layout.addWidget(self.show_connection_animation)
        
        self.auto_arrange_connections = QCheckBox("Auto-arrange connections")
        connection_layout.addWidget(self.auto_arrange_connections)
        
        connection_width_layout = QHBoxLayout()
        connection_width_layout.addWidget(QLabel("Connection width:"))
        self.connection_width = QSpinBox()
        self.connection_width.setRange(1, 10)
        self.connection_width.setValue(2)
        connection_width_layout.addWidget(self.connection_width)
        connection_width_layout.addStretch()
        connection_layout.addLayout(connection_width_layout)
        
        layout.addWidget(connection_group)
        
        # Node group
        node_group = QGroupBox("Nodes")
        node_layout = QVBoxLayout(node_group)
        
        self.show_node_preview = QCheckBox("Show node preview on hover")
        node_layout.addWidget(self.show_node_preview)
        
        self.auto_resize_nodes = QCheckBox("Auto-resize nodes based on content")
        node_layout.addWidget(self.auto_resize_nodes)
        
        layout.addWidget(node_group)
        
        # Validation group
        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        
        self.real_time_validation = QCheckBox("Real-time validation")
        validation_layout.addWidget(self.real_time_validation)
        
        self.show_validation_tooltips = QCheckBox("Show validation tooltips")
        validation_layout.addWidget(self.show_validation_tooltips)
        
        layout.addWidget(validation_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Editor")
        
    def create_performance_tab(self):
        """Create performance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Rendering group
        rendering_group = QGroupBox("Rendering")
        rendering_layout = QVBoxLayout(rendering_group)
        
        self.enable_antialiasing = QCheckBox("Enable anti-aliasing")
        rendering_layout.addWidget(self.enable_antialiasing)
        
        self.use_opengl = QCheckBox("Use OpenGL acceleration (requires restart)")
        rendering_layout.addWidget(self.use_opengl)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Maximum FPS:"))
        self.max_fps = QSpinBox()
        self.max_fps.setRange(30, 120)
        self.max_fps.setValue(60)
        fps_layout.addWidget(self.max_fps)
        fps_layout.addStretch()
        rendering_layout.addLayout(fps_layout)
        
        layout.addWidget(rendering_group)
        
        # Memory group
        memory_group = QGroupBox("Memory")
        memory_layout = QVBoxLayout(memory_group)
        
        undo_levels_layout = QHBoxLayout()
        undo_levels_layout.addWidget(QLabel("Undo levels:"))
        self.undo_levels = QSpinBox()
        self.undo_levels.setRange(10, 1000)
        self.undo_levels.setValue(100)
        undo_levels_layout.addWidget(self.undo_levels)
        undo_levels_layout.addStretch()
        memory_layout.addLayout(undo_levels_layout)
        
        self.cache_thumbnails = QCheckBox("Cache node thumbnails")
        memory_layout.addWidget(self.cache_thumbnails)
        
        layout.addWidget(memory_group)
        
        # Threading group
        threading_group = QGroupBox("Threading")
        threading_layout = QVBoxLayout(threading_group)
        
        self.enable_multithreading = QCheckBox("Enable multi-threading")
        threading_layout.addWidget(self.enable_multithreading)
        
        thread_count_layout = QHBoxLayout()
        thread_count_layout.addWidget(QLabel("Worker threads:"))
        self.thread_count = QSpinBox()
        self.thread_count.setRange(1, 16)
        self.thread_count.setValue(4)
        thread_count_layout.addWidget(self.thread_count)
        thread_count_layout.addStretch()
        threading_layout.addLayout(thread_count_layout)
        
        layout.addWidget(threading_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Performance")
        
    def create_language_tab(self):
        """Create language settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Language group
        language_group = QGroupBox("Language")
        language_layout = QVBoxLayout(language_group)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "中文 (Chinese)", "日本語 (Japanese)"])
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        language_layout.addWidget(QLabel(language_manager.get_text('interface_language')))
        language_layout.addWidget(self.language_combo)
        
        restart_note = QLabel(language_manager.get_text('restart_note'))
        restart_note.setStyleSheet("color: #ffd43b; font-style: italic;")
        language_layout.addWidget(restart_note)
        
        layout.addWidget(language_group)
        
        # Code generation group
        code_gen_group = QGroupBox("Code Generation")
        code_gen_layout = QVBoxLayout(code_gen_group)
        
        self.code_language_combo = QComboBox()
        self.code_language_combo.addItems(["English", "中文 (Chinese)", "日本語 (Japanese)"])
        code_gen_layout.addWidget(QLabel(language_manager.get_text('code_comments_language')))
        code_gen_layout.addWidget(self.code_language_combo)
        
        self.include_type_hints = QCheckBox(language_manager.get_text('include_type_hints'))
        code_gen_layout.addWidget(self.include_type_hints)
        
        self.include_docstrings = QCheckBox(language_manager.get_text('include_docstrings'))
        code_gen_layout.addWidget(self.include_docstrings)
        
        layout.addWidget(code_gen_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, language_manager.get_text('language'))
        
    def apply_styling(self):
        """Apply modern styling to the dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #2b2b2b;
            }
            
            QTabWidget::tab-bar {
                alignment: left;
            }
            
            QTabBar::tab {
                background-color: #404040;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #555555;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #404040;
            }
            
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            
            QComboBox, QSpinBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                color: #ffffff;
                min-height: 20px;
            }
            
            QComboBox:hover, QSpinBox:hover {
                border-color: #0078d4;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
            }
            
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 6px 12px;
                color: #ffffff;
            }
            
            QPushButton:hover {
                background-color: #555555;
                border-color: #0078d4;
            }
            
            QPushButton:pressed {
                background-color: #0078d4;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background-color: #404040;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background-color: #106ebe;
            }
        """)
        
    def choose_color(self, color_type):
        """Choose a color for the specified type"""
        current_color = getattr(self, color_type, QColor(255, 255, 255))
        color = QColorDialog.getColor(current_color, self, f"Choose {color_type.replace('_', ' ').title()}")
        
        if color.isValid():
            setattr(self, color_type, color)
            button = getattr(self, f"{color_type}_btn")
            button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #555555;")
            
    def choose_font(self):
        """Choose application font"""
        current_font = getattr(self, 'selected_font', QFont())
        font, ok = QFontDialog.getFont(current_font, self, "Choose Font")
        
        if ok:
            self.selected_font = font
            self.font_label.setText(f"Current: {font.family()} {font.pointSize()}pt")
            
    def clear_recent_files(self):
        """Clear recent files list"""
        reply = QMessageBox.question(
            self, "Clear Recent Files",
            "Are you sure you want to clear the recent files list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Implementation to clear recent files
            QMessageBox.information(self, "Cleared", "Recent files list has been cleared.")
            
    def load_settings(self):
        """Load settings from file"""
        default_settings = {
            'auto_save_enabled': True,
            'auto_save_interval': 5,
            'recent_files_count': 10,
            'restore_session': True,
            'show_start_page': True,
            'theme': 'Dark',
            'grid_color': '#404040',
            'background_color': '#2b2b2b',
            'selection_color': '#0078d4',
            'font_family': 'System',
            'font_size': 9,
            'enable_animations': True,
            'animation_speed': 5,
            'show_grid': True,
            'snap_to_grid': True,
            'grid_size': 20,
            'show_connection_animation': True,
            'auto_arrange_connections': False,
            'connection_width': 2,
            'show_node_preview': True,
            'auto_resize_nodes': True,
            'real_time_validation': True,
            'show_validation_tooltips': True,
            'enable_antialiasing': True,
            'use_opengl': False,
            'max_fps': 60,
            'undo_levels': 100,
            'cache_thumbnails': True,
            'enable_multithreading': True,
            'thread_count': 4,
            'language': 'English',
            'code_language': 'English',
            'include_type_hints': True,
            'include_docstrings': True
        }
        
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    default_settings.update(loaded_settings)
        except Exception as e:
            print(f"Error loading settings: {e}")
            
        return default_settings
        
    def save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")
            
    def load_current_settings(self):
        """Load current settings into UI controls"""
        # General tab
        self.auto_save_enabled.setChecked(self.settings.get('auto_save_enabled', True))
        self.auto_save_interval.setValue(self.settings.get('auto_save_interval', 5))
        self.recent_files_count.setValue(self.settings.get('recent_files_count', 10))
        self.restore_session.setChecked(self.settings.get('restore_session', True))
        self.show_start_page.setChecked(self.settings.get('show_start_page', True))
        
        # Appearance tab
        theme_index = ['Dark', 'Light', 'Auto'].index(self.settings.get('theme', 'Dark'))
        self.theme_combo.setCurrentIndex(theme_index)
        
        # Editor tab
        self.show_grid.setChecked(self.settings.get('show_grid', True))
        self.snap_to_grid.setChecked(self.settings.get('snap_to_grid', True))
        self.grid_size.setValue(self.settings.get('grid_size', 20))
        
        # Language tab
        lang_index = ['English', '中文 (Chinese)', '日本語 (Japanese)'].index(self.settings.get('language', 'English'))
        self.language_combo.setCurrentIndex(lang_index)
        
    def apply_settings(self):
        """Apply current settings"""
        # Collect settings from UI
        self.settings.update({
            'auto_save_enabled': self.auto_save_enabled.isChecked(),
            'auto_save_interval': self.auto_save_interval.value(),
            'recent_files_count': self.recent_files_count.value(),
            'restore_session': self.restore_session.isChecked(),
            'show_start_page': self.show_start_page.isChecked(),
            'theme': self.theme_combo.currentText(),
            'show_grid': self.show_grid.isChecked(),
            'snap_to_grid': self.snap_to_grid.isChecked(),
            'grid_size': self.grid_size.value(),
            'language': self.language_combo.currentText()
        })
        
        self.save_settings()
        
        # Apply settings to parent window if available
        if self.parent_window:
            self.parent_window.apply_settings(self.settings)
            
    def accept_settings(self):
        """Accept and apply settings"""
        self.apply_settings()
        self.accept()
        
    def on_language_changed(self, language_text):
        """Handle language change in combo box"""
        # Map display text to language code
        lang_map = {
            'English': 'en',
            '中文 (Chinese)': 'zh', 
            '日本語 (Japanese)': 'ja'
        }
        
        lang_code = lang_map.get(language_text, 'en')
        language_manager.set_language(lang_code)
        
    def update_language(self):
        """Update all dialog texts when language changes"""
        # Update window title
        self.setWindowTitle(language_manager.get_text('settings'))
        
        # Update tab titles
        self.tab_widget.setTabText(0, language_manager.get_text('general'))
        self.tab_widget.setTabText(1, language_manager.get_text('appearance'))
        self.tab_widget.setTabText(2, language_manager.get_text('editor'))
        self.tab_widget.setTabText(3, language_manager.get_text('performance'))
        self.tab_widget.setTabText(4, language_manager.get_text('language'))
        
        # Update current language selection
        current_lang = language_manager.get_current_language()
        lang_display_map = {
            'en': 'English',
            'zh': '中文 (Chinese)',
            'ja': '日本語 (Japanese)'
        }
        
        display_text = lang_display_map.get(current_lang, 'English')
        index = self.language_combo.findText(display_text)
        if index >= 0:
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(index)
            self.language_combo.blockSignals(False)
        
    def get_settings(self):
        """Get current settings"""
        return self.settings