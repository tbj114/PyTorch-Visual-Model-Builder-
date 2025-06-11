#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Model Builder Window
"""

import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import custom components
from canvas import ModelCanvas
from module_palette import ModulePalette
from property_panel import PropertyPanel
from toolbar import MainToolBar
from menu_bar import MainMenuBar
from status_bar import StatusBar
from settings_dialog import SettingsDialog
from file_manager import FileManager
from language_manager import language_manager

class ModelBuilderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize file manager
        self.file_manager = FileManager(self)
        
        # 初始化当前文件和修改状态
        self.current_file = None
        self.is_modified = False
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        # Initialize UI
        self.init_ui()
        self.setup_shortcuts()
        self.setup_connections()
        
        # Load settings
        self.load_settings()
        
        # 设置定期清理缓存的定时器
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self.cleanup_temp_files)
        # 每30分钟清理一次缓存
        self.cleanup_timer.start(30 * 60 * 1000)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(language_manager.get_text('app_title'))
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Set window icon
        self.setWindowIcon(QIcon('resources/icons/app_icon.svg'))
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create left panel (module palette)
        self.module_palette = ModulePalette()
        self.module_palette.setFixedWidth(250)
        
        # Create center panel (canvas)
        self.canvas = ModelCanvas()
        
        # Create right panel (properties)
        self.property_panel = PropertyPanel()
        self.property_panel.setFixedWidth(300)
        
        # Create splitters for resizable panels
        left_splitter = QSplitter(Qt.Horizontal)
        left_splitter.addWidget(self.module_palette)
        left_splitter.addWidget(self.canvas)
        left_splitter.setSizes([250, 800])
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.property_panel)
        main_splitter.setSizes([1050, 300])
        
        main_layout.addWidget(main_splitter)
        
        # Create menu bar
        self.menu_bar = MainMenuBar(self)
        self.setMenuBar(self.menu_bar)
        
        # Create toolbar
        self.toolbar = MainToolBar(self)
        self.addToolBar(self.toolbar)
        
        # Create status bar
        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)
        
        # Apply modern styling
        self.apply_modern_style()
        
    def apply_modern_style(self):
        """Apply modern styling to the window"""
        style = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QSplitter::handle {
            background-color: #404040;
            width: 2px;
            height: 2px;
        }
        
        QSplitter::handle:hover {
            background-color: #0078d4;
        }
        
        QToolBar {
            background-color: #3c3c3c;
            border: none;
            spacing: 3px;
            padding: 5px;
        }
        
        QMenuBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border: none;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 12px;
        }
        
        QMenuBar::item:selected {
            background-color: #0078d4;
        }
        
        QStatusBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-top: 1px solid #555555;
        }
        """
        self.setStyleSheet(style)
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # File operations
        QShortcut(QKeySequence.New, self, self.new_model)
        QShortcut(QKeySequence.Open, self, self.open_model)
        QShortcut(QKeySequence.Save, self, self.save_model)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self.save_model_as)
        QShortcut(QKeySequence("Ctrl+E"), self, self.export_to_python)
        
        # Edit operations
        QShortcut(QKeySequence.Delete, self, self.delete_selection)
        QShortcut(QKeySequence.SelectAll, self, self.select_all)
        
        # View operations
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)

        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        
        # Other operations

        QShortcut(QKeySequence("Ctrl+,"), self, self.open_settings)
        
    def setup_connections(self):
        """Setup signal connections"""
        # Canvas signals
        self.canvas.selection_changed.connect(self.property_panel.update_properties)
        self.canvas.model_modified.connect(self.on_model_modified)
        
        # Module palette signals
        self.module_palette.module_selected.connect(self.canvas.set_current_module)
        
    def new_model(self):
        """Create a new model"""
        if self.check_unsaved_changes():
            self.canvas.clear()
            self.current_file = None
            self.is_modified = False
            self.update_window_title()
            
    def open_model(self):
        """Open an existing model"""
        if self.check_unsaved_changes():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Model", "", "PyTorch Model Builder Files (*.ptmb);;Python Files (*.py);;All Files (*)"
            )
            if file_path:
                try:
                    self.file_manager.load_model(file_path, self.canvas)
                    self.current_file = file_path
                    self.is_modified = False
                    self.update_window_title()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
                    
    def save_model(self):
        """Save the current model"""
        if self.current_file:
            try:
                self.file_manager.save_model(self.canvas.serialize(), self.current_file)
                self.is_modified = False
                self.update_window_title()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
        else:
            self.save_model_as()
            
    def save_model_as(self):
        """Save the current model with a new name"""
        try:
            success = self.file_manager.save_model_as(self.canvas.serialize())
            if success:
                self.current_file = self.file_manager.current_file
                self.is_modified = False
                self.update_window_title()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
                
    def export_to_python(self):
        """Export model to Python code"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to Python", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            try:
                self.file_manager.export_to_python(self.canvas.serialize(), file_path)
                QMessageBox.information(self, "Success", "Model exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
                
    def check_unsaved_changes(self):
        """Check for unsaved changes and prompt user"""
        if self.is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_model()
                return not self.is_modified
            elif reply == QMessageBox.Cancel:
                return False
        return True
        
    def on_model_modified(self):
        """Handle model modification"""
        self.is_modified = True
        self.update_window_title()
        
    def update_window_title(self):
        """Update window title"""
        title = "PyTorch Visual Model Builder"
        if self.current_file:
            title += f" - {os.path.basename(self.current_file)}"
        if self.is_modified:
            title += " *"
        self.setWindowTitle(title)
        
    def undo(self):
        """Undo last action"""
        self.canvas.undo()
        
    def redo(self):
        """Redo last undone action"""
        self.canvas.redo()
        
    def copy_selection(self):
        """Copy selected items"""
        self.canvas.copy_selection()
        
    def cut_selection(self):
        """Cut selected items"""
        self.canvas.cut_selection()
        
    def paste_selection(self):
        """Paste items from clipboard"""
        self.canvas.paste_selection()
        
    def delete_selection(self):
        """Delete selected items"""
        self.canvas.delete_selection()
        
    def select_all(self):
        """Select all items"""
        self.canvas.select_all()
        
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.canvas.reset_zoom()
        
    def zoom_in(self):
        """Zoom in"""
        self.canvas.zoom_in()
        
    def zoom_out(self):
        """Zoom out"""
        self.canvas.zoom_out()
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def validate_model(self):
        """Validate the current model"""
        errors = self.canvas.validate_model()
        if errors:
            error_text = "\n".join(errors)
            QMessageBox.warning(self, "Model Validation", f"Model has errors:\n{error_text}")
        else:
            QMessageBox.information(self, "Model Validation", "Model is valid!")
            
    def load_settings(self):
        """Load application settings"""
        try:
            import json
            import os
            settings_file = 'settings.json'
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    # Apply loaded settings
                    self.apply_settings(settings)
        except Exception as e:
            print(f"Failed to load settings: {e}")
            
    def apply_settings(self, settings):
        """Apply settings to the application"""
        # Apply theme
        if 'theme' in settings:
            self.apply_theme(settings['theme'])
            
        # Apply language
        if 'language' in settings:
            self.apply_language(settings['language'])
            
    def apply_theme(self, theme):
        """Apply theme settings"""
        if theme == 'dark':
            self.apply_modern_style()
            
    def apply_language(self, language):
        """Apply language settings"""
        # Map display text to language code
        lang_map = {
            'English': 'en',
            '中文 (Chinese)': 'zh', 
            '日本語 (Japanese)': 'ja'
        }
        
        lang_code = lang_map.get(language, 'en')
        language_manager.set_language(lang_code)
            
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Apply settings
            settings = dialog.get_settings()
            self.apply_settings(settings)
            
    def update_language(self):
        """Update all UI texts when language changes"""
        # Update window title
        self.setWindowTitle(language_manager.get_text('app_title'))
        
        # Update status bar message
        self.status_bar.showMessage(language_manager.get_text('ready'))
        
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Error cleaning up {file_path}: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.is_modified:
            reply = QMessageBox.question(self, language_manager.get_text('unsaved_changes'), 
                                       language_manager.get_text('save_before_closing'),
                                       QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            
            if reply == QMessageBox.Save:
                if self.file_manager.save_file():
                    self.cleanup_temp_files()
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.Discard:
                self.cleanup_temp_files()
                event.accept()
            else:
                event.ignore()
        else:
            self.cleanup_temp_files()
            event.accept()