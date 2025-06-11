#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Menu Bar - Provides comprehensive menu functionality
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from language_manager import language_manager

class MainMenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.actions = {}  # Store actions for language updates
        self.menus = {}    # Store menus for language updates
        
        # Connect to language change signal
        language_manager.language_changed.connect(self.update_language)
        
        self.init_menus()
        
    def init_menus(self):
        """Initialize all menus"""
        self.create_file_menu()
        self.create_edit_menu()
        self.create_view_menu()
        self.create_model_menu()
        self.create_tools_menu()
        self.create_help_menu()
        
    def create_file_menu(self):
        """Create File menu"""
        file_menu = self.addMenu(language_manager.get_text('file'))
        self.menus['file'] = file_menu
        
        # New
        new_action = QAction(language_manager.get_text('new'), self)
        new_action.setShortcut(QKeySequence.New)
        new_action.setStatusTip(language_manager.get_text('new_model'))
        new_action.triggered.connect(self.parent_window.new_model)
        file_menu.addAction(new_action)
        self.actions['new'] = new_action
        
        # Open
        open_action = QAction(language_manager.get_text('open'), self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip(language_manager.get_text('open_model'))
        open_action.triggered.connect(self.parent_window.open_model)
        file_menu.addAction(open_action)
        self.actions['open'] = open_action
        
        # Recent files submenu
        recent_menu = file_menu.addMenu(language_manager.get_text('recent_files'))
        self.menus['recent_files'] = recent_menu
        self.update_recent_files_menu(recent_menu)
        
        file_menu.addSeparator()
        
        # Save
        save_action = QAction(language_manager.get_text('save'), self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setStatusTip(language_manager.get_text('save_model'))
        save_action.triggered.connect(self.parent_window.save_model)
        file_menu.addAction(save_action)
        self.actions['save'] = save_action
        
        # Save As
        save_as_action = QAction(language_manager.get_text('save_as'), self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.setStatusTip(language_manager.get_text('save_as'))
        save_as_action.triggered.connect(self.parent_window.save_model_as)
        file_menu.addAction(save_as_action)
        self.actions['save_as'] = save_as_action
        
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu(language_manager.get_text('export'))
        self.menus['export'] = export_menu
        
        export_python_action = QAction(language_manager.get_text('generate_code'), self)
        export_python_action.setShortcut(QKeySequence("Ctrl+E"))
        export_python_action.setStatusTip(language_manager.get_text('generate_code'))
        export_python_action.triggered.connect(self.parent_window.export_to_python)
        export_menu.addAction(export_python_action)
        self.actions['export_python'] = export_python_action
        
        export_onnx_action = QAction(language_manager.get_text('export_onnx'), self)
        export_onnx_action.setStatusTip(language_manager.get_text('export_onnx'))
        export_onnx_action.triggered.connect(self.export_to_onnx)
        export_menu.addAction(export_onnx_action)
        self.actions['export_onnx'] = export_onnx_action
        
        export_image_action = QAction("Export as Image...", self)
        export_image_action.setStatusTip("Export model diagram as image")
        export_image_action.triggered.connect(self.export_as_image)
        export_menu.addAction(export_image_action)
        self.actions['export_image'] = export_image_action
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction(language_manager.get_text('exit'), self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip(language_manager.get_text('exit'))
        exit_action.triggered.connect(self.parent_window.close)
        file_menu.addAction(exit_action)
        self.actions['exit'] = exit_action
        
    def create_edit_menu(self):
        """Create Edit menu"""
        edit_menu = self.addMenu(language_manager.get_text('edit'))
        self.menus['edit'] = edit_menu
        
        # Undo
        undo_action = QAction(language_manager.get_text('undo'), self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.setStatusTip(language_manager.get_text('undo_status_tip'))
        undo_action.triggered.connect(self.parent_window.undo)
        edit_menu.addAction(undo_action)
        
        # Redo
        redo_action = QAction(language_manager.get_text('redo'), self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.setStatusTip(language_manager.get_text('redo_status_tip'))
        redo_action.triggered.connect(self.parent_window.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        # Cutsh'w'i'jei
        cut_action = QAction(language_manager.get_text('cut'), self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.setStatusTip(language_manager.get_text('cut_status_tip'))
        cut_action.triggered.connect(self.parent_window.cut_selection)
        edit_menu.addAction(cut_action)
        
        # Copy
        copy_action = QAction(language_manager.get_text('copy'), self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setStatusTip(language_manager.get_text('copy_status_tip'))
        copy_action.triggered.connect(self.parent_window.copy_selection)
        edit_menu.addAction(copy_action)
        
        # Paste
        paste_action = QAction(language_manager.get_text('paste'), self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.setStatusTip(language_manager.get_text('paste_status_tip'))
        paste_action.triggered.connect(self.parent_window.paste_selection)
        edit_menu.addAction(paste_action)
        
        # Delete
        delete_action = QAction(language_manager.get_text('delete'), self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.setStatusTip(language_manager.get_text('delete_status_tip'))
        delete_action.triggered.connect(self.parent_window.delete_selection)
        edit_menu.addAction(delete_action)
        
        edit_menu.addSeparator()
        
        # Select All
        select_all_action = QAction(language_manager.get_text('select_all'), self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        select_all_action.setStatusTip(language_manager.get_text('select_all_status_tip'))
        select_all_action.triggered.connect(self.parent_window.select_all)
        edit_menu.addAction(select_all_action)
        
        edit_menu.addSeparator()
        
        # Find
        find_action = QAction(language_manager.get_text('find'), self)
        find_action.setShortcut(QKeySequence.Find)
        find_action.setStatusTip(language_manager.get_text('find_status_tip'))
        find_action.triggered.connect(self.show_find_dialog)
        edit_menu.addAction(find_action)
        
        edit_menu.addSeparator()
        
        # Preferences
        preferences_action = QAction(language_manager.get_text('preferences'), self)
        preferences_action.setShortcut(QKeySequence("Ctrl+,"))
        preferences_action.setStatusTip(language_manager.get_text('preferences_status_tip'))
        preferences_action.triggered.connect(self.parent_window.open_settings)
        edit_menu.addAction(preferences_action)
        
    def create_view_menu(self):
        """Create View menu"""
        view_menu = self.addMenu(language_manager.get_text('view'))
        self.menus['view'] = view_menu
        
        # Zoom
        zoom_in_action = QAction(language_manager.get_text('zoom_in'), self)
        zoom_in_action.setShortcut(QKeySequence("Ctrl++"))
        zoom_in_action.setStatusTip(language_manager.get_text('zoom_in_status_tip'))
        zoom_in_action.triggered.connect(self.parent_window.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction(language_manager.get_text('zoom_out'), self)
        zoom_out_action.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_action.setStatusTip(language_manager.get_text('zoom_out_status_tip'))
        zoom_out_action.triggered.connect(self.parent_window.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        reset_zoom_action = QAction(language_manager.get_text('reset_zoom'), self)
        reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        reset_zoom_action.setStatusTip(language_manager.get_text('reset_zoom_status_tip'))
        reset_zoom_action.triggered.connect(self.parent_window.reset_zoom)
        view_menu.addAction(reset_zoom_action)
        
        view_menu.addSeparator()
        
        # Fit to window
        fit_action = QAction(language_manager.get_text('fit_to_window'), self)
        fit_action.setShortcut(QKeySequence("Ctrl+F"))
        fit_action.setStatusTip(language_manager.get_text('fit_to_window_status_tip'))
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        view_menu.addSeparator()
        
        # Panel visibility
        toggle_palette_action = QAction(language_manager.get_text('toggle_module_palette'), self)
        toggle_palette_action.setShortcut(QKeySequence("F9"))
        toggle_palette_action.setStatusTip(language_manager.get_text('toggle_module_palette_status_tip'))
        toggle_palette_action.triggered.connect(self.toggle_module_palette)
        view_menu.addAction(toggle_palette_action)
        
        toggle_properties_action = QAction(language_manager.get_text('toggle_properties_panel'), self)
        toggle_properties_action.setShortcut(QKeySequence("F10"))
        toggle_properties_action.setStatusTip(language_manager.get_text('toggle_properties_panel_status_tip'))
        toggle_properties_action.triggered.connect(self.toggle_properties_panel)
        view_menu.addAction(toggle_properties_action)
        
        view_menu.addSeparator()
        
        # Fullscreen
        fullscreen_action = QAction(language_manager.get_text('fullscreen'), self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.setStatusTip(language_manager.get_text('fullscreen_status_tip'))
        fullscreen_action.triggered.connect(self.parent_window.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
    def create_model_menu(self):
        """Create Model menu"""
        model_menu = self.addMenu(language_manager.get_text('model'))
        self.menus['model'] = model_menu
        
        # Validate
        validate_action = QAction(language_manager.get_text('validate_model'), self)
        validate_action.setShortcut(QKeySequence("F5"))
        validate_action.setStatusTip(language_manager.get_text('validate_model_status_tip'))
        validate_action.triggered.connect(self.parent_window.validate_model)
        model_menu.addAction(validate_action)
        
        # Run
        run_action = QAction(language_manager.get_text('run_model'), self)
        run_action.setShortcut(QKeySequence("F6"))
        run_action.setStatusTip(language_manager.get_text('run_model_status_tip'))
        run_action.triggered.connect(self.run_model)
        model_menu.addAction(run_action)
        
        model_menu.addSeparator()
        
        # Model statistics
        stats_action = QAction(language_manager.get_text('model_statistics'), self)
        stats_action.setStatusTip(language_manager.get_text('model_statistics_status_tip'))
        stats_action.triggered.connect(self.show_model_statistics)
        model_menu.addAction(stats_action)
        
        # Visualize architecture
        visualize_action = QAction(language_manager.get_text('visualize_architecture'), self)
        visualize_action.setStatusTip(language_manager.get_text('visualize_architecture_status_tip'))
        visualize_action.triggered.connect(self.visualize_architecture)
        model_menu.addAction(visualize_action)
        
        model_menu.addSeparator()
        
        # Optimize
        optimize_action = QAction(language_manager.get_text('optimize_model'), self)
        optimize_action.setStatusTip(language_manager.get_text('optimize_model_status_tip'))
        optimize_action.triggered.connect(self.optimize_model)
        model_menu.addAction(optimize_action)
        
    def create_tools_menu(self):
        """Create Tools menu"""
        tools_menu = self.addMenu("&Tools")
        
        # Code generator
        code_gen_action = QAction("&Code Generator", self)
        code_gen_action.setStatusTip("Open code generator")
        code_gen_action.triggered.connect(self.open_code_generator)
        tools_menu.addAction(code_gen_action)
        
        # Model converter
        converter_action = QAction("Model &Converter", self)
        converter_action.setStatusTip("Convert between model formats")
        converter_action.triggered.connect(self.open_model_converter)
        tools_menu.addAction(converter_action)
        
        tools_menu.addSeparator()
        
        # Dataset tools
        dataset_menu = tools_menu.addMenu(language_manager.get_text('dataset_tools'))
        self.menus['dataset_tools'] = dataset_menu
        
        load_dataset_action = QAction(language_manager.get_text('load_dataset'), self)
        load_dataset_action.setStatusTip(language_manager.get_text('load_dataset_status_tip'))
        load_dataset_action.triggered.connect(self.load_dataset)
        dataset_menu.addAction(load_dataset_action)
        
        preview_data_action = QAction(language_manager.get_text('preview_data'), self)
        preview_data_action.setStatusTip(language_manager.get_text('preview_data_status_tip'))
        preview_data_action.triggered.connect(self.preview_dataset)
        dataset_menu.addAction(preview_data_action)
        
        tools_menu.addSeparator()
        
        # Plugin manager
        plugins_action = QAction(language_manager.get_text('plugin_manager'), self)
        plugins_action.setStatusTip(language_manager.get_text('plugin_manager_status_tip'))
        plugins_action.triggered.connect(self.open_plugin_manager)
        tools_menu.addAction(plugins_action)
        
    def create_help_menu(self):
        """Create Help menu"""
        help_menu = self.addMenu(language_manager.get_text('help'))
        self.menus['help'] = help_menu
        
        # Documentation
        docs_action = QAction(language_manager.get_text('documentation'), self)
        docs_action.setShortcut(QKeySequence("F1"))
        docs_action.setStatusTip(language_manager.get_text('documentation_status_tip'))
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
        
        # Tutorials
        tutorials_action = QAction(language_manager.get_text('tutorials'), self)
        tutorials_action.setStatusTip(language_manager.get_text('tutorials_status_tip'))
        tutorials_action.triggered.connect(self.open_tutorials)
        help_menu.addAction(tutorials_action)
        
        # Examples
        examples_action = QAction(language_manager.get_text('examples'), self)
        examples_action.setStatusTip(language_manager.get_text('examples_status_tip'))
        examples_action.triggered.connect(self.open_examples)
        help_menu.addAction(examples_action)
        
        help_menu.addSeparator()
        
        # Keyboard shortcuts
        shortcuts_action = QAction(language_manager.get_text('keyboard_shortcuts'), self)
        shortcuts_action.setStatusTip(language_manager.get_text('keyboard_shortcuts_status_tip'))
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        help_menu.addSeparator()
        
        # Check for updates
        updates_action = QAction(language_manager.get_text('check_for_updates'), self)
        updates_action.setStatusTip(language_manager.get_text('check_for_updates_status_tip'))
        updates_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(updates_action)
        
        # About
        about_action = QAction(language_manager.get_text('about'), self)
        about_action.setStatusTip(language_manager.get_text('about_status_tip'))
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def update_recent_files_menu(self, menu):
        """Update recent files menu"""
        # Placeholder for recent files functionality
        menu.addAction("No recent files")
        
    def export_to_onnx(self):
        """Export model to ONNX format"""
        QMessageBox.information(self.parent_window, "Export ONNX", "ONNX export functionality will be implemented.")
        
    def export_as_image(self):
        """Export model diagram as image"""
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent_window, "Export as Image", "", 
            "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf)"
        )
        if file_path:
            # Implementation for image export
            QMessageBox.information(self.parent_window, "Export Image", f"Image export to {file_path} will be implemented.")
            
    def show_find_dialog(self):
        """Show find dialog"""
        QMessageBox.information(self.parent_window, self.language_manager.get_text("find_dialog_title"), self.language_manager.get_text("find_dialog_message"))
        
    def fit_to_window(self):
        """Fit model to window"""
        if hasattr(self.parent_window, 'canvas'):
            canvas = self.parent_window.canvas
            if canvas.nodes:
                # Calculate bounding box of all nodes
                min_x = min(node.x for node in canvas.nodes)
                max_x = max(node.x + node.width for node in canvas.nodes)
                min_y = min(node.y for node in canvas.nodes)
                max_y = max(node.y + node.height for node in canvas.nodes)
                
                # Add some padding
                padding = 50
                scene_rect = QRectF(min_x - padding, min_y - padding, 
                                  max_x - min_x + 2 * padding, 
                                  max_y - min_y + 2 * padding)
                
                # Fit the view to the scene
                canvas.fitInView(scene_rect, Qt.KeepAspectRatio)
                canvas.zoom_factor = canvas.transform().m11()
            
    def toggle_module_palette(self):
        """Toggle module palette visibility"""
        if hasattr(self.parent_window, 'module_palette'):
            palette = self.parent_window.module_palette
            palette.setVisible(not palette.isVisible())
            
    def toggle_properties_panel(self):
        """Toggle properties panel visibility"""
        if hasattr(self.parent_window, 'property_panel'):
            panel = self.parent_window.property_panel
            panel.setVisible(not panel.isVisible())
            
    def run_model(self):
        """Run the model with sample data"""
        try:
            # 获取当前模型数据
            model_data = self.parent_window.canvas.serialize()
            
            if not model_data['nodes']:
                QMessageBox.warning(self.parent_window, self.language_manager.get_text("run_model_empty_title"), self.language_manager.get_text("run_model_empty_message"))
                return
                
            # 验证模型
            errors = self.parent_window.canvas.validate_model()
            if errors:
                error_text = "\n".join(errors)
                QMessageBox.warning(self.parent_window, self.language_manager.get_text("model_validation_failed_title"), f'{self.language_manager.get_text("model_validation_failed_message")}\n{error_text}')
                return
                
            # 生成Python代码
            from file_manager import FileManager
            file_manager = FileManager()
            python_code = file_manager.generate_python_code(model_data)
            
            # 创建临时文件并执行
            import tempfile
            import subprocess
            import sys
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(python_code)
                temp_file = f.name
                
            # 执行模型测试
            try:
                result = subprocess.run([sys.executable, temp_file], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    QMessageBox.information(self.parent_window, self.language_manager.get_text("model_run_success_title"), 
                                          f'{self.language_manager.get_text("model_run_success_message")}\n\n{self.language_manager.get_text("output_label")}:\n{result.stdout}')
                else:
                    QMessageBox.warning(self.parent_window, self.language_manager.get_text("model_run_failed_title"), 
                                       f'{self.language_manager.get_text("model_run_failed_message")}:\n{result.stderr}')
            except subprocess.TimeoutExpired:
                QMessageBox.warning(self.parent_window, self.language_manager.get_text("model_run_timeout_title"), self.language_manager.get_text("model_run_timeout_message"))
            finally:
                import os
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            QMessageBox.critical(self.parent_window, self.language_manager.get_text("runtime_error_title"), f'{self.language_manager.get_text("runtime_error_message")}:\n{str(e)}')
        
    def show_model_statistics(self):
        """Show detailed model statistics"""
        try:
            # 获取模型统计信息
            stats = self.parent_window.canvas.calculate_model_stats()
            
            # 格式化统计信息
            stats_text = f'{self.language_manager.get_text("model_statistics_info")}:\n\n'
            stats_text += f'{self.language_manager.get_text("total_nodes")}: {stats["total_nodes"]}\n'
            stats_text += f'- {self.language_manager.get_text("input_nodes")}: {stats["input_nodes"]}\n'
            stats_text += f'- {self.language_manager.get_text("output_nodes")}: {stats["output_nodes"]}\n'
            stats_text += f'- {self.language_manager.get_text("hidden_nodes")}: {stats["hidden_nodes"]}\n\n'
            stats_text += f'{self.language_manager.get_text("total_connections")}: {stats["total_connections"]}\n\n'
            stats_text += f'{self.language_manager.get_text("total_parameters")}: {stats["total_parameters"]:,} {self.language_manager.get_text("parameters_unit")}\n'
            stats_text += f'{self.language_manager.get_text("estimated_memory_usage")}: {stats["memory_usage"]:.2f} MB'
            
            # 显示统计信息
            QMessageBox.information(self.parent_window, self.language_manager.get_text("model_statistics_title"), stats_text)
            
        except Exception as e:
            QMessageBox.warning(self.parent_window, self.language_manager.get_text("statistics_error_title"), f'{self.language_manager.get_text("calculate_statistics_error_message")}:\n{str(e)}')
        
    def visualize_architecture(self):
        """Show model architecture visualization using ASCII art"""
        try:
            # 获取模型数据
            model_data = self.parent_window.canvas.serialize()
            
            if not model_data['nodes']:
                QMessageBox.warning(self.parent_window, self.language_manager.get_text("architecture_visualization_title"), self.language_manager.get_text("architecture_visualization_empty_message"))
                return
            
            # 生成ASCII图形
            nodes = model_data['nodes']
            connections = model_data['connections']
            
            # 按层级组织节点
            layers = {}
            for node in nodes:
                layer = self._calculate_node_layer(node, connections)
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node)
            
            # 生成ASCII图
            ascii_art = [f'{self.language_manager.get_text("model_architecture_diagram")}:\n', ""]
            max_layer = max(layers.keys())
            
            for layer in range(max_layer + 1):
                if layer in layers:
                    # 添加节点
                    layer_nodes = layers[layer]
                    node_line = "   ".join([f"[{node['name']}]" for node in layer_nodes])
                    ascii_art.append(node_line)
                    
                    # 添加连接线（如果不是最后一层）
                    if layer < max_layer:
                        connections_line = ""
                        for node in layer_nodes:
                            out_connections = [c for c in connections 
                                              if c['source_node'] == node['id']]
                            if out_connections:
                                connections_line += "   |   "
                            else:
                                connections_line += "       "
                        ascii_art.append(connections_line)
                        
                        # 添加箭头
                        arrows_line = ""
                        for node in layer_nodes:
                            out_connections = [c for c in connections 
                                              if c['source_node'] == node['id']]
                            if out_connections:
                                arrows_line += "   v   "
                            else:
                                arrows_line += "       "
                        ascii_art.append(arrows_line)
            
            # 显示ASCII图
            dialog = QDialog(self.parent_window)
            dialog.setWindowTitle(self.language_manager.get_text("model_architecture_diagram_title"))
            dialog.setMinimumWidth(600)
            dialog.setMinimumHeight(400)
            
            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setFont(QFont("Courier New", 10))
            text_edit.setPlainText("\n".join(ascii_art))
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self.parent_window, self.language_manager.get_text("visualization_error_title"), f'{self.language_manager.get_text("generate_architecture_error_message")}:\n{str(e)}')
    
    def _calculate_node_layer(self, node, connections):
        """Calculate the layer number for a node based on its connections"""
        if not any(c for c in connections if c['target_node'] == node['id']):
            return 0
        
        input_nodes = [c['source_node'] for c in connections 
                      if c['target_node'] == node['id']]
        max_input_layer = -1
        
        for input_id in input_nodes:
            input_node = next(n for n in node if n['id'] == input_id)
            input_layer = self._calculate_node_layer(input_node, connections)
            max_input_layer = max(max_input_layer, input_layer)
        
        return max_input_layer + 1
        
    def optimize_model(self):
        """Optimize model for performance"""
        try:
            # 创建优化选项对话框
            dialog = QDialog(self.parent_window)
            dialog.setWindowTitle(self.language_manager.get_text("model_optimization_title"))
            dialog.setMinimumWidth(400)
            
            layout = QVBoxLayout()
            
            # 优化选项
            group_box = QGroupBox("优化选项")
            options_layout = QVBoxLayout()
            
            # 模型压缩
            compression_check = QCheckBox("模型压缩")
            compression_combo = QComboBox()
            compression_combo.addItems(["权重剪枝", "通道剪枝", "知识蒸馏"])
            compression_combo.setEnabled(False)
            compression_check.toggled.connect(compression_combo.setEnabled)
            options_layout.addWidget(compression_check)
            options_layout.addWidget(compression_combo)
            
            # 模型量化
            quantization_check = QCheckBox("模型量化")
            quantization_combo = QComboBox()
            quantization_combo.addItems(["动态量化", "静态量化", "量化感知训练"])
            quantization_combo.setEnabled(False)
            quantization_check.toggled.connect(quantization_combo.setEnabled)
            options_layout.addWidget(quantization_check)
            options_layout.addWidget(quantization_combo)
            
            # 性能分析
            profiling_check = QCheckBox("性能分析")
            profiling_options = QListWidget()
            profiling_options.addItems(["计算量分析", "内存使用分析", "推理延迟分析", "瓶颈检测"])
            profiling_options.setSelectionMode(QAbstractItemView.MultiSelection)
            profiling_options.setEnabled(False)
            profiling_check.toggled.connect(profiling_options.setEnabled)
            options_layout.addWidget(profiling_check)
            options_layout.addWidget(profiling_options)
            
            group_box.setLayout(options_layout)
            layout.addWidget(group_box)
            
            # 优化目标
            target_group = QGroupBox("优化目标")
            target_layout = QVBoxLayout()
            
            speed_radio = QRadioButton("速度优先")
            memory_radio = QRadioButton("内存优先")
            balance_radio = QRadioButton("平衡模式")
            balance_radio.setChecked(True)
            
            target_layout.addWidget(speed_radio)
            target_layout.addWidget(memory_radio)
            target_layout.addWidget(balance_radio)
            
            target_group.setLayout(target_layout)
            layout.addWidget(target_group)
            
            # 按钮
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            # 显示对话框
            if dialog.exec_() == QDialog.Accepted:
                # 收集优化选项
                options = {
                    'compression': {
                        'enabled': compression_check.isChecked(),
                        'method': compression_combo.currentText() if compression_check.isChecked() else None
                    },
                    'quantization': {
                        'enabled': quantization_check.isChecked(),
                        'method': quantization_combo.currentText() if quantization_check.isChecked() else None
                    },
                    'profiling': {
                        'enabled': profiling_check.isChecked(),
                        'methods': [item.text() for item in profiling_options.selectedItems()] 
                                  if profiling_check.isChecked() else []
                    },
                    'target': 'speed' if speed_radio.isChecked() else 
                              'memory' if memory_radio.isChecked() else 'balance'
                }
                
                # 开始优化
                QMessageBox.information(self.parent_window, "优化进行中", 
                    "模型优化功能正在开发中。\n\n" + 
                    "选择的优化选项：\n" + 
                    f"- 压缩方法：{options['compression']['method'] if options['compression']['enabled'] else '无'}\n" + 
                    f"- 量化方法：{options['quantization']['method'] if options['quantization']['enabled'] else '无'}\n" + 
                    f"- 性能分析：{', '.join(options['profiling']['methods']) if options['profiling']['enabled'] else '无'}\n" + 
                    f"- 优化目标：{options['target']}")
                
        except Exception as e:
            QMessageBox.warning(self.parent_window, "优化错误", f"模型优化过程中发生错误：\n{str(e)}")
        
    def open_code_generator(self):
        """Open advanced code generator"""
        try:
            # 创建代码生成器对话框
            dialog = QDialog(self.parent_window)
            dialog.setWindowTitle("高级代码生成器")
            dialog.setMinimumWidth(500)
            
            layout = QVBoxLayout()
            
            # 框架选择
            framework_group = QGroupBox("目标框架")
            framework_layout = QVBoxLayout()
            
            pytorch_radio = QRadioButton("PyTorch")
            tensorflow_radio = QRadioButton("TensorFlow")
            jax_radio = QRadioButton("JAX")
            pytorch_radio.setChecked(True)
            
            framework_layout.addWidget(pytorch_radio)
            framework_layout.addWidget(tensorflow_radio)
            framework_layout.addWidget(jax_radio)
            
            framework_group.setLayout(framework_layout)
            layout.addWidget(framework_group)
            
            # 代码生成选项
            options_group = QGroupBox("代码生成选项")
            options_layout = QVBoxLayout()
            
            # 基础选项
            basic_check = QCheckBox("生成基础模型代码")
            basic_check.setChecked(True)
            basic_check.setEnabled(False)  # 必选项
            options_layout.addWidget(basic_check)
            
            # 训练代码
            training_check = QCheckBox("包含训练代码")
            training_options = QListWidget()
            training_options.addItems(["数据加载器", "损失函数", "优化器配置", 
                                     "训练循环", "验证循环", "早停机制", 
                                     "学习率调度器", "模型保存/加载"])
            training_options.setSelectionMode(QAbstractItemView.MultiSelection)
            training_options.setEnabled(False)
            training_check.toggled.connect(training_options.setEnabled)
            options_layout.addWidget(training_check)
            options_layout.addWidget(training_options)
            
            # 评估代码
            evaluation_check = QCheckBox("包含评估代码")
            evaluation_options = QListWidget()
            evaluation_options.addItems(["性能指标计算", "混淆矩阵", 
                                       "预测可视化", "模型解释"])
            evaluation_options.setSelectionMode(QAbstractItemView.MultiSelection)
            evaluation_options.setEnabled(False)
            evaluation_check.toggled.connect(evaluation_options.setEnabled)
            options_layout.addWidget(evaluation_check)
            options_layout.addWidget(evaluation_options)
            
            # 部署选项
            deployment_check = QCheckBox("包含部署相关代码")
            deployment_options = QListWidget()
            deployment_options.addItems(["模型导出(ONNX)", "批处理推理", 
                                       "服务器API", "性能优化"])
            deployment_options.setSelectionMode(QAbstractItemView.MultiSelection)
            deployment_options.setEnabled(False)
            deployment_check.toggled.connect(deployment_options.setEnabled)
            options_layout.addWidget(deployment_check)
            options_layout.addWidget(deployment_options)
            
            options_group.setLayout(options_layout)
            layout.addWidget(options_group)
            
            # 代码风格选项
            style_group = QGroupBox("代码风格")
            style_layout = QVBoxLayout()
            
            # 注释级别
            comment_label = QLabel("注释详细程度：")
            comment_combo = QComboBox()
            comment_combo.addItems(["简洁", "标准", "详细"])
            comment_combo.setCurrentText("标准")
            style_layout.addWidget(comment_label)
            style_layout.addWidget(comment_combo)
            
            # 类型注解
            type_hints_check = QCheckBox("包含类型注解")
            type_hints_check.setChecked(True)
            style_layout.addWidget(type_hints_check)
            
            # 代码格式化
            formatting_check = QCheckBox("自动格式化代码")
            formatting_check.setChecked(True)
            style_layout.addWidget(formatting_check)
            
            style_group.setLayout(style_layout)
            layout.addWidget(style_group)
            
            # 按钮
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            # 显示对话框
            if dialog.exec_() == QDialog.Accepted:
                # 收集生成选项
                options = {
                    'framework': 'pytorch' if pytorch_radio.isChecked() else
                                'tensorflow' if tensorflow_radio.isChecked() else 'jax',
                    'training': {
                        'enabled': training_check.isChecked(),
                        'options': [item.text() for item in training_options.selectedItems()]
                    },
                    'evaluation': {
                        'enabled': evaluation_check.isChecked(),
                        'options': [item.text() for item in evaluation_options.selectedItems()]
                    },
                    'deployment': {
                        'enabled': deployment_check.isChecked(),
                        'options': [item.text() for item in deployment_options.selectedItems()]
                    },
                    'style': {
                        'comment_level': comment_combo.currentText(),
                        'type_hints': type_hints_check.isChecked(),
                        'formatting': formatting_check.isChecked()
                    }
                }
                
                # 开始生成代码
                QMessageBox.information(self.parent_window, "代码生成", 
                    "高级代码生成器正在开发中。\n\n" + 
                    "选择的生成选项：\n" + 
                    f"- 框架：{options['framework']}\n" + 
                    f"- 训练代码：{'、'.join(options['training']['options']) if options['training']['enabled'] else '无'}\n" + 
                    f"- 评估代码：{'、'.join(options['evaluation']['options']) if options['evaluation']['enabled'] else '无'}\n" + 
                    f"- 部署代码：{'、'.join(options['deployment']['options']) if options['deployment']['enabled'] else '无'}\n" + 
                    f"- 注释级别：{options['style']['comment_level']}\n" + 
                    f"- 类型注解：{'是' if options['style']['type_hints'] else '否'}\n" + 
                    f"- 代码格式化：{'是' if options['style']['formatting'] else '否'}")
                
        except Exception as e:
            QMessageBox.warning(self.parent_window, "生成错误", f"代码生成过程中发生错误：\n{str(e)}")
        
    def open_model_converter(self):
        """Open model converter dialog"""
        try:
            # 创建模型转换器对话框
            dialog = QDialog(self.parent_window)
            dialog.setWindowTitle("模型转换器")
            dialog.setMinimumWidth(500)
            
            layout = QVBoxLayout()
            
            # 源格式选择
            source_group = QGroupBox("源模型格式")
            source_layout = QVBoxLayout()
            
            source_combo = QComboBox()
            source_combo.addItems(["PyTorch (.pth, .pt)", "TensorFlow (.pb, .h5)", 
                                  "ONNX (.onnx)", "TensorFlow Lite (.tflite)", 
                                  "Keras (.h5, .keras)", "Caffe (.caffemodel)"])
            source_layout.addWidget(source_combo)
            
            source_group.setLayout(source_layout)
            layout.addWidget(source_group)
            
            # 目标格式选择
            target_group = QGroupBox("目标格式")
            target_layout = QVBoxLayout()
            
            target_combo = QComboBox()
            target_combo.addItems(["ONNX (.onnx)", "TensorFlow Lite (.tflite)", 
                                  "TorchScript (.pt)", "OpenVINO (IR)", 
                                  "TensorRT Engine", "CoreML (.mlmodel)"])
            target_layout.addWidget(target_combo)
            
            target_group.setLayout(target_layout)
            layout.addWidget(target_group)
            
            # 转换选项
            options_group = QGroupBox("转换选项")
            options_layout = QVBoxLayout()
            
            # 动态形状支持
            dynamic_shapes_check = QCheckBox("支持动态输入形状")
            options_layout.addWidget(dynamic_shapes_check)
            
            # 优化级别
            optimization_label = QLabel("优化级别：")
            optimization_combo = QComboBox()
            optimization_combo.addItems(["O0 (无优化)", "O1 (基础优化)", 
                                       "O2 (中等优化)", "O3 (完全优化)"])
            optimization_combo.setCurrentText("O1 (基础优化)")
            options_layout.addWidget(optimization_label)
            options_layout.addWidget(optimization_combo)
            
            # 量化选项
            quantization_check = QCheckBox("启用量化")
            quantization_combo = QComboBox()
            quantization_combo.addItems(["INT8", "FP16", "动态范围量化"])
            quantization_combo.setEnabled(False)
            quantization_check.toggled.connect(quantization_combo.setEnabled)
            options_layout.addWidget(quantization_check)
            options_layout.addWidget(quantization_combo)
            
            # 运行时优化
            runtime_group = QGroupBox("运行时优化")
            runtime_layout = QVBoxLayout()
            
            runtime_options = QListWidget()
            runtime_options.addItems(["内存优化", "算子融合", "图优化", 
                                     "并行计算", "缓存优化"])
            runtime_options.setSelectionMode(QAbstractItemView.MultiSelection)
            runtime_layout.addWidget(runtime_options)
            
            runtime_group.setLayout(runtime_layout)
            options_layout.addWidget(runtime_group)
            
            options_group.setLayout(options_layout)
            layout.addWidget(options_group)
            
            # 验证选项
            validation_group = QGroupBox("转换验证")
            validation_layout = QVBoxLayout()
            
            # 数值精度验证
            accuracy_check = QCheckBox("数值精度验证")
            accuracy_check.setChecked(True)
            validation_layout.addWidget(accuracy_check)
            
            # 性能基准测试
            benchmark_check = QCheckBox("性能基准测试")
            validation_layout.addWidget(benchmark_check)
            
            validation_group.setLayout(validation_layout)
            layout.addWidget(validation_group)
            
            # 按钮
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            # 显示对话框
            if dialog.exec_() == QDialog.Accepted:
                # 收集转换选项
                options = {
                    'source_format': source_combo.currentText(),
                    'target_format': target_combo.currentText(),
                    'dynamic_shapes': dynamic_shapes_check.isChecked(),
                    'optimization_level': optimization_combo.currentText(),
                    'quantization': {
                        'enabled': quantization_check.isChecked(),
                        'type': quantization_combo.currentText() if quantization_check.isChecked() else None
                    },
                    'runtime_optimizations': [item.text() for item in runtime_options.selectedItems()],
                    'validation': {
                        'accuracy': accuracy_check.isChecked(),
                        'benchmark': benchmark_check.isChecked()
                    }
                }
                
                # 开始转换
                QMessageBox.information(self.parent_window, "模型转换", 
                    "模型转换器正在开发中。\n\n" + 
                    "选择的转换选项：\n" + 
                    f"- 源格式：{options['source_format']}\n" + 
                    f"- 目标格式：{options['target_format']}\n" + 
                    f"- 动态形状：{'是' if options['dynamic_shapes'] else '否'}\n" + 
                    f"- 优化级别：{options['optimization_level']}\n" + 
                    f"- 量化：{options['quantization']['type'] if options['quantization']['enabled'] else '无'}\n" + 
                    f"- 运行时优化：{'、'.join(options['runtime_optimizations']) if options['runtime_optimizations'] else '无'}\n" + 
                    f"- 精度验证：{'是' if options['validation']['accuracy'] else '否'}\n" + 
                    f"- 性能测试：{'是' if options['validation']['benchmark'] else '否'}")
                
        except Exception as e:
            QMessageBox.warning(self.parent_window, "转换错误", f"模型转换过程中发生错误：\n{str(e)}")
        
    def load_dataset(self):
        """Load and manage datasets"""
        try:
            # 创建数据集管理对话框
            dialog = QDialog(self.parent_window)
            dialog.setWindowTitle("数据集管理器")
            dialog.setMinimumWidth(600)
            
            layout = QVBoxLayout()
            
            # 数据集来源选择
            source_group = QGroupBox("数据集来源")
            source_layout = QVBoxLayout()
            
            # 本地文件
            local_radio = QRadioButton("本地文件")
            local_path = QLineEdit()
            local_path.setPlaceholder("选择数据集文件夹或文件")
            browse_button = QPushButton("浏览...")
            browse_button.clicked.connect(lambda: local_path.setText(
                QFileDialog.getExistingDirectory(dialog, "选择数据集目录")))
            
            local_layout = QHBoxLayout()
            local_layout.addWidget(local_path)
            local_layout.addWidget(browse_button)
            
            source_layout.addWidget(local_radio)
            source_layout.addLayout(local_layout)
            
            # 内置数据集
            builtin_radio = QRadioButton("内置数据集")
            builtin_combo = QComboBox()
            builtin_combo.addItems(["MNIST", "CIFAR-10", "CIFAR-100", 
                                   "ImageNet (mini)", "VOC2012"])
            builtin_combo.setEnabled(False)
            builtin_radio.toggled.connect(builtin_combo.setEnabled)
            
            source_layout.addWidget(builtin_radio)
            source_layout.addWidget(builtin_combo)
            
            # 在线数据集
            online_radio = QRadioButton("在线数据集")
            online_combo = QComboBox()
            online_combo.addItems(["Hugging Face Datasets", "TensorFlow Datasets", 
                                  "Torchvision Datasets"])
            online_combo.setEnabled(False)
            online_radio.toggled.connect(online_combo.setEnabled)
            
            online_search = QLineEdit()
            online_search.setPlaceholder("搜索数据集...")
            online_search.setEnabled(False)
            online_radio.toggled.connect(online_search.setEnabled)
            
            source_layout.addWidget(online_radio)
            source_layout.addWidget(online_combo)
            source_layout.addWidget(online_search)
            
            local_radio.setChecked(True)
            source_group.setLayout(source_layout)
            layout.addWidget(source_group)
            
            # 数据预处理选项
            preprocess_group = QGroupBox("数据预处理")
            preprocess_layout = QVBoxLayout()
            
            # 基础预处理
            basic_list = QListWidget()
            basic_list.addItems(["调整大小", "标准化", "数据增强", 
                                "标签编码", "缺失值处理"])
            basic_list.setSelectionMode(QAbstractItemView.MultiSelection)
            preprocess_layout.addWidget(basic_list)
            
            # 高级预处理
            advanced_check = QCheckBox("启用高级预处理")
            advanced_list = QListWidget()
            advanced_list.addItems(["特征提取", "降维", "类别平衡", 
                                   "噪声过滤", "异常检测"])
            advanced_list.setSelectionMode(QAbstractItemView.MultiSelection)
            advanced_list.setEnabled(False)
            advanced_check.toggled.connect(advanced_list.setEnabled)
            
            preprocess_layout.addWidget(advanced_check)
            preprocess_layout.addWidget(advanced_list)
            
            preprocess_group.setLayout(preprocess_layout)
            layout.addWidget(preprocess_group)
            
            # 数据集分割
            split_group = QGroupBox("数据集分割")
            split_layout = QGridLayout()
            
            train_label = QLabel("训练集比例：")
            train_spin = QSpinBox()
            train_spin.setRange(0, 100)
            train_spin.setValue(70)
            train_spin.setSuffix("%")
            
            val_label = QLabel("验证集比例：")
            val_spin = QSpinBox()
            val_spin.setRange(0, 100)
            val_spin.setValue(15)
            val_spin.setSuffix("%")
            
            test_label = QLabel("测试集比例：")
            test_spin = QSpinBox()
            test_spin.setRange(0, 100)
            test_spin.setValue(15)
            test_spin.setSuffix("%")
            
            split_layout.addWidget(train_label, 0, 0)
            split_layout.addWidget(train_spin, 0, 1)
            split_layout.addWidget(val_label, 1, 0)
            split_layout.addWidget(val_spin, 1, 1)
            split_layout.addWidget(test_label, 2, 0)
            split_layout.addWidget(test_spin, 2, 1)
            
            split_group.setLayout(split_layout)
            layout.addWidget(split_group)
            
            # 数据可视化选项
            viz_group = QGroupBox("数据可视化")
            viz_layout = QVBoxLayout()
            
            viz_list = QListWidget()
            viz_list.addItems(["样本预览", "类别分布", "特征统计", 
                              "相关性分析", "数据质量报告"])
            viz_list.setSelectionMode(QAbstractItemView.MultiSelection)
            viz_layout.addWidget(viz_list)
            
            viz_group.setLayout(viz_layout)
            layout.addWidget(viz_group)
            
            # 按钮
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            # 显示对话框
            if dialog.exec_() == QDialog.Accepted:
                # 收集数据集选项
                options = {
                    'source': {
                        'type': 'local' if local_radio.isChecked() else
                                'builtin' if builtin_radio.isChecked() else 'online',
                        'path': local_path.text() if local_radio.isChecked() else None,
                        'dataset': builtin_combo.currentText() if builtin_radio.isChecked() else
                                  online_combo.currentText() if online_radio.isChecked() else None,
                        'search': online_search.text() if online_radio.isChecked() else None
                    },
                    'preprocessing': {
                        'basic': [item.text() for item in basic_list.selectedItems()],
                        'advanced': {
                            'enabled': advanced_check.isChecked(),
                            'methods': [item.text() for item in advanced_list.selectedItems()] 
                                      if advanced_check.isChecked() else []
                        }
                    },
                    'split': {
                        'train': train_spin.value(),
                        'val': val_spin.value(),
                        'test': test_spin.value()
                    },
                    'visualization': [item.text() for item in viz_list.selectedItems()]
                }
                
                # 开始加载和处理数据集
                QMessageBox.information(self.parent_window, "数据集管理", 
                    "数据集管理器正在开发中。\n\n" + 
                    "选择的数据集选项：\n" + 
                    f"- 数据来源：{options['source']['type']}\n" + 
                    (f"  路径：{options['source']['path']}\n" if options['source']['path'] else 
                     f"  数据集：{options['source']['dataset']}\n") + 
                    f"- 基础预处理：{'、'.join(options['preprocessing']['basic']) if options['preprocessing']['basic'] else '无'}\n" + 
                    f"- 高级预处理：{'、'.join(options['preprocessing']['advanced']['methods']) if options['preprocessing']['advanced']['enabled'] else '无'}\n" + 
                    f"- 数据集分割：训练集{options['split']['train']}%、验证集{options['split']['val']}%、测试集{options['split']['test']}%\n" + 
                    f"- 可视化选项：{'、'.join(options['visualization']) if options['visualization'] else '无'}")
                
        except Exception as e:
            QMessageBox.warning(self.parent_window, "数据集错误", f"加载数据集时发生错误：\n{str(e)}")
        
    def preview_dataset(self):
        """Preview loaded dataset"""
        QMessageBox.information(self.parent_window, "Preview Dataset", "Dataset preview will be implemented.")
        
    def open_plugin_manager(self):
        """Open plugin manager"""
        QMessageBox.information(self.parent_window, "Plugin Manager", "Plugin manager will be implemented.")
        
    def open_documentation(self):
        """Open documentation"""
        QMessageBox.information(self.parent_window, "Documentation", "Documentation will be implemented.")
        
    def open_tutorials(self):
        """Open tutorials"""
        QMessageBox.information(self.parent_window, "Tutorials", "Tutorials will be implemented.")
        
    def open_examples(self):
        """Open example models"""
        QMessageBox.information(self.parent_window, "Examples", "Example models will be implemented.")
        
    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
        Keyboard Shortcuts:
        
        File Operations:
        Ctrl+N - New Model
        Ctrl+O - Open Model
        Ctrl+S - Save Model
        Ctrl+Shift+S - Save As
        Ctrl+E - Export to Python
        
        Edit Operations:
        Ctrl+Z - Undo
        Ctrl+Y - Redo
        Ctrl+C - Copy
        Ctrl+X - Cut
        Ctrl+V - Paste
        Del - Delete
        Ctrl+A - Select All
        
        View Operations:
        Ctrl++ - Zoom In
        Ctrl+- - Zoom Out
        Ctrl+0 - Reset Zoom
        F11 - Fullscreen
        F9 - Toggle Module Palette
        F10 - Toggle Properties Panel
        
        Model Operations:
        F5 - Validate Model
        F6 - Run Model
        
        Other:
        Ctrl+, - Settings
        F1 - Help
        """
        QMessageBox.information(self.parent_window, "Keyboard Shortcuts", shortcuts_text)
        
    def check_for_updates(self):
        """Check for software updates"""
        QMessageBox.information(self.parent_window, "Updates", "Update checking will be implemented.")
        
    def update_language(self):
        """Update all menu texts when language changes"""
        # Update menu titles
        if 'file' in self.menus:
            self.menus['file'].setTitle(language_manager.get_text('file'))
        if 'edit' in self.menus:
            self.menus['edit'].setTitle(language_manager.get_text('edit'))
        if 'view' in self.menus:
            self.menus['view'].setTitle(language_manager.get_text('view'))
        if 'tools' in self.menus:
            self.menus['tools'].setTitle(language_manager.get_text('tools'))
        if 'help' in self.menus:
            self.menus['help'].setTitle(language_manager.get_text('help'))
        if 'export' in self.menus:
            self.menus['export'].setTitle(language_manager.get_text('export'))
        if 'recent_files' in self.menus:
            self.menus['recent_files'].setTitle(language_manager.get_text('recent_files'))
            
        # Update action texts
        if 'new' in self.actions:
            self.actions['new'].setText(language_manager.get_text('new'))
            self.actions['new'].setStatusTip(language_manager.get_text('new_model'))
        if 'open' in self.actions:
            self.actions['open'].setText(language_manager.get_text('open'))
            self.actions['open'].setStatusTip(language_manager.get_text('open_model'))
        if 'save' in self.actions:
            self.actions['save'].setText(language_manager.get_text('save'))
            self.actions['save'].setStatusTip(language_manager.get_text('save_model'))
        if 'save_as' in self.actions:
            self.actions['save_as'].setText(language_manager.get_text('save_as'))
            self.actions['save_as'].setStatusTip(language_manager.get_text('save_as'))
        if 'exit' in self.actions:
            self.actions['exit'].setText(language_manager.get_text('exit'))
            self.actions['exit'].setStatusTip(language_manager.get_text('exit'))
        if 'export_python' in self.actions:
            self.actions['export_python'].setText(language_manager.get_text('generate_code'))
            self.actions['export_python'].setStatusTip(language_manager.get_text('generate_code'))
        if 'export_onnx' in self.actions:
            self.actions['export_onnx'].setText(language_manager.get_text('export_onnx'))
            self.actions['export_onnx'].setStatusTip(language_manager.get_text('export_onnx'))
        
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
        <h2>{language_manager.get_text('app_title')}</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Description:</b> {language_manager.get_text('app_description')}</p>
        <p><b>Author:</b> AI Assistant</p>
        <p><b>License:</b> MIT License</p>
        <br>
        <p>{language_manager.get_text('about_description')}</p>
        """
        
        QMessageBox.about(self, language_manager.get_text('about'), about_text)