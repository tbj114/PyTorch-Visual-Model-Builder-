#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Manager - Handles internationalization and localization
"""

import json
import os
from PyQt5.QtCore import QObject, pyqtSignal

class LanguageManager(QObject):
    language_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.current_language = 'en'
        self.translations = {}
        self.load_translations()
        
    def load_translations(self):
        """Load all translation files"""
        self.translations = {
            'en': {
                # Main Window
                'app_title': 'PyTorch Visual Model Builder',
                'app_description': 'A modern, visual interface for building PyTorch models through drag-and-drop operations.',
                'about_description': 'Built with PyQt5 and PyTorch. Features visual model construction, real-time validation, code generation, and multi-language support.',
                'ready': 'Ready',
                'file': 'File',
                'edit': 'Edit',
                'view': 'View',
                'tools': 'Tools',
                'help': 'Help',
                
                # File Menu
                'new': 'New',
                'open': 'Open',
                'save': 'Save',
                'save_as': 'Save As',
                'export': 'Export',
                'export_python': 'Export to Python',
                'export_onnx': 'Export to ONNX',
                'export_image': 'Export as Image',
                'import': 'Import',
                'recent_files': 'Recent Files',
                'exit': 'Exit',
                
                # Edit Menu
                'undo': 'Undo',
                'redo': 'Redo',
                'cut': 'Cut',
                'copy': 'Copy',
                'paste': 'Paste',
                'delete': 'Delete',
                'select_all': 'Select All',
                'find': 'Find',
                'preferences': 'Preferences',
                
                # View Menu
                'zoom_in': 'Zoom In',
                'zoom_out': 'Zoom Out',
                'zoom_reset': 'Reset Zoom',
                'reset_zoom': 'Reset Zoom',
                'fit_to_window': 'Fit to Window',
                'show_grid': 'Show Grid',
                'show_palette': 'Show Module Palette',
                'show_properties': 'Show Properties Panel',
                'fullscreen': 'Full Screen',
                
                # Tools Menu
                'validate_model': 'Validate Model',
                'generate_code': 'Generate Code',
                'model_converter': 'Model Converter',
                'settings': 'Settings',
                'model': 'Model',
                'validate_model_status_tip': 'Validate the current model',
                'run_model': 'Run Model',
                'run_model_status_tip': 'Run the model with sample data',
                'model_statistics': 'Model Statistics',
                'model_statistics_status_tip': 'Show detailed model statistics',
                'visualize_architecture': 'Visualize Architecture',
                'visualize_architecture_status_tip': 'Show model architecture visualization',
                'optimize_model': 'Optimize Model',
                'optimize_model_status_tip': 'Optimize model for performance',
                'code_generator': 'Code Generator',
                'code_generator_status_tip': 'Open code generator',
                'model_converter_status_tip': 'Convert between model formats',
                'dataset_tools': 'Dataset Tools',
                'load_dataset': 'Load Dataset',
                'load_dataset_status_tip': 'Load dataset for testing',
                'preview_data': 'Preview Data',
                'preview_data_status_tip': 'Preview loaded dataset',
                'plugin_manager': 'Plugin Manager',
                'plugin_manager_status_tip': 'Manage plugins',
                'check_for_updates': 'Check for Updates',
                'check_for_updates_status_tip': 'Check for software updates',
                
                # Help Menu
                'documentation': 'Documentation',
                'tutorials': 'Tutorials',
                'keyboard_shortcuts': 'Keyboard Shortcuts',
                'about': 'About',
                
                # Status Bar
                'nodes': 'Nodes',
                'connections': 'Connections',
                'parameters': 'Parameters',
                'memory': 'Memory',
                'cpu': 'CPU',
                
                # Module Palette
                'input_output': 'Input/Output',
                'linear_layers': 'Linear Layers',
                'convolution': 'Convolution',
                'pooling': 'Pooling',
                'normalization': 'Normalization',
                'activation': 'Activation',
                'dropout': 'Dropout',
                'recurrent': 'Recurrent',
                'attention': 'Attention',
                'custom': 'Custom',
                
                # Property Panel
                'properties': 'Properties',
                'node_properties': 'Node Properties',
                'connection_properties': 'Connection Properties',
                'model_properties': 'Model Properties',
                
                # Context Menu
                'add_node': 'Add Node',
                'delete_node': 'Delete Node',
                'duplicate_node': 'Duplicate Node',
                'delete_connection': 'Delete Connection',
                
                # Settings Dialog
                'general': 'General',
                'appearance': 'Appearance',
                'editor': 'Editor',
                'performance': 'Performance',
                'language': 'Language',
                'auto_save': 'Auto-save',
                'enable_auto_save': 'Enable auto-save',
                'auto_save_interval': 'Auto-save interval (minutes)',
                'recent_files_settings': 'Recent Files',
                'maximum_recent_files': 'Maximum recent files',
                'clear_recent_files': 'Clear Recent Files',
                'theme': 'Theme',
                'dark': 'Dark',
                'light': 'Light',
                'auto': 'Auto',
                'interface_language': 'Interface language',
                'code_comments_language': 'Code comments language',
                'code_generation_language': 'Code Generation Language',
                'include_type_hints': 'Include type hints in generated code',
                'include_docstrings': 'Include docstrings in generated code',
                'restart_note': 'Note: Language changes require application restart',
                
                # Messages
                'file_saved': 'File saved successfully',
                'file_loaded': 'File loaded successfully',
                'model_validated': 'Model validation completed',
                'code_generated': 'Code generated successfully',
                'error': 'Error',
                'warning': 'Warning',
                'success': 'Success',
                'info': 'Information',
                
                # Dialogs
                'confirm_exit': 'Are you sure you want to exit?',
                'unsaved_changes': 'You have unsaved changes. Do you want to save before closing?',
                'overwrite_file': 'File already exists. Do you want to overwrite it?',
                'save_before_closing': 'You have unsaved changes. Do you want to save before closing?',
                
                # Toolbar
                'new_model': 'New Model',
                'open_model': 'Open Model',
                'save_model': 'Save Model',
                'run_model': 'Run Model',
                'validate': 'Validate',
                'generate': 'Generate Code',
                
                # Additional UI elements
                'modules': 'modules',
                'no_model': 'No model',
                'new_tooltip': 'Create new model',
                'open_tooltip': 'Open existing model',
                'save_tooltip': 'Save current model',
                'undo_tooltip': 'Undo last action',
                'redo_tooltip': 'Redo last action',
                'copy_tooltip': 'Copy selection',
                'cut_tooltip': 'Cut selection',
                'paste_tooltip': 'Paste from clipboard',
                'delete_tooltip': 'Delete selection',
                'zoom_in_tooltip': 'Zoom in',
                'zoom_out_tooltip': 'Zoom out',
                'reset_zoom_tooltip': 'Reset zoom to 100%',
                'settings_tooltip': 'Open settings',
                'close_before_save': 'Do you want to save your changes before closing?',
                'module_palette': 'Module Palette',
                'search_modules': 'Search modules...',
                'category_basic': 'Basic Layers',
                'category_convolution': 'Convolution',
                'category_pooling': 'Pooling',
                'category_normalization': 'Normalization',
                'category_activation': 'Activation',
                'category_recurrent': 'Recurrent',
                'category_transformer': 'Transformer',
                'category_loss': 'Loss Functions'
            },
            
            'zh': {
                # Main Window
                'app_title': 'PyTorch 可视化模型构建器',
                'app_description': '通过拖放操作构建PyTorch模型的现代化可视界面。',
                'about_description': '使用PyQt5和PyTorch构建。具有可视化模型构建、实时验证、代码生成和多语言支持功能。',
                'ready': '就绪',
                'file': '文件',
                'edit': '编辑',
                'view': '视图',
                'tools': '工具',
                'help': '帮助',
                'undo_status_tip': '撤销上一步操作',
                'redo_status_tip': '重做上一步撤销的操作',
                'cut_status_tip': '剪切选中项',
                'copy_status_tip': '复制选中项',
                'paste_status_tip': '从剪贴板粘贴',
                'delete_status_tip': '删除选中项',
                'select_all_status_tip': '选择所有项',
                'find_status_tip': '查找模块',
                'preferences_status_tip': '打开偏好设置',
                'zoom_in_status_tip': '放大',
                'zoom_out_status_tip': '缩小',
                'reset_zoom_status_tip': '重置缩放比例为100%',
                'fit_to_window_status_tip': '将模型适应窗口大小',
                'toggle_module_palette': '切换模块面板',
                'toggle_module_palette_status_tip': '显示/隐藏模块面板',
                'toggle_properties_panel': '切换属性面板',
                'toggle_properties_panel_status_tip': '显示/隐藏属性面板',
                'fullscreen': '全屏',
                'fullscreen_status_tip': '切换全屏模式',
                'validate_model': '验证模型',
                'run_model': '运行模型',
                'model_statistics': '模型统计',
                'visualize_architecture': '可视化架构',
                'optimize_model': '优化模型',
                'code_generator': '代码生成器',
                'model_converter': '模型转换器',
                'dataset_tools': '数据集工具',
                'load_dataset': '加载数据集',
                'preview_data': '预览数据',
                'plugin_manager': '插件管理器',
                'documentation': '文档',
                'documentation_status_tip': '打开文档',
                'tutorials': '教程',
                'tutorials_status_tip': '打开教程',
                'examples': '示例',
                'examples_status_tip': '打开示例模型',
                'keyboard_shortcuts': '键盘快捷键',
                'keyboard_shortcuts_status_tip': '显示键盘快捷键',
                'check_for_updates': '检查更新',
                'check_for_updates_status_tip': '检查软件更新',
                'about': '关于',
                'about_status_tip': '关于此应用程序',
                
                # File Menu
                'new': '新建',
                'open': '打开',
                'save': '保存',
                'save_as': '另存为',
                'export': '导出',
                'export_python': '导出为Python',
                'export_onnx': '导出为ONNX',
                'export_image': '导出为图像',
                'import': '导入',
                'recent_files': '最近文件',
                'exit': '退出',
                
                # Edit Menu
                'undo': '撤销',
                'redo': '重做',
                'cut': '剪切',
                'copy': '复制',
                'paste': '粘贴',
                'delete': '删除',
                'select_all': '全选',
                'find': '查找',
                'preferences': '首选项',
                
                # View Menu
                'zoom_in': '放大',
                'zoom_out': '缩小',
                'zoom_reset': '重置缩放',
                'reset_zoom': '重置缩放',
                'fit_to_window': '适应窗口',
                'show_grid': '显示网格',
                'show_palette': '显示模块面板',
                'show_properties': '显示属性面板',
                'fullscreen': '全屏',
                
                # Tools Menu
                'validate_model': '验证模型',
                'generate_code': '生成代码',
                'model_converter': '模型转换器',
                'settings': '设置',
                'model': '模型',
                
                # Help Menu
                'documentation': '文档',
                'tutorials': '教程',
                'keyboard_shortcuts': '键盘快捷键',
                'about': '关于',
                
                # Status Bar
                'nodes': '节点',
                'connections': '连接',
                'parameters': '参数',
                'memory': '内存',
                'cpu': '处理器',
                
                # Module Palette
                'input_output': '输入/输出',
                'linear_layers': '线性层',
                'convolution': '卷积',
                'pooling': '池化',
                'normalization': '归一化',
                'activation': '激活函数',
                'dropout': 'Dropout',
                'recurrent': '循环网络',
                'attention': '注意力机制',
                'custom': '自定义',
                
                # Property Panel
                'properties': '属性',
                'node_properties': '节点属性',
                'connection_properties': '连接属性',
                'model_properties': '模型属性',
                
                # Context Menu
                'add_node': '添加节点',
                'delete_node': '删除节点',
                'duplicate_node': '复制节点',
                'delete_connection': '删除连接',
                
                # Settings Dialog
                'general': '常规',
                'appearance': '外观',
                'editor': '编辑器',
                'performance': '性能',
                'language': '语言',
                'auto_save': '自动保存',
                'enable_auto_save': '启用自动保存',
                'auto_save_interval': '自动保存间隔（分钟）',
                'recent_files_settings': '最近文件',
                'maximum_recent_files': '最大最近文件数',
                'clear_recent_files': '清除最近文件',
                'theme': '主题',
                'dark': '深色',
                'light': '浅色',
                'auto': '自动',
                'interface_language': '界面语言',
                'code_comments_language': '代码注释语言',
                'code_generation_language': '代码生成语言',
                'include_type_hints': '在生成的代码中包含类型提示',
                'include_docstrings': '在生成的代码中包含文档字符串',
                'restart_note': '注意：语言更改需要重启应用程序',
                
                # Messages
                'file_saved': '文件保存成功',
                'file_loaded': '文件加载成功',
                'model_validated': '模型验证完成',
                'code_generated': '代码生成成功',
                'error': '错误',
                'warning': '警告',
                'success': '成功',
                'info': '信息',
                
                # Dialogs
                'confirm_exit': '确定要退出吗？',
                'unsaved_changes': '您有未保存的更改。是否要在关闭前保存？',
                'overwrite_file': '文件已存在。是否要覆盖它？',
                'save_before_closing': '您有未保存的更改。是否要在关闭前保存？',
                
                # Toolbar
                'new_model': '新建模型',
                'open_model': '打开模型',
                'save_model': '保存模型',
                'run_model': '运行模型',
                'validate': '验证',
                'generate': '生成代码',
                
                # Additional UI elements
                'modules': '模块',
                'no_model': '无模型',
                'new_tooltip': '创建新模型',
                'open_tooltip': '打开现有模型',
                'save_tooltip': '保存当前模型',
                'undo_tooltip': '撤销上一个操作',
                'redo_tooltip': '重做上一个操作',
                'copy_tooltip': '复制选择',
                'cut_tooltip': '剪切选择',
                'paste_tooltip': '从剪贴板粘贴',
                'delete_tooltip': '删除选择',
                'zoom_in_tooltip': '放大',
                'zoom_out_tooltip': '缩小',
                'reset_zoom_tooltip': '重置缩放到100%',
                'settings_tooltip': '打开设置',
                'close_before_save': '关闭前是否要保存更改？',
                'module_palette': '模块面板',
                'search_modules': '搜索模块...',
                'category_basic': '基础层',
                'category_convolution': '卷积',
                'category_pooling': '池化',
                'category_normalization': '归一化',
                'category_activation': '激活',
                'category_recurrent': '循环',
                'category_transformer': 'Transformer',
                'category_loss': '损失函数'
            },
            
            'ja': {
                # Main Window
                'app_title': 'PyTorch ビジュアルモデルビルダー',
                'app_description': 'ドラッグアンドドロップ操作でPyTorchモデルを構築するモダンなビジュアルインターフェース。',
                'about_description': 'PyQt5とPyTorchで構築。ビジュアルモデル構築、リアルタイム検証、コード生成、多言語サポート機能を備えています。',
                'ready': '準備完了',
                'file': 'ファイル',
                'edit': '編集',
                'view': '表示',
                'tools': 'ツール',
                'help': 'ヘルプ',
                
                # File Menu
                'new': '新規',
                'open': '開く',
                'save': '保存',
                'save_as': '名前を付けて保存',
                'export': 'エクスポート',
                'export_python': 'Pythonにエクスポート',
                'export_onnx': 'ONNXにエクスポート',
                'export_image': '画像としてエクスポート',
                'import': 'インポート',
                'recent_files': '最近のファイル',
                'exit': '終了',
                
                # Edit Menu
                'undo': '元に戻す',
                'redo': 'やり直し',
                'cut': '切り取り',
                'copy': 'コピー',
                'paste': '貼り付け',
                'delete': '削除',
                'select_all': 'すべて選択',
                'find': '検索',
                'preferences': '環境設定',
                
                # View Menu
                'zoom_in': '拡大',
                'zoom_out': '縮小',
                'zoom_reset': 'ズームリセット',
                'reset_zoom': 'ズームリセット',
                'fit_to_window': 'ウィンドウに合わせる',
                'show_grid': 'グリッド表示',
                'show_palette': 'モジュールパレット表示',
                'show_properties': 'プロパティパネル表示',
                'fullscreen': 'フルスクリーン',
                
                # Tools Menu
                'validate_model': 'モデル検証',
                'generate_code': 'コード生成',
                'model_converter': 'モデルコンバーター',
                'settings': '設定',
                'model': 'モデル',
                
                # Help Menu
                'documentation': 'ドキュメント',
                'tutorials': 'チュートリアル',
                'keyboard_shortcuts': 'キーボードショートカット',
                'about': 'について',
                
                # Status Bar
                'nodes': 'ノード',
                'connections': '接続',
                'parameters': 'パラメータ',
                'memory': 'メモリ',
                'cpu': 'CPU',
                
                # Module Palette
                'input_output': '入力/出力',
                'linear_layers': '線形層',
                'convolution': '畳み込み',
                'pooling': 'プーリング',
                'normalization': '正規化',
                'activation': '活性化関数',
                'dropout': 'ドロップアウト',
                'recurrent': 'リカレント',
                'attention': 'アテンション',
                'custom': 'カスタム',
                
                # Property Panel
                'properties': 'プロパティ',
                'node_properties': 'ノードプロパティ',
                'connection_properties': '接続プロパティ',
                'model_properties': 'モデルプロパティ',
                
                # Context Menu
                'add_node': 'ノード追加',
                'delete_node': 'ノード削除',
                'duplicate_node': 'ノード複製',
                'delete_connection': '接続削除',
                
                # Settings Dialog
                'general': '一般',
                'appearance': '外観',
                'editor': 'エディタ',
                'performance': 'パフォーマンス',
                'language': '言語',
                'auto_save': '自動保存',
                'enable_auto_save': '自動保存を有効にする',
                'auto_save_interval': '自動保存間隔（分）',
                'recent_files_settings': '最近のファイル',
                'maximum_recent_files': '最大最近ファイル数',
                'clear_recent_files': '最近のファイルをクリア',
                'theme': 'テーマ',
                'dark': 'ダーク',
                'light': 'ライト',
                'auto': '自動',
                'interface_language': 'インターフェース言語',
                'code_comments_language': 'コードコメント言語',
                'code_generation_language': 'コード生成言語',
                'include_type_hints': '生成されたコードに型ヒントを含める',
                'include_docstrings': '生成されたコードにドキュメント文字列を含める',
                'restart_note': '注意：言語の変更にはアプリケーションの再起動が必要です',
                
                # Messages
                'file_saved': 'ファイルが正常に保存されました',
                'file_loaded': 'ファイルが正常に読み込まれました',
                'model_validated': 'モデル検証が完了しました',
                'code_generated': 'コードが正常に生成されました',
                'error': 'エラー',
                'warning': '警告',
                'success': '成功',
                'info': '情報',
                
                # Dialogs
                'confirm_exit': '本当に終了しますか？',
                'unsaved_changes': '保存されていない変更があります。閉じる前に保存しますか？',
                'overwrite_file': 'ファイルが既に存在します。上書きしますか？',
                'save_before_closing': '未保存の変更があります。閉じる前に保存しますか？',
                
                # Toolbar
                'new_model': '新しいモデル',
                'open_model': 'モデルを開く',
                'save_model': 'モデルを保存',
                'run_model': 'モデルを実行',
                'validate': '検証',
                'generate': 'コード生成',
                
                # Additional UI elements
                'modules': 'モジュール',
                'no_model': 'モデルなし',
                'new_tooltip': '新しいモデルを作成',
                'open_tooltip': '既存のモデルを開く',
                'save_tooltip': '現在のモデルを保存',
                'undo_tooltip': '最後の操作を元に戻す',
                'redo_tooltip': '最後の操作をやり直し',
                'copy_tooltip': '選択をコピー',
                'cut_tooltip': '選択を切り取り',
                'paste_tooltip': 'クリップボードから貼り付け',
                'delete_tooltip': '選択を削除',
                'zoom_in_tooltip': '拡大',
                'zoom_out_tooltip': '縮小',
                'reset_zoom_tooltip': 'ズームを100%にリセット',
                'settings_tooltip': '設定を開く',
                'close_before_save': '閉じる前に変更を保存しますか？',
                'module_palette': 'モジュールパレット',
                'search_modules': 'モジュールを検索...',
                'category_basic': '基本レイヤー',
                'category_convolution': '畳み込み',
                'category_pooling': 'プーリング',
                'category_normalization': '正規化',
                'category_activation': '活性化',
                'category_recurrent': '再帰',
                'category_transformer': 'Transformer',
                'category_loss': '損失関数'
            }
        }
        
        self.translations['en'].update({
            "find_dialog_title": "Find",
            "find_dialog_message": "Find functionality will be implemented.",
            "run_model_empty_title": "Run Model",
            "run_model_empty_message": "Model is empty, please add some nodes first.",
            "model_validation_failed_title": "Model Validation Failed",
            "model_validation_failed_message": "The model has the following issues",
            "model_run_success_title": "Model Run Success",
            "model_run_success_message": "Model test completed!",
            "output_label": "Output",
            "model_run_failed_title": "Model Run Failed",
            "model_run_failed_message": "Model execution error",
            "model_run_timeout_title": "Model Run Timeout",
            "model_run_timeout_message": "Model execution timed out (30 seconds), please check model complexity.",
            "runtime_error_title": "Runtime Error",
            "runtime_error_message": "An error occurred during model runtime",
            "model_statistics_info": "Model Statistics",
            "total_nodes": "Total Nodes",
            "input_nodes": "- Input Nodes",
            "output_nodes": "- Output Nodes",
            "hidden_nodes": "- Hidden Nodes",
            "total_connections": "Total Connections",
            "total_parameters": "Model Parameters",
            "parameters_unit": "Parameters",
            "estimated_memory_usage": "Estimated Memory Usage",
            "model_statistics_title": "Model Statistics",
            "statistics_error_title": "Statistics Error",
            "calculate_statistics_error_message": "An error occurred while calculating model statistics",
            "architecture_visualization_title": "Architecture Visualization",
            "architecture_visualization_empty_message": "Model is empty, please add some nodes first.",
            "model_architecture_diagram": "Model Architecture Diagram",
            "model_architecture_diagram_title": "Model Architecture Diagram",
            "visualization_error_title": "Visualization Error",
            "generate_architecture_error_message": "An error occurred while generating the architecture diagram",
            "model_optimization_title": "Model Optimization"
        })

        self.translations['zh'].update({
            "find_dialog_title": "查找",
            "find_dialog_message": "查找功能将在此处实现。",
            "run_model_empty_title": "运行模型",
            "run_model_empty_message": "模型为空，请先添加一些节点。",
            "model_validation_failed_title": "模型验证失败",
            "model_validation_failed_message": "模型存在以下问题",
            "model_run_success_title": "模型运行成功",
            "model_run_success_message": "模型测试完成！",
            "output_label": "输出",
            "model_run_failed_title": "模型运行失败",
            "model_run_failed_message": "模型执行出错",
            "model_run_timeout_title": "模型运行超时",
            "model_run_timeout_message": "模型执行超时（30秒），请检查模型复杂度。",
            "runtime_error_title": "运行错误",
            "runtime_error_message": "模型运行时发生错误",
            "model_statistics_info": "模型统计信息",
            "total_nodes": "总节点数",
            "input_nodes": "- 输入节点",
            "output_nodes": "- 输出节点",
            "hidden_nodes": "- 隐藏节点",
            "total_connections": "总连接数",
            "total_parameters": "模型参数量",
            "parameters_unit": "参数",
            "estimated_memory_usage": "预估内存占用",
            "model_statistics_title": "模型统计",
            "statistics_error_title": "统计错误",
            "calculate_statistics_error_message": "计算模型统计信息时发生错误",
            "architecture_visualization_title": "架构可视化",
            "architecture_visualization_empty_message": "模型为空，请先添加一些节点。",
            "model_architecture_diagram": "模型架构图",
            "model_architecture_diagram_title": "模型架构图",
            "visualization_error_title": "可视化错误",
            "generate_architecture_error_message": "生成架构图时发生错误",
            "model_optimization_title": "模型优化"
        })

    def set_language(self, language_code):
        """Set the current language"""
        if language_code in self.translations:
            self.current_language = language_code
            self.language_changed.emit(language_code)
            
    def get_text(self, key, default=None):
        """Get translated text for the given key"""
        if default is None:
            default = key
        return self.translations.get(self.current_language, {}).get(key, default)
        
    def get_current_language(self):
        """Get the current language code"""
        return self.current_language
        
    def get_available_languages(self):
        """Get list of available languages"""
        return list(self.translations.keys())
        
# Global language manager instance
language_manager = LanguageManager()