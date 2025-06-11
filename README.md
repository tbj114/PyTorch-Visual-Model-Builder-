# PyTorch Visual Model Builder

一个现代化的可视化PyTorch模型构建软件，通过拖拽操作来设计和构建深度学习模型。

## 功能特性

### 🎨 可视化建模
- **拖拽式界面**: 通过简单的拖拽操作添加和连接模型组件
- **实时预览**: 实时显示模型结构和参数统计
- **多路径支持**: 支持复杂的多分支模型架构
- **智能连接**: 自动验证连接的兼容性和数据流

### 🎯 模块管理
- **丰富的模块库**: 包含所有常用的PyTorch模块
  - 线性层 (Linear, Bilinear)
  - 卷积层 (Conv1d, Conv2d, Conv3d, ConvTranspose2d)
  - 归一化层 (BatchNorm, LayerNorm, GroupNorm)
  - 激活函数 (ReLU, Sigmoid, Tanh, GELU等)
  - 池化层 (MaxPool, AvgPool, AdaptivePool)
  - 循环层 (LSTM, GRU, RNN)
  - 注意力机制 (MultiheadAttention)
  - 正则化 (Dropout, AlphaDropout)
- **分类管理**: 模块按功能分类，便于查找
- **搜索功能**: 快速定位所需模块

### 🔧 智能编辑
- **属性面板**: 直观编辑模块参数
- **实时验证**: 即时检查模型的有效性
- **撤销/重做**: 完整的操作历史管理
- **复制/粘贴**: 快速复制模型组件
- **网格对齐**: 精确的布局控制

### 💾 文件管理
- **多格式支持**: 
  - `.ptmb` - 加密的项目文件格式
  - `.py` - 导出为Python代码
- **加密保护**: 项目文件采用密码加密
- **代码生成**: 自动生成完整的PyTorch代码
- **自动保存**: 防止数据丢失

### 🌍 多语言支持
- **界面语言**: 支持中文、英文、日文
- **代码注释**: 生成的代码支持多语言注释

### 🎨 现代化界面
- **深色主题**: 护眼的深色界面设计
- **响应式布局**: 自适应不同屏幕尺寸
- **流畅动画**: 丰富的视觉反馈
- **高DPI支持**: 完美支持高分辨率显示器

## 安装要求

### 系统要求
- Windows 10/11, macOS 10.14+, 或 Linux
- Python 3.7+
- 至少 4GB RAM
- 支持OpenGL的显卡（可选，用于硬件加速）

### Python依赖
```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch>=1.9.0` - PyTorch深度学习框架
- `PyQt5>=5.15.0` - GUI框架
- `cryptography>=3.4.8` - 文件加密
- `psutil>=5.8.0` - 系统信息监控

## 快速开始

### 1. 安装依赖
```bash
git clone <repository-url>
cd pytorch-visual-model-builder
pip install -r requirements.txt
```

### 2. 启动应用
```bash
python main.py
```

### 3. 创建第一个模型
1. 从模块面板拖拽一个"Input"模块到画布
2. 添加所需的层（如Linear、ReLU等）
3. 连接模块的输入输出端口
4. 在属性面板中调整参数
5. 验证模型（F5）
6. 导出为Python代码（Ctrl+E）

## 使用指南

### 基本操作

#### 添加模块
- 从左侧模块面板拖拽模块到画布
- 双击模块面板中的模块快速添加
- 使用搜索框快速查找模块

#### 连接模块
- 从输出端口（绿色）拖拽到输入端口（蓝色）
- 系统会自动验证连接的兼容性
- 不兼容的连接会显示警告

#### 编辑属性
- 选中模块后在右侧属性面板编辑参数
- 支持多选编辑公共属性
- 实时预览参数变化的影响

#### 模型验证
- 按F5或点击验证按钮检查模型
- 错误和警告会在状态栏显示
- 悬停在问题模块上查看详细信息

### 快捷键

#### 文件操作
- `Ctrl+N` - 新建模型
- `Ctrl+O` - 打开模型
- `Ctrl+S` - 保存模型
- `Ctrl+Shift+S` - 另存为
- `Ctrl+E` - 导出Python代码

#### 编辑操作
- `Ctrl+Z` - 撤销
- `Ctrl+Y` - 重做
- `Ctrl+C` - 复制
- `Ctrl+X` - 剪切
- `Ctrl+V` - 粘贴
- `Delete` - 删除选中项
- `Ctrl+A` - 全选

#### 视图操作
- `Ctrl++` - 放大
- `Ctrl+-` - 缩小
- `Ctrl+0` - 重置缩放
- `F11` - 全屏模式
- `F9` - 切换模块面板
- `F10` - 切换属性面板

#### 模型操作
- `F5` - 验证模型
- `F6` - 运行模型（测试）

### 高级功能

#### 自定义模块
可以通过继承基础模块类来创建自定义模块：

```python
class CustomModule(ModelNode):
    def __init__(self):
        super().__init__()
        self.module_type = "Custom"
        self.display_name = "My Custom Module"
        # 定义输入输出端口
        # 设置默认参数
```

#### 模型模板
保存常用的模型结构作为模板，快速创建新项目。

#### 批量操作
选中多个模块进行批量属性编辑、复制、删除等操作。

## 项目结构

```
pytorch-visual-model-builder/
├── main.py                 # 应用程序入口
├── model_builder.py        # 主窗口类
├── canvas.py              # 画布组件
├── model_node.py          # 模型节点类
├── port.py                # 连接端口类
├── connection.py          # 连接线类
├── grid_background.py     # 网格背景
├── module_palette.py      # 模块面板
├── property_panel.py      # 属性面板
├── toolbar.py             # 工具栏
├── menu_bar.py            # 菜单栏
├── status_bar.py          # 状态栏
├── settings_dialog.py     # 设置对话框
├── file_manager.py        # 文件管理器
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

## 开发指南

### 添加新模块类型

1. 在`model_node.py`中的`get_default_config`方法添加新模块配置
2. 在`module_palette.py`中添加模块按钮
3. 在`file_manager.py`中添加代码生成逻辑

### 自定义主题

修改各组件的`apply_styling`方法来自定义界面样式。

### 扩展文件格式

在`file_manager.py`中添加新的文件格式支持。

## 故障排除

### 常见问题

**Q: 应用启动失败**
A: 检查Python版本和依赖包是否正确安装

**Q: 模块连接失败**
A: 确保输出端口连接到输入端口，检查数据类型兼容性

**Q: 生成的代码运行错误**
A: 验证模型结构，确保所有必需参数都已设置

**Q: 文件加密/解密失败**
A: 确保密码正确，检查文件是否损坏

### 性能优化

- 在设置中禁用动画可提高性能
- 使用OpenGL加速（如果支持）
- 减少撤销历史级别以节省内存

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

### 代码规范

- 使用Python PEP 8代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 确保向后兼容性

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [your-email@example.com]

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 基础的可视化建模功能
- 支持常用PyTorch模块
- 文件加密和代码生成
- 多语言界面支持

---

**感谢使用PyTorch Visual Model Builder！** 🚀