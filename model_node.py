#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Node - Represents a PyTorch module in the visual editor
"""

import math
import uuid
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from port import Port

class ModelNode(QGraphicsItem):
    
    def __init__(self, module_type, position=None):
        super().__init__()
        self.module_type = module_type
        self.is_active = True
        self.node_width = 120
        self.node_height = 80
        self.corner_radius = 8
        self.node_id = str(uuid.uuid4()) # Add a unique ID for each node
        
        # Module configuration
        self.config = self.get_default_config(module_type)
        
        # Ports
        self.input_ports = []
        self.output_ports = []
        self.create_ports()
        
        # Visual properties
        self.title_font = QFont("Arial", 10, QFont.Bold)
        self.param_font = QFont("Arial", 8)
        
        # Set flags
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # Set position
        if position:
            self.setPos(position)
            
    def get_default_config(self, module_type):
        """Get default configuration for module type"""
        configs = {
            'Input': {
                'shape': [1, 3, 224, 224],
                'name': 'input'
            },
            'Linear': {
                'in_features': 512,
                'out_features': 256,
                'bias': True
            },
            'Conv2d': {
                'in_channels': 3,
                'out_channels': 64,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'bias': True
            },
            'Conv1d': {
                'in_channels': 1,
                'out_channels': 64,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'bias': True
            },
            'ConvTranspose2d': {
                'in_channels': 64,
                'out_channels': 3,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'output_padding': 0,
                'bias': True
            },
            'BatchNorm2d': {
                'num_features': 64,
                'eps': 1e-5,
                'momentum': 0.1
            },
            'BatchNorm1d': {
                'num_features': 64,
                'eps': 1e-5,
                'momentum': 0.1
            },
            'LayerNorm': {
                'normalized_shape': [256],
                'eps': 1e-5,
                'elementwise_affine': True
            },
            'GroupNorm': {
                'num_groups': 32,
                'num_channels': 256,
                'eps': 1e-5,
                'affine': True
            },
            'ReLU': {
                'inplace': False
            },
            'LeakyReLU': {
                'negative_slope': 0.01,
                'inplace': False
            },
            'GELU': {
                'approximate': 'none'
            },
            'Swish': {
                'beta': 1.0
            },
            'Sigmoid': {},
            'Tanh': {},
            'Softmax': {
                'dim': -1
            },
            'Dropout': {
                'p': 0.5,
                'inplace': False
            },
            'Dropout2d': {
                'p': 0.5,
                'inplace': False
            },
            'AlphaDropout': {
                'p': 0.5,
                'inplace': False
            },
            'MaxPool2d': {
                'kernel_size': 2,
                'stride': 2,
                'padding': 0
            },
            'AvgPool2d': {
                'kernel_size': 2,
                'stride': 2,
                'padding': 0
            },
            'AdaptiveAvgPool2d': {
                'output_size': [1, 1]
            },
            'AdaptiveMaxPool2d': {
                'output_size': [1, 1]
            },
            'Flatten': {
                'start_dim': 1,
                'end_dim': -1
            },
            'Reshape': {
                'shape': [-1, 256]
            },
            'Permute': {
                'dims': [0, 2, 1]
            },
            'Squeeze': {
                'dim': None
            },
            'Unsqueeze': {
                'dim': 0
            },
            'LSTM': {
                'input_size': 128,
                'hidden_size': 256,
                'num_layers': 1,
                'bias': True,
                'batch_first': True,
                'dropout': 0.0,
                'bidirectional': False
            },
            'GRU': {
                'input_size': 128,
                'hidden_size': 256,
                'num_layers': 1,
                'bias': True,
                'batch_first': True,
                'dropout': 0.0,
                'bidirectional': False
            },
            'RNN': {
                'input_size': 128,
                'hidden_size': 256,
                'num_layers': 1,
                'nonlinearity': 'tanh',
                'bias': True,
                'batch_first': True,
                'dropout': 0.0,
                'bidirectional': False
            },
            'Embedding': {
                'num_embeddings': 10000,
                'embedding_dim': 128,
                'padding_idx': None,
                'max_norm': None,
                'norm_type': 2.0,
                'scale_grad_by_freq': False,
                'sparse': False
            },
            'MultiheadAttention': {
                'embed_dim': 512,
                'num_heads': 8,
                'dropout': 0.0,
                'bias': True,
                'add_bias_kv': False,
                'add_zero_attn': False,
                'kdim': None,
                'vdim': None,
                'batch_first': True
            },
            'TransformerEncoderLayer': {
                'd_model': 512,
                'nhead': 8,
                'dim_feedforward': 2048,
                'dropout': 0.1,
                'activation': 'relu',
                'layer_norm_eps': 1e-5,
                'batch_first': False,
                'norm_first': False
            },
            'TransformerDecoderLayer': {
                'd_model': 512,
                'nhead': 8,
                'dim_feedforward': 2048,
                'dropout': 0.1,
                'activation': 'relu',
                'layer_norm_eps': 1e-5,
                'batch_first': False,
                'norm_first': False
            },
            'Output': {
                'name': 'output'
            }
        }
        return configs.get(module_type, {})
        
    def create_ports(self):
        """Create input and output ports based on module type"""
        # Clear existing ports
        self.input_ports.clear()
        self.output_ports.clear()
        
        # Define port configurations
        port_configs = {
            'Input': {'inputs': 0, 'outputs': 1},
            'Linear': {'inputs': 1, 'outputs': 1},
            'Conv2d': {'inputs': 1, 'outputs': 1},
            'Conv1d': {'inputs': 1, 'outputs': 1},
            'BatchNorm2d': {'inputs': 1, 'outputs': 1},
            'BatchNorm1d': {'inputs': 1, 'outputs': 1},
            'ReLU': {'inputs': 1, 'outputs': 1},
            'LeakyReLU': {'inputs': 1, 'outputs': 1},
            'Sigmoid': {'inputs': 1, 'outputs': 1},
            'Tanh': {'inputs': 1, 'outputs': 1},
            'Softmax': {'inputs': 1, 'outputs': 1},
            'Dropout': {'inputs': 1, 'outputs': 1},
            'MaxPool2d': {'inputs': 1, 'outputs': 1},
            'AvgPool2d': {'inputs': 1, 'outputs': 1},
            'AdaptiveAvgPool2d': {'inputs': 1, 'outputs': 1},
            'Flatten': {'inputs': 1, 'outputs': 1},
            'LSTM': {'inputs': 1, 'outputs': 2},  # output, (h_n, c_n)
            'GRU': {'inputs': 1, 'outputs': 2},   # output, h_n
            'Embedding': {'inputs': 1, 'outputs': 1},
            'MultiheadAttention': {'inputs': 3, 'outputs': 2},  # query, key, value -> attn_output, attn_weights
            'TransformerEncoderLayer': {'inputs': 1, 'outputs': 1},
            'Output': {'inputs': 1, 'outputs': 0}
        }
        
        config = port_configs.get(self.module_type, {'inputs': 1, 'outputs': 1})
        
        # Create input ports
        for i in range(config['inputs']):
            port = Port('input', self)
            port.setParentItem(self)
            self.input_ports.append(port)
            
        # Create output ports
        for i in range(config['outputs']):
            port = Port('output', self)
            port.setParentItem(self)
            self.output_ports.append(port)
            
        self.update_port_positions()
        
    def update_port_positions(self):
        """Update positions of ports"""
        # Position input ports on the left
        if self.input_ports:
            port_spacing = self.node_height / (len(self.input_ports) + 1)
            for i, port in enumerate(self.input_ports):
                y = port_spacing * (i + 1) - port.port_radius
                port.setPos(-port.port_radius, y)
                
        # Position output ports on the right
        if self.output_ports:
            port_spacing = self.node_height / (len(self.output_ports) + 1)
            for i, port in enumerate(self.output_ports):
                y = port_spacing * (i + 1) - port.port_radius
                port.setPos(self.node_width - port.port_radius, y)
                
    def boundingRect(self):
        """Return the bounding rectangle"""
        return QRectF(0, 0, self.node_width, self.node_height)
        
    def paint(self, painter, option, widget):
        """Paint the node"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get colors based on state
        if self.isSelected():
            border_color = QColor(100, 150, 255)
            border_width = 2
        else:
            border_color = QColor(80, 80, 80)
            border_width = 1
            
        if self.is_active:
            fill_color = self.get_module_color()
        else:
            fill_color = QColor(60, 60, 60)
            
        # Draw main rectangle
        rect = QRectF(0, 0, self.node_width, self.node_height)
        
        # Draw shadow
        shadow_rect = rect.translated(2, 2)
        painter.setBrush(QBrush(QColor(0, 0, 0, 50)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(shadow_rect, self.corner_radius, self.corner_radius)
        
        # Draw main body
        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(rect, self.corner_radius, self.corner_radius)
        
        # Draw title
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(self.title_font)
        title_rect = QRectF(5, 5, self.node_width - 10, 20)
        painter.drawText(title_rect, Qt.AlignCenter, self.module_type)
        
        # Draw parameters
        painter.setFont(self.param_font)
        painter.setPen(QPen(QColor(200, 200, 200)))
        
        y_offset = 25
        for key, value in list(self.config.items())[:3]:  # Show first 3 parameters
            if isinstance(value, (int, float)):
                text = f"{key}: {value}"
            elif isinstance(value, bool):
                text = f"{key}: {value}"
            elif isinstance(value, list):
                text = f"{key}: {value}"
            else:
                text = f"{key}: {str(value)[:10]}..."
                
            param_rect = QRectF(5, y_offset, self.node_width - 10, 12)
            painter.drawText(param_rect, Qt.AlignLeft, text)
            y_offset += 12
            
        # Draw activation indicator
        if not self.is_active:
            painter.setBrush(QBrush(QColor(255, 0, 0, 100)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, self.corner_radius, self.corner_radius)
            
            # Draw "INACTIVE" text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, "INACTIVE")
            
    def get_module_color(self):
        """Get color based on module type"""
        color_map = {
            'Input': QColor(100, 200, 100),      # Green
            'Output': QColor(200, 100, 100),     # Red
            'Linear': QColor(100, 150, 200),     # Blue
            'Conv2d': QColor(150, 100, 200),     # Purple
            'Conv1d': QColor(140, 90, 190),      # Purple variant
            'BatchNorm2d': QColor(200, 150, 100), # Orange
            'BatchNorm1d': QColor(190, 140, 90),  # Orange variant
            'ReLU': QColor(200, 200, 100),       # Yellow
            'LeakyReLU': QColor(190, 190, 90),   # Yellow variant
            'Sigmoid': QColor(180, 180, 80),     # Yellow variant
            'Tanh': QColor(170, 170, 70),        # Yellow variant
            'Softmax': QColor(160, 160, 60),     # Yellow variant
            'Dropout': QColor(150, 150, 150),    # Gray
            'MaxPool2d': QColor(100, 200, 200),  # Cyan
            'AvgPool2d': QColor(90, 190, 190),   # Cyan variant
            'AdaptiveAvgPool2d': QColor(80, 180, 180), # Cyan variant
            'Flatten': QColor(200, 100, 200),    # Magenta
            'LSTM': QColor(150, 200, 150),       # Light green
            'GRU': QColor(140, 190, 140),        # Light green variant
            'Embedding': QColor(200, 150, 200),  # Light magenta
            'MultiheadAttention': QColor(100, 100, 200), # Dark blue
            'TransformerEncoderLayer': QColor(120, 120, 220) # Dark blue variant
        }
        return color_map.get(self.module_type, QColor(120, 120, 120))
        
    def set_active(self, active):
        """Set the activation state of the node"""
        self.is_active = active
        self.update()
        
    def itemChange(self, change, value):
        """Handle item changes"""
        if change == QGraphicsItem.ItemPositionHasChanged:
            # Update connections
            for port in self.input_ports + self.output_ports:
                port.update_connections()
        return super().itemChange(change, value)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            # Check if clicking on a port
            for port in self.input_ports + self.output_ports:
                if port.contains(port.mapFromParent(event.pos())):
                    # Let the port handle the event
                    return
                    
        super().mousePressEvent(event)
        
    def get_config_value(self, key):
        """Get a configuration value"""
        return self.config.get(key)
        
    def set_config_value(self, key, value):
        """Set a configuration value"""
        self.config[key] = value
        self.update()
        
    def get_input_shape(self):
        """Get expected input shape"""
        # This would be calculated based on the module type and configuration
        # For now, return a placeholder
        return None
        
    def get_output_shape(self, input_shape=None):
        """Calculate output shape given input shape"""
        # This would calculate the actual output shape
        # For now, return a placeholder
        return None
        
    def calculate_parameters(self):
        """Calculate number of parameters for this module"""
        if self.module_type == 'Linear':
            in_features = self.config.get('in_features', 512)
            out_features = self.config.get('out_features', 256)
            bias = self.config.get('bias', True)
            params = in_features * out_features
            if bias:
                params += out_features
            return params
            
        elif self.module_type == 'Conv2d':
            in_channels = self.config.get('in_channels', 3)
            out_channels = self.config.get('out_channels', 64)
            kernel_size = self.config.get('kernel_size', 3)
            bias = self.config.get('bias', True)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            params = in_channels * out_channels * kernel_size[0] * kernel_size[1]
            if bias:
                params += out_channels
            return params
            
        elif self.module_type == 'Conv1d':
            in_channels = self.config.get('in_channels', 1)
            out_channels = self.config.get('out_channels', 64)
            kernel_size = self.config.get('kernel_size', 3)
            bias = self.config.get('bias', True)
            params = in_channels * out_channels * kernel_size
            if bias:
                params += out_channels
            return params
            
        elif self.module_type == 'BatchNorm2d' or self.module_type == 'BatchNorm1d':
            num_features = self.config.get('num_features', 64)
            return num_features * 2  # weight and bias
            
        elif self.module_type == 'LSTM':
            input_size = self.config.get('input_size', 128)
            hidden_size = self.config.get('hidden_size', 256)
            num_layers = self.config.get('num_layers', 1)
            bidirectional = self.config.get('bidirectional', False)
            bias = self.config.get('bias', True)
            num_directions = 2 if bidirectional else 1
            
            # For each layer: 4 sets of weights and biases for input, hidden state
            params = 0
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                params += 4 * (layer_input_size * hidden_size + hidden_size * hidden_size)
                if bias:
                    params += 4 * 2 * hidden_size
            return params * num_directions
            
        elif self.module_type == 'GRU':
            input_size = self.config.get('input_size', 128)
            hidden_size = self.config.get('hidden_size', 256)
            num_layers = self.config.get('num_layers', 1)
            bidirectional = self.config.get('bidirectional', False)
            bias = self.config.get('bias', True)
            num_directions = 2 if bidirectional else 1
            
            # For each layer: 3 sets of weights and biases for input, hidden state
            params = 0
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                params += 3 * (layer_input_size * hidden_size + hidden_size * hidden_size)
                if bias:
                    params += 3 * 2 * hidden_size
            return params * num_directions
            
        elif self.module_type == 'Embedding':
            num_embeddings = self.config.get('num_embeddings', 10000)
            embedding_dim = self.config.get('embedding_dim', 128)
            return num_embeddings * embedding_dim
            
        elif self.module_type == 'MultiheadAttention':
            embed_dim = self.config.get('embed_dim', 512)
            num_heads = self.config.get('num_heads', 8)
            bias = self.config.get('bias', True)
            # 3 linear layers for Q,K,V + 1 output linear layer
            params = 4 * embed_dim * embed_dim
            if bias:
                params += 4 * embed_dim
            return params
            
        elif self.module_type == 'TransformerEncoderLayer':
            d_model = self.config.get('d_model', 512)
            dim_feedforward = self.config.get('dim_feedforward', 2048)
            # Self attention + feedforward + layer norm
            attention_params = 4 * d_model * d_model + 4 * d_model
            feedforward_params = d_model * dim_feedforward + dim_feedforward + dim_feedforward * d_model + d_model
            layer_norm_params = 4 * d_model  # 2 layer norms with weight and bias each
            return attention_params + feedforward_params + layer_norm_params
            
        return 0  # Return 0 for modules without parameters
        
    def serialize(self):
        """Serialize node to dictionary"""
        return {
            'type': 'ModelNode',
            'module_type': self.module_type,
            'position': [self.pos().x(), self.pos().y()],
            'config': self.config,
            'is_active': self.is_active,
            'node_id': self.node_id
        }
        
    @staticmethod
    def deserialize(data):
        """Deserialize node from dictionary"""
        node = ModelNode(data['module_type'])
        node.setPos(QPointF(data['position'][0], data['position'][1]))
        node.config = data['config']
        node.is_active = data.get('is_active', True)
        node.node_id = data.get('node_id', str(uuid.uuid4())) # Ensure old models can be loaded
        return node
        
    def generate_pytorch_code(self, var_name):
        """Generate PyTorch code for this module"""
        if self.module_type == 'Input':
            shape = self.config.get('shape', [1, 3, 224, 224])
            return f"torch.randn{tuple(shape)}"
            
        elif self.module_type == 'Linear':
            in_features = self.config.get('in_features', 512)
            out_features = self.config.get('out_features', 256)
            bias = self.config.get('bias', True)
            return f"nn.Linear({in_features}, {out_features}, bias={bias})"
            
        elif self.module_type == 'Conv2d':
            in_channels = self.config.get('in_channels', 3)
            out_channels = self.config.get('out_channels', 64)
            kernel_size = self.config.get('kernel_size', 3)
            stride = self.config.get('stride', 1)
            padding = self.config.get('padding', 1)
            bias = self.config.get('bias', True)
            return f"nn.Conv2d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, padding={padding}, bias={bias})"
            
        elif self.module_type == 'ReLU':
            inplace = self.config.get('inplace', False)
            return f"nn.ReLU(inplace={inplace})"
            
        elif self.module_type == 'MaxPool2d':
            kernel_size = self.config.get('kernel_size', 2)
            stride = self.config.get('stride', 2)
            padding = self.config.get('padding', 0)
            return f"nn.MaxPool2d({kernel_size}, stride={stride}, padding={padding})"
            
        elif self.module_type == 'Conv1d':
            in_channels = self.config.get('in_channels', 1)
            out_channels = self.config.get('out_channels', 64)
            kernel_size = self.config.get('kernel_size', 3)
            stride = self.config.get('stride', 1)
            padding = self.config.get('padding', 1)
            bias = self.config.get('bias', True)
            return f"nn.Conv1d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, padding={padding}, bias={bias})"

        elif self.module_type == 'ConvTranspose2d':
            in_channels = self.config.get('in_channels', 64)
            out_channels = self.config.get('out_channels', 3)
            kernel_size = self.config.get('kernel_size', 4)
            stride = self.config.get('stride', 2)
            padding = self.config.get('padding', 1)
            output_padding = self.config.get('output_padding', 0)
            bias = self.config.get('bias', True)
            return f"nn.ConvTranspose2d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding}, bias={bias})"

        elif self.module_type == 'LayerNorm':
            normalized_shape = self.config.get('normalized_shape', [256])
            eps = self.config.get('eps', 1e-5)
            elementwise_affine = self.config.get('elementwise_affine', True)
            return f"nn.LayerNorm({normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine})"

        elif self.module_type == 'GroupNorm':
            num_groups = self.config.get('num_groups', 32)
            num_channels = self.config.get('num_channels', 256)
            eps = self.config.get('eps', 1e-5)
            affine = self.config.get('affine', True)
            return f"nn.GroupNorm({num_groups}, {num_channels}, eps={eps}, affine={affine})"

        elif self.module_type == 'GELU':
            approximate = self.config.get('approximate', 'none')
            return f"nn.GELU(approximate='{approximate}')"

        elif self.module_type == 'Swish':
            beta = self.config.get('beta', 1.0)
            return f"nn.SiLU() if {beta} == 1.0 else nn.ReLU() * torch.sigmoid(beta * nn.ReLU())"

        elif self.module_type == 'AdaptiveMaxPool2d':
            output_size = self.config.get('output_size', (1, 1))
            return f"nn.AdaptiveMaxPool2d({output_size})"

        elif self.module_type == 'RNN':
            input_size = self.config.get('input_size', 128)
            hidden_size = self.config.get('hidden_size', 256)
            num_layers = self.config.get('num_layers', 1)
            nonlinearity = self.config.get('nonlinearity', 'tanh')
            bias = self.config.get('bias', True)
            batch_first = self.config.get('batch_first', True)
            dropout = self.config.get('dropout', 0)
            bidirectional = self.config.get('bidirectional', False)
            return f"nn.RNN({input_size}, {hidden_size}, num_layers={num_layers}, nonlinearity='{nonlinearity}', bias={bias}, batch_first={batch_first}, dropout={dropout}, bidirectional={bidirectional})"

        elif self.module_type == 'TransformerDecoderLayer':
            d_model = self.config.get('d_model', 512)
            nhead = self.config.get('nhead', 8)
            dim_feedforward = self.config.get('dim_feedforward', 2048)
            dropout = self.config.get('dropout', 0.1)
            activation = self.config.get('activation', 'relu')
            layer_norm_eps = self.config.get('layer_norm_eps', 1e-5)
            batch_first = self.config.get('batch_first', False)
            norm_first = self.config.get('norm_first', False)
            return f"nn.TransformerDecoderLayer({d_model}, {nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}', layer_norm_eps={layer_norm_eps}, batch_first={batch_first}, norm_first={norm_first})"

        elif self.module_type == 'Dropout2d':
            p = self.config.get('p', 0.5)
            inplace = self.config.get('inplace', False)
            return f"nn.Dropout2d(p={p}, inplace={inplace})"

        elif self.module_type == 'AlphaDropout':
            p = self.config.get('p', 0.5)
            inplace = self.config.get('inplace', False)
            return f"nn.AlphaDropout(p={p}, inplace={inplace})"

        elif self.module_type == 'Reshape':
            shape = self.config.get('shape', [-1, 256])
            return f"lambda x: x.reshape({shape})"

        elif self.module_type == 'Permute':
            dims = self.config.get('dims', [0, 2, 1])
            return f"lambda x: x.permute({dims})"

        elif self.module_type == 'Squeeze':
            dim = self.config.get('dim', None)
            return f"lambda x: x.squeeze({dim})" if dim is not None else "lambda x: x.squeeze()"

        elif self.module_type == 'Unsqueeze':
            dim = self.config.get('dim', 0)
            return f"lambda x: x.unsqueeze({dim})"

            padding = self.config.get('padding', 1)
            bias = self.config.get('bias', True)
            return f"nn.Conv1d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, padding={padding}, bias={bias})"

        elif self.module_type == 'BatchNorm2d':
            num_features = self.config.get('num_features', 64)
            eps = self.config.get('eps', 1e-5)
            momentum = self.config.get('momentum', 0.1)
            affine = self.config.get('affine', True)
            track_running_stats = self.config.get('track_running_stats', True)
            return f"nn.BatchNorm2d({num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats})"

        elif self.module_type == 'BatchNorm1d':
            num_features = self.config.get('num_features', 64)
            eps = self.config.get('eps', 1e-5)
            momentum = self.config.get('momentum', 0.1)
            affine = self.config.get('affine', True)
            track_running_stats = self.config.get('track_running_stats', True)
            return f"nn.BatchNorm1d({num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats})"

        elif self.module_type == 'LeakyReLU':
            negative_slope = self.config.get('negative_slope', 0.01)
            inplace = self.config.get('inplace', False)
            return f"nn.LeakyReLU(negative_slope={negative_slope}, inplace={inplace})"

        elif self.module_type == 'Sigmoid':
            return "nn.Sigmoid()"

        elif self.module_type == 'Tanh':
            return "nn.Tanh()"

        elif self.module_type == 'Softmax':
            dim = self.config.get('dim', 1)
            return f"nn.Softmax(dim={dim})"

        elif self.module_type == 'Dropout':
            p = self.config.get('p', 0.5)
            inplace = self.config.get('inplace', False)
            return f"nn.Dropout(p={p}, inplace={inplace})"

        elif self.module_type == 'AvgPool2d':
            kernel_size = self.config.get('kernel_size', 2)
            stride = self.config.get('stride', 2)
            padding = self.config.get('padding', 0)
            return f"nn.AvgPool2d({kernel_size}, stride={stride}, padding={padding})"

        elif self.module_type == 'AdaptiveAvgPool2d':
            output_size = self.config.get('output_size', (1, 1))
            return f"nn.AdaptiveAvgPool2d({output_size})"

        elif self.module_type == 'Flatten':
            start_dim = self.config.get('start_dim', 1)
            end_dim = self.config.get('end_dim', -1)
            return f"nn.Flatten(start_dim={start_dim}, end_dim={end_dim})"

        elif self.module_type == 'LSTM':
            input_size = self.config.get('input_size', 128)
            hidden_size = self.config.get('hidden_size', 256)
            num_layers = self.config.get('num_layers', 1)
            bias = self.config.get('bias', True)
            batch_first = self.config.get('batch_first', True)
            dropout = self.config.get('dropout', 0)
            bidirectional = self.config.get('bidirectional', False)
            return f"nn.LSTM({input_size}, {hidden_size}, num_layers={num_layers}, bias={bias}, batch_first={batch_first}, dropout={dropout}, bidirectional={bidirectional})"

        elif self.module_type == 'GRU':
            input_size = self.config.get('input_size', 128)
            hidden_size = self.config.get('hidden_size', 256)
            num_layers = self.config.get('num_layers', 1)
            bias = self.config.get('bias', True)
            batch_first = self.config.get('batch_first', True)
            dropout = self.config.get('dropout', 0)
            bidirectional = self.config.get('bidirectional', False)
            return f"nn.GRU({input_size}, {hidden_size}, num_layers={num_layers}, bias={bias}, batch_first={batch_first}, dropout={dropout}, bidirectional={bidirectional})"

        elif self.module_type == 'Embedding':
            num_embeddings = self.config.get('num_embeddings', 10000)
            embedding_dim = self.config.get('embedding_dim', 128)
            padding_idx = self.config.get('padding_idx', None)
            max_norm = self.config.get('max_norm', None)
            norm_type = self.config.get('norm_type', 2.0)
            scale_grad_by_freq = self.config.get('scale_grad_by_freq', False)
            sparse = self.config.get('sparse', False)
            return f"nn.Embedding({num_embeddings}, {embedding_dim}, padding_idx={padding_idx}, max_norm={max_norm}, norm_type={norm_type}, scale_grad_by_freq={scale_grad_by_freq}, sparse={sparse})"

        elif self.module_type == 'MultiheadAttention':
            embed_dim = self.config.get('embed_dim', 512)
            num_heads = self.config.get('num_heads', 8)
            dropout = self.config.get('dropout', 0.0)
            bias = self.config.get('bias', True)
            add_bias_kv = self.config.get('add_bias_kv', False)
            add_zero_attn = self.config.get('add_zero_attn', False)
            kdim = self.config.get('kdim', None)
            vdim = self.config.get('vdim', None)
            return f"nn.MultiheadAttention({embed_dim}, {num_heads}, dropout={dropout}, bias={bias}, add_bias_kv={add_bias_kv}, add_zero_attn={add_zero_attn}, kdim={kdim}, vdim={vdim})"

        elif self.module_type == 'TransformerEncoderLayer':
            d_model = self.config.get('d_model', 512)
            nhead = self.config.get('nhead', 8)
            dim_feedforward = self.config.get('dim_feedforward', 2048)
            dropout = self.config.get('dropout', 0.1)
            activation = self.config.get('activation', 'relu')
            layer_norm_eps = self.config.get('layer_norm_eps', 1e-5)
            batch_first = self.config.get('batch_first', False)
            norm_first = self.config.get('norm_first', False)
            return f"nn.TransformerEncoderLayer({d_model}, {nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}', layer_norm_eps={layer_norm_eps}, batch_first={batch_first}, norm_first={norm_first})"

        elif self.module_type == 'Output':
            return "# Output layer - pass through"

        else:
            return f"# {self.module_type} - not implemented"