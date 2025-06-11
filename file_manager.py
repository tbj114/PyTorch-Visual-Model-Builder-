#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Manager - Handles model file operations including encryption
"""

import json
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import hashlib
import time
import collections
from model_node import ModelNode
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_node import ModelNode

class FileManager(QObject):
    """Manages file operations for the model builder"""
    
    file_saved = pyqtSignal(str)  # Emitted when file is saved
    file_loaded = pyqtSignal(str)  # Emitted when file is loaded
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.is_modified = False
        self.encryption_key = None
        self.file_format_version = "1.0"
        
    def new_model(self):
        """Create a new model"""
        self.current_file = None
        self.is_modified = False
        return True
        
    def load_model(self, file_path=None, canvas=None):
        """Open a model file"""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                None, "Open Model", "",
                "PyTorch Model Builder Files (*.ptmb);;Python Files (*.py);;All Files (*)"
            )
            
        if not file_path:
            return None
            
        try:
            if file_path.endswith('.ptmb'):
                model_data = self.load_encrypted_file(file_path)
            elif file_path.endswith('.py'):
                model_data = self.load_python_file(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            self.current_file = file_path
            self.is_modified = False
            self.file_loaded.emit(file_path)
            if canvas and model_data:
                canvas.deserialize(model_data)
            return model_data
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to open file: {str(e)}")
            return None
            
    def save_model(self, model_data, file_path=None):
        """Save model to file"""
        if not file_path:
            file_path = self.current_file
            
        if not file_path:
            return self.save_model_as(model_data)
            
        try:
            if file_path.endswith('.ptmb'):
                self.save_encrypted_file(model_data, file_path)
            elif file_path.endswith('.py'):
                self.save_python_file(model_data, file_path)
            else:
                raise ValueError("Unsupported file format")
                
            self.current_file = file_path
            self.is_modified = False
            self.file_saved.emit(file_path)
            return True
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save file: {str(e)}")
            return False
            
    def save_model_as(self, model_data):
        """Save model with new filename"""
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Model As", "",
            "PyTorch Model Builder Files (*.ptmb);;Python Files (*.py)"
        )
        
        if not file_path:
            return False

        # Ensure file has an extension
        if not (file_path.endswith('.ptmb') or file_path.endswith('.py')):
            # Default to .ptmb if no recognized extension is provided
            file_path += '.ptmb'
            
        return self.save_model(model_data, file_path)
        
    def export_to_python(self, model_data, file_path=None):
        """Export model as Python code"""
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                None, "Export to Python", "",
                "Python Files (*.py)"
            )
            
        if not file_path:
            return False
            
        try:
            python_code = self.generate_python_code(model_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(python_code)
                
            return True
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to export Python code: {str(e)}")
            return False
            
    def load_encrypted_file(self, file_path):
        """Load encrypted .ptmb file"""
        # Get password from user
        password, ok = QInputDialog.getText(
            None, "Password Required", "Enter password to decrypt file:",
            QLineEdit.Password
        )
        
        if not ok or not password:
            raise ValueError("Password required to decrypt file")
            
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
                
            # Extract salt and encrypted content
            salt = encrypted_data[:16]
            encrypted_content = encrypted_data[16:]
            
            # Derive key from password
            key = self.derive_key_from_password(password, salt)
            fernet = Fernet(key)
            
            # Decrypt and decompress
            decrypted_data = fernet.decrypt(encrypted_content)
            json_data = decrypted_data.decode('utf-8')
            
            model_data = json.loads(json_data)
            
            # Validate file format
            if 'file_format_version' not in model_data:
                raise ValueError("Invalid file format")
                
            return model_data
            
        except Exception as e:
            raise ValueError(f"Failed to decrypt file: {str(e)}")
            
    def save_encrypted_file(self, model_data, file_path):
        """Save encrypted .ptmb file"""
        # Get password from user
        password, ok = QInputDialog.getText(
            None, "Set Password", "Enter password to encrypt file:",
            QLineEdit.Password
        )
        
        if not ok or not password:
            raise ValueError("Password required to encrypt file")
            
        # Confirm password
        confirm_password, ok = QInputDialog.getText(
            None, "Confirm Password", "Confirm password:",
            QLineEdit.Password
        )
        
        if not ok or password != confirm_password:
            raise ValueError("Passwords do not match")
            
        try:
            # Add metadata
            model_data['file_format_version'] = self.file_format_version
            model_data['created_time'] = time.time()
            model_data['modified_time'] = time.time()
            
            # Convert to JSON
            json_data = json.dumps(model_data, indent=2, ensure_ascii=False)
            
            # Generate salt and derive key
            salt = os.urandom(16)
            key = self.derive_key_from_password(password, salt)
            fernet = Fernet(key)
            
            # Encrypt data
            encrypted_data = fernet.encrypt(json_data.encode('utf-8'))
            
            # Save salt + encrypted data
            with open(file_path, 'wb') as f:
                f.write(salt + encrypted_data)
                
        except Exception as e:
            raise ValueError(f"Failed to encrypt file: {str(e)}")
            
    def derive_key_from_password(self, password, salt):
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
        
    def load_python_file(self, file_path):
        """Load Python file and extract model information"""
        # This is a simplified implementation
        # In a real application, you might want to parse the Python AST
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create a basic model data structure
        model_data = {
            'file_format_version': self.file_format_version,
            'nodes': [],
            'connections': [],
            'metadata': {
                'name': os.path.basename(file_path),
                'description': 'Imported from Python file',
                'python_code': content
            }
        }
        
        return model_data
        
    def save_python_file(self, model_data, file_path):
        """Save model as Python file"""
        python_code = self.generate_python_code(model_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
            
    def generate_python_code(self, model_data):
        """Generate Python code from model data with improved topology handling"""
        nodes_data = model_data.get('nodes', [])
        connections_data = model_data.get('connections', [])

        if not nodes_data:
            return "# Empty model\npass"

        # Create a map from node_id to node_data for easy lookup
        node_map = {node['node_id']: node for node in nodes_data}

        # Build adjacency list for topological sort
        adj = collections.defaultdict(list)
        in_degree = collections.defaultdict(int)

        for conn in connections_data:
            output_node_id = conn['output_node_id']
            input_node_id = conn['input_node_id']
            adj[output_node_id].append(input_node_id)
            in_degree[input_node_id] += 1

        # Identify all nodes that are part of any connection
        connected_nodes_in_graph = set()
        for conn in connections_data:
            connected_nodes_in_graph.add(conn['output_node_id'])
            connected_nodes_in_graph.add(conn['input_node_id'])

        # Initialize the queue with nodes that have 0 in-degree AND are part of the connected graph
        queue = collections.deque()
        for node_id in connected_nodes_in_graph:
            if node_id not in in_degree or in_degree[node_id] == 0: # If not in in_degree, it means in_degree is 0
                queue.append(node_id)

        topological_order = []

        while queue:
            u = queue.popleft()
            topological_order.append(u) # Add to topological order

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # Cycle detection: If there are nodes in `connected_nodes_in_graph` that are not in `topological_order`,
        # it means those nodes are part of a cycle.
        if len(topological_order) != len(connected_nodes_in_graph):
            raise ValueError("Model contains a cycle among connected nodes, cannot generate code.")

        # Generate __init__ method
        init_lines = []
        forward_lines = []
        layer_names = {}
        output_vars = {}
        layer_counter = {}

        for i, node_id in enumerate(topological_order):
            node_data = node_map[node_id]
            module_type = node_data['module_type']
            config = node_data.get('config', {})

            if module_type == 'Input':
                continue

            # Generate unique layer names
            if module_type not in layer_counter:
                layer_counter[module_type] = 0
            layer_counter[module_type] += 1
            
            layer_var_name = f"self.{module_type.lower()}_{layer_counter[module_type]}"
            layer_names[node_id] = layer_var_name

            # Generate layer instantiation code
            layer_def = self.generate_layer_definition_from_config(module_type, config)
            if layer_def:
                init_lines.append(f"        {layer_var_name} = {layer_def}")

        # Generate forward method
        for node_id in topological_order:
            node_data = node_map[node_id]
            module_type = node_data['module_type']
            
            output_var = f"x_{node_id.replace('-', '_')}" # Unique output variable for each layer
            output_vars[node_id] = output_var

            if module_type == 'Input':
                forward_lines.append(f"        {output_var} = x")
            else:
                layer_var_name = layer_names.get(node_id)
                if not layer_var_name:
                    continue

                # Collect all inputs for the current node
                input_vars = []
                input_node_ids = []
                for conn in connections_data:
                    if conn['input_node_id'] == node_id:
                        # Find the output variable of the connected previous node
                        prev_node_output_var = output_vars.get(conn['output_node_id'])
                        prev_node_id = conn['output_node_id']
                        if prev_node_output_var and prev_node_id not in input_node_ids:
                            input_vars.append(prev_node_output_var)
                            input_node_ids.append(prev_node_id)

                if not input_vars:
                    # Use model input as fallback
                    current_input_str = 'x'
                elif len(input_vars) == 1:
                    current_input_str = input_vars[0]
                else:
                    # Handle multiple inputs based on module type
                    if module_type == 'Add':
                        current_input_str = ' + '.join(input_vars)
                    elif module_type == 'Multiply':
                        current_input_str = ' * '.join(input_vars)
                    elif module_type == 'Concatenate':
                        current_input_str = f"torch.cat([{', '.join(input_vars)}], dim=1)"
                    else:
                        # For modules with multiple inputs, create a list of inputs
                        current_input_str = f"[{', '.join(input_vars)}]"
                        forward_lines.append(f"        # Note: {module_type} module receives multiple inputs")

                if module_type in ['Add', 'Concatenate', 'Multiply'] and len(input_vars) > 1:
                    forward_lines.append(f"        {output_var} = {current_input_str}")
                else:
                    if len(input_vars) > 1:
                        # For modules with multiple inputs, pass them as a list
                        forward_lines.append(f"        {output_var} = {layer_var_name}(*{current_input_str})")
                    else:
                        forward_lines.append(f"        {output_var} = {layer_var_name}({current_input_str})")

        # Find all output nodes (nodes with no outgoing connections)
        output_nodes = []
        for node_id in topological_order:
            has_output = any(conn['output_node_id'] == node_id for conn in connections_data)
            if not has_output:
                output_nodes.append(node_id)
        
        # If no explicit output nodes, use the last node
        if not output_nodes and topological_order:
            output_nodes = [topological_order[-1]]
            
        # Handle multiple outputs
        if len(output_nodes) > 1:
            output_vars_str = ', '.join([output_vars[node_id] for node_id in output_nodes])
            forward_lines.append(f"        return {output_vars_str}")
        else:
            forward_lines.append(f"        return {output_vars[output_nodes[0]]}")
            
        # Initialize code list
        code = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class GeneratedModel(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()"
        ]
        code.extend(init_lines)
        code.extend([
            "",
            "    def forward(self, x):"
        ])
        code.extend(forward_lines)
        
        # Add test code generation
        test_code = [
            "\n# Test the model",
            "if __name__ == '__main__':",
            "    # Create model instance",
            "    model = GeneratedModel()",
            "    model.eval()  # Set to evaluation mode",
            "    ",
            "    # Create sample input",
            "    batch_size = 1"
        ]
        
        # Find input shapes from input nodes
        input_shapes = []
        for node_id in topological_order:
            if node_map[node_id]['module_type'] == 'Input':
                if 'config' in node_map[node_id] and 'shape' in node_map[node_id]['config']:
                    input_shapes.append(node_map[node_id]['config']['shape'])
                    
        if input_shapes:
            shape_str = str(input_shapes[0])
            test_code.extend([
                f"    # Sample input with shape {shape_str}",
                f"    x = torch.randn(batch_size, *{shape_str})",
                "    ",
                "    # Forward pass",
                "    with torch.no_grad():",
                "        outputs = model(x)",
                "    ",
                "    # Print output shape(s)",
                "    if isinstance(outputs, tuple):",
                "        print('Model outputs:')",
                "        for i, output in enumerate(outputs):",
                "            print(f'Output {i + 1} shape: {output.shape}')",
                "    else:",
                "        print(f'Output shape: {outputs.shape}')"
            ])
        else:
            test_code.extend([
                "    # Please specify input shape according to your model requirements",
                "    x = torch.randn(batch_size, 1, 28, 28)  # Example shape",
                "    ",
                "    # Forward pass",
                "    with torch.no_grad():",
                "        outputs = model(x)",
                "    ",
                "    print(f'Output shape: {outputs.shape}')"
            ])
            
        # Add test code to the final code
        code.extend([line for line in test_code])

        # Add comprehensive test code
        code.extend([
            "",
            "# Model Testing Example",
            "if __name__ == '__main__':",
            "    # Create model instance",
            "    model = GeneratedModel()",
            "    model.eval()",
            "    ",
            "    # Create sample input (adjust dimensions as needed)",
            "    batch_size = 1",
            "    input_size = 784  # Example: MNIST flattened image size",
            "    sample_input = torch.randn(batch_size, input_size)",
            "    ",
            "    # Forward pass test",
            "    try:",
            "        with torch.no_grad():",
            "            output = model(sample_input)",
            "        print(f'Model test successful!')",
            "        print(f'Input shape: {sample_input.shape}')",
            "        if isinstance(output, list):",
            "            print(f'Number of outputs: {len(output)}')",
            "            for i, out in enumerate(output):",
            "                print(f'Output {i+1} shape: {out.shape}')",
            "        else:",
            "            print(f'Output shape: {output.shape}')",
            "    except Exception as e:",
            "        print(f'Model test failed: {e}')",
            "        print('Please check model structure and input dimensions')",
            "    ",
            "    # Calculate model parameters",
            "    total_params = sum(p.numel() for p in model.parameters())",
            "    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
            "    print(f'Total parameters: {total_params:,}')",
            "    print(f'Trainable parameters: {trainable_params:,}')"
        ])

        return "\n".join(code)
    
    def generate_layer_definition_from_config(self, module_type, config):
        """Generate layer definition from module type and config"""
        if module_type == 'Linear':
            in_features = config.get('in_features', 128)
            out_features = config.get('out_features', 64)
            bias = config.get('bias', True)
            return f"nn.Linear({in_features}, {out_features}, bias={bias})"
        elif module_type == 'Conv2d':
            in_channels = config.get('in_channels', 3)
            out_channels = config.get('out_channels', 64)
            kernel_size = config.get('kernel_size', 3)
            stride = config.get('stride', 1)
            padding = config.get('padding', 1)
            return f"nn.Conv2d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, padding={padding})"
        elif module_type == 'BatchNorm2d':
            num_features = config.get('num_features', 64)
            return f"nn.BatchNorm2d({num_features})"
        elif module_type == 'ReLU':
            inplace = config.get('inplace', True)
            return f"nn.ReLU(inplace={inplace})"
        elif module_type == 'MaxPool2d':
            kernel_size = config.get('kernel_size', 2)
            stride = config.get('stride', 2)
            return f"nn.MaxPool2d({kernel_size}, stride={stride})"
        elif module_type == 'Dropout':
            p = config.get('p', 0.5)
            return f"nn.Dropout(p={p})"
        elif module_type == 'LSTM':
            input_size = config.get('input_size', 128)
            hidden_size = config.get('hidden_size', 64)
            num_layers = config.get('num_layers', 1)
            batch_first = config.get('batch_first', True)
            return f"nn.LSTM({input_size}, {hidden_size}, {num_layers}, batch_first={batch_first})"
        elif module_type in ['Add', 'Concatenate', 'Multiply']:
            return None  # These are handled in forward pass
        else:
            return f"# TODO: Implement {module_type}"
        
    def generate_module_code(self, node):
        """Generate PyTorch module code for a node"""
        module_type = node.get('module_type', '')
        config = node.get('config', {})
        
        if module_type == 'Linear':
            in_features = config.get('in_features', 128)
            out_features = config.get('out_features', 64)
            bias = config.get('bias', True)
            return f"nn.Linear({in_features}, {out_features}, bias={bias})"
            
        elif module_type == 'Conv2d':
            in_channels = config.get('in_channels', 3)
            out_channels = config.get('out_channels', 64)
            kernel_size = config.get('kernel_size', 3)
            stride = config.get('stride', 1)
            padding = config.get('padding', 1)
            return f"nn.Conv2d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, padding={padding})"
            
        elif module_type == 'BatchNorm2d':
            num_features = config.get('num_features', 64)
            return f"nn.BatchNorm2d({num_features})"
            
        elif module_type == 'ReLU':
            inplace = config.get('inplace', True)
            return f"nn.ReLU(inplace={inplace})"
            
        elif module_type == 'MaxPool2d':
            kernel_size = config.get('kernel_size', 2)
            stride = config.get('stride', 2)
            return f"nn.MaxPool2d({kernel_size}, stride={stride})"
            
        elif module_type == 'Dropout':
            p = config.get('p', 0.5)
            return f"nn.Dropout(p={p})"
            
        elif module_type == 'LSTM':
            input_size = config.get('input_size', 128)
            hidden_size = config.get('hidden_size', 64)
            num_layers = config.get('num_layers', 1)
            batch_first = config.get('batch_first', True)
            return f"nn.LSTM({input_size}, {hidden_size}, {num_layers}, batch_first={batch_first})"
            
        # Add more module types as needed
        return f"# TODO: Implement {module_type}"
        
    def generate_forward_code(self, model_data):
        """Generate forward pass code"""
        nodes = model_data.get('nodes', [])
        connections = model_data.get('connections', [])
        
        # Find input nodes
        input_nodes = [node for node in nodes if node.get('module_type') == 'Input']
        
        if not input_nodes:
            return ["# No input nodes found", "return x"]
            
        # Simple linear execution for now
        # In a more sophisticated implementation, you would build a proper execution graph
        code_lines = []
        
        # Start with input
        current_var = "x"
        
        # Process nodes in order (simplified)
        active_nodes = [node for node in nodes if node.get('active', True) and node.get('module_type') != 'Input']
        
        for i, node in enumerate(active_nodes):
            node_id = node['id']
            module_type = node.get('module_type', '')
            
            if module_type in ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']:
                # Activation functions
                if module_type == 'ReLU':
                    code_lines.append(f"{current_var} = F.relu({current_var})")
                elif module_type == 'Sigmoid':
                    code_lines.append(f"{current_var} = torch.sigmoid({current_var})")
                elif module_type == 'Tanh':
                    code_lines.append(f"{current_var} = torch.tanh({current_var})")
                elif module_type == 'Softmax':
                    dim = node.get('config', {}).get('dim', -1)
                    code_lines.append(f"{current_var} = F.softmax({current_var}, dim={dim})")
            else:
                # Regular modules
                code_lines.append(f"{current_var} = self.{node_id}({current_var})")
                
        code_lines.append(f"return {current_var}")
        
        return code_lines
        
    def get_current_file(self):
        """Get current file path"""
        return self.current_file
        
    def is_file_modified(self):
        """Check if file has been modified"""
        return self.is_modified
        
    def set_modified(self, modified=True):
        """Set file modification status"""
        self.is_modified = modified
        
    def get_file_info(self, file_path):
        """Get file information"""
        if not os.path.exists(file_path):
            return None
            
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_encrypted': file_path.endswith('.ptmb')
        }
        
    def create_backup(self, file_path):
        """Create backup of file"""
        if not os.path.exists(file_path):
            return False
            
        backup_path = f"{file_path}.backup"
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            return True
        except Exception:
            return False