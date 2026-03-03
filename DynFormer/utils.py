import torch
from thop import profile, clever_format
import os
import json
import numpy as np
import json
import os
import importlib

class ConfigObject:
    """
    A class that converts a dictionary into an object with attributes.
    Nested dictionaries are recursively converted to ConfigObject instances.
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries to ConfigObject
                setattr(self, key, ConfigObject(value))
            else:
                # Set value as attribute
                setattr(self, key, value)
    
    def __repr__(self):
        # Create a nice string representation
        attrs = ', '.join(f"{key}={repr(value)}" for key, value in self.__dict__.items())
        return f"ConfigObject({attrs})"

def load_config(config_path):
    """
    Load a JSON configuration file and convert it to a ConfigObject.
    
    Args:
        config_path (str): Path to the JSON configuration file
        
    Returns:
        ConfigObject: An object with attributes from the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return ConfigObject(config_dict)

def update_args_from_config(args, config_obj):
    """
    Update an args object with values from a config object.
    
    Args:
        args: The argparse.Namespace object to update
        config_obj: The ConfigObject containing configuration values
        
    Returns:
        Updated args object
    """
    # Convert the ConfigObject back to a flat dictionary with dot notation for nested keys
    flat_config = {}
    
    def _flatten_config(obj, prefix=''):
        for key, value in obj.__dict__.items():
            if isinstance(value, ConfigObject):
                # Recursively flatten nested ConfigObjects
                _flatten_config(value, f"{prefix}{key}.")
            else:
                # Add the flattened key-value pair
                flat_config[f"{prefix}{key}"] = value
    
    _flatten_config(config_obj)
    
    # Update args with the flattened config
    for key, value in flat_config.items():
        setattr(args, key, value)
    
    return args

def count_flops_and_params(model, input, static_data):
    flops, params = profile(model, inputs=(input, static_data))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


class Noiser(object):
    def __init__(self, dim, gain=0.05):
        self.dim = dim
        self.gain = gain

    def add_noise(self, truth):
        # require truth shape: [batch, seq, channel, ...]
        assert len(truth.shape) == 3 + self.dim
        torch.manual_seed(0)

        channel_num = truth.shape[2]
        noise_truth = torch.zeros_like(truth)
        for num in range(channel_num):
            truth_channel = truth[:, :, num:num + 1, ...]
            std_data = torch.std(truth_channel)
            noise_level = std_data * self.gain
            noise = noise_level * torch.randn_like(truth_channel, device=truth.device)
            noise_truth[:, :, num:num + 1, ...] = truth_channel + noise
        return noise_truth


class Normalizer(object):
    def __init__(self, source, dim, kind=None):
        self.min = source
        self.max = source
        for d in dim:
            self.min = torch.min(self.min, dim=d, keepdim=True).values
            self.max = torch.max(self.max, dim=d, keepdim=True).values

        self.mean = torch.mean(source, dim=dim, keepdim=True)
        self.var = torch.var(source, dim=dim, keepdim=True)

        self.kind = kind

    def normalize(self, data):
        if self.kind == 'min-max':
            data = self._normalize_minmax(data)
        elif self.kind == 'mean-var':
            data = self._normalize_meanvar(data)
        elif self.kind is None:
            data = data
        else:
            raise Exception(rf"Normalization '{self.kind}' does not exist. Only 'min-max', 'mean-var' and None are provided.")
        return data

    def inverse_normalize(self, data):
        if self.kind == 'min-max':
            data = self._inverse_normalize_minmax(data)
        elif self.kind == 'mean-var':
            data = self._inverse_normalize_meanvar(data)
        elif self.kind is None:
            data = data
        else:
            raise Exception(rf"Normalization '{self.kind}' does not exist. Only 'min-max', 'mean-var' and None are provided.")
        return data

    def _normalize_minmax(self, data):
        self.max = self.max.to(data.device)
        self.min = self.min.to(data.device)
        norm_inp = (data - self.min) / (self.max - self.min + 1e-6)
        return (norm_inp - 0)*1

    def _inverse_normalize_minmax(self, data):
        self.max = self.max.to(data.device)
        self.min = self.min.to(data.device)
        inverse_norm_inp = (data / 1 + 0) * (self.max - self.min) + self.min
        return inverse_norm_inp

    def _normalize_meanvar(self, data):
        self.mean = self.mean.to(data.device)
        self.var = self.var.to(data.device)
        norm_inp = (data - self.mean) / self.var
        return norm_inp

    def _inverse_normalize_meanvar(self, data):
        self.mean = self.mean.to(data.device)
        self.var = self.var.to(data.device)
        inverse_norm_inp = data * self.var + self.mean
        return inverse_norm_inp


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, dynamic_data, targets, static_data_list=None):
        """
        Args:
            dynamic_data: Data that changes with each iteration
            targets: Target outputs
            static_data_list: List of static data items, each can be:
                - Per-sample tensor [batch, ...]
                - Global tensor [...]
        """
        self.dynamic_data = dynamic_data
        self.targets = targets
        self.static_data_list = static_data_list or []
        
    def __len__(self):
        return len(self.dynamic_data)
        
    def __getitem__(self, idx):
        dynamic = self.dynamic_data[idx]
        target = self.targets[idx]
        
        # Process each static data item
        static_items = []
        for item in self.static_data_list:
            if item.dim() > 0 and item.size(0) == len(self.dynamic_data):
                # Per-sample static data
                static_items.append(item[idx])
            else:
                # Global static data (same for all samples)
                static_items.append(item)
                    
        # Include static data with the batch
        if static_items:
            return dynamic, target, static_items
        else:
            return dynamic, target

def create_or_load_optimizer(model, checkpoint_path=None, optimizer=None, lr=0.001):
    """
    Load a trained optimizer from a saved checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model: The loaded model
        
    Returns:
        Loaded optimizer
    """
    
    # Create optimizer
    if optimizer is None:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    else:
        optimizer = optimizer

    # Load optimizer if necessary
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}")
            return None 
    
    return optimizer


class ModelCreator:
    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose
    
    def import_class_from_path(self, file_path, class_name):
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    def create_or_load_model(self, model_config=None, model=None, checkpoint_path=None):
        """
        Create or load a model based on model name and path
        
        Args:
            model_config: Configuration arguments including data_space_dim for dimensionality
            model: Model instance to load
            checkpoint_path: Path to model checkpoint (None for new models)
            
        Returns:
            Instantiated model
        """

        if model is None:
            if model_config is None:
                raise ValueError("Must provide either `model` instance or `model_config` to load model.")
            else:
                # Create the model
                model_class = self.import_class_from_path(model_config.model_path, model_config.model_name)
                # Use the device specified in model_config if available, otherwise use the default device
                model_device = getattr(model_config, 'device', self.device)
                model = model_class(model_config, model_device)
            model_name = model_config.model_name
        else:
            model = model
            model_name = model.__class__.__name__
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            if self.verbose > 0:
                print(f"Loading checkpoint: \n {os.path.abspath(checkpoint_path)} \nfor {model_name}...")
            
            try:
                # Use the device specified in model_config if available, otherwise use the default device
                model_device = getattr(model_config, 'device', self.device) if model_config else self.device
                checkpoint = torch.load(checkpoint_path, map_location=model_device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            except FileNotFoundError:
                print(f"Checkpoint file not found at {checkpoint_path}")
                return None 
        
        return model


# ===============================================================================
# Static Data generated funcgtion
# ===============================================================================
def generate_structure_change_coeff(data, config, train):
    """
    Compute the structure change coefficient based on cosine similarity between consecutive time steps.

    This function calculates how much the structure of the data changes over time by computing
    the cosine similarity between consecutive time steps and returning the average difference from 1.
    Higher values indicate greater structural changes in the data over time.

    Args:
        data (dict): Dictionary containing the dataset with input_key and output_key as specified in config
        config (ConfigObject): Configuration object containing data parameters 
                                including res_step, ntrain, ntest, and input_key
        train (bool): Boolean flag indicating whether to generate coordinates 
                        for training data (True) or test data (False)

    Returns:
        float: Structure change coefficient computed as (1 - mean cosine similarity) between 
                consecutive time steps. Higher values indicate greater structural changes.
    """
    # Extract the relevant data based on train flag
    input_key = config.data.input_key
    ntrain = config.data.ntrain
    ntest = config.data.ntest

    # Get the appropriate data slice
    data_tensor = data[input_key][:ntrain]

    B, T = data_tensor.shape[0], data_tensor.shape[1]

    # Flatten spatial dimensions
    flat_data = torch.tensor(data_tensor).reshape(B, T, -1)
    # Dot product between consecutive time steps
    dot_prod = torch.sum(flat_data[:, 1:] * flat_data[:, :-1], dim=-1)
    # Product of norms for consecutive time steps
    norms = torch.norm(flat_data[:, 1:], dim=-1) * torch.norm(flat_data[:, :-1], dim=-1)
    # Compute cosine similarity
    cosine_sim = dot_prod / (norms + 1e-8)

    # Structure change coefficient = 1 - cosine similarity (higher value means greater structural change)
    structure_change_coeff = (1 - cosine_sim).mean().item()

    return structure_change_coeff

def generate_coor_input(data, config, train):
    """
    Generate coordinate inputs for models based on data dimensionality.

    This function creates coordinate tensors that match the spatial dimensions of the input data.
    It handles 1D, 2D, and 3D data by creating appropriate coordinate grids spanning [0, 1].
    The function determines the spatial dimensions by examining the shape of the input data.

    Args:
        data (dict): Dictionary containing the dataset with input_key and output_key as specified in config
        config (ConfigObject): Configuration object containing data parameters 
                                including res_step, ntrain, ntest, and input_key
        train (bool): Boolean flag indicating whether to generate coordinates 
                        for training data (True) or test data (False)
        
    Returns:
        torch.Tensor: Coordinate input tensor of appropriate dimension:
            - For 1D: [batch, 1, res_x] where res_x is the resolution in x-dimension
            - For 2D: [batch, 2, res_x, res_y] where res_x, res_y are resolutions in x,y-dimensions
            - For 3D: [batch, 3, res_x, res_y, res_z] where res_x, res_y, res_z are resolutions in x,y,z-dimensions
    """
    res_step = config.data.res_step
    ntrain = config.data.ntrain
    ntest = config.data.ntest
    input_key = config.data.input_key

    data_shape = data[input_key][:ntrain].shape if train else data[input_key][ntrain : ntrain + ntest].shape

    # require data obeying the shape [batch, length, channel, res_x, res_y, res_z]
    if len(data_shape) == 4:
        res_x = data_shape[-1]
        res_y = None
        res_z = None
    elif len(data_shape) == 5:
        res_x = data_shape[-2]
        res_y = data_shape[-1]
        res_z = None
    elif len(data_shape) == 6:
        res_x = data_shape[-3]
        res_y = data_shape[-2]
        res_z = data_shape[-1]
    else:
        raise ValueError("Invalid dimension configuration, must be 4, 5 or 6 dimensions and the mesh resolution must locate in the last dimension.")

    device= torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 1D case
    if res_y is None and res_z is None:
        x = np.linspace(0, 1, int(np.ceil(res_x/res_step)), endpoint=False)
        coor_input = np.expand_dims(x, axis=(0, 1))  # [1,1,res_x]
        coor_input = torch.tensor(coor_input, dtype=torch.float32)
        if device is not None:
            coor_input = coor_input.to(device)
        return coor_input.repeat(data_shape[0], 1, 1)  # [batch,1,res_x]

    # 2D case
    elif res_y is not None and res_z is None:
        x = np.linspace(0, 1, int(np.ceil(res_x/res_step)), endpoint=False)
        y = np.linspace(0, 1, int(np.ceil(res_y/res_step)), endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        coor_input = np.concatenate((X[None, None, ...], Y[None, None, ...]), 1)  # [1,2,res_x,res_y]
        coor_input = torch.tensor(coor_input, dtype=torch.float32)
        if device is not None:
            coor_input = coor_input.to(device)
        return coor_input.repeat(data_shape[0], 1, 1, 1)  # [batch,2,res_x,res_y]

    # 3D case
    elif res_y is not None and res_z is not None:
        x = np.linspace(0, 1, int(np.ceil(res_x/res_step)), endpoint=False)
        y = np.linspace(0, 1, int(np.ceil(res_y/res_step)), endpoint=False)
        z = np.linspace(0, 1, int(np.ceil(res_z/res_step)), endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coor_input = np.concatenate((X[None, None, ...], Y[None, None, ...], Z[None, None, ...]), 1)  # [1,3,res_x,res_y,res_z]
        coor_input = torch.tensor(coor_input, dtype=torch.float32)
        if device is not None:
            coor_input = coor_input.to(device)
        return coor_input.repeat(data_shape[0], 1, 1, 1, 1)  # [batch,3,res_x,res_y,res_z]

    else:
        raise ValueError("Invalid dimension configuration, must be 4, 5 or 6 dimensions.")

