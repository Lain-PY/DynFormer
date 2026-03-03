from utils import *
import scipy.io as scio
import numpy as np

class Preprocessor:
    """
    Wraps data loading and preprocessing.
    """
    def __init__(self, config):
        self.config = config
        data_config = config.data
        model_config = config.model
        training_config = config.training

        self.device = 'cpu' # config.device
        self.verbose = config.verbose

        self.static_data_generators = data_config.static_data_generators if hasattr(data_config, 'static_data_generators') else None
        self.static_data_keys = data_config.static_data_keys if hasattr(data_config, 'static_data_keys') else []

        self.pre_shuffle = data_config.pre_shuffle if hasattr(data_config, 'pre_shuffle') else False
        self.dataset_path = data_config.dataset_path
        self.input_key = data_config.input_key if hasattr(data_config, 'input_key') else 'coeff'
        self.output_key = data_config.output_key if hasattr(data_config, 'output_key') else 'sol'
        self.data_space_dim = data_config.data_space_dim if hasattr(data_config, 'data_space_dim') else None
        self.time_start_idx = data_config.time_start_idx if hasattr(data_config, 'time_start_idx') else 0
        self.input_dim = data_config.input_dim
        self.inp_involve_history_step = data_config.inp_involve_history_step
        self.output_dim = data_config.output_dim
        self.ar_nseq_train = data_config.ar_nseq_train
        self.ar_nseq_test = data_config.ar_nseq_test
        self.res_x = data_config.res_x
        self.res_y = data_config.res_y
        self.res_z = data_config.res_z
        self.res_step = data_config.res_step if hasattr(data_config, 'res_step') else None
        self.norm_kind = data_config.norm_kind
        self.noise_level_ic = data_config.noise_level_ic
        self.noise_level_label = data_config.noise_level_label
        self.ntrain = data_config.ntrain
        self.ntest = data_config.ntest

        self.batch_size = training_config.batch_size

        if data_config.noise_level_ic > 0:
            self.noiser_ic = Noiser(gain=data_config.noise_level_ic)
        elif data_config.noise_level_ic < 0:
            raise ValueError('noise_level_ic should be positive')
        else:
            self.noiser_ic = None

        if data_config.noise_level_label > 0:
            self.noiser_label = Noiser(gain=data_config.noise_level_label)
        elif data_config.noise_level_label < 0:
            raise ValueError('noise_level_label should be positive')
        else:
            self.noiser_label = None

    def normalize(self, inp, out):
        if self.normalize_inp is not None:
            inp = self.normalize_input(inp)
        if self.normalize_out is not None:
            out = self.normalize_output(out)
        return inp, out

    def normalize_input(self, inp):
        if self.normalize_inp is not None:
            inp = self.normalize_inp.normalize(inp)
        return inp

    def normalize_output(self, out):
        if self.normalize_out is not None:
            out = self.normalize_out.normalize(out)
        return out

    def inverse_normalize(self, inp, out):
        if self.normalize_inp is not None:
            inp = self.inverse_normalize_input(inp)
        if self.normalize_out is not None:
            out = self.inverse_normalize_output(out)
        return inp, out

    def inverse_normalize_input(self, inp):
        if self.normalize_inp is not None:
            inp = self.normalize_inp.inverse_normalize(inp)
        return inp

    def inverse_normalize_output(self, out):
        if self.normalize_out is not None:
            out = self.normalize_out.inverse_normalize(out)
        return out

    def add_noise(self, inp, out):
        if self.noiser_ic is not None:
            inp = self.noiser_ic.add_noise(inp)
        if self.noiser_label is not None:
            out = self.noiser_label.add_noise(out)
        
        return inp, out

    def load_additional_data(self, data, train=True):
        # Handle static data with flexible options
        # static_data_list: List of static data items, each can be:
        # - Per-sample tensor [batch, ...]
        # - Global tensor [...]
        static_data_list = []
        
        # Option 1: Load static data from the loading data file(s)
        if self.static_data_keys:
            for key in self.static_data_keys:
                if key in data:
                    static_item = data[key][:self.ntrain] if train else data[key][self.ntrain : self.ntrain + self.ntest]
                    static_data_list.append(torch.as_tensor(static_item, dtype=torch.float32).to(self.device))
                else:
                    print(f"Warning: Static data key '{key}' not found in data")
        
        # Option 2: Generate static data from custom function that use the properties of dataset
        if self.static_data_generators:
            for generator in self.static_data_generators:
                # Import the function dynamically
                import importlib
                module_path, func_name = generator.rsplit('.', 1)
                module = importlib.import_module(module_path)
                generator_func = getattr(module, func_name)
            
                # The generator might return a single tensor or a list of tensors
                generated_data = generator_func(
                    data=data, config=self.config, train=train
                )
            
                if isinstance(generated_data, (list, tuple)):
                    static_data_list.extend([torch.as_tensor(item, dtype=torch.float32).to(self.device) 
                                            if not isinstance(item, torch.Tensor) else item.to(self.device)
                                            for item in generated_data])
                else:
                    item = generated_data
                    static_data_list.append(
                        torch.as_tensor(item, dtype=torch.float32).to(self.device)
                        if not isinstance(item, torch.Tensor) else item.to(self.device)
                    )

        return static_data_list

    def load_and_preprocess_data(self):
        # Load data
        if self.dataset_path.endswith('.mat'):
            data = scio.loadmat(self.dataset_path)
        elif self.dataset_path.endswith('.nc'):
            import netCDF4 as nc
            data = nc.Dataset(self.dataset_path)
        else:
            raise ValueError(f'Unsupported dataset forma. Only .mat and .nc are supported')
        
        # Extract data according to dataset configuration
        # Shuffle the first dimension of the data
        if self.pre_shuffle:
            indices = np.random.permutation(data[self.input_key].shape[0])
        else:
            indices = np.arange(data[self.input_key].shape[0])

        if self.input_key != self.output_key:
            origin_data_x = data[self.input_key][indices][...]
            origin_data_y = data[self.output_key][indices][...]
        else:
            origin_data_x = data[self.input_key][indices][...]
            origin_data_y = origin_data_x

        # Get dataset dimensionality
        if self.data_space_dim is None:
            self.data_space_dim = len(origin_data_x.shape) - 3
        
        # Handle concatenation if needed
        if self.input_key != self.output_key:
            if origin_data_x.shape[:1] == origin_data_y.shape[:1] and origin_data_x.shape[2:] == origin_data_y.shape[2:]:
                # Use time_start_idx as the starting index for concatenation
                origin_data = np.concatenate([origin_data_x, origin_data_y], axis=1)
            
                # Calculate indices for input considering time starting index
                input_start = self.time_start_idx
                input_end = input_start + self.inp_involve_history_step + 1
                
                # Extract input data
                origin_data_x = origin_data[:self.ntrain + self.ntest, 
                                        input_start:input_end,
                                        :self.input_dim]
                
                # Calculate indices for output
                output_start = input_end
                output_end = output_start + (self.ar_nseq_train + self.ar_nseq_test)
                
                # Extract output data
                origin_data_y = origin_data[:self.ntrain + self.ntest, 
                                        output_start:output_end,
                                        :self.input_dim]
            else:
                # For non-concatenated data, apply time_start_idx directly
                if isinstance(self.time_start_idx, list):
                    inp_time_start_idx = self.time_start_idx[0]
                    out_time_start_idx = self.time_start_idx[1]
                else:
                    inp_time_start_idx = self.time_start_idx
                    out_time_start_idx = self.time_start_idx
                
                # Calculate input slice with time starting index
                input_start = inp_time_start_idx
                input_end = input_start + self.inp_involve_history_step + 1
                
                # Calculate output slice with time starting index 
                output_start = out_time_start_idx  # For direct output data (may need adjustment based on your use case)
                output_end = output_start + (self.ar_nseq_train + self.ar_nseq_test)
                
                # Slice the data along time dimension
                origin_data_x = origin_data_x[:self.ntrain + self.ntest, 
                                            input_start:input_end,
                                            :self.input_dim]
                
                origin_data_y = origin_data_y[:self.ntrain + self.ntest, 
                                            output_start:output_end,
                                            :self.input_dim]
        else:
            input_start = self.time_start_idx
            input_end = input_start + self.inp_involve_history_step + 1
            
            # Calculate output slice with time starting index 
            output_start = input_end
            output_end = output_start + (self.ar_nseq_train + self.ar_nseq_test)
                
            # Slice the data along time dimension
            origin_data_x = origin_data_x[:self.ntrain + self.ntest, 
                                        input_start:input_end,
                                        :self.input_dim]
                
            origin_data_y = origin_data_y[:self.ntrain + self.ntest, 
                                            output_start:output_end,
                                            :self.input_dim]
        
        # Apply spatial resolution steps based on dimensionality
        res_steps = []
        if self.res_step is not None:
            # Single resolution step for all dimensions
            res_steps = [self.res_step] * self.data_space_dim
        else:
            # Individual resolution steps for each dimension
            for i in range(self.data_space_dim):
                res_steps.append(1)  # Default: no downsampling
        
        # Build the slicing for spatial dimensions based on data_space_dim
        spatial_slices = tuple([slice(None, None, step) for step in res_steps])
        
        # Apply slicing based on dimensionality
        if self.data_space_dim == 1:
            # For 1D data: [batch, time, channel, x]
            origin_data_x = origin_data_x[..., 
                                        spatial_slices[0]]
            
            origin_data_y = origin_data_y[..., 
                                        spatial_slices[0]]
        elif self.data_space_dim == 2:
            # For 2D data: [batch, time, channel, x, y]
            origin_data_x = origin_data_x[..., 
                                        spatial_slices[0], spatial_slices[1]]
            
            origin_data_y = origin_data_y[..., 
                                        spatial_slices[0], spatial_slices[1]]
        elif self.data_space_dim == 3:
            # For 3D data: [batch, time, channel, x, y, z]
            origin_data_x = origin_data_x[..., 
                                        spatial_slices[0], spatial_slices[1], spatial_slices[2]]
            
            origin_data_y = origin_data_y[..., 
                                        spatial_slices[0], spatial_slices[1], spatial_slices[2]]
        
        # Convert to tensors
        tensor_data_x = torch.as_tensor(origin_data_x, dtype=torch.float32).to(self.device)
        tensor_data_y = torch.as_tensor(origin_data_y, dtype=torch.float32).to(self.device)

        # Add noise
        x_train, y_train = self.add_noise(tensor_data_x, tensor_data_y)

        # Normalize
        if self.norm_kind is not None:
            # Dynamically determine normalization dimensions based on data_space_dim
            norm_dims = [0, 1]  # Always normalize batch and time dimensions
            for i in range(self.data_space_dim):
                norm_dims.append(3 + i)  # Add spatial dimensions (3+0 for 1D, 3+0,3+1 for 2D, etc.)
                
            self.normalize_inp = Normalizer(source=x_train[:self.ntrain], dim=norm_dims, kind=self.norm_kind)
            self.normalize_out = Normalizer(source=y_train[:self.ntrain], dim=norm_dims, kind=self.norm_kind)
        else:
            self.normalize_inp = None
            self.normalize_out = None
        # x_train, y_train = self.normalize(x_train, y_train)
        x_train = self.normalize_input(x_train)

        # Calculate sub-resolution sizes for each dimension
        if self.data_space_dim >= 1:
            self.sub_res_x = int(np.ceil(self.res_x / res_steps[0]))
        if self.data_space_dim >= 2:
            self.sub_res_y = int(np.ceil(self.res_y / res_steps[1]))
        if self.data_space_dim >= 3:
            self.sub_res_z = int(np.ceil(self.res_z / res_steps[2]))

        # Prepare data for model input - reshape based on dimensionality
        # This flattens time and channel dimensions together
        if self.data_space_dim == 1:
            x_train = x_train.reshape(
                self.ntrain + self.ntest, -1, self.sub_res_x)
        elif self.data_space_dim == 2:
            x_train = x_train.reshape(
                self.ntrain + self.ntest, -1, self.sub_res_x, self.sub_res_y)
        elif self.data_space_dim == 3:
            x_train = x_train.reshape(
                self.ntrain + self.ntest, -1, self.sub_res_x, self.sub_res_y, self.sub_res_z)
        
        train_static_data_list = self.load_additional_data(data, train=True)
        test_static_data_list = self.load_additional_data(data, train=False)
        
        # Create data loaders with mixed data
        train_dataset = MixedDataset(
            x_train[:self.ntrain], 
            y_train[:self.ntrain],
            train_static_data_list
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True,
        )
        
        # Similar for test data
        test_dataset = MixedDataset(
            x_train[self.ntrain:self.ntrain + self.ntest],
            y_train[self.ntrain:self.ntrain + self.ntest],
            test_static_data_list
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )

        return train_loader, test_loader