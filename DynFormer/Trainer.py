from Evaluator import Evaluator
import torch
from Loss import DyMixOp_Loss
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from utils import *
import copy

class Trainer:
    def __init__(self, config,
                 train_loader, test_loader, 
                 model,
                 optimizer, loss_func=None, scheduler=None, preprocessor=None,
                 gpu_id=None,
                 ):

        # arguments
        self.config = config
        self.data_config = config.data
        self.model_config = config.model
        self.training_config = config.training

        self.device = config.device
        self.verbose = config.verbose
        self.gpu_id = config.device if gpu_id is None else gpu_id

        # trainer component
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

        # datasets
        self.train_loader = train_loader
        self.test_loader = test_loader

        # model
        self.model_name = self.model_config.model_name
        self.model_path = self.model_config.model_path

        # path and dir   
        self.restart_checkpoint_path = self.training_config.restart_checkpoint_path
        self.save_checkpoint_dir_name = self.training_config.save_checkpoint_dir_name
        self.save_info_dir_name = self.training_config.save_info_dir_name

        # Check if base directory exists
        base_dir = os.path.join(self.save_info_dir_name, self.data_config.dataset_name, self.model_name)
        
        # If the variant_id is specified by user, use it directly as model_path/model_name identifier
        self.variant_id = self.training_config.variant_id
        if self.variant_id != "default":
            self.save_info_dir = os.path.join(base_dir, self.variant_id)
        else:
            # Extract numbers from 'VariantX' format
            self.max_variant_num = 0
            # Find the highest variant number if base directory exists
            if os.path.exists(base_dir):
                variant_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('Variant')]
                if variant_dirs:
                    for dir_name in variant_dirs:
                        # Extract the number after 'Variant'
                        variant_num = int(dir_name[7:])  # 'Variant' has 7 characters
                        self.max_variant_num = max(self.max_variant_num, variant_num)     
            # Increment the highest variant number
            self.variant_id = f"Variant{self.max_variant_num + 1}"
            
            # Create the final directory path
            self.save_info_dir = os.path.join(base_dir, self.variant_id)

        self.save_checkpoint_dir = os.path.join(self.save_info_dir, self.save_checkpoint_dir_name)

        # Assign frequently used arguments to new local variables
        self.hidden_dim = self.model_config.hidden_dim
        self.num_layers = self.model_config.num_layers
        self.input_dim = self.model_config.input_dim
        self.inp_involve_history_step = self.model_config.inp_involve_history_step
        self.output_dim = self.model_config.output_dim
        self.ar_nseq_train = self.model_config.ar_nseq_train
        self.ar_nseq_test = self.model_config.ar_nseq_test
        
        self.batch_size = self.training_config.batch_size
        self.num_epochs = self.training_config.num_epochs
        self.random_seed = self.training_config.random_seed
        self.save_best_checkpoint_on_val_dataset = self.training_config.save_best_checkpoint_on_val_dataset
        self.save_best_checkpoint_on_test_dataset = self.training_config.save_best_checkpoint_on_test_dataset
        self.save_checkpoints_at_end = self.training_config.save_checkpoints_at_end
        self.save_checkpoints_ep = self.training_config.save_checkpoints_ep
        self.step_size = self.training_config.step_size
        self.max_grad_norm = self.training_config.max_grad_norm

        self.dataset_path = self.data_config.dataset_path
        self.data_space_dim = self.data_config.data_space_dim

        self.pre_num_epochs = None
        self.min_loss_train_dataset = None
        self.min_loss_val_dataset = None
        self.min_loss_test_dataset = None

        self.preprocessor = preprocessor
        self.evaluator = Evaluator(self.config, self.model, self.test_loader, self.device, self.verbose, self.preprocessor)

        self.restart = True if self.restart_checkpoint_path is not None else False

        # Get dataset name
        if self.data_config.dataset_name is None:
            self.dataset_name = os.path.abspath(self.dataset_path).split('/')[-1]
        else:
            self.dataset_name = self.data_config.dataset_name

        # use built-in default setting if some components are None
        self._default_setting()

    def _default_setting(self):
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=0.97)
        if self.loss_func is None:
            self.loss_func = DyMixOp_Loss(reccons_type='mse', consist_type='mse', weight_reccons=1.0, weight_consist=0)

    def train(self):
        # Check if restart by loading restart_checkpoint_path
        if self.restart and self.restart_checkpoint_path is not None:
            checkpoint_info = torch.load(self.restart_checkpoint_path)
            self.pre_num_epochs = checkpoint_info['epoch'] + 1
            self.min_loss_train_dataset = checkpoint_info['min_loss_train_dataset']
            self.min_loss_val_dataset = checkpoint_info['min_loss_val_dataset']
            self.min_loss_test_dataset = checkpoint_info['min_loss_test_dataset']
            if self.num_epochs <= self.pre_num_epochs:
                raise ValueError("When 'restart' is True, 'num_epochs' should be larger than previous number of epochs {}.".format(self.pre_num_epochs))
        else:
            self.pre_num_epochs = 0
            self.min_loss_train_dataset = np.inf
            self.min_loss_val_dataset = np.inf
            self.min_loss_test_dataset = np.inf

        # At the beginning of training
        torch.cuda.reset_peak_memory_stats()  # Reset stats before training
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if self.verbose > 0:
            print(f"Start training for the {self.model_name} model on the {self.dataset_name} dataset...")

        # Training loop
        for epoch in tqdm(range(self.pre_num_epochs, self.num_epochs)):
            self.model.train()
            
            # Record start time
            start_event.record()

            # batch data for train
            losses_train_dataset = torch.tensor(0., dtype=torch.float32).to(self.device)
            losses_val_dataset = torch.tensor(0., dtype=torch.float32).to(self.device)
            losses_test_dataset = torch.tensor(0., dtype=torch.float32).to(self.device)
            for num_iter, (input_data_train, truth_data, static_data) in enumerate(self.train_loader):
                # input_data (batch, length*channel, res_x, res_y); truth_data (batch, length, channel, res_x, res_y)
                input_data_train = input_data_train.to(self.device)
                truth_data = truth_data.to(self.device)
                for i in range(len(static_data)):
                    static_data[i] = static_data[i].to(self.device)

                # get outputs
                predictions = self.model(input_data_train, static_data)
                
                # inverse normalize
                if self.preprocessor is not None:
                    predictions = self.preprocessor.inverse_normalize_output(predictions)

                predictions_train, predictions_val = predictions[:, :self.ar_nseq_train], predictions[:, self.ar_nseq_train:self.ar_nseq_train + self.ar_nseq_test]
                truth_train, truth_val = truth_data[:, :self.ar_nseq_train], truth_data[:, self.ar_nseq_train:self.ar_nseq_train + self.ar_nseq_test]

                # calculate loss on train and validation dataset
                metrics_train = self.loss_func(pred=predictions_train, truth=truth_train)
                losses_train_dataset += metrics_train
                if torch.numel(truth_val) > 0:
                    metrics_val = self.loss_func(pred=predictions_val, truth=truth_val)
                    losses_val_dataset += metrics_val

                # loss for backward
                loss = metrics_train

                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients separately for real and complex parameters
                # for p in model.parameters():
                #     if p.grad is not None:
                #         if p.grad.is_complex():
                #             # Handle complex gradients by clipping real and imaginary parts separately
                #             real_grad = p.grad.real
                #             imag_grad = p.grad.imag
                #             torch.clamp_(real_grad, min=-0.001, max=0.001)
                #             torch.clamp_(imag_grad, min=-0.001, max=0.001)
                #             p.grad = torch.complex(real_grad, imag_grad)
                #         else:
                #             # Normal gradient clipping for real parameters
                #             torch.clamp_(p.grad, min=-0.001, max=0.001)

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
            self.scheduler.step()
            train_num_iter = num_iter + 1

            # batch data for test
            if self.verbose > 1:
                print(f"Running evaluation for {self.model_name} at epoch {epoch}...")  
            inference_dict = self.evaluator.evaluate(loss_func=self.loss_func)
            
            # calculate loss on test dataset
            losses_test_dataset = inference_dict['metrics']

            # update the minimum loss and record the checkpoints with minimal loss
            if losses_train_dataset.item() / train_num_iter < self.min_loss_train_dataset:
                self.min_loss_train_dataset = losses_train_dataset.item() / train_num_iter
                epoch_min_loss_train_dataset = epoch
            if losses_val_dataset.item() / train_num_iter < self.min_loss_val_dataset:
                self.min_loss_val_dataset = losses_val_dataset.item() / train_num_iter
                epoch_min_loss_val_dataset = epoch
                if self.save_best_checkpoint_on_val_dataset:
                    best_checkpoint_on_val_dataset = copy.deepcopy(self.model.state_dict())
                    optimizer_by_best_checkpoint_on_val_dataset = copy.deepcopy(self.optimizer.state_dict())
            if losses_test_dataset.item() < self.min_loss_test_dataset:
                self.min_loss_test_dataset = losses_test_dataset.item()
                epoch_min_loss_test_dataset = epoch
                if self.save_best_checkpoint_on_test_dataset:
                    best_checkpoint_on_test_dataset = copy.deepcopy(self.model.state_dict())
                    optimizer_by_best_checkpoint_on_test_dataset = copy.deepcopy(self.optimizer.state_dict())

            # Record end time
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

            if self.verbose > 0: 
                print(
                    f"Epoch: {epoch} / {self.num_epochs}; Time: {elapsed_time};",
                    # train
                    f"Losses on train dataset: {losses_train_dataset.item() / train_num_iter};",
                    # validation
                    f"Losses on validation dataset: {losses_val_dataset.item() / train_num_iter};",
                    # test
                    f"Losses on test dataset: {losses_test_dataset.item()};",
                    # gpu id
                    f"Running on {self.gpu_id}",
                    flush=True)

            # Save the checkpoints
            if epoch == self.pre_num_epochs and self.save_checkpoint_dir is not None:
                os.makedirs(self.save_checkpoint_dir, exist_ok=True)
                checkpoints_num = 0

            if os.path.exists(self.save_checkpoint_dir):
                # Save checkpoints at end
                if self.save_checkpoints_at_end and epoch == self.num_epochs - 1:
                    checkpoints_info = {
                        'epoch': epoch,
                        'losses_train_dataset': losses_train_dataset.item() / train_num_iter,
                        'losses_val_dataset': losses_val_dataset.item() / train_num_iter,
                        'losses_test_dataset': losses_test_dataset.item(),
                        'random_seed': self.random_seed,
                        'width': self.hidden_dim,
                        'layer': self.num_layers,
                        'dataset_name': self.dataset_name,
                        'model_name': self.model_name,
                        'model_state_dict': copy.deepcopy(self.model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict())
                    }

                    checkpoint_path = os.path.join(
                        self.save_checkpoint_dir,
                        f'checkpoint_at_end.pth')
                    torch.save(checkpoints_info, checkpoint_path)
                    
                    if self.verbose > 0:
                        print(f"Checkpoints at end saved at {checkpoint_path}")
                # Save checkpoints in terms of save_checkpoints_ep. if the checkpoint at the end of epoch is saved, then skip the following checkpoint saving.
                elif self.save_checkpoints_ep is not None:
                    if self.save_checkpoints_ep > 0:
                        if epoch % self.save_checkpoints_ep == 0:
                            checkpoints_info = {
                                'epoch': epoch,
                                'losses_train_dataset': losses_train_dataset.item() / train_num_iter,
                                'losses_val_dataset': losses_val_dataset.item() / train_num_iter,
                                'losses_test_dataset': losses_test_dataset.item(),
                                'random_seed': self.random_seed,
                                'width': self.hidden_dim,
                                'layer': self.num_layers,
                                'dataset_name': self.dataset_name,
                                'model_name': self.model_name,
                                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict())
                            }

                            checkpoint_path = os.path.join(
                                self.save_checkpoint_dir,
                                f'checkpoint_{checkpoints_num}.pth')
                            checkpoints_num += 1
                            torch.save(checkpoints_info, checkpoint_path)
                            
                            if self.verbose > 0:
                                print(f"Checkpoint saved at {checkpoint_path}")
                    elif self.save_checkpoints_ep == 0:
                        pass
                    else:
                        raise ValueError("save_checkpoints_ep must be a positive integer or None")
                else:
                    pass

                # Save best checkpoints on test dataset
                if self.save_best_checkpoint_on_test_dataset:
                    if epoch == self.num_epochs - 1:
                        checkpoints_info = {
                            'epoch': epoch_min_loss_test_dataset,
                            'min_loss_test_dataset': self.min_loss_test_dataset,
                            'random_seed': self.random_seed,
                            'width': self.hidden_dim,
                            'layer': self.num_layers,
                            'dataset_name': self.dataset_name,
                            'model_name': self.model_name,
                            'model_state_dict': best_checkpoint_on_test_dataset,
                            'optimizer_state_dict': optimizer_by_best_checkpoint_on_test_dataset
                        }

                        checkpoint_path = os.path.join(
                            self.save_checkpoint_dir,
                            f'best_checkpoint_on_test_dataset.pth')
                        torch.save(checkpoints_info, checkpoint_path)
                        
                        if self.verbose > 0:
                            print(f"Best checkpoint on test dataset at epoch {epoch_min_loss_test_dataset} saved at {checkpoint_path}")
                
                # Save best checkpoints on validation dataset
                if self.save_best_checkpoint_on_val_dataset:
                    if epoch == self.num_epochs - 1:
                        checkpoints_info = {
                            'epoch': epoch_min_loss_val_dataset,
                            'min_loss_val_dataset': self.min_loss_val_dataset,
                            'random_seed': self.random_seed,
                            'width': self.hidden_dim,
                            'layer': self.num_layers,
                            'dataset_name': self.dataset_name,
                            'model_name': self.model_name,
                            'model_state_dict': best_checkpoint_on_val_dataset,
                            'optimizer_state_dict': optimizer_by_best_checkpoint_on_val_dataset
                        }

                        checkpoint_path = os.path.join(
                            self.save_checkpoint_dir,
                            f'best_checkpoint_on_val_dataset.pth')
                        torch.save(checkpoints_info, checkpoint_path)
                        
                        if self.verbose > 0:
                            print(f"Best checkpoint on validation dataset at epoch {epoch_min_loss_val_dataset} saved at {checkpoint_path}")
            
            else:
                if self.verbose > 0:
                    print("Checkpoint directory is not set. No checkpoints will be saved in this process.")

            # Save training information such as loss during training
            if epoch == self.pre_num_epochs and self.save_info_dir is not None:
                os.makedirs(self.save_info_dir, exist_ok=True)

            # Create data entry for this epoch
            train_info_entry = {
                'epoch': epoch,
                'time': elapsed_time,
                'train_loss': losses_train_dataset.item() / train_num_iter,
                'val_loss': losses_val_dataset.item() / train_num_iter,
                'test_loss': losses_test_dataset.item()
            }

            # If restarting or the file exists, append to it
            if epoch == self.pre_num_epochs:
                train_info_file_path = os.path.join(self.save_info_dir, 'train_info.csv')
                if self.restart:
                    # Read existing data
                    try:
                        df_existing = pd.read_csv(train_info_file_path, sep=';')
                        # Append new data
                        df_new = pd.DataFrame([train_info_entry])
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    except:
                        # If reading fails, create new DataFrame with just this entry
                        df_combined = pd.DataFrame([train_info_entry])

                        if self.verbose > 0:
                            print(f"Failed to read existing data from {train_info_file_path}. Created new DataFrame that will be saved in {train_info_file_path}.")
                else:
                    # Create new DataFrame
                    df_combined = pd.DataFrame([train_info_entry])
            else:
                df_new = pd.DataFrame([train_info_entry])
                df_combined = pd.concat([df_combined, df_new], ignore_index=True)

        # Save DataFrame to CSV
        df_combined.to_csv(train_info_file_path, sep=';', index=False)

        # After training completes, get the maximum memory used
        max_gpu_memory = torch.cuda.max_memory_reserved(device=self.device) / (1024 ** 2)  # MB
        # or in GB: torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Save model information
        model_info_file_path = os.path.join(self.save_info_dir, 'model_info.csv')
        model_info_entry = pd.DataFrame([{
            'model_name': self.model_name,
            'width': self.hidden_dim,
            'layer': self.num_layers,
            'random_seed': self.random_seed,
            'dataset': self.dataset_name,
            'model_path': self.model_path,
            'parameters': count_flops_and_params(self.model, torch.ones(torch.Size([1]) + input_data_train.shape[1:]).to(self.device), [torch.ones(torch.Size([1]) + static.shape[1:]).to(self.device) for static in static_data]),
            'gpu_memory_usage': max_gpu_memory,
            'avg_epoch_time': df_combined['time'].mean(),
            'min_loss_train_dataset': self.min_loss_train_dataset,
            'min_loss_test_dataset': self.min_loss_test_dataset,
            'min_loss_val_dataset': self.min_loss_val_dataset
        }])
        model_info_entry.to_csv(model_info_file_path, sep=';', index=False)
        
        if self.verbose > 0:
            print("Total FLOPs: ", model_info_entry['parameters'][0][0], flush=True)
            print("Parameters: ", model_info_entry['parameters'][0][1], flush=True)
            print("GPU Memory Usage: ", model_info_entry['gpu_memory_usage'][0], flush=True)
            print(f"Model information saved at \n {os.path.abspath(model_info_file_path)}")
            print("Training completed.")