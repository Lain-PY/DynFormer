import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cartopy.crs as ccrs

class Visualizer:
    """
    Wraps visualization functions.
    """
    def __init__(self):
        pass

    def compare(self, inference_dicts,
                save_path=None, sample_idx=0, time_idx=0, channel_idx=0):
        """
        Visualize comparison between true data and predictions from multiple models.
        
        Args:
            inference_dicts: List of dictionaries containing model predictions and truths
            save_path: Path to save the visualization
            sample_idx, time_idx, channel_idx: Indices for visualization
        """
        # Check if this is the 3dsw dataset based on data shape
        input_data = inference_dicts[0]['inputs']
        true_data = inference_dicts[0]['truths']
        
        # Check if this is 3dsw dataset (64x32 spatial resolution)
        if (input_data.shape[-2] == 64 and input_data.shape[-1] == 32) or \
           (true_data.shape[-2] == 64 and true_data.shape[-1] == 32):
            # Use specialized 3dsw visualization
            self._compare_3dsw(inference_dicts, save_path, sample_idx, time_idx, channel_idx)
        else:
            # Use standard visualization
            self._compare_standard(inference_dicts, save_path, sample_idx, time_idx, channel_idx)
    
    def _compare_3dsw(self, inference_dicts, save_path=None, sample_idx=0, time_idx=0, channel_idx=0):
        """
        Specialized visualization for 3dsw dataset using cartopy projections.
        """
        # Determine number of models to compare
        input_data = inference_dicts[0]['inputs']
        true_data = inference_dicts[0]['truths']
        num_models = len(inference_dicts)
        
        # Convert data to numpy arrays if needed
        if isinstance(input_data, torch.Tensor):
            input_np = input_data[sample_idx, time_idx, channel_idx].cpu().numpy()
            true_np = true_data[sample_idx, time_idx, channel_idx].cpu().numpy()
        else:
            input_np = input_data[sample_idx, time_idx, channel_idx]
            true_np = true_data[sample_idx, time_idx, channel_idx]
        
        # Create coordinate grids for lat-lon projection
        nlon, nlat = input_np.shape
        lats = np.linspace(-90, 90, nlat)
        lons = np.linspace(0, 360, nlon)
        lon, lat = np.meshgrid(lons, lats, indexing='ij')
        
        # Create figure with 2 rows: first row for data, second row for errors
        fig = plt.figure(figsize=(5 * (num_models + 1), 5))
        
        # Field names based on channel index (0: height, 1: vorticity)
        field_names = ['Height', 'Vorticity']
        field_name = field_names[channel_idx] if channel_idx < len(field_names) else f'Channel {channel_idx}'
        cmaps = ['RdYlBu_r', 'RdBu_r']
        cmap = cmaps[channel_idx] if channel_idx < len(cmaps) else 'viridis'
        
        # Collect all errors for consistent colorbar and Pre-calculate all prediction arrays and errors
        errors_np = []
        error_max = -np.inf
        error_min = np.inf
        pred_arrays = []
        for inference_dict in inference_dicts:
            if isinstance(inference_dict['predictions'], torch.Tensor):
                pred_np = inference_dict['predictions'][sample_idx, time_idx, channel_idx].cpu().numpy()
            else:
                pred_np = inference_dict['predictions'][sample_idx, time_idx, channel_idx]
            pred_arrays.append(pred_np)

            # Calculate error
            error_np = np.abs(pred_np - true_np)
            errors_np.append(error_np)
            error_max = max(error_max, np.max(error_np))
            error_min = min(error_min, np.min(error_np))
        
        # First column: Input and Ground Truth with cartopy projection
        # Plot input data
        ax1 = fig.add_subplot(2, num_models + 1, 1, projection=ccrs.Robinson())
        ax1.coastlines()
        ax1.gridlines()
        
        input_plot = ax1.contourf(lon, lat, input_np,
                                 transform=ccrs.PlateCarree(),
                                 levels=50,
                                 cmap=cmap,
                                 extend='both')
        plt.colorbar(input_plot, ax=ax1, label=f'Input {field_name}')
        ax1.set_title(f'Input {field_name}')
        
        # Plot ground truth
        ax2 = fig.add_subplot(2, num_models + 1, num_models + 2, projection=ccrs.Robinson())
        ax2.coastlines()
        ax2.gridlines()
        
        true_plot = ax2.contourf(lon, lat, true_np,
                                transform=ccrs.PlateCarree(),
                                levels=50,
                                cmap=cmap,
                                extend='both')
        plt.colorbar(true_plot, ax=ax2, label=f'Ground Truth {field_name}')
        ax2.set_title(f'Ground Truth {field_name}')
        
        # Plot each model prediction and error
        for i, inference_dict in enumerate(inference_dicts):
            pred_np = pred_arrays[i]
            error_np = errors_np[i]
            
            # Plot prediction with cartopy projection
            ax_pred = fig.add_subplot(2, num_models + 1, i + 2, projection=ccrs.Robinson())
            ax_pred.coastlines()
            ax_pred.gridlines()
            
            pred_plot = ax_pred.contourf(lon, lat, pred_np,
                                        transform=ccrs.PlateCarree(),
                                        levels=np.linspace(true_np.min(), true_np.max(), 50),
                                        cmap=cmap,
                                        extend='both',
                                        vmin=true_np.min(), vmax=true_np.max())
            plt.colorbar(pred_plot, ax=ax_pred, label=f'Predicted {field_name}')
            
            # Add model name to title
            title = f'{inference_dict["model_name"]}'
            ax_pred.set_title(title)
        
        # Plot errors with consistent colorbar
        for i, error_np in enumerate(errors_np):
            # Plot error with cartopy projection
            ax_error = fig.add_subplot(2, num_models + 1, num_models + 3 + i, projection=ccrs.Robinson())
            ax_error.coastlines()
            ax_error.gridlines()
            
            # Calculate error metrics
            mse = np.mean(np.square(error_np))
            mae = np.mean(np.abs(error_np))
            
            error_plot = ax_error.contourf(lon, lat, error_np,
                                          transform=ccrs.PlateCarree(),
                                          levels=np.linspace(error_min, error_max, 50),
                                          cmap='hot',
                                          extend='both',
                                          vmin=error_min, vmax=error_max)
            plt.colorbar(error_plot, ax=ax_error, label=f'Error')
            
            # Add error metrics to title
            title = f'Error: MSE={mse:.4e}, MAE={mae:.4e}'
            ax_error.set_title(title)

        # Adjust layout and save figure
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    def _compare_standard(self, inference_dicts, save_path=None, sample_idx=0, time_idx=0, channel_idx=0):
        """
        Standard visualization for non-3dsw datasets.
        """
        # Determine number of models to compare
        input_data = inference_dicts[0]['inputs']
        true_data = inference_dicts[0]['truths']

        num_models = len(inference_dicts)
        space_dim = len(input_data.shape) - 3  # Subtract batch, time, channel dimensions
        
        # Create figure with subplots (num_models columns, 2 rows)
        # First column: input and ground truth
        # Other columns: model predictions and errors
        fig = plt.figure(figsize=(4 * (num_models + 1), 6))
        # fig = plt.figure(figsize=(1 * (num_models + 1), 3))
        
        # Slice for time dimension
        if space_dim == 1:
            time_slice = slice(None, None)
        elif space_dim == 2 or space_dim == 3:
            time_slice = time_idx
        else:
            raise ValueError("Invalid space dimension, must be 1, 2 or 3.")
        
        # Convert data to numpy arrays if needed
        if isinstance(input_data, torch.Tensor):
            input_np = input_data[sample_idx, time_slice, channel_idx].cpu().numpy()
            true_np = true_data[sample_idx, time_slice, channel_idx].cpu().numpy()
        else:
            input_np = input_data[sample_idx, time_slice, channel_idx]
            true_np = true_data[sample_idx, time_slice, channel_idx]
        
        # Create meshgrid for plotting
        if space_dim == 1:
            T, X = np.meshgrid(
                np.linspace(0, 1, input_np.shape[0], endpoint=False),
                np.linspace(0, 1, input_np.shape[1], endpoint=False),
                indexing='ij'
            )
            meshgrids = [T.flatten(), X.flatten()]
            projection_mode = None
        elif space_dim == 2:
            X, Y = np.meshgrid(
                np.linspace(0, 1, input_np.shape[-2], endpoint=False),
                np.linspace(0, 1, input_np.shape[-1], endpoint=False),
                indexing='ij'
            )
            meshgrids = [X.flatten(), Y.flatten()]
            projection_mode = None
        elif space_dim == 3:
            X, Y, Z = np.meshgrid(
                np.linspace(0, 1, input_np.shape[-3], endpoint=False),
                np.linspace(0, 1, input_np.shape[-2], endpoint=False),
                np.linspace(0, 1, input_np.shape[-1], endpoint=False),
                indexing='ij'
            )
            meshgrids = [X.flatten(), Y.flatten(), Z.flatten()]
            projection_mode = '3d'
        else:
            raise ValueError("Invalid space dimension, must be 1, 2 or 3.")

        # Collect all errors for consistent colorbar and Pre-calculate all prediction arrays and errors
        errors_np = []
        error_max = -np.inf
        error_min = np.inf
        pred_arrays = []
        for inference_dict in inference_dicts:
            if isinstance(inference_dict['predictions'], torch.Tensor):
                pred_np = inference_dict['predictions'][sample_idx, time_slice, channel_idx].cpu().numpy()
            else:
                pred_np = inference_dict['predictions'][sample_idx, time_slice, channel_idx]
            pred_arrays.append(pred_np)

            # Calculate error
            error_np = np.abs(pred_np - true_np)
            errors_np.append(error_np)
            error_max = max(error_max, np.max(error_np))
            error_min = min(error_min, np.min(error_np))
        
        # First column: Input and Ground Truth
        # Plot input data
        ax1 = fig.add_subplot(2, num_models + 1, 1, projection=projection_mode)
        
        if space_dim == 1:
            # Use contourf for 1D data to create a heatmap-like visualization
            T_grid = np.linspace(0, 1, input_np.shape[0])
            X_grid = np.linspace(0, 1, input_np.shape[1])
            input_plot = ax1.contourf(X_grid, T_grid, input_np, levels=50, cmap='viridis', extend='both')
            plt.colorbar(input_plot, ax=ax1, label='Value')
        else:
            # Use scatter for 2D and 3D data
            scatter = ax1.scatter(*meshgrids, c=input_np.flatten(), cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, ax=ax1)
            
        ax1.set_title('Input Data')
        if space_dim == 1:
            ax1.set_xlabel('X')
            ax1.set_ylabel('Time')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
        elif space_dim == 2:
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
        elif space_dim == 3:
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_zlim(0, 1)
        
        # Plot ground truth
        ax2 = fig.add_subplot(2, num_models + 1, num_models + 2, projection=projection_mode)
        
        if space_dim == 1:
            # Use contourf for 1D data to create a heatmap-like visualization
            true_plot = ax2.contourf(X_grid, T_grid, true_np, levels=50, cmap='viridis', extend='both')
            plt.colorbar(true_plot, ax=ax2, label='Value')
        else:
            # Use scatter for 2D and 3D data
            scatter = ax2.scatter(*meshgrids, c=true_np.flatten(), cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, ax=ax2)
            
        ax2.set_title('Ground Truth')
        if space_dim == 1:
            ax2.set_xlabel('X')
            ax2.set_ylabel('Time')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        elif space_dim == 2:
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        elif space_dim == 3:
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_zlim(0, 1)
        
        # Plot each model prediction and error
        for i, inference_dict in enumerate(inference_dicts):
            pred_np = pred_arrays[i]
            error_np = errors_np[i]
            
            # Plot prediction
            ax_pred = fig.add_subplot(2, num_models + 1, i + 2, projection=projection_mode)
            
            if space_dim == 1:
                # Use contourf for 1D data
                pred_plot = ax_pred.contourf(X_grid, T_grid, pred_np, levels=50, 
                                           cmap='viridis', extend='both',
                                           vmin=true_np.min(), vmax=true_np.max())
                plt.colorbar(pred_plot, ax=ax_pred, label='Value')
            else:
                # Use scatter for 2D and 3D data
                scatter = ax_pred.scatter(*meshgrids, c=pred_np.flatten(), 
                                        cmap='viridis', alpha=0.8, 
                                        vmin=true_np.min(), vmax=true_np.max())
                plt.colorbar(scatter, ax=ax_pred)
            
            # Add model name to title
            title = f'{inference_dict["model_name"]}'
            ax_pred.set_title(title)
            if space_dim == 1:
                ax_pred.set_xlabel('X')
                ax_pred.set_ylabel('Time')
                ax_pred.set_xlim(0, 1)
                ax_pred.set_ylim(0, 1)
            elif space_dim == 2:
                ax_pred.set_xlabel('X')
                ax_pred.set_ylabel('Y')
                ax_pred.set_xlim(0, 1)
                ax_pred.set_ylim(0, 1)
            elif space_dim == 3:
                ax_pred.set_xlabel('X')
                ax_pred.set_ylabel('Y')
                ax_pred.set_zlabel('Z')
                ax_pred.set_xlim(0, 1)
                ax_pred.set_ylim(0, 1)
                ax_pred.set_zlim(0, 1)
        
        for i, error_np in enumerate(errors_np):
            # Plot error
            ax_error = fig.add_subplot(2, num_models + 1, num_models + 3 + i, projection=projection_mode)
            
            if space_dim == 1:
                # Use contourf for 1D data
                error_plot = ax_error.contourf(X_grid, T_grid, error_np, levels=50, 
                                             cmap='hot', extend='both',
                                             vmin=error_min, vmax=error_max)
                plt.colorbar(error_plot, ax=ax_error, label='Error')
            else:
                # Use scatter for 2D and 3D data
                scatter = ax_error.scatter(*meshgrids, c=error_np.flatten(), 
                                         cmap='hot', alpha=0.8, 
                                         vmin=error_min, vmax=error_max)
                plt.colorbar(scatter, ax=ax_error)
            
            # Add error metrics to title
            mse = np.mean(np.square(error_np))
            mae = np.mean(np.abs(error_np))
            title = f'Error: MSE={mse:.4e}, MAE={mae:.4e}'
            ax_error.set_title(title)
            if space_dim == 1:
                ax_error.set_xlabel('X')
                ax_error.set_ylabel('Time')
                ax_error.set_xlim(0, 1)
                ax_error.set_ylim(0, 1)
            elif space_dim == 2:
                ax_error.set_xlabel('X')
                ax_error.set_ylabel('Y')
                ax_error.set_xlim(0, 1)
                ax_error.set_ylim(0, 1)
            elif space_dim == 3:
                ax_error.set_xlabel('X')
                ax_error.set_ylabel('Y')
                ax_error.set_zlabel('Z')
                ax_error.set_xlim(0, 1)
                ax_error.set_ylim(0, 1)
                ax_error.set_zlim(0, 1)
        
        # Adjust layout and save figure
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        plt.show()
        plt.close()
