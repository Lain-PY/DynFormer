
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from scipy.ndimage import zoom

# Set default font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


class Visualizer:
    """
    Wraps visualization functions.
    Refactored to unify static and animation layouts.
    """
    def __init__(self, config=None, dataset_name=None, verbose=1):
        """
        Initialize Visualizer.
        
        Args:
            config: Optional config object with data.dataset_name and verbose attributes
            dataset_name: Optional dataset name string (used if config not provided)
            verbose: Verbosity level (default 1, used if config not provided)
        """
        if config is not None:
            self.verbose = config.verbose
            self.dataset_name = config.data.dataset_name
        elif dataset_name is not None:
            self.verbose = verbose
            self.dataset_name = dataset_name
        else:
            raise ValueError("Either config or dataset_name must be provided")

    def _get_data_at_step(self, data, sample_idx, time_idx, channel_idx):
        """Helper to extract numpy data at a specific step."""
        if self.dataset_name.lower() == "1dks" and self._get_space_dim(data) == 2:
                    data = data.squeeze(-1)

        if isinstance(data, torch.Tensor):
            return data[sample_idx, time_idx, channel_idx].cpu().numpy()
        return data[sample_idx, time_idx, channel_idx]

    def _get_space_dim(self, input_data):
        """Determine spatial dimension from input data shape (Batch, Time, Channel, space...)."""
        return len(input_data.shape) - 3

    def _setup_plot_grid(self, fig, num_models, space_dim):
        """Setup GridSpec for the interleaved layout (5 plotting columns)."""
        # Fixed 5 columns for plots + 1 column for colorbar
        self.ncols = 4
        
        # Calculate batches needed (each batch is a pair of rows: Values, Errors)
        # Items: 1 GT + num_models
        n_items = 1 + num_models
        self.n_batches = int(np.ceil(n_items / self.ncols))
        
        total_rows = self.n_batches * 2  # 1 Value row + 1 Error row per batch
        
        # GridSpec: 5 plot cols (weight 1), 1 cbar col (weight 0.05)
        gs = GridSpec(total_rows, self.ncols + 1, figure=fig, 
                      width_ratios=[1]*self.ncols + [0.05],
                      wspace=0.05, hspace=0.28)
        return gs

    def _plot_field(self, ax, data, space_dim, X, Y, Z=None, T=None, title=None, 
                    vmin=None, vmax=None, cmap='viridis', is_error=False):
        """Unified plotter for a single field (1D/2D/3D)."""
        plot_obj = None
        
        if self.dataset_name == '3dSW':
            # 3D Sphere Projection
            norm = plt.Normalize(vmin, vmax)
            m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            fcolors = m.to_rgba(data)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, shade=False)
            ax.view_init(elev=30, azim=45)
            plot_obj = m
            
        elif space_dim == 2:
            levels = np.linspace(vmin, vmax, 50) if vmin is not None and vmax is not None else 50
            plot_obj = ax.contourf(X, Y, data, levels=levels, cmap=cmap, extend='both', vmin=vmin, vmax=vmax)
            
        elif space_dim == 1:
             if len(data.shape) == 1:
                 # Multicolor Line Plot
                 points = np.array([X, data]).T.reshape(-1, 1, 2)
                 segments = np.concatenate([points[:-1], points[1:]], axis=1)
                 
                 norm = plt.Normalize(vmin, vmax)
                 lc = LineCollection(segments, cmap=cmap, norm=norm)
                 lc.set_array(data)
                 lc.set_linewidth(2)
                 ax.add_collection(lc)
                 
                 ax.set_xlim(X.min(), X.max())
                 ax.set_ylim(vmin, vmax)
                 plot_obj = lc

        elif space_dim == 3:
            # 3D Volume Plot (Pseudo-Volumetric via Slices)
            # data shape: (X, Y, Z)
            target_z = 30 # Reduced from 100 for performance/visibility
            current_z = data.shape[2]
            z_factor = target_z / current_z if current_z < target_z else 1.0
            
            target_xy = 60
            xy_factor = target_xy / data.shape[0] if data.shape[0] < target_xy else 1.0
            
            data_dense = zoom(data, (xy_factor, xy_factor, z_factor), order=1)
            
            nx, ny, nz = data_dense.shape
            x = np.linspace(X.min(), X.max(), nx)
            y = np.linspace(Y.min(), Y.max(), ny)
            z = np.linspace(Z.min(), Z.max(), nz)
            X_dense, Y_dense = np.meshgrid(x, y, indexing='ij')
            
            if vmin is None: vmin = data.min()
            if vmax is None: vmax = data.max()
            
            # Plot slices
            for i in range(nz):
                z_height = z[i]
                slice_data = data_dense[:, :, i]
                # Alpha depends on value magnitude or constant?
                # Original used simple alpha, maybe 0.1-0.3
                cset = ax.contourf(X_dense, Y_dense, slice_data, zdir='z', offset=z_height, 
                                  levels=20, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.15)
            
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
            ax.set_zlim(Z.min(), Z.max())
            ax.view_init(elev=30, azim=45)
            
            # For colorbar, we need a ScalarMappable
            norm = plt.Normalize(vmin, vmax)
            plot_obj = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        if title:
            ax.set_title(title)
        ax.axis('off')
        return plot_obj

    def _shrink_colorbar(self, cax, shrink_ratio=0.7):
        pos = cax.get_position()
        new_height = pos.height * shrink_ratio
        new_bottom = pos.y0 + (pos.height - new_height) / 2
        cax.set_position([pos.x0, new_bottom, pos.width, new_height])
        
    def _render_scene(self, fig, true_np, pred_dicts, pred_nps, errors_np, 
                      grid_coords, space_dim, 
                      gt_pred_min, gt_pred_max, error_min, error_max,
                      cmap='viridis', mse_history=None, current_time_idx=None, input_np=None, show_input=False):
        """
        Renders the full scene (GT + Predictions + Errors) into the figure.
        Interleaved Layout: Row 1 Values, Row 2 Errors, Row 3 Values...
        """
        num_models = len(pred_dicts)
        gs = self._setup_plot_grid(fig, num_models, space_dim)
        
        X, Y, Z, T = grid_coords
        
        # Store handles/labels for global legend
        legend_handles = []
        legend_labels = []
        
        # Consolidate items into flat lists for slicing
        # Values: [GT] + [Pred 1, Pred 2, ...]
        value_items = [{'type': 'gt', 'data': true_np, 'title': 'Ground Truth'}]
        for d, p in zip(pred_dicts, pred_nps):
             value_items.append({'type': 'pred', 'data': p, 'title': d['model_name']})
             
        # Errors: [Curve/Input] + [Error 1, Error 2, ...]
        if show_input or (space_dim == 3 and self.dataset_name != '3dSW'):
            # Use Input Field
            title = "Input Field"
            error_items = [{'type': 'input', 'data': input_np, 'title': title}]
        else:
            error_items = [{'type': 'curve', 'data': None, 'title': 'Error Evolution'}]
            
        for error_np in errors_np:
            mse = np.mean(np.square(error_np))
            error_items.append({'type': 'error', 'data': error_np, 'title': f'MSE={mse:.4e}'})
            
        # Iterate by batch (rows)
        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.ncols
            end_idx = start_idx + self.ncols
            
            # --- Value Row ---
            value_row_idx = batch_idx * 2
            batch_vals = value_items[start_idx:end_idx]
            
            sm_data = None
            for col_idx, item in enumerate(batch_vals):
                ax = fig.add_subplot(gs[value_row_idx, col_idx], projection='3d' if (self.dataset_name == '3dSW' or space_dim == 3) else None)
                sm_curr = self._plot_field(ax, item['data'], space_dim, X, Y, Z, T, 
                                          title=item['title'], vmin=gt_pred_min, vmax=gt_pred_max, cmap=cmap)
                if sm_curr: sm_data = sm_curr
            
            # Shared Data Colorbar for this row
            if sm_data:
                cax = fig.add_subplot(gs[value_row_idx, -1])
                self._shrink_colorbar(cax, shrink_ratio=0.7)
                cbar = fig.colorbar(sm_data, cax=cax)
                # Force exact number of ticks (start, end, +3 intermediate = 5 ticks)
                cbar.set_ticks(np.linspace(gt_pred_min, gt_pred_max, 5))
                cbar.update_ticks()
                
            # --- Error Row ---
            error_row_idx = value_row_idx + 1
            batch_errs = error_items[start_idx:end_idx]
            
            sm_err = None
            for col_idx, item in enumerate(batch_errs):
                if item['type'] == 'curve':
                    # Plot Curve
                    ax_loss = fig.add_subplot(gs[error_row_idx, col_idx])
                    if mse_history is not None:
                        colors = plt.cm.tab10(np.linspace(0, 1, num_models))
                        max_time = len(next(iter(mse_history.values())))
                        time_axis = np.arange(max_time)
                        
                        for i, pred_dict in enumerate(pred_dicts):
                            model_name = pred_dict["model_name"]
                            if model_name in mse_history:
                                curve = mse_history[model_name]
                                if current_time_idx is not None:
                                     ax_loss.semilogy(time_axis[:current_time_idx+1], curve[:current_time_idx+1], 
                                                     color=colors[i], linewidth=2, label=model_name)
                                     if current_time_idx < max_time - 1:
                                          ax_loss.semilogy(time_axis[current_time_idx:], curve[current_time_idx:], 
                                                      color=colors[i], linewidth=1, alpha=0.3, linestyle='--')
                                     ax_loss.axvline(x=current_time_idx, color='gray', linestyle=':', alpha=0.7)
                                else:
                                    ax_loss.semilogy(time_axis, curve, color=colors[i], linewidth=2, label=model_name)
                        
                        ax_loss.set_title(item['title'])
                        # Capture handles for global legend
                        h, l = ax_loss.get_legend_handles_labels()
                        if h:
                            legend_handles = h
                            legend_labels = l
                        
                        ax_loss.grid(True, which="both", ls="-", alpha=0.2)
                        
                elif item['type'] == 'input':
                    # Plot Input Field
                    ax_in = fig.add_subplot(gs[error_row_idx, col_idx], projection='3d' if (self.dataset_name == '3dSW' or space_dim == 3) else None)
                    # Use provided input data. If None, skip?
                    if item['data'] is not None:
                        self._plot_field(ax_in, item['data'], space_dim, X, Y, Z, T, 
                                        title=item['title'], vmin=gt_pred_min, vmax=gt_pred_max, cmap=cmap)
                    else:
                        ax_in.text(0.5, 0.5, "No Input Data", ha='center')
                        
                else:
                    # Plot Error Map
                    ax_err = fig.add_subplot(gs[error_row_idx, col_idx], projection='3d' if (self.dataset_name == '3dSW' or space_dim == 3) else None)
                    sm_curr = self._plot_field(ax_err, item['data'], space_dim, X, Y, Z, T, 
                                              title=item['title'], vmin=error_min, vmax=error_max, cmap='hot', is_error=True)
                    if sm_curr: sm_err = sm_curr
                    
            # Shared Error Colorbar for this row
            if sm_err:
                cax = fig.add_subplot(gs[error_row_idx, -1])
                self._shrink_colorbar(cax, shrink_ratio=0.8)
                cbar = fig.colorbar(sm_err, cax=cax)
                # Force exact number of ticks
                cbar.set_ticks(np.linspace(error_min, error_max, 5))
                cbar.update_ticks()
        
        # Global Legend
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='lower center', 
                       ncol=9, bbox_to_anchor=(0.5, 0.07))
            
    def _prepare_grid(self, shape, space_dim):
        """Prepare meshgrid based on shape and plot type."""
        X, Y, Z, T = None, None, None, None
        
        if self.dataset_name == '3dSW':
            # 3dSW Sphere logic
            nlon, nlat = shape
            lats = np.linspace(-90, 90, nlat)
            lons = np.linspace(0, 360, nlon)
            lon, lat = np.meshgrid(lons, lats, indexing='ij')
            lon_rad = np.radians(lon)
            lat_rad = np.radians(lat)
            X = np.cos(lat_rad) * np.cos(lon_rad)
            Y = np.cos(lat_rad) * np.sin(lon_rad)
            Z = np.sin(lat_rad)
            
        elif space_dim == 2:
            X, Y = np.meshgrid(
                np.linspace(0, 1, shape[-2], endpoint=True),
                np.linspace(0, 1, shape[-1], endpoint=True),
                indexing='ij'
            )
        elif space_dim == 1:
            X = np.linspace(0, 1, shape[-1], endpoint=True)
            
        elif space_dim == 3:
            # 3D Grid
            X, Y, Z = np.meshgrid(
                np.linspace(0, 1, shape[-3], endpoint=True),
                np.linspace(0, 1, shape[-2], endpoint=True),
                np.linspace(0, 1, shape[-1], endpoint=True),
                indexing='ij'
            )
            
        return X, Y, Z, T

    def compare(self, inference_dicts, save_path=None, sample_idx=0, time_idx=0, channel_idx=0, show_input=False, plot_xt=False):
        """
        Visualize a single frame (snapshot) with the animation layout.
        """
        input_data = inference_dicts[0]['inputs']
        true_data = inference_dicts[0]['truths']
        space_dim = self._get_space_dim(input_data)
        # Extract Data
        pred_nps = []
        errors_np = []
        
        # Extract Data
        if plot_xt:
            # Space-Time Plot Logic
            # Extract FULL time series: (Time, Space...)
            # Assuming data is (Batch, Time, Channel, Space...)
            # We want (Time, Space...)
            
            def extract_xt(data):
                if self.dataset_name.lower() == "1dks" and space_dim == 2:
                    data = data.squeeze(-1) # To reduced the pseudo spatial dimension

                if isinstance(data, torch.Tensor):
                    return data[sample_idx, :, channel_idx].cpu().numpy()
                return data[sample_idx, :, channel_idx]

            true_np = extract_xt(true_data)
            input_np = extract_xt(input_data) if input_data is not None else None
            
            for d in inference_dicts:
                p = extract_xt(d['predictions'])
                pred_nps.append(p)
                errors_np.append(np.abs(p - true_np))
                
            # Increase space_dim for visualization
            # 1D -> 2D (heatmap), 2D -> 3D (volume)
            if self.dataset_name.lower() != "1dks":
                space_dim += 1
            
        else:
            # Standard Frame Logic
            true_np = self._get_data_at_step(true_data, sample_idx, time_idx, channel_idx)
            # Extract Input for 3D viz
            input_np = self._get_data_at_step(input_data, sample_idx, min(time_idx, input_data.shape[1]-1), channel_idx) if input_data is not None else None
            
            for d in inference_dicts:
                p = self._get_data_at_step(d['predictions'], sample_idx, time_idx, channel_idx)
                pred_nps.append(p)
                errors_np.append(np.abs(p - true_np))
            
            if self.dataset_name.lower() == "1dks":
                space_dim -= 1
            
        # Limits (local for static comparison)
        all_vals = [true_np] + pred_nps
        gt_pred_min = np.min([np.min(v) for v in all_vals])
        gt_pred_max = np.max([np.max(v) for v in all_vals])
        
        all_errs = errors_np
        error_min = np.min([np.min(e) for e in all_errs])
        error_max = np.max([np.max(e) for e in all_errs])
        
        # Setup Figure
        # num_models = len(inference_dicts)
        
        # ncols = 4
        # n_batches = int(np.ceil((1 + num_models) / ncols))
        # n_rows = n_batches * 2
        
        figsize_w = 12 # 3.2 * (ncols + 0.5) #2.7
        figsize_h = 11 # 3 * n_rows  #2.5
        
        fig = plt.figure(figsize=(figsize_w, figsize_h))
        
        # Grid
        grid_coords = self._prepare_grid(true_np.shape, space_dim)
        
        # Colormap selection
        cmaps = ['RdYlBu_r', 'RdBu_r']
        cmap = 'Blues' # cmaps[channel_idx] if self.dataset_name == '3dSW' and channel_idx < len(cmaps) else 'viridis'
        
        # Prepare MSE History
        mse_history = {}
        # We need the full time series to compute the history
        # Note: compare() is often used for one frame, but we need context for the curve.
        # Check if we can get full data.
        num_time_steps = true_data.shape[1]
        for d in inference_dicts:
             model_name = d['model_name']
             hist = []
             for t in range(num_time_steps):
                 t_true = self._get_data_at_step(true_data, sample_idx, t, channel_idx)
                 t_pred = self._get_data_at_step(d['predictions'], sample_idx, t, channel_idx)
                 hist.append(np.mean(np.square(t_pred - t_true)))
             mse_history[model_name] = hist
        
        if plot_xt:
            # Disable current time marker for XT plots as it shows full evolution
            time_idx = None 
        
        # Render
        self._render_scene(fig, true_np, inference_dicts, pred_nps, errors_np, 
                           grid_coords, space_dim, 
                           gt_pred_min, gt_pred_max, error_min, error_max, 
                           cmap=cmap, mse_history=mse_history, current_time_idx=time_idx, input_np=input_np, show_input=show_input)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig_name = os.path.join(save_path, f'Batch_{sample_idx}_TimeStep_{time_idx}_Channel_{channel_idx}.png')
            plt.savefig(fig_name, dpi=400, bbox_inches='tight', transparent=True)
            if self.verbose > 0:
                print(f'Saved figure to \n {os.path.abspath(fig_name)}')
        
        plt.show()
        plt.close()

    def animate(self, inference_dicts, save_path=None, sample_idx=0, channel_idx=0, 
                fps=5, dpi=150, format='gif', show_input=False):
        """
        Create animation reusing the frame rendering logic.
        """
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        
        input_data = inference_dicts[0]['inputs']
        true_data = inference_dicts[0]['truths']
        num_time_steps = true_data.shape[1]
        space_dim = self._get_space_dim(input_data)
        num_models = len(inference_dicts)

        if self.dataset_name.lower() == "1dks":
            space_dim -= 1
        
        # Extract Input for 3D viz (assuming static input across time or extracting frame 0)
        # Usually input is initial condition (t=0) or forcing.
        # We'll extract frame 0 for static visualization in 3D slot
        input_np_base = self._get_data_at_step(input_data, sample_idx, 0, channel_idx) if input_data is not None else None
        
        # 1. Pre-compute/Extract ALL data to get Global Limits
        all_true = []
        all_preds = [] # list of lists
        
        # We iteration over time to get global min/max
        # For memory efficiency, we might just scan min/max, but storing might speed up plotting if RAM allows.
        # Given potential size, we should probably just scan indices.
        
        # Global Min/Max Scanning
        g_min = np.inf
        g_max = -np.inf
        e_min = np.inf
        e_max = -np.inf
        
        # This can be heavy. Let's try to be smart.
        # Just use the tensor min/max if possible assuming fit in memory.
        if isinstance(true_data, torch.Tensor):
            subset_true = true_data[sample_idx, :, channel_idx]
            g_min = min(g_min, subset_true.min().item())
            g_max = max(g_max, subset_true.max().item())
        else:
             subset_true = true_data[sample_idx, :, channel_idx]
             g_min = min(g_min, np.min(subset_true))
             g_max = max(g_max, np.max(subset_true))
             
        for d in inference_dicts:
            pred = d['predictions']
            if isinstance(pred, torch.Tensor):
                p_sub = pred[sample_idx, :, channel_idx]
                g_min = min(g_min, p_sub.min().item())
                g_max = max(g_max, p_sub.max().item())
                
                # Error min/max
                # Error requires temporal alignment calc
                # We can approximate or just loop
                # Just loop mainly for error to avoid huge tensor creation
            else:
                p_sub = pred[sample_idx, :, channel_idx]
                g_min = min(g_min, np.min(p_sub))
                g_max = max(g_max, np.max(p_sub))

        # Precision scan for errors & MSE History
        mse_history = {d['model_name']: [] for d in inference_dicts}
        for t in range(num_time_steps):
            t_true = self._get_data_at_step(true_data, sample_idx, t, channel_idx)
            for d in inference_dicts:
                t_pred = self._get_data_at_step(d['predictions'], sample_idx, t, channel_idx)
                err = np.abs(t_pred - t_true)
                e_min = min(e_min, np.min(err))
                e_max = max(e_max, np.max(err))
                
                # Compute MSE
                mse = np.mean(np.square(err))
                mse_history[d['model_name']].append(mse)
        
        # Setup Figure
        # ncols = 4
        # n_batches = int(np.ceil((1 + num_models) / ncols))
        # n_rows = n_batches * 2
        
        figsize_w = 12 # 3.2 * (ncols + 0.5)
        figsize_h = 11 # 3 * n_rows
        
        fig = plt.figure(figsize=(figsize_w, figsize_h))
        
        # 3D SW Pre-computation (Sphere coords)
        # Assuming Data is consistent shape across time
        dummy_data = self._get_data_at_step(true_data, sample_idx, 0, channel_idx)
        grid_coords = self._prepare_grid(dummy_data.shape, space_dim)
        
        cmaps = ['RdYlBu_r', 'RdBu_r']
        cmap = cmaps[channel_idx] if self.dataset_name == '3dSW' and channel_idx < len(cmaps) else 'viridis'

        def update(frame):
            fig.clear()
            
            # Extract Frame Data
            t_true = self._get_data_at_step(true_data, sample_idx, frame, channel_idx)
            t_preds = []
            t_errors = []
            
            for d in inference_dicts:
                p = self._get_data_at_step(d['predictions'], sample_idx, frame, channel_idx)
                t_preds.append(p)
                t_errors.append(np.abs(p - t_true))
                
            self._render_scene(fig, t_true, inference_dicts, t_preds, t_errors, 
                               grid_coords, space_dim,
                               g_min, g_max, e_min, e_max, cmap,
                               mse_history=mse_history, current_time_idx=frame, input_np=input_np_base, show_input=show_input)
            
            # Title with Time Calculation? (Optional but good)
            fig.suptitle(f'Time Step: {frame}', fontsize=16, y=0.93)

        anim = FuncAnimation(fig, update, frames=num_time_steps, interval=200)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f'Animation_Batch_{sample_idx}_Channel_{channel_idx}.{format}')
            if format == 'gif':
                anim.save(filename, writer=PillowWriter(fps=fps))
            elif format == 'mp4':
                anim.save(filename, writer=FFMpegWriter(fps=fps))
            
            if self.verbose > 0:
                print(f'Saved animation to \n {os.path.abspath(filename)}')
        
        # plt.show() # Animation show loops forever, usually checking save
        plt.close()
