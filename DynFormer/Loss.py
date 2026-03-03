import torch
import torch.nn as nn

class DyMixOp_Loss(nn.Module):
    def __init__(self, reccons_type='mse', consist_type='mse', weight_reccons=1.0, weight_consist=0):
        super(DyMixOp_Loss, self).__init__()
        self.reccons_type = reccons_type
        self.consist_type = consist_type
        self.weight_reccons = weight_reccons
        self.weight_consist = weight_consist

        self.mse_loss = nn.MSELoss()
    
    def relative_mse_loss(self, pred, truth):
        """
        Calculate the relative mean squared error loss
        
        Args:
            pred: Model predictions
            truth: Ground truth values
        
        Returns:
            Relative MSE loss
        """
        return self.mse_loss(pred, truth) / self.mse_loss(torch.zeros_like(truth), truth)

    def dymixop_loss(self, pred, truth):
        """
        Calculate the loss including reconstruct and consistency for the DyMixOp model training
        
        Args:
            pred: Model predictions
            truth: Ground truth values
        
        Returns:
            Total loss
        """
        # reconstruct
        if self.reccons_type == 'mse':
            reconstruct_loss = self.mse_loss(pred, truth)
        elif self.reccons_type == 'relative_mse':
            reconstruct_loss = self.relative_mse_loss(pred, truth)
        else:
            raise Exception(rf"Reconstruct type '{self.reccons_type}' does not exist. Only 'mse' and 'relative_mse' are provided.")

        # consistency
        if self.consist_type == 'mse':
            consistency_loss = self.mse_loss(pred, truth)
        elif self.consist_type == 'relative_mse':
            consistency_loss = self.relative_mse_loss(pred, truth)
        else:
            raise Exception(rf"Consistency type '{self.consist_type}' does not exist. Only 'mse' and 'relative_mse' are provided.")

        return self.weight_reccons * reconstruct_loss + self.weight_consist * consistency_loss

    def calculate_metrics(self, pred, truth):
        """
        Calculate reconstruct metrics between true and predicted data for evaluation
        
        Args:
            truth: Ground truth data tensor
            pred: Model prediction tensor
        
        Returns:
            Dictionary of metrics (MSE, MAE, RMSE, NRMSE, R2)
        """
        if truth.shape.numel() == 0 or pred.shape.numel() == 0:
            return {
                'MSE': torch.tensor(0.0, dtype=torch.float32),
                'Rel. MSE': torch.tensor(0.0, dtype=torch.float32),
                'MAE': torch.tensor(0.0, dtype=torch.float32),
                'RMSE': torch.tensor(0.0, dtype=torch.float32),
                'NRMSE': torch.tensor(0.0, dtype=torch.float32),
                'R2': torch.tensor(0.0, dtype=torch.float32)
            }
    
        # Calculate metrics
        mse = torch.mean((truth - pred) ** 2)
        rel_mse = mse / torch.mean(truth ** 2)
        mae = torch.mean(torch.abs(truth - pred))
        rmse = torch.sqrt(mse)
    
        # Normalized RMSE
        data_range = torch.max(truth) - torch.min(truth)
        nrmse = rmse / data_range if data_range > 0 else rmse
    
        # R^2 score
        ss_tot = torch.sum((truth - torch.mean(truth)) ** 2)
        ss_res = torch.sum((truth - pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
        return {
            'MSE': mse.cpu().numpy(),
            'Rel. MSE': rel_mse.cpu().numpy(),
            'MAE': mae.cpu().numpy(),
            'RMSE': rmse.cpu().numpy(),
            'NRMSE': nrmse.cpu().numpy(),
            'R2': r2.cpu().numpy()
        }

    def __call__(self, pred, truth):
        return self.dymixop_loss(pred, truth)