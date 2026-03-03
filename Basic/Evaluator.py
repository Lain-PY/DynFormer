import torch
import os
from Loss import DyMixOp_Loss
import pandas as pd

class Evaluator:
    """
    Wraps evaluation and inference functions.
    """
    def __init__(self, config, model, test_loader, device, verbose, preprocessor=None):
        self.config = config

        self.model_name = self.config.model.model_name
        self.num_channel = self.config.data.input_dim

        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.verbose = verbose
        self.preprocessor = preprocessor

    def evaluate(self, loss_func):
        """
        Run evaluation if predictions don't exist
        
        Args:
            loss_func: Loss function
            
        Returns:
            Dictionary of model names mapped to their predictions
        """
        
        predictions = {}
        metrics = {}
        
        with torch.no_grad():
            self.model.eval()
            
            # Initialize prediction tensor
            inp = []
            predictions = []
            truths = []
            
            # Run evaluation
            for num_iter, (input_data, truth_data, static_data) in enumerate(self.test_loader):
                # input_data (batch, length*channel, res_x, res_y); truth_data (batch, length, channel, res_x, res_y)
                # truth_data = truth_data.permute(1, 0, 2, 3, 4)  # (length, batch, channel, res_x, res_y);

                prediction_data = self.model(input_data, static_data, truth_data)
                
                # inverse normalize
                if self.preprocessor is not None:
                    prediction_data = self.preprocessor.inverse_normalize_output(prediction_data)

                predictions.append(prediction_data)
                truths.append(truth_data)

                input_data_shape = input_data.shape
                inp.append(input_data.reshape(input_data_shape[0], -1, *input_data_shape[2:]))
            
            test_num_iter = num_iter + 1

        # Stack predictions along time dimension
        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)
        inp = torch.cat(inp, dim=0)

        metrics = loss_func(pred=predictions, truth=truths)
        
        inference_dict = {"predictions": predictions, "truths": truths, "metrics": metrics, "inputs": inp}
        
        return inference_dict


    def infer(self, save_info_dir, loss_func=None):
        """
        Load predictions from files if they exist, otherwise run inference if predictions don't exist
        
        Args:
            inference_output_dir: Directory to save inference results
            loss_func: Loss function
            
        Returns:
            Dictionary of model names mapped to their predictions
        """

        inference_path = os.path.join(save_info_dir, 'predictions.pt')
        loss_func = DyMixOp_Loss(reccons_type='mse', consist_type='mse', weight_reccons=1.0, weight_consist=0) if loss_func is None else loss_func
        
        # First try to load predictions from file
        if os.path.exists(inference_path):
            if self.verbose > 0:
                print(f"Loading pre-computed predictions for {self.model_name} from: \n {inference_path}")

            inference_dict = torch.load(inference_path)
            metrics = inference_dict['metrics']
        else:
            # Create output directory
            os.makedirs(save_info_dir, exist_ok=True)

            # Run inference if predictions don't exist
            if self.verbose > 0:
                print(f"Running inference for {self.model_name}...")
            
            inference_dict = self.evaluate(loss_func)
            metrics = loss_func.calculate_metrics(pred=inference_dict['predictions'], truth=inference_dict['truths'])
            inference_dict['metrics'] = metrics
            inference_dict['model_name'] = self.model_name

            # inverse normalize for saving inference and later visualization
            # if self.preprocessor is not None:
            #     inference_dict['predictions'] = self.preprocessor.inverse_normalize_output(inference_dict['predictions'])
            #     inference_dict['truths'] = self.preprocessor.inverse_normalize_output(inference_dict['truths'])

            torch.save(inference_dict, inference_path)

            # Save inference information
            infer_info_file_path = os.path.join(save_info_dir, 'inference_metrics.csv')
            infer_info_entry = pd.DataFrame([inference_dict['metrics']])
            infer_info_entry.to_csv(infer_info_file_path, sep=';', index=False)

        # print
        if self.verbose > 0:
            print(f"Saved predictions for {self.model_name} to {inference_path}")
            print(f"\nMetrics for {self.model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}") 
        
        return inference_dict
