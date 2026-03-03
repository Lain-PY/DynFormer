import argparse
import os
from Preprocessor import Preprocessor
from Trainer import Trainer
from Evaluator import Evaluator
from utils import *
from Loss import DyMixOp_Loss
from Visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='DyMixOp model training and evaluation')

    # Add configuration options
    parser.add_argument('--config', type=str, default='Configs/FNO/config_3dbrusselator_FNO.json', 
                        help='Path to general JSON configuration file')
    parser.add_argument('--physical_gpu_id', type=int, default=0, 
                        help='Physical GPU ID for logging (passed from bash script)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load configuration as an object with attributes
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Extract sections from config
    data_config = config.data
    model_config = config.model
    training_config = config.training
    vis_config = config.visualization
    loss_config = config.loss
    
    # Preprocess
    torch.manual_seed(training_config.random_seed)
    np.random.seed(training_config.random_seed)
    pre = Preprocessor(config)
    train_loader, test_loader = pre.load_and_preprocess_data()

    # Create or load model
    model_creator = ModelCreator(config.device, config.verbose)

    # Train
    if getattr(training_config, 'train', False):
        model = model_creator.create_or_load_model(model_config, model=None, checkpoint_path=training_config.restart_checkpoint_path)
        optimizer = create_or_load_optimizer(model=model, checkpoint_path=training_config.restart_checkpoint_path, optimizer=None, lr=training_config.lr)  # default setting is AdamW or change it through optimizer
        loss_func = DyMixOp_Loss(reccons_type=loss_config.reccons_type, consist_type=loss_config.consist_type, weight_reccons=loss_config.weight_reccons, weight_consist=loss_config.weight_consist)

        trainer = Trainer(
            config=config,
            train_loader=train_loader, 
            test_loader=test_loader, 
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=None,
            preprocessor=pre,
            gpu_id=args.physical_gpu_id,
        )
        trainer.train()

    # Inference    
    inference_dicts = []  
    if getattr(training_config, 'inference', False):
        if training_config.variant_id != "default":
            variant_id = training_config.variant_id
        elif training_config.variant_id == "default" and training_config.train:
            variant_id = trainer.variant_id
        else:
            variant_id = "Variant1"

        model_names = [model_config.model_name]

        for model_name in model_names:
            # Construct path for loading model checkpoint (always use original Results directory)
            checkpoint_save_info_dir = os.path.join(training_config.save_info_dir_name, data_config.dataset_name, model_name, variant_id)
            
            infer_checkpoint_path = os.path.join(checkpoint_save_info_dir, training_config.save_checkpoint_dir_name, training_config.infer_checkpoint_path)
            if not os.path.exists(infer_checkpoint_path):
                raise Exception("Checkpoint not found. Please run Training first.")
            
            # If inference follows training, it is not necessary to load model
            if not getattr(training_config, 'train', False):
                model = model_creator.create_or_load_model(model_config, model=None, checkpoint_path=infer_checkpoint_path)
            
            # Construct path for saving inference results
            # If scaled_inference_output_dir is specified, use it; otherwise use the original directory
            if hasattr(training_config, 'scaled_inference_output_dir') and training_config.scaled_inference_output_dir:
                # For scaled inference, create output directory with scaling factor in the name
                scaling_factor = getattr(training_config, 'scaled_inference_factor', 1.0)
                scaled_variant_id = f"{variant_id}_scale{scaling_factor}"
                inference_save_info_dir = os.path.join(
                    training_config.scaled_inference_output_dir, 
                    data_config.dataset_name, 
                    model_name, 
                    scaled_variant_id
                )
            else:
                # Normal inference, save to original Results directory
                inference_save_info_dir = checkpoint_save_info_dir
            
            evalr = Evaluator(
                config=config,
                model=model,
                test_loader=test_loader,
                device=config.device,
                verbose=config.verbose,
                preprocessor=pre,
            )

            # inference metrics (save to the appropriate directory)
            inference_dict = evalr.infer(inference_save_info_dir)
            inference_dicts.append(inference_dict)

    # Visualization
    if getattr(training_config, 'visualize', False):
        if not inference_dicts:
            for model_name in vis_config.vis_models:
                save_info_dir = os.path.join(training_config.save_info_dir_name, data_config.dataset_name, model_name, variant_id)
                inference_path = os.path.join(save_info_dir, 'predictions.pt')
                inference_dict = torch.load(inference_path)
                inference_dicts.append(inference_dict)

        vis = Visualizer(config)
        
        # Compare visualization
        sample_idx = vis_config.sample_idx
        time_idx = vis_config.time_idx
        channel_idx = vis_config.channel_idx
        visual_compare_path = os.path.join(training_config.save_info_dir_name, data_config.dataset_name, model_config.model_name, variant_id, vis_config.visual_compare_dir_name)
        
        vis.compare(
            inference_dicts=inference_dicts, 
            save_path=visual_compare_path,
            sample_idx=sample_idx,
            time_idx=time_idx,
            channel_idx=channel_idx,
        )

    print("Processing complete!")


if __name__ == '__main__':
    main()
