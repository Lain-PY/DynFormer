
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
    parser.add_argument('--config', type=str, default='Configs/2dNS/config_2dns_tdymixop.json', 
                        help='Path to general JSON configuration file')
    
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

    # Specified the optimizer (optional) If set None, the default component will be used
    optimizer = None
    
    # Train
    if getattr(training_config, 'train', False):
        model = model_creator.create_or_load_model(model_config, model=None, checkpoint_path=training_config.restart_checkpoint_path)
        optimizer = create_or_load_optimizer(model=model, checkpoint_path=training_config.restart_checkpoint_path, optimizer=optimizer, lr=training_config.lr)  # default setting is AdamW or change it through optimizer
        
        # Specified the scheduler (optional) If set None, the default component will be used
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=training_config.lr, epochs=training_config.num_epochs,
                                                        steps_per_epoch=len(train_loader))
        
        loss_func = DyMixOp_Loss(reccons_type=loss_config.reccons_type, consist_type=loss_config.consist_type, weight_reccons=loss_config.weight_reccons, weight_consist=loss_config.weight_consist)

        trainer = Trainer(
            config=config,
            train_loader=train_loader, 
            test_loader=test_loader, 
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=pre,
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

        if vis_config.visual_models and training_config.visualize:
            vis_models = vis_config.visual_models
        else:
            vis_models = [model_config.model_name]

        for model_name in vis_models:
            save_info_dir = os.path.join(training_config.save_info_dir_name, data_config.dataset_name, model_name, variant_id)
            
            if not os.path.exists(os.path.join(save_info_dir, 'predictions.pt')):
                infer_checkpoint_path = os.path.join(save_info_dir, training_config.save_checkpoint_dir_name, training_config.infer_checkpoint_path)
                model = model_creator.create_or_load_model(model_config, model=None, checkpoint_path=infer_checkpoint_path)
            else:
                model = None
            
            evalr = Evaluator(
                config=config,
                model=model,
                test_loader=test_loader,
                device=config.device,
                verbose=config.verbose,
                preprocessor=pre,
            )
            inference_dict = evalr.infer(save_info_dir)
            inference_dicts.append(inference_dict)

        # Visualization
        if getattr(training_config, 'visualize', False) and inference_dict:
            vis = Visualizer()
            
            # Compare visualization
            sample_idx = vis_config.sample_idx
            time_idx = vis_config.time_idx
            channel_idx = vis_config.channel_idx
            visual_compare_path = os.path.join(training_config.save_info_dir_name, data_config.dataset_name, vis_config.visual_compare_dir_name)
            
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
