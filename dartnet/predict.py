import argparse
import os
import sys
import torch
torch.backends.cuda.matmul.allow_tf32 = False   ## May introduce numeric diffs. TF32 (TensorFloat-32) on NVIDIA Ampere is faster but lower precision and may be non-deterministic
torch.backends.cudnn.allow_tf32 = False
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger  # unused; infer uses logger=False
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch_geometric.seed import seed_everything

# Import project dependencies
sys.path.append(os.path.realpath("."))
from dartnet.model import DartNet
from data_loading.data_loading import get_dataset_train_val_test
from dartnet.config import load_gnn_arguments_from_json



def fix_seed(seed):
    """
    Fixed version of the random seed setup function.
    
    Args:
        seed: Random seed
        full_deterministic: Whether to enable fully deterministic mode (may reduce performance)
    """
    if seed is None:
        seed = random.randint(1, 10000)
    
    # 1. Basic random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 2. Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Important for CUDA 10.2+
    
    # 3. GPU-related settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # if full_deterministic:
        # Fully deterministic mode (trade performance for reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        
        # Enable CUDA deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        # else:
        #     # Performance mode (may retain some randomness)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
    
    # 4. Thread settings
    torch.set_num_threads(1)  # Keep; helps determinism
    
    # print(f"[Info] Random seed set to: {seed}")
    # print(f"[Info] Deterministic mode: {full_deterministic}")
    
    return seed

def main():
    parser = argparse.ArgumentParser(description="Load a trained model with PyTorch Lightning and run prediction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer", action='store_true', help='Inference mode')
    parser.add_argument("--ckpt-path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to data for prediction")
    parser.add_argument("--infer-file-name", type=str, required=True, help="Inference file name")
    parser.add_argument("--output-path", type=str, default="./predictions", help="Path to save prediction results")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--use-gpu", action="store_true", help="Whether to use GPU for prediction")
    parser.add_argument(
        "--gpu-devices",
        type=int,
        default=2,
        help="GPU index when --use-gpu is set (same meaning as dartnet.train --gpu-devices)",
    )
    parser.add_argument("--fingerprint-type", type=str, default="morgan2048")
    parser.add_argument(
        "--molformer-ckpt-path",
        type=str,
        default=None,
        help="MolFormer checkpoint; only needed if {infer}_emb.pt is missing and must be generated",
    )
    parser.add_argument(
        "--molformer-hparams-path",
        type=str,
        default=None,
        help="MolFormer hparams.yaml (optional, uses project default if omitted)",
    )
    parser.add_argument(
        "--molformer-env",
        type=str,
        default="MolTran_CUDA11",
        help="Conda env for embedding subprocess; only used if embeddings are missing",
    )

    args = parser.parse_args()

    fix_seed(args.seed)
    
    # Create output directory
    Path(args.output_path).mkdir(exist_ok=True, parents=True)

    ############## Load model config ##############
    ckpt_dir = os.path.dirname(args.ckpt_path)
    config_json_path = [f for f in os.listdir(ckpt_dir) if f.endswith(".json")][0]
    config_json_path = os.path.join(ckpt_dir, config_json_path)
    argsdict = load_gnn_arguments_from_json(config_json_path)

    # print("argsdict:", argsdict)
    ############## Load data ##############
    dataset = argsdict["dataset"]
    test_data, num_classes, task_type, scaler = get_dataset_train_val_test(
        infer=args.infer,
        dataset=dataset,
        dataset_dir=args.dataset_dir,
        infer_file_name=args.infer_file_name,
        one_hot=argsdict["dataset_one_hot"],
        target_name=argsdict["dataset_target_name"],
        # Pass fingerprint computation args (optional; omit to use defaults)
        fingerprint_type=argsdict["fingerprint_type"],    # Compute Morgan fingerprints
        morgan_radius=2,              # Radius=2 (ECFP4)
        morgan_nBits=2048,            # Bits=2048 (default 2048; comment historically said 1024)
        useChirality=True,            # Consider chirality
        num_processes=4,              # Parallel processes (tune to CPU cores; historically noted as 8)
        fp_chunk_size=2000,           # Process 2000 samples per chunk
        molformer_ckpt_path=args.molformer_ckpt_path,
        molformer_hparams_path=args.molformer_hparams_path,
        molformer_env=args.molformer_env,
        gpu_devices=args.gpu_devices if args.use_gpu else None,
    )
    
    # print("type(test_data)", type(test_data))
    # print("test_data[0]", test_data[0])
    test_ids = [data.smiles_id for data in test_data]  # Sample IDs
    num_features = test_data[0].x.shape[-1]
    # print("num_features:", num_features)
    edge_dim = None
    if hasattr(test_data[0], "edge_attr") and test_data[0].edge_attr is not None:   # Check sample has non-empty edge_attr
        edge_dim = test_data[0].edge_attr.shape[-1]    # Per-edge feature dimension
        # print("edge_dim:", edge_dim)
    # Create data loader
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.use_gpu else False
    )

    ############## Load model ##############
    model = DartNet.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        task_type=task_type,
        num_features=num_features,
        # gnn_intermediate_dim=argsdict["gnn_intermediate_dim"],
        # output_node_dim=argsdict["output_node_dim"],
        # num_layers=argsdict["num_layers"],
        # conv_type=argsdict["conv_type"],
        linear_output_size=num_classes,
        scaler=scaler,
        edge_dim=edge_dim,
        **argsdict
    )
    print(model)

    ############## Configure Trainer ##############
    # Configure trainer for inference
    trainer_kwargs = {
        "logger": False,  # No logger needed for inference
        "enable_checkpointing": False,  # Do not save checkpoints during inference
        "enable_progress_bar": True,  # Show progress bar
        "enable_model_summary": True,  # Show model summary
        "max_epochs": 1,  # Inference needs only one epoch
        "callbacks": [TQDMProgressBar(refresh_rate=10)],  # Progress bar callback
    }
    if args.use_gpu:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = [args.gpu_devices]
    else:
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1

    trainer = Trainer(**trainer_kwargs)

    ############## Run prediction with Lightning ##############
    # Run inference with trainer.predict()
    predictions = trainer.predict(
        model=model,
        dataloaders=test_loader,
        return_predictions=True  # Return prediction results
    )

    ############## Process prediction results ##############
    # Only rank-0 process merges and saves results (critical for DDP)
    if trainer.is_global_zero:  # Whether this is the main process (rank 0)
        # Merge all batch results (predictions already include all GPUs)
        all_preds = []
        for batch_pred in predictions:
            # print("batch_pred:", batch_pred)
            # Handle model output format (same logic as before)
            if isinstance(batch_pred, tuple):
                batch_pred = batch_pred[2]
            if hasattr(batch_pred, "logits"):
                batch_pred = batch_pred.logits
            batch_pred = batch_pred.cpu().numpy()
            
            if num_classes == 1:
                batch_pred = 1 / (1 + np.exp(-batch_pred))
            else:
                batch_pred = np.exp(batch_pred) / np.sum(np.exp(batch_pred), axis=1, keepdims=True)
            
            # print("batch_pred:", batch_pred)
            batch_pred = batch_pred.flatten()
            # print("batch_pred:", batch_pred)
            all_preds.append(batch_pred)
        # print("all_preds:", all_preds)        
        # Concatenate all results (should contain all samples)
        all_preds = np.concatenate(all_preds, axis=0)
        # print("all_preds:", all_preds)    
        print(f"Total predicted samples: {len(all_preds)}")  # Sanity-check sample count

        ############## Save prediction results ##############
        preds_path = os.path.join(args.output_path, "infer", f"predictions_{args.infer_file_name}_({np.sum(all_preds > 0.5)}).csv")
        directory = os.path.dirname(preds_path)
        Path(directory).mkdir(exist_ok=True, parents=True)
        
        # # Save only on main process to avoid overwrite
        # pd.DataFrame(all_preds).to_csv(preds_path, index=False, header=False, float_format="%.6f")
        # print(f"Main process saved predictions to: {preds_path}")

    
        # Combine IDs and predictions into a list of dicts
        results = [
            {"ID": id, "Prediction": pred} 
            for id, pred in zip(test_ids, all_preds)
        ]
        
        pd.DataFrame(results).to_csv(
            preds_path, 
            index=False, 
            float_format="%.6f"  # Keep prediction value precision
        )
        print(f"Main process saved predictions to: {preds_path}")
    
    
if __name__ == "__main__":
    main()
