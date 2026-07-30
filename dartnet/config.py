import os
import re
import json
import torch
import pandas as pd


def save_gnn_arguments_to_json(argsdict, out_path):
    json_out_path = os.path.join(out_path, "gnn_hyperparameters.json")
    with open(json_out_path, "w", encoding="UTF-8") as f:
        json.dump(argsdict, f)
    return json_out_path


def load_gnn_arguments_from_json(json_path):
    with open(json_path, "r", encoding="UTF-8") as f:
        argsdict = json.load(f)

    return argsdict


def validate_gnn_argparse_arguments(argsdict):
    assert "seed" in argsdict
    assert "dataset_download_dir" in argsdict
    assert "lr" in argsdict
    assert "batch_size" in argsdict
    assert "early_stopping_patience" in argsdict
    assert "output_node_dim" in argsdict
    assert "num_layers" in argsdict
    assert "gnn_intermediate_dim" in argsdict
    assert "out_path" in argsdict
    argsdict.setdefault("conv_type", "GAT")
    assert argsdict["gat_attn_heads"] >= 1


def get_gnn_wandb_name(argsdict):
    if "dataset" not in argsdict:
        dataset = None
    else:
        dataset = argsdict['dataset']
    name = f"GNN+{dataset}+T={argsdict['dataset_target_name']}+S={argsdict['seed']}+GAT"
    name += f"+GC={argsdict['gradient_clip_val']}"
    name += f"+OPTD={argsdict['optimiser_weight_decay']}"

    if "dataset_one_hot" not in argsdict:
        dataset_one_hot = None
    else:
        dataset_one_hot = argsdict['dataset_one_hot']
    name += f"+OH={dataset_one_hot}"
    name += f"+NDIM={argsdict['output_node_dim']}"
    name += f"+NL={argsdict['num_layers']}+GIDIM={argsdict['gnn_intermediate_dim']}"
    name += f"+GATH={argsdict['gat_attn_heads']}+GATD={argsdict['gat_dropout']}"
    name += f"+BS={argsdict['batch_size']}"
    name += f"+ESP={argsdict['early_stopping_patience']}"
    name += f"+lr={argsdict['lr']}"
    name += "_atteCat"
    name += f"_embdrop{argsdict['emb_feat_out_drop']}"
    name += f"_fpdrop{argsdict['fp_feat_out_drop']}"
    name += f"_gnndrop{argsdict['gnn_feat_out_drop']}"
    return name


# def save_classification_metrics(model_test_metrics, metrics_path):
#     """Save classification-task metrics"""
#     metrics_data = []
#     # Classification metric names (aligned with model output tuple order)
#     metric_names = ["AUROC", "MCC", "Accuracy", "F1", "TP", "TN", "FP", "FN"]
    
#     for key, metrics_tuple in model_test_metrics.items():
#         for idx, tensor_value in enumerate(metrics_tuple):
#             # Convert tensor to Python numeric value
#             if isinstance(tensor_value, torch.Tensor):
#                 metric_value = tensor_value.cpu().item() if tensor_value.device.type != 'cpu' else tensor_value.item()
#             else:
#                 metric_value = tensor_value  # Keep integer metrics (e.g. TP, TN) as-is
            
#             # Ensure index stays within metric name list bounds
#             if idx < len(metric_names):
#                 metrics_data.append({
#                     "key": key,
#                     "metric": metric_names[idx],
#                     "value": round(metric_value, 6) if isinstance(metric_value, float) else metric_value
#                 })
    
#     pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)




def get_epochs_from_ckpt_dir(output_save_dir):
    """
    Extract epoch values from all .ckpt files under the given directory.

    Args:
        output_save_dir (str): Directory path containing ckpt files
    Returns:
        dict: Mapping from filename to epoch value (e.g. {"epoch=044.ckpt": 44})
    """
    if not os.path.isdir(output_save_dir):
        raise NotADirectoryError(f"Directory does not exist: {output_save_dir}")
    epoch_dict = {}
    for filename in os.listdir(output_save_dir):
        if filename.endswith(".ckpt"):
            match = re.search(r"epoch=(\d+)", filename)   # Extract epoch number via regex (matches "epoch=<digits>")
            if match:
                epoch = int(match.group(1))  # Convert to int (leading zeros stripped)
                epoch_dict[filename] = epoch + 1
                best_ckpt_path = os.path.join(output_save_dir, filename)  # Full path to best-model ckpt (used to load stage-1 model)
                print(f"Stage-1 best model path: {best_ckpt_path}")
            else:
                print(f"Warning: file {filename} has no epoch info, skipped")
    if not epoch_dict:
        raise FileNotFoundError(f"No .ckpt files with epoch info found in directory {output_save_dir}")
    return epoch_dict, best_ckpt_path


    

def save_classification_metrics_val_best(model_test_metrics, metrics_path, epoch_type=None, hyperparameters=None):
    """Save classification metrics (including derived Sensitivity/Specificity) and hyperparameters"""
    metrics_data = []
    # Metric names strictly aligned with get_cls_metrics_binary_pt return order
    metric_names = [
        "AUROC",          # 0: returned by original function
        "PRAUC",          # 1: returned by original function
        "MCC",            # 2: returned by original function
        "Accuracy",       # 3: returned by original function
        "F1",             # 4: returned by original function
        "Sensitivity",    # 5: derived metric (TP/(TP+FN))
        "Specificity",    # 6: derived metric (TN/(TN+FP))
        "TP",             # 7: returned by original function
        "TN",             # 8: returned by original function
        "FP",             # 9: returned by original function
        "FN"              # 10: returned by original function
    ]
    
    # Ensure hyperparameters is a dict; default to empty if not provided
    hyperparameters = hyperparameters or {}
    
    for key, metrics_tuple in model_test_metrics.items():
        # 1. Unpack metrics returned by the original function
        auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn = metrics_tuple
        
        # 2. Derive Sensitivity and Specificity (avoid division by zero)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity = Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        
        # 3. Build full metric list (including derived metrics)
        full_metrics = [
            auroc, prauc, mcc, accuracy, f1,
            sensitivity, specificity,  # Insert derived metrics
            tp, tn, fp, fn
        ]
        
        # 4. Process and save metrics, also attach hyperparameters
        metric_dict = {}
        # Add hyperparameters
        for hp_name, hp_value in hyperparameters.items():
            metric_dict[hp_name] = hp_value
        
        # Add key (usually epoch info)
        metric_dict["key"] = key
        
        # Add metrics
        for idx, metric_value in enumerate(full_metrics):
            # Convert tensor to Python numeric value
            if isinstance(metric_value, torch.Tensor):
                processed_value = metric_value.cpu().item() if metric_value.device.type != 'cpu' else metric_value.item()
            else:
                processed_value = metric_value  # Keep int/float as-is
            
            # Format values (floats rounded to 3 decimals)
            if isinstance(processed_value, float):
                processed_value = round(processed_value, 3)
            
            # Store metric
            if idx < len(metric_names):
                metric_dict[metric_names[idx]] = processed_value
        
        metrics_data.append(metric_dict)
    
    # Save as CSV in append mode
    df = pd.DataFrame(metrics_data)
    # Write header only if file does not exist yet
    file_exists = os.path.exists(metrics_path)
    df.to_csv(metrics_path, index=False, mode='a', header=not file_exists)
    print(f"{epoch_type} classification metrics and hyperparameters saved to: {metrics_path}")


def save_classification_metrics(model_test_metrics, metrics_path, epoch_type=None):
    """Save classification metrics (including derived Sensitivity/Specificity)"""
    metrics_data = []
    # Metric names strictly aligned with get_cls_metrics_binary_pt return order
    # First metrics from original function, next 2 derived, last 4 confusion-matrix metrics
    metric_names = [
        "AUROC",          # 0: returned by original function
        "PRAUC",          # 1: returned by original function
        "MCC",            # 2: returned by original function
        "Accuracy",       # 3: returned by original function
        "F1",             # 4: returned by original function
        "Sensitivity",    # 5: derived metric (TP/(TP+FN))
        "Specificity",    # 6: derived metric (TN/(TN+FP))
        "TP",             # 7: returned by original function
        "TN",             # 8: returned by original function
        "FP",             # 9: returned by original function
        "FN"              # 10: returned by original function
    ]
    
    for key, metrics_tuple in model_test_metrics.items():
        # 1. Unpack the metrics returned by the original function
        auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn = metrics_tuple
        
        # 2. Derive Sensitivity and Specificity (avoid division by zero)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity = Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        
        # 3. Build full metric list (including derived metrics)
        full_metrics = [
            auroc, prauc, mcc, accuracy, f1,
            sensitivity, specificity,  # Insert derived metrics
            tp, tn, fp, fn
        ]
        
        # 4. Process and save metrics
        for idx, metric_value in enumerate(full_metrics):
            # Convert tensor to Python numeric value
            if isinstance(metric_value, torch.Tensor):
                processed_value = metric_value.cpu().item() if metric_value.device.type != 'cpu' else metric_value.item()
            else:
                processed_value = metric_value  # Keep int/float as-is
            
            # Format values (floats rounded to 3 decimals; comment historically said 6)
            if isinstance(processed_value, float):
                processed_value = round(processed_value, 3)
            
            # Store metric
            if idx < len(metric_names):
                metrics_data.append({
                    "key": key,
                    "metric": metric_names[idx],
                    "value": processed_value
                })
    
    # Save as CSV
    pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)
    print(f"{epoch_type} classification metrics saved to: {metrics_path}")
    

def save_classification_metrics_tra_val(model_metrics, metrics_path, epoch_type="Validation"):
    """
    Save classification metrics (adapted to model.val_metrics as a dict).
    Args:
        model_metrics: dict, key=epoch index (0-based), value=per-epoch metric tuple
        metrics_path: CSV save path
        epoch_type: metric type label (e.g. "Validation")
    """
    # Metric names (strictly aligned with tuple order: auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn)
    metric_names = [
        "Epoch",          # Epoch column (1-based counting, more conventional)
        "AUROC",          # 0: tensor metric from original tuple
        "PRAUC",          # 1: tensor metric from original tuple
        "MCC",            # 2: tensor metric from original tuple
        "Accuracy",       # 3: tensor metric from original tuple
        "F1",             # 4: tensor metric from original tuple
        "Sensitivity",    # Derived metric (TP/(TP+FN))
        "Specificity",    # Derived metric (TN/(TN+FP))
        "TP",             # 5: integer from original tuple
        "TN",             # 6: integer from original tuple
        "FP",             # 7: integer from original tuple
        "FN"              # 8: integer from original tuple
    ]
    
    all_epochs_data = []  # Store processed data for all epochs
    
    # Iterate dict: key=epoch index (0-based), value=metric tuple
    for epoch_idx, metrics_tuple in model_metrics.items():
        # 1. Validate metric tuple length (expect 9 elements)
        expected_tuple_len = 9
        if len(metrics_tuple) != expected_tuple_len:
            raise ValueError(
                f"Metric tuple length error at epoch {epoch_idx}!\n"
                f"Expected {expected_tuple_len} elements, got {len(metrics_tuple)}."
            )
        
        # 2. Unpack metric tuple (strict order)
        auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn = metrics_tuple
        
        # 3. Derive Sensitivity and Specificity, avoid division by zero
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 4. Process metric values (Tensor -> CPU numeric, round to 3 decimals)
        def process_value(val):
            if isinstance(val, torch.Tensor):
                # Tensor to Python numeric (move to CPU first to avoid device issues)
                return round(val.cpu().item(), 3)
            elif isinstance(val, (int, float)):
                # Keep int/float; round floats to 3 decimals
                return val if isinstance(val, int) else round(val, 3)
            else:
                return val
        
        # Assemble full row for current epoch (epochs counted from 1)
        current_epoch_data = [
            epoch_idx + 1,  # Epoch: 0->1, 1->2, ... (matches training convention)
            process_value(auroc),
            process_value(prauc),
            process_value(mcc),
            process_value(accuracy),
            process_value(f1),
            round(sensitivity, 3),  # Derived metric, 3 decimals
            round(specificity, 3),  # Derived metric, 3 decimals
            tp, tn, fp, fn  # Confusion-matrix metrics (integers)
        ]
        
        all_epochs_data.append(current_epoch_data)
    
    # 5. Convert to DataFrame and save CSV
    df = pd.DataFrame(all_epochs_data, columns=metric_names)
    df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"{epoch_type} classification metrics saved to: {metrics_path}")
    print(f"Saved metrics for {len(all_epochs_data)} epochs")
