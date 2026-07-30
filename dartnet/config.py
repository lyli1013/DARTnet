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
#     """保存分类任务的指标"""
#     metrics_data = []
#     # 分类指标名称（与模型输出的元组顺序对应）
#     metric_names = ["AUROC", "MCC", "Accuracy", "F1", "TP", "TN", "FP", "FN"]
    
#     for key, metrics_tuple in model_test_metrics.items():
#         for idx, tensor_value in enumerate(metrics_tuple):
#             # 转换张量为Python数值
#             if isinstance(tensor_value, torch.Tensor):
#                 metric_value = tensor_value.cpu().item() if tensor_value.device.type != 'cpu' else tensor_value.item()
#             else:
#                 metric_value = tensor_value  # 整数指标（如TP、TN）直接保留
            
#             # 确保索引不超出指标名称列表范围
#             if idx < len(metric_names):
#                 metrics_data.append({
#                     "key": key,
#                     "metric": metric_names[idx],
#                     "value": round(metric_value, 6) if isinstance(metric_value, float) else metric_value
#                 })
    
#     pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)




def get_epochs_from_ckpt_dir(output_save_dir):
    """
    从指定目录下的所有.ckpt文件中提取epoch值
    
    参数:
        output_save_dir (str): 存放ckpt文件的目录路径
    返回:
        dict: 文件名到epoch值的映射（如{"epoch=044.ckpt": 44}）
    """
    if not os.path.isdir(output_save_dir):
        raise NotADirectoryError(f"目录不存在：{output_save_dir}")
    epoch_dict = {}
    for filename in os.listdir(output_save_dir):
        if filename.endswith(".ckpt"):
            match = re.search(r"epoch=(\d+)", filename)   # 用正则提取epoch数值（匹配"epoch=数字"格式）
            if match:
                epoch = int(match.group(1))  # 转换为整数（自动去除前导0）
                epoch_dict[filename] = epoch + 1
                best_ckpt_path = os.path.join(output_save_dir, filename)  # 新增：获取最优模型的完整ckpt路径（核心！用于加载阶段1模型）
                print(f"阶段1最优模型路径：{best_ckpt_path}")
            else:
                print(f"警告：文件 {filename} 不含epoch信息，已跳过")
    if not epoch_dict:
        raise FileNotFoundError(f"目录 {output_save_dir} 中未找到含epoch的.ckpt文件")
    return epoch_dict, best_ckpt_path


    

def save_classification_metrics_val_best(model_test_metrics, metrics_path, epoch_type=None, hyperparameters=None):
    """保存分类任务指标（含推导的敏感率和特异性）及超参数"""
    metrics_data = []
    # 指标名称与 get_cls_metrics_binary_pt 返回顺序严格对应
    metric_names = [
        "AUROC",          # 0: 原函数返回
        "PRAUC",          # 1: 原函数返回
        "MCC",            # 2: 原函数返回
        "Accuracy",       # 3: 原函数返回
        "F1",             # 4: 原函数返回
        "Sensitivity",    # 5: 推导指标（TP/(TP+FN)）
        "Specificity",    # 6: 推导指标（TN/(TN+FP)）
        "TP",             # 7: 原函数返回
        "TN",             # 8: 原函数返回
        "FP",             # 9: 原函数返回
        "FN"              # 10: 原函数返回
    ]
    
    # 确保超参数是字典，如果未提供则为空字典
    hyperparameters = hyperparameters or {}
    
    for key, metrics_tuple in model_test_metrics.items():
        # 1. 提取原函数返回的指标
        auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn = metrics_tuple
        
        # 2. 推导计算敏感率和特异性（避免除0错误）
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 敏感率=召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异性
        
        # 3. 组合完整指标列表（包含推导指标）
        full_metrics = [
            auroc, prauc, mcc, accuracy, f1,
            sensitivity, specificity,  # 插入推导指标
            tp, tn, fp, fn
        ]
        
        # 4. 处理并保存指标，同时加入超参数
        metric_dict = {}
        # 添加超参数
        for hp_name, hp_value in hyperparameters.items():
            metric_dict[hp_name] = hp_value
        
        # 添加key（通常是epoch信息）
        metric_dict["key"] = key
        
        # 添加指标
        for idx, metric_value in enumerate(full_metrics):
            # 转换张量为Python数值
            if isinstance(metric_value, torch.Tensor):
                processed_value = metric_value.cpu().item() if metric_value.device.type != 'cpu' else metric_value.item()
            else:
                processed_value = metric_value  # 整数/浮点数直接保留
            
            # 格式化数值（浮点数保留3位小数）
            if isinstance(processed_value, float):
                processed_value = round(processed_value, 3)
            
            # 保存指标
            if idx < len(metric_names):
                metric_dict[metric_names[idx]] = processed_value
        
        metrics_data.append(metric_dict)
    
    # 保存为CSV，使用追加模式
    df = pd.DataFrame(metrics_data)
    # 检查文件是否存在，不存在则写入表头
    file_exists = os.path.exists(metrics_path)
    df.to_csv(metrics_path, index=False, mode='a', header=not file_exists)
    print(f"{epoch_type}分类指标及超参数已保存至: {metrics_path}")


def save_classification_metrics(model_test_metrics, metrics_path, epoch_type=None):
    """保存分类任务指标（含推导的敏感率和特异性）"""
    metrics_data = []
    # 指标名称与 get_cls_metrics_binary_pt 返回顺序严格对应
    # 前4个为原函数直接返回，中间2个为推导指标，最后4个为混淆矩阵指标
    metric_names = [
        "AUROC",          # 0: 原函数返回
        "PRAUC",          # 1: 原函数返回
        "MCC",            # 2: 原函数返回
        "Accuracy",       # 3: 原函数返回
        "F1",             # 4: 原函数返回
        "Sensitivity",    # 5: 推导指标（TP/(TP+FN)）
        "Specificity",    # 6: 推导指标（TN/(TN+FP)）
        "TP",             # 7: 原函数返回
        "TN",             # 8: 原函数返回
        "FP",             # 9: 原函数返回
        "FN"              # 10: 原函数返回
    ]
    
    for key, metrics_tuple in model_test_metrics.items():
        # 1. 提取原函数返回的8个指标
        auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn = metrics_tuple
        
        # 2. 推导计算敏感率和特异性（避免除0错误）
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 敏感率=召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异性
        
        # 3. 组合完整指标列表（包含推导指标）
        full_metrics = [
            auroc, prauc, mcc, accuracy, f1,
            sensitivity, specificity,  # 插入推导指标
            tp, tn, fp, fn
        ]
        
        # 4. 处理并保存指标
        for idx, metric_value in enumerate(full_metrics):
            # 转换张量为Python数值
            if isinstance(metric_value, torch.Tensor):
                processed_value = metric_value.cpu().item() if metric_value.device.type != 'cpu' else metric_value.item()
            else:
                processed_value = metric_value  # 整数/浮点数直接保留
            
            # 格式化数值（浮点数保留6位小数）
            if isinstance(processed_value, float):
                processed_value = round(processed_value, 3)
            
            # 保存指标
            if idx < len(metric_names):
                metrics_data.append({
                    "key": key,
                    "metric": metric_names[idx],
                    "value": processed_value
                })
    
    # 保存为CSV
    pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)
    print(f"{epoch_type}分类指标已保存至: {metrics_path}")
    

def save_classification_metrics_tra_val(model_metrics, metrics_path, epoch_type="Validation"):
    """
    保存分类任务指标（适配 model.val_metrics 为字典的结构）
    参数:
        model_metrics: 字典，key=轮次索引（0开始），value=每轮指标元组
        metrics_path: CSV保存路径
        epoch_type: 指标类型（如"Validation"）
    """
    # 指标名称（与元组顺序严格对应：auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn）
    metric_names = [
        "Epoch",          # 轮次列（1开始计数，更符合习惯）
        "AUROC",          # 0: 原元组中的tensor指标
        "PRAUC",          # 1: 原元组中的tensor指标
        "MCC",            # 2: 原元组中的tensor指标
        "Accuracy",       # 3: 原元组中的tensor指标
        "F1",             # 4: 原元组中的tensor指标
        "Sensitivity",    # 推导指标（TP/(TP+FN)）
        "Specificity",    # 推导指标（TN/(TN+FP)）
        "TP",             # 5: 原元组中的整数
        "TN",             # 6: 原元组中的整数
        "FP",             # 7: 原元组中的整数
        "FN"              # 8: 原元组中的整数
    ]
    
    all_epochs_data = []  # 存储所有轮次的处理后数据
    
    # 遍历字典：key=轮次索引（0开始），value=指标元组
    for epoch_idx, metrics_tuple in model_metrics.items():
        # 1. 验证指标元组长度（确保是9个元素，与预期一致）
        expected_tuple_len = 9
        if len(metrics_tuple) != expected_tuple_len:
            raise ValueError(
                f"轮次 {epoch_idx} 的指标元组长度错误！\n"
                f"期望 {expected_tuple_len} 个元素，实际 {len(metrics_tuple)} 个。"
            )
        
        # 2. 解包指标元组（与元组顺序严格对应）
        auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn = metrics_tuple
        
        # 3. 推导敏感率（Sensitivity）和特异度（Specificity），避免除0错误
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 4. 处理指标值（Tensor转CPU数值，保留3位小数）
        def process_value(val):
            if isinstance(val, torch.Tensor):
                # Tensor转Python数值（先移到CPU，避免设备问题）
                return round(val.cpu().item(), 3)
            elif isinstance(val, (int, float)):
                # 整数/浮点数直接保留（整数无需小数，浮点数保留3位）
                return val if isinstance(val, int) else round(val, 3)
            else:
                return val
        
        # 组合当前轮次的完整数据（轮次从1开始计数）
        current_epoch_data = [
            epoch_idx + 1,  # 轮次：0→1，1→2，...（更符合实际训练习惯）
            process_value(auroc),
            process_value(prauc),
            process_value(mcc),
            process_value(accuracy),
            process_value(f1),
            round(sensitivity, 3),  # 推导指标保留3位小数
            round(specificity, 3),  # 推导指标保留3位小数
            tp, tn, fp, fn  # 混淆矩阵指标（整数）
        ]
        
        all_epochs_data.append(current_epoch_data)
    
    # 5. 转换为DataFrame并保存CSV
    df = pd.DataFrame(all_epochs_data, columns=metric_names)
    df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"{epoch_type}分类指标已保存至: {metrics_path}")
    print(f"共保存 {len(all_epochs_data)} 轮验证指标")
