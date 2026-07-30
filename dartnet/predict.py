import argparse
import os
import sys
import torch
torch.backends.cuda.matmul.allow_tf32 = False   ## 可能引入数值差异.TF32（TensorFloat-32）是NVIDIA Ampere架构引入的，虽然计算速度快，但精度较低且可能产生非确定性结果
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

# 导入项目中的依赖
sys.path.append(os.path.realpath("."))
from dartnet.model import DartNet
from data_loading.data_loading import get_dataset_train_val_test
from dartnet.config import load_gnn_arguments_from_json



def fix_seed(seed):
    """
    修正版本的随机种子设置函数
    
    Args:
        seed: 随机种子
        full_deterministic: 是否启用完全确定性模式（可能降低性能）
    """
    if seed is None:
        seed = random.randint(1, 10000)
    
    # 1. 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 2. 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 重要：对于CUDA 10.2+
    
    # 3. GPU相关设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # if full_deterministic:
        # 完全确定性模式（牺牲性能换取可重复性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        
        # 启用CUDA确定性操作
        torch.use_deterministic_algorithms(True, warn_only=True)
        # else:
        #     # 性能模式（可能有一些随机性）
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
    
    # 4. 线程设置
    torch.set_num_threads(1)  # 保持，有助于确定性
    
    # print(f"[Info] 随机种子已设置为: {seed}")
    # print(f"[Info] 确定性模式: {full_deterministic}")
    
    return seed

def main():
    parser = argparse.ArgumentParser(description="使用PyTorch Lightning加载训练好的模型并进行预测")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer", action='store_true', help='推理模式')
    parser.add_argument("--ckpt-path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data-path", type=str, required=True, help="待预测数据路径")
    parser.add_argument("--infer-file-name", type=str, required=True, help="推理文件名")
    parser.add_argument("--output-path", type=str, default="./predictions", help="预测结果保存路径")
    parser.add_argument("--batch-size", type=int, default=32, help="预测批次大小")
    parser.add_argument("--use-gpu", action="store_true", help="是否使用GPU进行预测")
    parser.add_argument("--fingerprint-type", type=str, default="morgan2048")
    parser.add_argument(
        "--molformer-ckpt-path",
        type=str,
        required=True,
        help="MolFormer pretrained checkpoint; embeddings are generated in data-path if missing",
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
        help="Conda env for MolFormer embedding extraction (subprocess)",
    )

    args = parser.parse_args()

    fix_seed(args.seed)
    
    # 创建输出目录
    Path(args.output_path).mkdir(exist_ok=True, parents=True)

    ############## 加载模型配置 ##############
    ckpt_dir = os.path.dirname(args.ckpt_path)
    config_json_path = [f for f in os.listdir(ckpt_dir) if f.endswith(".json")][0]
    config_json_path = os.path.join(ckpt_dir, config_json_path)
    argsdict = load_gnn_arguments_from_json(config_json_path)

    # print("argsdict:", argsdict)
    ############## 加载数据 ##############
    dataset = argsdict["dataset"]
    test_data, num_classes, task_type, scaler = get_dataset_train_val_test(
        infer=args.infer,
        dataset=dataset,
        dataset_dir=args.data_path,
        infer_file_name=args.infer_file_name,
        one_hot=argsdict["dataset_one_hot"],
        target_name=argsdict["dataset_target_name"],
        # 传递指纹计算参数（可选，使用默认值则无需传递）
        fingerprint_type=argsdict["fingerprint_type"],    # 计算Morgan指纹
        morgan_radius=2,              # 半径=2（ECFP4）
        morgan_nBits=2048,            # 维度=1024（默认2048）
        useChirality=True,            # 考虑手性
        num_processes=4,              # 8个并行进程（根据CPU核心数调整）
        fp_chunk_size=2000,           # 每块处理2000个样本
        molformer_ckpt_path=args.molformer_ckpt_path,
        molformer_hparams_path=args.molformer_hparams_path,
        molformer_env=args.molformer_env,
        gpu_devices=0 if args.use_gpu else None,
    )
    
    # print("type(test_data)", type(test_data))
    # print("test_data[0]", test_data[0])
    test_ids = [data.smiles_id for data in test_data]  # 样本id
    num_features = test_data[0].x.shape[-1]
    # print("num_features:", num_features)
    edge_dim = None
    if hasattr(test_data[0], "edge_attr") and test_data[0].edge_attr is not None:   #判断样本中是否具有edge_attr属性且不为空
        edge_dim = test_data[0].edge_attr.shape[-1]    # 每条边的维度
        # print("edge_dim:", edge_dim)
    # 创建数据加载器
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.use_gpu else False
    )

    ############## 加载模型 ##############
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

    ############## 配置Trainer ##############
    # 配置训练器，用于推理
    trainer_kwargs = {
        "logger": False,  # 推理时不需要日志
        "enable_checkpointing": False,  # 推理时不保存检查点
        "enable_progress_bar": True,  # 显示进度条
        "enable_model_summary": True,  # 显示模型摘要
        "max_epochs": 1,  # 推理只需要一个epoch
        "accelerator": "auto",  # 自动选择加速器
        # "devices": "auto" if args.use_gpu else 1,  # 根据use_gpu决定设备
        # "devices": 1,#gpu数量
        "devices": [2],  #指定 GPU 编号
        "callbacks": [TQDMProgressBar(refresh_rate=10)],  # 进度条回调
    }

    trainer = Trainer(**trainer_kwargs)

    ############## 使用Lightning进行预测 ##############
    # 使用trainer.predict()进行推理
    predictions = trainer.predict(
        model=model,
        dataloaders=test_loader,
        return_predictions=True  # 返回预测结果
    )

    ############## 处理预测结果 ##############
    # 仅主进程执行结果合并和保存（关键修改）
    if trainer.is_global_zero:  # 判断是否为主进程（rank 0）
        # 合并所有批次的结果（此时predictions已包含所有GPU的结果）
        all_preds = []
        for batch_pred in predictions:
            # print("batch_pred:", batch_pred)
            # 处理模型输出格式（与之前逻辑一致）
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
        # 合并所有结果（此时应包含6个样本）
        all_preds = np.concatenate(all_preds, axis=0)
        # print("all_preds:", all_preds)    
        print(f"总预测样本数：{len(all_preds)}")  # 确认输出为6

        ############## 保存预测结果 ##############
        preds_path = os.path.join(args.output_path, "infer", f"predictions_{args.infer_file_name}_({np.sum(all_preds > 0.5)}).csv")
        directory = os.path.dirname(preds_path)
        Path(directory).mkdir(exist_ok=True, parents=True)
        
        # # 仅主进程保存，避免覆盖
        # pd.DataFrame(all_preds).to_csv(preds_path, index=False, header=False, float_format="%.6f")
        # print(f"主进程已保存预测值至：{preds_path}")

    
        # 组合ID和预测值为字典列表
        results = [
            {"ID": id, "Prediction": pred} 
            for id, pred in zip(test_ids, all_preds)
        ]
        
        pd.DataFrame(results).to_csv(
            preds_path, 
            index=False, 
            float_format="%.6f"  # 保持预测值的精度格式
        )
        print(f"主进程已保存预测值至：{preds_path}")
    
    
if __name__ == "__main__":
    main()
